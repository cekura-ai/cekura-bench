"""Smallest AI Hydra over ``/waves/v1/s2s``.

Hydra's events are OpenAI Realtime's names for the same things (audio deltas,
``speech_started`` / ``speech_stopped``, streamed tool-call arguments,
``response.done`` with a status), so this adapter inherits the event handling
and overrides what differs:

- **Handshake.** The server speaks first with ``session.created`` (the session
  id at the top level, not under ``session``), then waits for exactly one
  ``session.configure``. ``session.configured`` echoes what was applied and is
  the acknowledgement.
- **Credential in the URL.** The key is a query parameter, so it is scrubbed
  from anything this adapter raises. Nothing else here records the URL.
- **Turn detection has no settings.** The server decides every turn and takes
  no silence duration, threshold or semantic mode. Hydra therefore runs only the
  bench's default ``server_vad`` cell. The duration in that cell's label was
  not sent, and the acknowledged session shows it. Every other turn
  configuration is declared unsupported rather than run under a label that
  does not describe it. There is no manual commit and no text output.
- **Transcripts.** Hydra documents none. It does stream the text of each reply
  in ``response.output_audio_transcript.delta`` frames, with no ``.done``, so a
  reply's text is collected per response and recorded when the response ends.
  That is the agent's transcript, as it is on the other rows. A reply that
  carried audio but no text falls back to Smallest's own speech-to-text (Pulse
  Pro), so a frame Hydra stops sending costs a transcriber, not a transcript.
  The caller is never transcribed by Hydra: each completed caller turn goes to
  Pulse Pro, with the same key, which is the vendor's documented answer and the
  same arrangement as the other rows, whose caller transcripts also come from
  the vendor's separate transcription model. Every transcript event names its
  source.
- **One reply per tool turn.** The model can call several tools in one turn,
  and a ``response.create`` after each result makes it narrate before all
  results are in. Results are posted as they come, and a single
  ``response.create`` follows the last of them after a short quiet window.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any

import websockets

from service import events as ev
from service.adapters.base import AdapterError, SessionConfig
from service.adapters.openai_realtime import CONNECT_TIMEOUT_S, OpenAIRealtimeAdapter

URL = "wss://api.smallest.ai/waves/v1/s2s"
TRANSCRIBE_URL = "https://api.smallest.ai/waves/v1/stt/"
TRANSCRIPTION_MODEL = "pulse-pro"
PROVIDER_TEXT = "provider"  # the source named on an agent transcript Hydra sent itself
TRANSCRIBE_TIMEOUT_S = 30.0
# The vendor's figure for collecting several tool results before asking for one reply.
RESPONSE_CREATE_DEBOUNCE_S = 0.2
# Caller audio kept from before the endpointer announced speech: the announcement
# arrives after the first word has begun.
PREROLL_S = 1.0
# How long closing waits for transcripts of speech already received.
CLOSE_DRAIN_S = 10.0
TURN_DETECTION_APPLIED = "server default; no settings exist"


class SmallestHydraAdapter(OpenAIRealtimeAdapter):
    name = "smallest-hydra"
    input_rate = 16000
    output_rate = 24000
    supports_manual_commit = False
    supports_text_modality = False
    emits_vad_events = True

    url = URL

    def __init__(self, *, model: str = "hydra-v1.1", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._response_pcm: dict[str, bytearray] = {}
        self._response_text: dict[str, list[str]] = {}
        # Caller audio up to here has been handed to the transcriber. The
        # endpointer's own offsets are not used to cut turns: they run several
        # hundred milliseconds ahead of the audio actually sent, and a cut taken
        # from them loses the turn's first word.
        self._caller_cursor = 0
        # Where the turn being spoken starts: a second before the endpointer
        # heard it. The silence before a turn is left out, because the
        # transcriber invents words in long stretches of it.
        self._caller_mark: int | None = None
        # (side, ref, audio, rate, text): text already known skips the transcriber,
        # and still waits its turn so transcripts stay in the order turns happened.
        self._transcribe_queue: asyncio.Queue[tuple[str, str, bytes, int, str | None]] = asyncio.Queue()
        self._transcribing = 0
        self._transcriber: asyncio.Task | None = None
        self._response_create: asyncio.Task | None = None

    @classmethod
    def unsupported_reason(cls, config: SessionConfig) -> str | None:
        reason = super().unsupported_reason(config)
        if reason:
            return reason
        detection = config.turn_detection
        if (
            detection.mode != "server_vad"
            or detection.threshold is not None
            or detection.prefix_padding_ms is not None
            or detection.silence_duration_ms not in (None, 500)
        ):
            return f"{cls.name} decides turns on its server with no settings; it runs the default server_vad cell only"
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        query = urllib.parse.urlencode({"model": self.model, "api_key": self._api_key})
        try:
            self._ws = await websockets.connect(f"{self.url}?{query}", open_timeout=CONNECT_TIMEOUT_S, max_size=None)
        except Exception as exc:  # noqa: BLE001 -- the reason is what we want to publish
            raise AdapterError(f"{self.name} connect failed: {self._redact(str(exc))}") from None

        first = json.loads(await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT_S))
        self.log.raw(first)
        if first.get("type") == "warning":  # a deprecated model tag warns before the session opens
            first = json.loads(await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT_S))
            self.log.raw(first)
        if first.get("type") != "session.created":
            raise AdapterError(f"{self.name} refused the session: {first}")
        self.session_id = first.get("session_id")
        self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model)

        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        self._transcriber = asyncio.create_task(self._transcribe_worker(), name=f"{self.name}-transcribe")
        self.session_sent = self._session_payload()
        await self._send_json({"type": "session.configure", "session": self.session_sent})
        try:
            await asyncio.wait_for(self._configured.wait(), CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise AdapterError(f"{self.name} never acknowledged session.configure") from exc
        if self.config.first_message:
            await self._send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": self.config.first_message}],
                    },
                }
            )

    def _redact(self, text: str) -> str:
        if not self._api_key:
            return text
        # The URL carries the key as urlencode wrote it, which differs from the raw key
        # wherever the key has a character that needs escaping.
        for form in (self._api_key, urllib.parse.quote_plus(self._api_key), urllib.parse.quote(self._api_key, safe="")):
            text = text.replace(form, "<redacted>")
        return text

    def _session_payload(self) -> dict[str, Any]:
        config = self.config
        session: dict[str, Any] = {
            "instructions": config.instructions,
            "generate_initial_response": False,
            "output_audio_sample_rate": self.output_rate,
        }
        if config.voice:
            session["voice"] = config.voice
        if config.tools:
            session["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in config.tools
            ]
        return session

    async def close(self) -> None:
        if self._response_create:
            self._response_create.cancel()
        # Speech already received is still owed its transcript.
        deadline = time.monotonic() + CLOSE_DRAIN_S
        while self._transcribing and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if self._transcribing:
            self.log.emit(ev.SESSION_ERROR, error=f"{self._transcribing} transcript(s) unfinished at close",
                          where="transcriber")
        if self._transcriber:
            self._transcriber.cancel()
        await super().close()

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output if isinstance(output, str) else json.dumps(output),
                },
            }
        )
        if self._response_create:
            self._response_create.cancel()
        self._response_create = asyncio.create_task(self._create_response_after_quiet())

    async def _create_response_after_quiet(self) -> None:
        await asyncio.sleep(RESPONSE_CREATE_DEBOUNCE_S)
        await self._send_json({"type": "response.create"})

    # ── receiving ────────────────────────────────────────────────────────────

    def _handle(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type", "")
        if kind == "response.output_audio.delta":
            self._response_pcm.setdefault(payload.get("response_id", ""), bytearray()).extend(
                base64.b64decode(payload["delta"])
            )
        elif kind == "response.output_audio_transcript.delta":
            self._response_text.setdefault(payload.get("response_id", ""), []).append(payload.get("delta") or "")
        elif kind == "session.configured":
            self.log.raw(payload)
            self.session_ack = payload.get("session")
            self._configured.set()
            self.log.emit(
                ev.SESSION_CONFIGURED,
                turn_detection=self.config.turn_detection.label,
                turn_detection_applied=TURN_DETECTION_APPLIED,
            )
            return
        elif kind == "input_audio_buffer.speech_started" and self._caller_mark is None:
            preroll = int(PREROLL_S * self.input_rate) * 2
            self._caller_mark = max(self._caller_cursor, len(self.caller_pcm) - preroll)
        elif kind == "conversation.item.done":
            self._on_item_done(payload.get("item") or {})
        elif kind == "response.done":
            response = payload.get("response") or {}
            ref = response.get("id", "")
            pcm = self._response_pcm.pop(ref, None)
            text = " ".join(" ".join(self._response_text.pop(ref, [])).split())
            if text:
                self._queue_transcription("agent", ref, b"", self.output_rate, text=text)
            elif pcm:
                self._queue_transcription("agent", ref, bytes(pcm), self.output_rate)
        super()._handle(payload)

    def _on_item_done(self, item: dict[str, Any]) -> None:
        """A caller turn the endpointer kept is transcribed from where it started.

        A turn it discarded as noise moves nothing, so its audio is heard again
        with the next kept turn rather than lost.
        """
        if item.get("role") != "user" or item.get("type") != "message" or item.get("status") != "completed":
            return
        if not self.config.transcribe_input:
            return
        start = self._caller_cursor if self._caller_mark is None else self._caller_mark
        end = len(self.caller_pcm)
        self._caller_cursor, self._caller_mark = end, None
        if end > start:
            self._queue_transcription("caller", item.get("id", ""), bytes(self.caller_pcm[start:end]), self.input_rate)

    # ── transcription ────────────────────────────────────────────────────────

    @property
    def transcripts_pending(self) -> bool:
        return self._transcribing > 0

    def _queue_transcription(self, side: str, ref: str, pcm: bytes, rate: int, text: str | None = None) -> None:
        self._transcribing += 1
        self._transcribe_queue.put_nowait((side, ref, pcm, rate, text))

    async def _transcribe_worker(self) -> None:
        """One request at a time, so transcripts land in the order the turns happened."""
        while True:
            side, ref, pcm, rate, text = await self._transcribe_queue.get()
            source = PROVIDER_TEXT if text is not None else TRANSCRIPTION_MODEL
            started = time.monotonic()
            try:
                if text is None:
                    text = await asyncio.to_thread(self._transcribe, pcm, rate)
            except Exception as exc:  # noqa: BLE001 -- a lost transcript is recorded, not fatal
                self.log.emit(ev.SESSION_ERROR, error=self._redact(repr(exc)), where="transcriber", side=side, ref=ref)
                text = None
            finally:
                self._transcribing -= 1
            if text is None:
                continue
            meta = {"transcriber": source, "ref": ref, "transcribe_s": round(time.monotonic() - started, 3)}
            if side == "agent":
                self.agent_text.append(text)
                self.log.emit(ev.AGENT_TRANSCRIPT, text=text, **meta)
            else:
                self.caller_text.append(text)
                self.log.emit(ev.CALLER_TRANSCRIPT, text=text, **meta)

    def _transcribe(self, pcm: bytes, rate: int) -> str:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm)
        request = urllib.request.Request(
            f"{TRANSCRIBE_URL}?{urllib.parse.urlencode({'model': TRANSCRIPTION_MODEL, 'language': 'en'})}",
            data=buffer.getvalue(),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=TRANSCRIBE_TIMEOUT_S) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:  # surface the vendor's reason, not a stack trace
            raise AdapterError(f"{TRANSCRIPTION_MODEL} HTTP {exc.code}: {exc.read()[:200]!r}") from None
        return (body.get("transcription") or "").strip()

    def state(self) -> dict[str, Any]:
        state = super().state()
        state["transcripts"]["transcriber"] = {
            "agent": f"{PROVIDER_TEXT}; {TRANSCRIPTION_MODEL} for a reply sent without text",
            "caller": TRANSCRIPTION_MODEL,
        }
        state["turn_detection_applied"] = TURN_DETECTION_APPLIED
        return state
