"""Gemini Live over its raw websocket (``BidiGenerateContent``).

Written against the wire rather than the vendor SDK, for the same reason no
framework sits in the measured path: the SDK's own buffering and reconnection
would be read as provider behaviour. The shapes below were read off live
frames, not documentation.

Four things this protocol does differently from an OpenAI-shaped one, each of
which changed the harness rather than only this file:

* **Frames arrive as bytes.** Every server message is a binary frame carrying
  JSON, not a text frame.
* **No turn-detection events.** The provider never says when it heard speech
  start or stop; it only says ``interrupted`` when it yielded. Probes that need
  the provider's own endpointer decisions have to void here, and say why.
* **Faster than realtime, and the first chunk is large.** The first audio frame
  of a reply has been close to a second long, and the rest arrives well ahead
  of playout. ``turnComplete`` is paced to when playout would finish, seconds
  after the last audio frame. A caller that took "no more audio arriving" as
  "the agent has finished" would be talking over a reply the listener was still
  hearing -- which is why the caller's control clock follows the playout model.
* **It thinks by default.** A reply may open with a reasoning part flagged as a
  thought before any audio. That is part of the configuration a customer gets,
  so it is measured as such and the thought tokens are recorded, not suppressed.

Input is 16 kHz, output 24 kHz; the corpus master is resampled by the published
filter on the way in.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import websockets

from service import events as ev
from service.adapters.base import AdapterError, RealtimeAdapter, SessionConfig

URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
CONNECT_TIMEOUT_S = 20.0


class GeminiLiveAdapter(RealtimeAdapter):
    name = "gemini-live"
    input_rate = 16000
    output_rate = 24000
    supports_manual_commit = True
    supports_text_modality = True
    emits_vad_events = False

    def __init__(self, *, model: str = "gemini-2.5-flash-native-audio-preview-12-2025", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._ws: websockets.ClientConnection | None = None
        self._agent_transcript: list[str] = []   # fragments of the reply under way
        self._caller_transcript: list[str] = []  # fragments of what it heard us say
        self._agent_text_parts: list[str] = []   # text-modality reply under way
        self._pending_usage: dict[str, Any] | None = None
        self._call_names: dict[str, str] = {}    # call id -> tool name, for the response
        self._open_window_next = False           # manual mode: bracket the coming utterance

    @classmethod
    def unsupported_reason(cls, config: SessionConfig) -> str | None:
        reason = super().unsupported_reason(config)
        if reason:
            return reason
        detection = config.turn_detection
        if detection.mode == "semantic_vad":
            return "gemini-live has no semantic turn detection"
        if detection.threshold is not None:
            return "gemini-live has no numeric VAD threshold; it exposes sensitivity levels"
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        try:
            self._ws = await websockets.connect(
                URL,
                additional_headers={"x-goog-api-key": self._api_key},
                open_timeout=CONNECT_TIMEOUT_S,
                max_size=None,
            )
        except Exception as exc:  # noqa: BLE001 -- the reason is what we want to publish
            raise AdapterError(f"gemini-live connect failed: {exc}") from exc

        self.session_sent = self._setup_payload()
        await self._send_json({"setup": self.session_sent})
        try:
            first = _decode(await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT_S))
        except asyncio.TimeoutError as exc:
            raise AdapterError("gemini-live never completed setup") from exc
        self.log.raw(first)
        if "setupComplete" not in first:
            raise AdapterError(f"gemini-live refused the session: {first}")
        # The Live API acknowledges setup without echoing the configuration, so
        # the record says exactly that rather than leaving the field looking
        # like an omission on our side.
        self.session_ack = {"setupComplete": first["setupComplete"], "echoes_configuration": False}
        self.log.emit(ev.SESSION_OPEN, session_id=None, model=self.model)
        self.log.emit(ev.SESSION_CONFIGURED, turn_detection=self.config.turn_detection.label)

        self._receiver = asyncio.create_task(self._receive_loop(), name="gemini-live-recv")
        if self.config.first_message:
            await self._send_json(
                {
                    "clientContent": {
                        "turns": [{"role": "model", "parts": [{"text": self.config.first_message}]}],
                        "turnComplete": False,
                    }
                }
            )

    def _setup_payload(self) -> dict[str, Any]:
        config = self.config
        # The native-audio models refuse a TEXT response modality outright, so
        # the text arm here is text *in*, audio out: the caller's words arrive as
        # text and the reply is graded from the provider's transcript of its own
        # speech. That still removes the caller-side speech pathway, which is
        # what the arm exists to isolate.
        generation: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if config.voice:
            generation["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": config.voice}}}
        if config.temperature is not None:
            generation["temperature"] = config.temperature

        setup: dict[str, Any] = {
            "model": f"models/{self.model}",
            "generationConfig": generation,
            "realtimeInputConfig": {"automaticActivityDetection": self._activity_detection()},
        }
        if config.instructions:
            setup["systemInstruction"] = {"parts": [{"text": config.instructions}]}
        if config.tools:
            setup["tools"] = [
                {
                    "functionDeclarations": [
                        {"name": t.name, "description": t.description, "parameters": t.parameters}
                        for t in config.tools
                    ]
                }
            ]
        setup["outputAudioTranscription"] = {}
        if config.modality == "audio" and config.transcribe_input:
            setup["inputAudioTranscription"] = {}
        return setup

    def _activity_detection(self) -> dict[str, Any]:
        detection = self.config.turn_detection
        if detection.is_manual:
            return {"disabled": True}
        payload: dict[str, Any] = {"disabled": False}
        if detection.silence_duration_ms is not None:
            payload["silenceDurationMs"] = detection.silence_duration_ms
        if detection.prefix_padding_ms is not None:
            payload["prefixPaddingMs"] = detection.prefix_padding_ms
        return payload

    async def close(self) -> None:
        if self._receiver:
            self._receiver.cancel()
        if self._ws:
            await self._ws.close()
        self._flush_transcripts()
        self.closed.set()
        self.log.emit(ev.SESSION_CLOSED)

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        self.log.raw(payload, direction="out")
        await self._ws.send(json.dumps(payload))

    def note_utterance_start(self) -> None:
        if self.config.turn_detection.is_manual:
            self._open_window_next = True

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        if self._open_window_next:
            # With automatic detection off the model hears only what sits inside
            # an activity window, and ``activityStart`` itself means "the caller
            # began speaking". Sent at connect time, or re-sent straight after a
            # commit, it interrupts the reply the commit just asked for; sent
            # here it brackets exactly the authored utterance.
            self._open_window_next = False
            await self._send_json({"realtimeInput": {"activityStart": {}}})
        # Not routed through _send_json: the base64 would double the raw log for
        # no audit value. The caller timeline and caller.wav already have it.
        await self._ws.send(
            json.dumps(
                {
                    "realtimeInput": {
                        "audio": {
                            "data": base64.b64encode(pcm).decode(),
                            "mimeType": f"audio/pcm;rate={self.input_rate}",
                        }
                    }
                }
            )
        )

    async def _commit(self) -> None:
        """Close the activity window at this sample. The next utterance opens the next."""
        await self._send_json({"realtimeInput": {"activityEnd": {}}})
        self._open_window_next = False

    async def _send_text(self, text: str) -> None:
        await self._send_json(
            {"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": True}}
        )

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        # The response is a Struct, so a bare value is wrapped rather than sent as is.
        response = output if isinstance(output, dict) else {"result": output}
        await self._send_json(
            {
                "toolResponse": {
                    "functionResponses": [
                        {"id": call_id, "name": self._call_names.get(call_id, ""), "response": response}
                    ]
                }
            }
        )

    # ── receiving ────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                self._handle(_decode(message))
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            self.log.emit(ev.SESSION_CLOSED, code=exc.code, reason=str(exc.reason))
            self.closed.set()
        except Exception as exc:  # noqa: BLE001
            self.log.emit(ev.SESSION_ERROR, error=repr(exc))
            self.closed.set()

    def _handle(self, payload: dict[str, Any]) -> None:
        content = payload.get("serverContent") or {}
        audio_parts = [p for p in (content.get("modelTurn") or {}).get("parts", []) if "inlineData" in p]
        if not audio_parts or len(payload) > 1 or set(content) - {"modelTurn"}:
            self.log.raw(_without_audio(payload))  # audio bytes live in the wav, not the log

        for part in (content.get("modelTurn") or {}).get("parts", []):
            if "inlineData" in part:
                self._on_agent_audio(base64.b64decode(part["inlineData"]["data"]))
            elif part.get("text"):
                if part.get("thought"):
                    self.log.emit(ev.AGENT_THOUGHT, text=part["text"])
                else:
                    self._agent_text_parts.append(part["text"])

        if content.get("outputTranscription", {}).get("text"):
            self._agent_transcript.append(content["outputTranscription"]["text"])
        if content.get("inputTranscription", {}).get("text"):
            self._caller_transcript.append(content["inputTranscription"]["text"])

        if content.get("interrupted"):
            self._on_agent_interrupted()
        if content.get("generationComplete"):
            self._on_agent_audio_done(reason="generation_complete")

        if payload.get("usageMetadata"):
            self._pending_usage = _flatten_usage_metadata(payload["usageMetadata"])

        if content.get("turnComplete"):
            self._on_agent_audio_done(reason="turn_complete")
            self._flush_transcripts()
            self.log.emit(ev.RESPONSE_DONE, status="completed", usage=self._pending_usage)
            self._pending_usage = None

        for call in (payload.get("toolCall") or {}).get("functionCalls", []):
            record = {
                "name": call.get("name", ""),
                "call_id": call.get("id", ""),
                "arguments": call.get("args") or {},
            }
            self._call_names[record["call_id"]] = record["name"]
            self.tool_calls.append(record)
            self.log.emit(ev.TOOL_CALL, **record)

        if "goAway" in payload:
            self.log.emit(ev.SESSION_ERROR, error={"goAway": payload["goAway"]})

    def _flush_transcripts(self) -> None:
        """Fragments become one entry per turn, so ``agent_text[-1]`` is a whole reply."""
        if self._agent_transcript or self._agent_text_parts:
            text = "".join(self._agent_transcript) or "".join(self._agent_text_parts)
            self._agent_transcript.clear()
            self._agent_text_parts.clear()
            self.agent_text.append(text)
            self.log.emit(ev.AGENT_TRANSCRIPT, text=text)
        if self._caller_transcript:
            text = "".join(self._caller_transcript).strip()
            self._caller_transcript.clear()
            self.caller_text.append(text)
            self.log.emit(ev.CALLER_TRANSCRIPT, text=text)


def _decode(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, (bytes, bytearray)):
        message = message.decode("utf-8")
    return json.loads(message)


def _without_audio(payload: dict[str, Any]) -> dict[str, Any]:
    """The frame with audio payloads replaced by their size, for the raw log."""
    content = payload.get("serverContent")
    if not content or not content.get("modelTurn"):
        return payload
    parts = []
    for part in content["modelTurn"].get("parts", []):
        if "inlineData" in part:
            data = part["inlineData"]
            parts.append({"inlineData": {"mimeType": data.get("mimeType"), "bytes": len(base64.b64decode(data.get("data", "")))}})
        else:
            parts.append(part)
    return {**payload, "serverContent": {**content, "modelTurn": {**content["modelTurn"], "parts": parts}}}


def _flatten_usage_metadata(usage: dict[str, Any]) -> dict[str, Any]:
    """The provider's usage block as counts the metrics layer can sum.

    The per-modality breakdown arrives as a list of ``{modality, tokenCount}``
    rows, which a generic flattener cannot add up. It becomes a nested dict
    keyed by modality, so audio and text tokens stay separable at costing time.
    """
    flat: dict[str, Any] = {}
    for key, value in usage.items():
        if isinstance(value, int) and not isinstance(value, bool):
            flat[key] = value
        elif isinstance(value, list):
            flat[key] = {
                str(row.get("modality", "?")): int(row.get("tokenCount", 0))
                for row in value
                if isinstance(row, dict)
            }
    return flat
