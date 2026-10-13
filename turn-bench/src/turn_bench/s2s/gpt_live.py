"""OpenAI GPT-Live 1 (``/v1/live/sessions``) as a service-bench adapter.

Implements ``service.adapters.base.RealtimeAdapter`` for ``gpt-live-1`` with a
Responses-delegated backend, configured as the published benchmark row
"GPT-Live 1 + GPT-6 Sol (low thinking)" (``reference-agents/pipecat-s2s/bot.py``
``_gpt_live``): backend ``gpt-6-sol`` at reasoning effort ``low``, the agent's
instructions given whole to the backend, voice ``marin``, 24 kHz PCM16.

GPT-Live is *not* the Realtime protocol, and the differences decide how this
adapter reads a turn (wire shapes from pipecat-ai 1.11.0
``services/openai/live/events.py``, confirmed by the live smoke):

* **Session.** The first client frame is ``session.start`` carrying the whole,
  mostly immutable configuration (``model``, ``instructions``, ``audio``,
  ``delegation``, optional prior ``input`` messages); the server answers
  ``session.started`` with the resolved session (recorded as ``session_ack``,
  its ``model`` as ``resolved_model``). Unset fields must be *omitted*: the
  schema is strict and ``null`` is not "absent".
* **Audio.** In: ``session.input_audio.append`` (base64, not acknowledged).
  Out: ``session.output_audio.delta``. The service resamples what it is given,
  but the rate is declared (``audio.format``) so the record says what was sent.
* **Full duplex, no turn events.** There is no ``speech_started`` /
  ``speech_stopped``, no ``response.created`` / ``response.done``, no cancel
  and no manual commit: the model listens and speaks at once and decides by
  itself when to talk and when to stop. Worse for a naive client, it **streams
  output continuously at real-time pace, silence included**. A reply is
  therefore delimited here by the audio itself: ``agent.audio.start`` (and
  ``response.start``) at the first 10 ms frame above ``VOICED_RMS`` after
  silence, ``agent.audio.end`` (and ``response.done``, status ``silence``) once
  ``HANGOVER_MS`` of sub-threshold output follows; the sample indices point at
  the voiced span. The whole stream, silence included, is still recorded in the
  agent timeline so sample positions stay arrival-anchored.
* **Yielding.** Being talked over is handled by the model with no signal, so
  ``agent.interrupted`` is never emitted; a yield shows up only as the audio
  going quiet (an ``agent.audio.end``).
* **Transcripts** are frame-aligned fragments (``session.input_transcript.delta``
  / ``session.output_transcript.delta`` with ``start_ms``/``end_ms`` on the
  session timeline); fragments are kept verbatim in the raw log and joined per
  voiced reply / per caller burst for the transcripts.
* **Usage/cost.** ``session.usage.updated`` carries cumulative live audio
  seconds (billed per minute); backend token usage arrives only inside
  ``response.event`` envelopes (nested ``response.completed``). Both are kept in
  ``state()["usage"]``. ``session.close`` asks for a graceful shutdown that
  ends with ``session.closed`` and the final usage; close waits briefly for it.
* The reference agent appends a delegation section to the live model's prompt
  (backchannel / interruption policy and when to delegate). It is off by
  default here (``prompt_addendum=False``) because it coaches the behaviour a
  turn-taking benchmark measures and these sessions have no tools; pass the
  text to reproduce the agent row's prompt. Either way ``state()`` records it.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Awaitable, Callable

import numpy as np

from turn_bench.author import service_bridge

service_bridge._ensure_path()

from service import events as ev  # noqa: E402
from service.adapters.base import AdapterError, RealtimeAdapter, SessionConfig  # noqa: E402

URL = "wss://api.openai.com/v1/live/sessions"
SAMPLE_RATE = 24000
CONNECT_TIMEOUT_S = 20.0
CLOSE_TIMEOUT_S = 5.0
DEFAULT_MODEL = "gpt-live-1"
DEFAULT_VOICE = "marin"
BACKEND_MODEL = "gpt-6-sol"
BACKEND_REASONING = "low"
CREDENTIAL_ENV = "OPENAI_API_KEY"

#: Normalized event marking a reply's first voiced frame; additive to ``service.events``.
RESPONSE_START = "response.start"

#: Voiced-frame gate over the continuous output stream: 10 ms frames, RMS in
#: int16 units (100 ~ -50 dBFS, well above the model's silence fill and well
#: below speech). A reply ends after HANGOVER_MS of unvoiced frames, longer than
#: ordinary inter-word gaps.
FRAME_MS = 10
VOICED_RMS = 100.0
HANGOVER_MS = 600

Connector = Callable[[str, dict[str, str]], Awaitable[Any]]


async def _websocket_connect(url: str, headers: dict[str, str]) -> Any:
    import websockets  # lazy: offline tests inject a fake connector

    return await websockets.connect(url, additional_headers=headers, open_timeout=CONNECT_TIMEOUT_S, max_size=None)


class GPTLiveAdapter(RealtimeAdapter):
    name = "gpt-live"
    input_rate = SAMPLE_RATE
    output_rate = SAMPLE_RATE
    supports_manual_commit = False   # the model decides when to speak; there is no commit
    supports_text_modality = False   # no text-in turn on this protocol
    emits_vad_events = False         # no speech_started / speech_stopped at all

    @classmethod
    def unsupported_reason(cls, config: SessionConfig) -> str | None:
        reason = super().unsupported_reason(config)
        if reason:
            return reason
        if config.turn_detection.mode != "server_vad" or config.turn_detection.silence_duration_ms is not None:
            return "gpt-live has no configurable turn detection; only its native behaviour runs"
        if config.tools:
            return "gpt-live tools run on the delegated backend; not wired in this adapter"
        return None

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        backend_model: str = BACKEND_MODEL,
        backend_reasoning: str = BACKEND_REASONING,
        prompt_addendum: str | None = None,
        connector: Connector | None = None,
        url: str = URL,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.backend_model = backend_model
        self.backend_reasoning = backend_reasoning
        self.prompt_addendum = prompt_addendum
        self._connector = connector or _websocket_connect
        self._url = url
        self._ws: Any = None
        self._started = asyncio.Event()
        self._closed_by_server = asyncio.Event()
        self._startup_error: Any = None
        self.resolved_model: str | None = None
        self.live_seconds: float | None = None
        self.backend_usage: list[dict[str, Any]] = []
        self.delegations: list[dict[str, Any]] = []
        # Voiced-frame gate state (see module docstring).
        self._frame_samples = SAMPLE_RATE * FRAME_MS // 1000
        self._pending = b""              # output bytes not yet a whole 10 ms frame
        self._gate_sample = 0            # sample index of the next frame the gate reads
        self._last_voiced_end = 0        # sample just past the last voiced frame
        self._unvoiced_run = 0           # consecutive unvoiced frames while speaking
        self._reply_text: list[str] = []
        self._caller_burst: list[str] = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        try:
            self._ws = await self._connector(self._url, {"Authorization": f"Bearer {self._api_key}"})
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"{self.name} connect failed: {type(exc).__name__}: {exc}") from exc

        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        self.session_sent = self._session_payload()
        await self._send_json({"type": "session.start", "session": self.session_sent})
        waiter = asyncio.ensure_future(self._started.wait())
        closed = asyncio.ensure_future(self.closed.wait())
        done, pending = await asyncio.wait({waiter, closed}, timeout=CONNECT_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if not self._started.is_set():
            reason = self._startup_error or ("socket closed" if self.closed.is_set() else "timeout")
            raise AdapterError(f"{self.name} never started the session: {reason}")

    def _session_payload(self) -> dict[str, Any]:
        config = self.config
        instructions = config.instructions or ""
        if self.prompt_addendum:
            instructions = f"{instructions.rstrip()}\n\n{self.prompt_addendum}"
        responses: dict[str, Any] = {"model": self.backend_model, "reasoning": {"effort": self.backend_reasoning}}
        if config.instructions:
            # The backend does the task, so it gets the agent's instructions whole.
            responses["instructions"] = config.instructions
        session: dict[str, Any] = {
            "model": self.model,
            "audio": {
                "output": {"voice": config.voice or DEFAULT_VOICE},
                "format": {"type": "audio/pcm", "rate": self.input_rate},
            },
            "delegation": {"type": "responses", "responses": responses},
        }
        if instructions:
            session["instructions"] = instructions
        if config.first_message:
            session["input"] = [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": config.first_message}]}
            ]
        return session

    async def close(self) -> None:
        if self._ws is not None and self._started.is_set() and not self._closed_by_server.is_set():
            try:
                await self._send_json({"type": "session.close"})
                await asyncio.wait_for(self._closed_by_server.wait(), CLOSE_TIMEOUT_S)
            except Exception:  # noqa: BLE001 -- best effort: the final usage is a nicety
                pass
        self._flush_gate(end_of_stream=True)
        if self._receiver:
            self._receiver.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        self.closed.set()
        self.log.emit(ev.SESSION_CLOSED, by="harness")

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        self.log.raw(payload, direction="out")
        await self._ws.send(json.dumps(payload))

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        await self._ws.send(
            json.dumps({"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode()})
        )

    async def _commit(self) -> None:
        raise AdapterError(f"{self.name} has no manual commit")

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        await self._send_json(
            {
                "type": "response.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output if isinstance(output, str) else json.dumps(output),
                },
            }
        )
        await self._send_json({"type": "response.create"})

    # ── receiving ────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                self._handle(json.loads(message))
            self.log.emit(ev.SESSION_CLOSED, by="provider", code=getattr(self._ws, "close_code", None),
                          reason=getattr(self._ws, "close_reason", None))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- includes websockets.ConnectionClosed
            self.log.emit(ev.SESSION_ERROR, error=f"{type(exc).__name__}: {exc}")
        finally:
            self.closed.set()

    def _handle(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type", "")
        if kind != "session.output_audio.delta":
            self.log.raw(payload)  # audio is in the wav, not the log

        if kind == "session.output_audio.delta":
            self._on_output_audio(base64.b64decode(payload.get("delta") or ""))
        elif kind == "session.started":
            session = payload.get("session") or {}
            self.session_ack = session
            self.session_id = session.get("id")
            self.resolved_model = session.get("model")
            self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model, resolved_model=self.resolved_model)
            self.log.emit(ev.SESSION_CONFIGURED, turn_detection="native")
            self._started.set()
        elif kind == "session.updated":
            self.session_ack = payload.get("session") or self.session_ack
        elif kind == "session.output_transcript.delta":
            if payload.get("delta"):
                self._reply_text.append(payload["delta"])
        elif kind == "session.input_transcript.delta":
            delta = payload.get("delta") or ""
            if delta:
                self._caller_burst.append(delta)
                self.log.emit("caller.transcript.delta", text=delta,
                              start_ms=payload.get("start_ms"), end_ms=payload.get("end_ms"))
        elif kind == "session.delegation.created":
            delegation = payload.get("delegation") or {}
            self.delegations.append(delegation)
            self.log.emit("agent.delegation", delegation_id=delegation.get("id"), target=delegation.get("target"))
        elif kind == "response.event":
            inner = payload.get("event") or {}
            if inner.get("type") in ("response.completed", "response.done", "response.incomplete", "response.failed"):
                usage = (inner.get("response") or {}).get("usage")
                if usage:
                    self.backend_usage.append(usage)
        elif kind == "session.usage.updated":
            self.live_seconds = (payload.get("usage") or {}).get("seconds", self.live_seconds)
        elif kind == "session.closed":
            usage = payload.get("usage") or {}
            if usage.get("seconds") is not None:
                self.live_seconds = usage["seconds"]
            self._closed_by_server.set()
        elif kind == "error":
            if not self._started.is_set():
                self._startup_error = payload.get("error")
            self.log.emit(ev.SESSION_ERROR, error=payload.get("error"))

    # ── the voiced-frame gate ────────────────────────────────────────────────

    def _on_output_audio(self, pcm: bytes) -> None:
        """Record every output chunk at arrival; open / close replies on voiced frames."""
        if not pcm:
            return
        at = self.log.clock.now()
        chunk = self.agent_timeline.record_pcm(pcm, at)
        self.agent_pcm.extend(pcm)
        data = self._pending + pcm
        frame_bytes = self._frame_samples * 2
        n_frames = len(data) // frame_bytes
        self._pending = data[n_frames * frame_bytes:]
        if not n_frames:
            return
        frames = np.frombuffer(data[: n_frames * frame_bytes], dtype="<i2").astype(np.float64)
        rms = np.sqrt(np.mean(frames.reshape(n_frames, self._frame_samples) ** 2, axis=1))
        for value in rms:
            start = self._gate_sample
            self._gate_sample += self._frame_samples
            if value >= VOICED_RMS:
                self._unvoiced_run = 0
                self._last_voiced_end = self._gate_sample
                if not self._speaking:
                    self._speaking = True
                    self.log.emit(RESPONSE_START, at=at, sample=start)
                    self.log.emit(
                        ev.AGENT_AUDIO_START,
                        at=at,
                        sample=start,
                        chunk_first_sample=chunk.first_sample,
                        chunk_ms=round(1000 * chunk.n_samples / self.output_rate, 2),
                    )
            elif self._speaking:
                self._unvoiced_run += 1
                if self._unvoiced_run * FRAME_MS >= HANGOVER_MS:
                    self._end_reply(at, reason="silence")

    def _end_reply(self, at: float, reason: str) -> None:
        self._speaking = False
        self._unvoiced_run = 0
        self.log.emit(ev.AGENT_AUDIO_END, at=at, sample=self._last_voiced_end, reason=reason)
        text = "".join(self._reply_text).strip()
        self._reply_text.clear()
        if text:
            self.agent_text.append(text)
            self.log.emit(ev.AGENT_TRANSCRIPT, text=text)
        if self._caller_burst:
            caller = "".join(self._caller_burst).strip()
            self._caller_burst.clear()
            if caller:
                self.caller_text.append(caller)
        self.log.emit(ev.RESPONSE_DONE, status=reason, usage=None)

    def _flush_gate(self, *, end_of_stream: bool) -> None:
        if self._speaking and end_of_stream:
            self._end_reply(self.log.clock.now(), reason="session_end")
        leftover_caller = "".join(self._caller_burst).strip()
        if leftover_caller:
            self.caller_text.append(leftover_caller)
            self._caller_burst.clear()
        leftover_reply = "".join(self._reply_text).strip()
        if leftover_reply:
            self.agent_text.append(leftover_reply)
            self._reply_text.clear()

    def state(self) -> dict[str, Any]:
        state = super().state()
        state["resolved_model"] = self.resolved_model
        state["backend"] = {"model": self.backend_model, "reasoning": self.backend_reasoning}
        state["prompt_addendum"] = self.prompt_addendum
        state["usage"] = {"live_audio_seconds": self.live_seconds, "backend": list(self.backend_usage)}
        state["delegations"] = list(self.delegations)
        state["reply_gate"] = {"frame_ms": FRAME_MS, "voiced_rms": VOICED_RMS, "hangover_ms": HANGOVER_MS}
        return state


def provider_entry():
    """A ``service.registry.ProviderEntry`` for this adapter (not registered;
    a coordinator adds it to whichever registry the run reads)."""
    from service.registry import ProviderEntry

    return ProviderEntry(
        key="gpt-live",
        adapter=GPTLiveAdapter,
        default_model=DEFAULT_MODEL,
        credential_env=CREDENTIAL_ENV,
        default_voice=DEFAULT_VOICE,
        discloses=(
            f"delegated backend {BACKEND_MODEL} at reasoning {BACKEND_REASONING}; backend tokens billed on top of "
            "per-minute live audio",
            "full duplex with no turn events: replies delimited by the output audio's own energy",
        ),
    )
