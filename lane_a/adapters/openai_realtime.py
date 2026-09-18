"""OpenAI Realtime (``/v1/realtime``) -- the reference implementation of an adapter.

One adapter serves every ``gpt-realtime-*`` model: the wire protocol is the unit
of work, the model is a parameter. Note that ``gpt-live-1`` is *not* served here.
It is refused on this endpoint ("not supported in realtime mode") and lives at
``/v1/live/sessions`` as a separate product with a delegated backend text model,
so it needs its own adapter and its published row has to disclose backend cost.

The session shape below was read off a live ``session.created`` rather than from
documentation, because the GA schema moved (``output_modalities``, nested
``audio.input`` / ``audio.output``) and a stale guess fails as a silent default
rather than as an error.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import websockets

from lane_a import events as ev
from lane_a.adapters.base import AdapterError, RealtimeAdapter, SessionConfig, parse_arguments

URL = "wss://api.openai.com/v1/realtime"
CONNECT_TIMEOUT_S = 20.0

# The GA names, with the pre-GA aliases kept: a provider that renames an event
# should show up as an explicit mapping here, never as a missing measurement.
_AUDIO_DELTA = {"response.output_audio.delta", "response.audio.delta"}
_AUDIO_DONE = {"response.output_audio.done", "response.audio.done"}
_AGENT_TEXT_DONE = {
    "response.output_audio_transcript.done",
    "response.audio_transcript.done",
    "response.output_text.done",
    "response.text.done",
}


class OpenAIRealtimeAdapter(RealtimeAdapter):
    name = "openai-realtime"
    input_rate = 24000   # this service does not resample; feeding it 16 kHz makes it hear us fast
    output_rate = 24000
    supports_manual_commit = True
    supports_text_modality = True

    url = URL  # subclasses serving the same protocol elsewhere override this

    def __init__(self, *, model: str = "gpt-realtime-2.1", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._ws: websockets.ClientConnection | None = None
        self._configured = asyncio.Event()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        try:
            self._ws = await websockets.connect(
                f"{self.url}?model={self.model}",
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                open_timeout=CONNECT_TIMEOUT_S,
                max_size=None,
            )
        except Exception as exc:  # noqa: BLE001 -- the reason is what we want to publish
            raise AdapterError(f"{self.name} connect failed: {exc}") from exc

        first = json.loads(await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT_S))
        self.log.raw(first)
        if first.get("type") != "session.created":
            raise AdapterError(f"{self.name} refused the session: {first}")
        self.session_id = first["session"].get("id")
        self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model)

        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        self.session_sent = self._session_payload()
        await self._send_json({"type": "session.update", "session": self.session_sent})
        try:
            await asyncio.wait_for(self._configured.wait(), CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise AdapterError(f"{self.name} never acknowledged session.update") from exc
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

    def _session_payload(self) -> dict[str, Any]:
        config: SessionConfig = self.config
        audio: dict[str, Any] = {
            "input": {
                "format": {"type": "audio/pcm", "rate": self.input_rate},
                "turn_detection": self._turn_detection_payload(),
            }
        }
        if config.transcribe_input:
            audio["input"]["transcription"] = {"model": "whisper-1"}
        if config.modality == "audio":
            audio["output"] = {"format": {"type": "audio/pcm", "rate": self.output_rate}}
            if config.voice:
                audio["output"]["voice"] = config.voice

        session: dict[str, Any] = {
            "type": "realtime",
            "instructions": config.instructions,
            "output_modalities": ["audio" if config.modality == "audio" else "text"],
            "audio": audio,
        }
        if config.tools:
            session["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in config.tools
            ]
            session["tool_choice"] = "auto"
        return session

    def _turn_detection_payload(self) -> dict[str, Any] | None:
        detection = self.config.turn_detection
        if detection.is_manual:
            return None  # we declare the boundary ourselves, at the exact sample
        payload: dict[str, Any] = {"type": detection.mode}
        if detection.silence_duration_ms is not None:
            payload["silence_duration_ms"] = detection.silence_duration_ms
        if detection.threshold is not None:
            payload["threshold"] = detection.threshold
        if detection.prefix_padding_ms is not None:
            payload["prefix_padding_ms"] = detection.prefix_padding_ms
        return payload

    async def close(self) -> None:
        if self._receiver:
            self._receiver.cancel()
        if self._ws:
            await self._ws.close()
        self.closed.set()
        self.log.emit(ev.SESSION_CLOSED)

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        self.log.raw(payload, direction="out")
        await self._ws.send(json.dumps(payload))

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        # Deliberately not routed through _send_json: the base64 payload would
        # double the raw log for no audit value. The timeline already has it.
        await self._ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
        )

    async def _commit(self) -> None:
        await self._send_json({"type": "input_audio_buffer.commit"})
        await self._send_json({"type": "response.create"})

    async def _send_text(self, text: str) -> None:
        """The text control arm: identical scenario, no audio anywhere."""
        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
            }
        )
        await self._send_json({"type": "response.create"})

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
        await self._send_json({"type": "response.create"})

    # ── receiving ────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                self._handle(json.loads(message))
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            self.log.emit(ev.SESSION_CLOSED, code=exc.code, reason=str(exc.reason))
            self.closed.set()
        except Exception as exc:  # noqa: BLE001
            self.log.emit(ev.SESSION_ERROR, error=repr(exc))
            self.closed.set()

    def _handle(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type", "")
        if kind not in _AUDIO_DELTA:
            self.log.raw(payload)  # audio deltas are in the wav, not the log

        if kind in _AUDIO_DELTA:
            self._on_agent_audio(base64.b64decode(payload["delta"]))
        elif kind in _AUDIO_DONE:
            self._on_agent_audio_done()
        elif kind in _AGENT_TEXT_DONE:
            text = payload.get("transcript") or payload.get("text") or ""
            self.agent_text.append(text)
            self.log.emit(ev.AGENT_TRANSCRIPT, text=text)
        elif kind == "session.updated":
            self._configured.set()
            self.session_ack = payload.get("session")
            self.log.emit(ev.SESSION_CONFIGURED, turn_detection=self.config.turn_detection.label)
        elif kind == "input_audio_buffer.speech_started":
            self._on_caller_speech_started(audio_ms=payload.get("audio_start_ms"))
        elif kind == "input_audio_buffer.speech_stopped":
            self.log.emit(ev.VAD_SPEECH_END, audio_ms=payload.get("audio_end_ms"))
        elif kind == "conversation.item.input_audio_transcription.completed":
            text = payload.get("transcript", "")
            self.caller_text.append(text)
            self.log.emit(ev.CALLER_TRANSCRIPT, text=text)
        elif kind == "response.function_call_arguments.done":
            call = {
                "name": payload.get("name", ""),
                "call_id": payload.get("call_id", ""),
                "arguments": parse_arguments(payload.get("arguments")),
            }
            self.tool_calls.append(call)
            self.log.emit(ev.TOOL_CALL, **call)
        elif kind == "response.done":
            response = payload.get("response", {})
            status = response.get("status")
            if status == "cancelled":
                self._on_agent_interrupted()
            self._on_agent_audio_done(reason=status or "done")
            self.log.emit(ev.RESPONSE_DONE, status=status, usage=response.get("usage"))
        elif kind == "error":
            self.log.emit(ev.SESSION_ERROR, error=payload.get("error"))
