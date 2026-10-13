"""Phonic speech-to-speech (``phonic_v1``) as a service-bench adapter.

Implements ``service.adapters.base.RealtimeAdapter`` over Phonic's
conversations websocket, matching the published "Phonic v1" row
(``reference-agents/pipecat-s2s/phonic_realtime.py`` and ``bot.py``): model
``phonic_v1``, voice ``sabrina``, 16 kHz PCM both ways, ``intelligence_level``
``high`` and Phonic's documented turn defaults (800 ms of silence ends a turn,
one word interrupts), all sent explicitly rather than left to the server.

Protocol: https://docs.phonic.co/api-reference/conversations/conversations

Protocol quirks, each silent when got wrong:

* **Auth and handshake.** ``wss://api.phonic.ai/v1/sts/ws`` with
  ``Authorization: Bearer``. The client sends one ``config`` message; the server
  answers ``conversation_created`` (carries ``conversation_id``) and then
  ``ready_to_start_conversation``. Audio sent before ``ready`` is refused or
  answered under defaults, so ``connect`` waits for it. Neither message echoes
  the configuration back, so ``session_ack`` holds those two frames verbatim and
  the resolved model is whatever they name (``resolved_model``), else the pinned
  id we sent -- the service does not report one independently.
* **The output stream never stops.** Audio is paced to real time and keeps
  arriving *between* replies as silence. Forwarded as received, the agent would
  never stop speaking. Only audio inside a reply is recorded: a reply opens on
  ``assistant_started_speaking`` (or the first ``audio_chunk`` that carries
  ``text``, which can precede it) and closes on ``assistant_finished_speaking``.
  Because the stream is real-time paced, arrival timestamps are playout times.
* **Barge-in is the service's decision.** It stops a reply only after
  ``min_words_to_interrupt`` caller words and says so with
  ``interrupted_response``; that is mapped to ``agent.interrupted``. A caller
  ``user_started_speaking`` is the endpointer's onset (``vad.speech.start``),
  not a cancel.
* **Turn detection is native only.** There is no commit message, so
  ``supports_manual_commit`` is False; ``TurnDetection.silence_duration_ms`` /
  ``threshold`` / ``prefix_padding_ms`` map onto ``vad_min_silence_duration_ms``
  / ``vad_threshold`` / ``vad_prebuffer_duration_ms``.
* **No text-only output** (``supports_text_modality`` False).
* **No silent seed of an agent turn.** ``SessionConfig.first_message`` ("already
  said") has no protocol equivalent that does not make the agent speak, so it
  is appended to the system prompt as context instead; the caller speaks first.
  No ``generate_reply`` is sent: the turn benchmark's caller opens the call.
* **Strict tools.** Phonic requires every parameter be required; optional ones
  are made nullable and nulls (and, on optional fields, the string ``"null"``)
  are dropped from the arguments before recording, as the reference agent does.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
from typing import Any, Awaitable, Callable

from turn_taking_bench.author import service_bridge

service_bridge._ensure_path()

from service import events as ev  # noqa: E402
from service.adapters.base import AdapterError, RealtimeAdapter, SessionClosed  # noqa: E402,F401

URL = "wss://api.phonic.ai/v1/sts/ws"
MODEL = "phonic_v1"
VOICE = "sabrina"
SAMPLE_RATE = 16000
AUDIO_FORMAT = "pcm_16000"
CONNECT_TIMEOUT_S = 20.0
READY_TIMEOUT_S = 15.0
TOOL_OUTPUT_TIMEOUT_MS = 60000

#: The published row's settings (reference-agents/pipecat-s2s/bot.py).
INTELLIGENCE_LEVEL = "high"
TURNS: dict[str, Any] = {
    "vad_threshold": 0.38,
    "vad_min_silence_duration_ms": 800,
    "vad_min_speech_duration_ms": 50,
    "vad_prebuffer_duration_ms": 500,
    "min_words_to_interrupt": 1,
}

Connector = Callable[..., Awaitable[Any]]


async def _websocket_connect(url: str, *, headers: dict[str, str]) -> Any:
    import websockets

    return await websockets.connect(url, additional_headers=headers, open_timeout=CONNECT_TIMEOUT_S, max_size=None)


def strict_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """A tool's parameters in Phonic's strict form: all required, closed, optional ones nullable."""
    schema = copy.deepcopy(schema)
    if schema.get("type") == "object":
        properties = schema.setdefault("properties", {})
        required = set(schema.get("required") or ())
        for name, prop in properties.items():
            properties[name] = strict_parameters(prop)
            if name not in required:
                kind = properties[name].get("type")
                if isinstance(kind, str):
                    properties[name]["type"] = [kind, "null"]
                elif isinstance(kind, list) and "null" not in kind:
                    properties[name]["type"] = [*kind, "null"]
                if "enum" in properties[name] and None not in properties[name]["enum"]:
                    properties[name]["enum"] = [*properties[name]["enum"], None]
        schema["required"] = list(properties)
        schema["additionalProperties"] = False
    elif schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        schema["items"] = strict_parameters(schema["items"])
    return schema


def without_nulls(arguments: dict[str, Any], optional: frozenset[str]) -> dict[str, Any]:
    return {k: v for k, v in arguments.items() if v is not None and not (k in optional and v == "null")}


def seeded_instructions(instructions: str, first_message: str | None) -> str:
    if not first_message:
        return instructions
    return f"{instructions}\n\nYou have already greeted the caller with: \"{first_message}\"".strip()


class PhonicAdapter(RealtimeAdapter):
    name = "phonic"
    input_rate = SAMPLE_RATE
    output_rate = SAMPLE_RATE
    supports_manual_commit = False
    supports_text_modality = False
    emits_vad_events = True

    def __init__(
        self,
        *,
        model: str = MODEL,
        api_key: str = "",
        connector: Connector | None = None,
        url: str = URL,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._connector = connector or _websocket_connect
        self._url = url
        self._ws: Any = None
        self._ready = asyncio.Event()
        self._replying = False
        self._reply_text: list[str] = []
        self._optional: dict[str, frozenset[str]] = {}
        self.resolved_model: str | None = None

    # ── session ──────────────────────────────────────────────────────────────

    def config_payload(self) -> dict[str, Any]:
        config = self.config
        detection = config.turn_detection
        turns = dict(TURNS)
        if detection.silence_duration_ms is not None:
            turns["vad_min_silence_duration_ms"] = detection.silence_duration_ms
        if detection.threshold is not None:
            turns["vad_threshold"] = detection.threshold
        if detection.prefix_padding_ms is not None:
            turns["vad_prebuffer_duration_ms"] = detection.prefix_padding_ms
        tools = []
        for tool in config.tools:
            parameters = tool.parameters or {"type": "object", "properties": {}}
            self._optional[tool.name] = frozenset(
                set(parameters.get("properties", {})) - set(parameters.get("required") or ())
            )
            tools.append(
                {
                    "type": "custom_websocket",
                    "tool_call_output_timeout_ms": TOOL_OUTPUT_TIMEOUT_MS,
                    "tool_schema": {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": strict_parameters(parameters),
                            "strict": True,
                        },
                    },
                }
            )
        return {
            "type": "config",
            "system_prompt": seeded_instructions(config.instructions, config.first_message),
            "voice_id": config.voice or VOICE,
            "phonic_model": self.model,
            "input_format": AUDIO_FORMAT,
            "output_format": AUDIO_FORMAT,
            "tools": tools,
            "intelligence_level": INTELLIGENCE_LEVEL,
            **turns,
        }

    async def connect(self) -> None:
        if not self._api_key:
            raise AdapterError(f"{self.name} has no API key")
        try:
            self._ws = await self._connector(self._url, headers={"Authorization": f"Bearer {self._api_key}"})
        except Exception as exc:  # noqa: BLE001 -- the reason is what we publish
            raise AdapterError(f"{self.name} connect failed: {exc}") from exc
        self.session_ack = {}
        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        self.session_sent = self.config_payload()
        await self._send_json(self.session_sent)
        ready = asyncio.ensure_future(self._ready.wait())
        closed = asyncio.ensure_future(self.closed.wait())
        done, pending = await asyncio.wait({ready, closed}, timeout=READY_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if not self._ready.is_set():
            errors = [e.data for e in self.log.of_kind(ev.SESSION_ERROR)]
            await self.close()
            raise AdapterError(f"{self.name} never became ready: {errors or 'timeout'}")
        self.log.emit(
            ev.SESSION_CONFIGURED,
            turn_detection=self.config.turn_detection.label,
            model=self.model,
            resolved_model=self.resolved_model,
        )

    async def close(self) -> None:
        if self._receiver:
            self._receiver.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        if not self.closed.is_set():
            self.closed.set()
            self.log.emit(ev.SESSION_CLOSED, by="harness")

    def state(self) -> dict[str, Any]:
        state = super().state()
        state["resolved_model"] = self.resolved_model
        return state

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        self.log.raw(payload, direction="out")
        await self._ws.send(json.dumps(payload))

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        if self.closed.is_set():
            raise SessionClosed(f"{self.name} session closed")
        await self._ws.send(json.dumps({"type": "audio_chunk", "audio": base64.b64encode(pcm).decode()}))

    async def _commit(self) -> None:
        raise AdapterError(f"{self.name} has no manual commit")

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        await self._send_json({"type": "tool_call_output", "tool_call_id": call_id, "output": output})

    # ── receiving ────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        ws = self._ws
        try:
            async for message in ws:
                try:
                    payload = json.loads(message)
                except (TypeError, json.JSONDecodeError):
                    self.log.emit(ev.SESSION_ERROR, error="non-JSON frame")
                    continue
                self._handle(payload)
            self.log.emit(ev.SESSION_CLOSED, by="provider", code=getattr(ws, "close_code", None),
                          reason=getattr(ws, "close_reason", None))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- includes websockets.ConnectionClosed
            self.log.emit(ev.SESSION_CLOSED, by="provider", code=getattr(ws, "close_code", None),
                          reason=getattr(ws, "close_reason", None), error=repr(exc))
        self._end_reply("closed")
        self.closed.set()

    def _start_reply(self) -> None:
        if not self._replying:
            self._replying = True
            self._reply_text = []

    def _end_reply(self, status: str) -> None:
        if not self._replying:
            return
        self._replying = False
        self._on_agent_audio_done(reason=status)
        text = "".join(self._reply_text).strip()
        if text:
            self.agent_text.append(text)
            self.log.emit(ev.AGENT_TRANSCRIPT, text=text)
        self.log.emit(ev.RESPONSE_DONE, status=status, usage=None)

    def _handle(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type", "")
        if kind == "audio_chunk":
            # Audio is in the wav; only the frame's text goes to the raw log.
            text = payload.get("text") or ""
            if text:
                self.log.raw({"type": kind, "text": text})
            if not self._replying:
                if not text:
                    return  # the silence the stream carries between replies
                self._start_reply()
            if text:
                self._reply_text.append(text)
            audio = base64.b64decode(payload.get("audio") or "")
            if audio:
                self._on_agent_audio(audio)
            return

        self.log.raw(payload)
        if kind == "assistant_started_speaking":
            self._start_reply()
        elif kind == "assistant_finished_speaking":
            self._end_reply("completed")
        elif kind == "interrupted_response":
            if self._replying:
                self._on_agent_interrupted()
                self._end_reply("cancelled")
        elif kind == "user_started_speaking":
            self._on_caller_speech_started()
        elif kind == "user_finished_speaking":
            self.log.emit(ev.VAD_SPEECH_END)
        elif kind == "input_text":
            text = payload.get("text", "")
            self.caller_text.append(text)
            self.log.emit(ev.CALLER_TRANSCRIPT, text=text)
        elif kind == "tool_call":
            name = payload.get("tool_name", "")
            call = {
                "name": name,
                "call_id": payload.get("tool_call_id", ""),
                "arguments": without_nulls(payload.get("parameters") or {}, self._optional.get(name, frozenset())),
            }
            self.tool_calls.append(call)
            self.log.emit(ev.TOOL_CALL, **call)
        elif kind == "conversation_created":
            self.session_id = payload.get("conversation_id")
            self.session_ack["conversation_created"] = payload
            self.resolved_model = payload.get("phonic_model") or payload.get("model") or self.resolved_model
            self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model)
        elif kind == "ready_to_start_conversation":
            self.session_ack["ready_to_start_conversation"] = payload
            self.resolved_model = payload.get("phonic_model") or payload.get("model") or self.resolved_model
            self._ready.set()
        elif kind == "error":
            self.log.emit(ev.SESSION_ERROR, error=payload.get("error"), param_errors=payload.get("param_errors"))
        elif kind == "assistant_ended_conversation":
            self.log.emit(ev.SESSION_ERROR, error="assistant ended the conversation")
