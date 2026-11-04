"""Deepgram Voice Agent over ``/v1/agent/converse``.

Not one speech model: a managed pipeline Deepgram runs behind one socket.
Flux hears the caller and decides when the turn ended, a named text model
replies, and an Aura voice speaks. The row therefore names all three, and the
model id says which text model answered (``<provider type>/<model>``, e.g.
``open_ai/gpt-4.1``), because the same socket serves a dozen of them.

The text model runs on our own OpenAI account, through the bring-your-own
endpoint Deepgram documents, not on Deepgram's hosted copy of it. A hosted
model's prompt is capped at 25,000 characters and Deepgram cuts whatever is
past that with only a warning; the medicare contract's prompt is longer, so a
hosted row would be scored against a different agent from every other row's.
Same model, whole prompt. The key goes to Deepgram on the wire only: the
settings the record keeps carry it redacted.

What this protocol does differently from an OpenAI-shaped one:

* **Audio is binary.** Both directions carry raw PCM frames, and every JSON
  message is a text frame. Nothing is base64.
* **The endpointer is Flux, and it says so.** ``UserStartedSpeaking`` marks the
  caller beginning and ``EndOfTurn`` the moment Flux decided they had finished,
  with what triggered it (``model`` when it was Flux's own confidence).
* **No cancel event.** On barge-in the service ends the reply with the same
  ``AgentAudioDone`` that closes a finished one, in the instant after
  ``UserStartedSpeaking``. A reply that was still arriving when the caller
  started and then ends that way is recorded as the provider yielding.
* **Tool calls can be withdrawn.** A call is sent as soon as the text model
  emits it, before Flux has confirmed the turn. If the caller keeps talking,
  ``FunctionCallCancelled`` withdraws it, and Deepgram drops any reply to it.
* **No manual turn boundary.** ``ForceEndTurn`` exists, but Flux keeps
  endpointing on its own, so it cannot give the zero-error floor that manual
  mode means here.

Input is 16 kHz (Flux's native rate), output 24 kHz.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from typing import Any

import websockets

from service import events as ev
from service.adapters.base import AdapterError, RealtimeAdapter, SessionConfig, parse_arguments

URL = "wss://agent.deepgram.com/v1/agent/converse"
CONNECT_TIMEOUT_S = 20.0
# ``KeepAlive`` holds a session open that is sending no audio: the text arm, or
# any carrier that pauses its stream. Sent on a timer in every session.
KEEPALIVE_S = 5.0
KEEPALIVE = json.dumps({"type": "KeepAlive"})

DEFAULT_MODEL = "open_ai/gpt-4.1"
DEFAULT_VOICE = "aura-2-thalia-en"
# Held fixed for every row, as the cascade's speech-to-text is: what varies
# between Deepgram rows is the text model, not how the caller is heard.
LISTEN = {"type": "deepgram", "version": "v2", "model": "flux-general-en"}
# Where each text-model provider type is reached on our own account, and the
# credential it takes. Only OpenAI is wired: it is the row's model.
THINK_ENDPOINTS = {"open_ai": "https://api.openai.com/v1/chat/completions"}
THINK_KEY_ENV = "OPENAI_API_KEY"
REDACTED = "<redacted>"
# Below this the service refuses the settings (INVALID_SETTINGS, read off the wire).
EOT_TIMEOUT_MIN_MS = 500


def split_model(model: str) -> tuple[str, str]:
    """``open_ai/gpt-4.1`` -> (``open_ai``, ``gpt-4.1``). Only the first slash splits:
    Groq's ids carry one of their own (``groq/openai/gpt-oss-20b``)."""
    kind, sep, name = model.partition("/")
    if not sep or not kind or not name:
        raise AdapterError(f"deepgram model must be '<provider type>/<model>', got {model!r}")
    return kind, name


class DeepgramVoiceAgentAdapter(RealtimeAdapter):
    name = "deepgram-voice-agent"
    input_rate = 16000
    output_rate = 24000
    supports_manual_commit = False
    supports_text_modality = True   # text in, audio out, as with Gemini
    emits_vad_events = True

    def __init__(self, *, model: str = DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._ws: websockets.ClientConnection | None = None
        self._agent_parts: list[str] = []    # sentences of the reply under way
        self._call_names: dict[str, str] = {}
        self._keepalive: asyncio.Task | None = None
        self._caller_cut_in = False   # the caller started while this reply was still arriving

    @classmethod
    def unsupported_reason(cls, config: SessionConfig) -> str | None:
        reason = super().unsupported_reason(config)
        if reason:
            return reason
        # Flux is a model-based end-of-turn detector, so ``semantic_vad`` and a
        # bare ``server_vad`` (the provider's defaults) both mean Flux as shipped.
        # A silence duration becomes Flux's ``eot_timeout_ms``: a ceiling rather
        # than a fixed hangover, since Flux may still end the turn sooner when it
        # is confident. The value sent is in the record.
        detection = config.turn_detection
        if detection.threshold is not None or detection.prefix_padding_ms is not None:
            return "deepgram-voice-agent ends turns on Flux's confidence, not a VAD threshold or prefix padding"
        if detection.silence_duration_ms is not None and detection.silence_duration_ms < EOT_TIMEOUT_MIN_MS:
            return f"deepgram-voice-agent's eot_timeout_ms starts at {EOT_TIMEOUT_MIN_MS} ms"
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        # A malformed id or a missing text-model key is refused before connecting.
        self.session_sent = self._settings_payload()
        wire = self._with_credentials(self.session_sent)
        try:
            self._ws = await websockets.connect(
                URL,
                additional_headers={"Authorization": f"Token {self._api_key}"},
                open_timeout=CONNECT_TIMEOUT_S,
                max_size=None,
            )
        except Exception as exc:  # noqa: BLE001 -- the reason is what we want to publish
            raise AdapterError(f"{self.name} connect failed: {exc}") from exc

        welcome = await self._recv_json("never said Welcome")
        if welcome.get("type") != "Welcome":
            raise AdapterError(f"{self.name} refused the session: {welcome}")
        self.session_id = welcome.get("request_id")
        self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model)

        await self._send_json(self.session_sent, wire=wire)
        applied = await self._recv_json("never applied Settings")
        if applied.get("type") != "SettingsApplied":
            raise AdapterError(f"{self.name} refused the settings: {applied}")
        # Applied, not echoed: the record says so rather than leaving the field
        # looking like an omission on our side.
        self.session_ack = {"SettingsApplied": True, "echoes_configuration": False}
        self.log.emit(ev.SESSION_CONFIGURED, turn_detection=self.config.turn_detection.label)

        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        self._keepalive = asyncio.create_task(self._keep_alive(), name=f"{self.name}-keepalive")

    async def _recv_json(self, what: str) -> dict[str, Any]:
        assert self._ws is not None
        try:
            message = await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise AdapterError(f"{self.name} {what}") from exc
        if isinstance(message, (bytes, bytearray)):
            raise AdapterError(f"{self.name} sent audio before the session was set up")
        payload = json.loads(message)
        self.log.raw(payload)
        return payload

    def _settings_payload(self) -> dict[str, Any]:
        config = self.config
        kind, think_model = split_model(self.model)
        think_provider: dict[str, Any] = {"type": kind, "model": think_model}
        if config.temperature is not None:
            think_provider["temperature"] = config.temperature
        if kind not in THINK_ENDPOINTS:
            raise AdapterError(f"{self.name} runs its text model on our own account; no endpoint is wired for {kind!r}")
        think: dict[str, Any] = {
            "provider": think_provider,
            "endpoint": {"url": THINK_ENDPOINTS[kind], "headers": {"authorization": REDACTED}},
            "prompt": config.instructions,
        }
        if config.tools:
            # No function ``endpoint``: a function without one is run by the client, so
            # every call reaches the harness's mock tools.
            think["functions"] = [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in config.tools
            ]
        listen = dict(LISTEN)
        if config.turn_detection.silence_duration_ms is not None:
            listen["eot_timeout_ms"] = config.turn_detection.silence_duration_ms
        agent: dict[str, Any] = {
            "listen": {"provider": listen},
            "think": think,
            "speak": {"provider": {"type": "deepgram", "model": config.voice or DEFAULT_VOICE}},
        }
        if config.first_message:
            # Seeded as already said, not as ``greeting``: a greeting would be
            # spoken, and the contracts open with the line behind us.
            agent["context"] = {
                "messages": [{"type": "History", "role": "assistant", "content": config.first_message}]
            }
        return {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "linear16", "sample_rate": self.input_rate},
                "output": {"encoding": "linear16", "sample_rate": self.output_rate, "container": "none"},
            },
            "agent": agent,
        }

    async def close(self) -> None:
        for task in (self._keepalive, self._receiver):
            if task:
                task.cancel()
        if self._ws:
            await self._ws.close()
        self._flush_agent_text()
        self.closed.set()
        self.log.emit(ev.SESSION_CLOSED, by="harness")

    # ── sending ──────────────────────────────────────────────────────────────

    def _with_credentials(self, settings: dict[str, Any]) -> dict[str, Any]:
        """The settings as sent: the recorded copy with the text-model key filled in."""
        key = os.environ.get(THINK_KEY_ENV)
        if not key:
            raise AdapterError(f"{self.name} runs its text model on our own account: {THINK_KEY_ENV} is not set")
        wire = copy.deepcopy(settings)
        wire["agent"]["think"]["endpoint"]["headers"]["authorization"] = f"Bearer {key}"
        return wire

    async def _send_json(self, payload: dict[str, Any], wire: dict[str, Any] | None = None) -> None:
        """``payload`` is what the record keeps; ``wire``, when given, is what is sent."""
        if self._ws is None:
            raise AdapterError("not connected")
        self.log.raw(payload, direction="out")
        await self._ws.send(json.dumps(wire if wire is not None else payload))

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        await self._ws.send(pcm)

    async def _commit(self) -> None:
        raise AdapterError(f"{self.name} has no manual commit")

    async def _send_text(self, text: str) -> None:
        await self._send_json({"type": "InjectUserMessage", "content": text})

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        await self._send_json(
            {
                "type": "FunctionCallResponse",
                "id": call_id,
                "name": self._call_names.get(call_id, ""),
                "content": output if isinstance(output, str) else json.dumps(output),
            }
        )

    async def _keep_alive(self) -> None:
        assert self._ws is not None
        while True:
            await asyncio.sleep(KEEPALIVE_S)
            try:
                # Not logged: it carries nothing but the session's own upkeep.
                await self._ws.send(KEEPALIVE)
            except websockets.ConnectionClosed:
                return  # the receive loop records the close

    # ── receiving ────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                if isinstance(message, (bytes, bytearray)):
                    self._on_agent_audio(bytes(message))  # in the wav, not the log
                else:
                    self._handle(json.loads(message))
            self.log.emit(ev.SESSION_CLOSED, by="provider", code=self._ws.close_code, reason=self._ws.close_reason)
        except websockets.ConnectionClosed:
            self.log.emit(ev.SESSION_CLOSED, code=self._ws.close_code, reason=self._ws.close_reason)
            self.closed.set()
        except Exception as exc:  # noqa: BLE001
            self.log.emit(ev.SESSION_ERROR, error=repr(exc))
            self.closed.set()

    def _handle(self, payload: dict[str, Any]) -> None:
        self.log.raw(payload)
        kind = payload.get("type", "")

        if kind == "ConversationText":
            text = payload.get("content", "").strip()
            if payload.get("role") == "assistant":
                self._agent_parts.append(text)
            else:
                self.caller_text.append(text)
                self.log.emit(ev.CALLER_TRANSCRIPT, text=text)
        elif kind == "UserStartedSpeaking":
            self._caller_cut_in = self._speaking
            self._on_caller_speech_started()
        elif kind == "EndOfTurn":
            self.log.emit(ev.VAD_SPEECH_END, trigger=payload.get("trigger"))
        elif kind == "AgentThinking":
            self.log.emit(ev.AGENT_THOUGHT, text=payload.get("content", ""))
        elif kind == "AgentAudioDone":
            cut_in, self._caller_cut_in = self._caller_cut_in, False
            if cut_in:
                self._on_agent_interrupted()
            status = "cancelled" if cut_in else "completed"
            self._on_agent_audio_done(reason=status)
            self._flush_agent_text()
            self.log.emit(ev.RESPONSE_DONE, status=status, usage=None)
        elif kind == "FunctionCallRequest":
            for function in payload.get("functions", []):
                record = {
                    "name": function.get("name", ""),
                    "call_id": function.get("id", ""),
                    "arguments": parse_arguments(function.get("arguments")),
                }
                self._call_names[record["call_id"]] = record["name"]
                self.tool_calls.append(record)
                self.log.emit(ev.TOOL_CALL, **record)
        elif kind == "FunctionCallCancelled":
            # Deepgram withdrew calls it had already sent: the caller kept talking
            # inside the speculative window, or barged in on the reply that made
            # them. Marked in place (the tool pump reads the list by position) so
            # the pump does not answer them and the task verdict leaves them out.
            cancelled = {function.get("id", "") for function in payload.get("functions", [])}
            for record in self.tool_calls:
                if record["call_id"] in cancelled:
                    record["cancelled"] = True
                    self.log.emit(ev.TOOL_CANCELLED, call_id=record["call_id"], name=record["name"])
        elif kind == "Error":
            self.log.emit(ev.SESSION_ERROR, error={"code": payload.get("code"), "description": payload.get("description")})

    def _flush_agent_text(self) -> None:
        """Sentences become one entry per reply, so ``agent_text[-1]`` is a whole reply."""
        if self._agent_parts:
            text = " ".join(part for part in self._agent_parts if part)
            self._agent_parts.clear()
            self.agent_text.append(text)
            self.log.emit(ev.AGENT_TRANSCRIPT, text=text)
