"""Deepslate Opal speech-to-speech as a service-bench adapter.

Implements ``service.adapters.base.RealtimeAdapter`` over Deepslate's realtime
websocket, matching the published "Deepslate Opal" row
(``reference-agents/pipecat-s2s/deepslate_realtime.py`` and ``bot.py``): model
``opal`` (not selectable; the service runs its current Opal), 16 kHz PCM both
ways, replies voiced by ElevenLabs ``eleven_flash_v2_5`` voice
``21m00Tcm4TlvDq8ikWAM`` (location US) called by the service on our ElevenLabs
key, the vendor SDK's default detector (390 ms of silence ends a turn) and
temperature 0.3, every field sent explicitly.

Protocol: https://docs.deepslate.eu/api-reference/realtime

Protocol quirks, each silent when got wrong:

* **Wire format is protobuf** (binary frames). Message classes come from the
  vendor SDK, pinned (``deepslate-core==0.1.20``); only the generated classes are
  used, not its session layer, which configures lazily at 24 kHz when anything
  but audio is sent first and then reconfigures only the *input* line -- replies
  then arrive at one rate labelled as another -- and which silently reconnects
  into a fresh session that remembers nothing. Here the session is configured
  once, before any audio, and a drop ends the session.
* **Auth and account path.**
  ``wss://app.deepslate.eu/api/v1/vendors/{vendor}/organizations/{org}/realtime``
  with ``Authorization: Bearer <DEEPSLATE_API_KEY>``. Vendor / organization ids
  and ``ELEVENLABS_API_KEY`` are read from the environment or ``.env`` unless
  passed as keyword arguments.
* **The acknowledgement is empty.** ``session_ready`` has no fields: nothing
  the service applied is echoed, and no model id is ever reported. So
  ``session_ack`` records that the session became ready plus the (non-secret)
  configuration we sent, and ``resolved_model`` stays ``None`` (unreported).
* **The service owns the caller's turn.** Its detector reports
  ``vad_state_event`` transitions; SPEECH_STARTING→SPEECH is the caller-onset
  decision (``vad.speech.start``) and SPEECH_ENDING→SILENCE the end
  (``vad.speech.end``). Audio is sent with trigger mode ``IMMEDIATE`` (as the
  SDK does): a closed turn starts a reply at once, replacing one in progress.
  No manual commit (``supports_manual_commit`` False).
* **Barge-in signal is unconditional.** ``playback_clear_buffer`` is sent every
  time the caller starts speaking, playing or not; it is an interruption
  (``agent.interrupted``) only while a turn is open (``response_begin`` without
  its ``response_end``) or its audio is still audible on the playout model.
  Audio still in flight for that cut turn (same ``turn_id``) is dropped so it
  cannot reopen the reply.
* **response_end is the end of generation, not of speech.** A reply's audio
  arrives in ~1 s chunks, often the whole reply within ~200 ms, and
  ``response_end`` follows the last chunk while most of it is still to play
  (seen live: ``model_speech_progress`` keeps counting played bytes afterwards).
  ``agent.audio.end`` therefore marks the last arrival; when speech stops is the
  base timeline's playout model, which a clear-buffer cuts. The service's own
  ``turn_snapshot`` (raw log) records how much of a cut reply it judged heard.
  ``model_speech_progress`` / ``inference_complete`` / ``turn_snapshot`` are
  sent even with playback reporting off; they go to the raw log only.
* **Turn ids**: caller and agent turns share one counter (agent turns came back
  odd live: 1, 3, 5, ...).
* **Speech is text first.** Opal produces text and the service voices it, so a
  reply's ``model_text_fragment``s lead its audio by up to ~1.5 s; a reply's
  start is its first audio chunk, never its text or ``response_begin``.
  ``turn_id`` is 0-based and optional; a piece without one belongs to the
  latest ``response_begin``. Audio arrives as fast as TTS produces it (not
  real-time paced); the base timeline's playout model handles that.
* **No text-only output and no silent seed.** ``first_message`` is appended to
  the system prompt as context; no ``trigger_inference`` is sent at start, so the
  caller opens the call.
* **Tool arguments are a protobuf Struct** (one number type): whole floats are
  turned back into integers before recording.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from turn_taking_bench.author import service_bridge
from turn_taking_bench.s2s._vendor_env import setting

service_bridge._ensure_path()

from service import events as ev  # noqa: E402
from service.adapters.base import AdapterError, RealtimeAdapter, SessionClosed  # noqa: E402

URL = "wss://app.deepslate.eu/api/v1/vendors/{vendor_id}/organizations/{organization_id}/realtime"
MODEL = "opal"
SAMPLE_RATE = 16000
CONNECT_TIMEOUT_S = 20.0
READY_TIMEOUT_S = 15.0

#: The published row's settings (reference-agents/pipecat-s2s/bot.py).
VAD: dict[str, Any] = {
    "confidence_threshold": 0.4,
    "min_volume": 0.0,
    "start_duration_ms": 150,
    "stop_duration_ms": 390,
    "backbuffer_duration_ms": 1000,
}
TEMPERATURE = 0.3
ELEVENLABS_VOICE = "21m00Tcm4TlvDq8ikWAM"
ELEVENLABS_MODEL = "eleven_flash_v2_5"
ELEVENLABS_LOCATION = "US"

Connector = Callable[..., Awaitable[Any]]


def _proto():
    try:
        from deepslate.core.proto import realtime_pb2
    except ImportError as exc:  # pragma: no cover - environment
        raise AdapterError("deepslate adapter needs deepslate-core==0.1.20 (its protobuf classes)") from exc
    return realtime_pb2


async def _websocket_connect(url: str, *, headers: dict[str, str]) -> Any:
    import websockets

    return await websockets.connect(url, additional_headers=headers, open_timeout=CONNECT_TIMEOUT_S, max_size=None)


def _duration(proto: Any, ms: int) -> Any:
    return proto.Duration(seconds=ms // 1000, nanos=(ms % 1000) * 1_000_000)


def whole_numbers(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: whole_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [whole_numbers(v) for v in value]
    return value


def seeded_instructions(instructions: str, first_message: str | None) -> str:
    if not first_message:
        return instructions
    return f"{instructions}\n\nYou have already greeted the caller with: \"{first_message}\"".strip()


class DeepslateAdapter(RealtimeAdapter):
    name = "deepslate"
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
        vendor_id: str | None = None,
        organization_id: str | None = None,
        elevenlabs_api_key: str | None = None,
        voice: str = ELEVENLABS_VOICE,
        connector: Connector | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._vendor_id = vendor_id or setting("DEEPSLATE_VENDOR_ID")
        self._organization_id = organization_id or setting("DEEPSLATE_ORGANIZATION_ID")
        self._elevenlabs_key = elevenlabs_api_key or setting("ELEVENLABS_API_KEY")
        self._voice = self.config.voice or voice
        self._connector = connector or _websocket_connect
        self._ws: Any = None
        self._ready = asyncio.Event()
        self._packet_id = 0
        self._replying = False
        self._turn: int | None = None
        self._cut_turn: int | None = None
        self._turn_open = False          # between response_begin and response_end
        self._audible_at_onset = False   # the agent could be heard when the caller began
        self._reply_text: list[str] = []
        self.resolved_model: str | None = None  # Deepslate never reports one

    # ── session ──────────────────────────────────────────────────────────────

    def vad_settings(self) -> dict[str, Any]:
        detection = self.config.turn_detection
        vad = dict(VAD)
        if detection.silence_duration_ms is not None:
            vad["stop_duration_ms"] = detection.silence_duration_ms
        if detection.threshold is not None:
            vad["confidence_threshold"] = detection.threshold
        if detection.prefix_padding_ms is not None:
            vad["backbuffer_duration_ms"] = detection.prefix_padding_ms
        return vad

    def initialize_request(self) -> Any:
        proto = _proto()
        vad = self.vad_settings()
        line = proto.AudioLineConfiguration(
            sample_rate=SAMPLE_RATE, channel_count=1, sample_format=proto.SampleFormat.SIGNED_16_BIT
        )
        return proto.InitializeSessionRequest(
            input_audio_line=line,
            output_audio_line=line,
            vad_configuration=proto.VadConfiguration(
                confidence_threshold=vad["confidence_threshold"],
                min_volume=vad["min_volume"],
                start_duration=_duration(proto, vad["start_duration_ms"]),
                stop_duration=_duration(proto, vad["stop_duration_ms"]),
                backbuffer_duration=_duration(proto, vad["backbuffer_duration_ms"]),
            ),
            inference_configuration=proto.InferenceConfiguration(
                system_prompt=seeded_instructions(self.config.instructions, self.config.first_message),
                temperature=TEMPERATURE if self.config.temperature is None else self.config.temperature,
            ),
            tts_configuration=proto.TtsConfiguration(
                eleven_labs=proto.ElevenLabsTtsConfiguration(
                    api_key=self._elevenlabs_key or "",
                    voice_id=self._voice,
                    model_id=ELEVENLABS_MODEL,
                    location=proto.ElevenLabsLocation.Value(ELEVENLABS_LOCATION),
                )
            ),
            supports_playback_reporting=False,
        )

    def tool_definitions(self) -> Any:
        proto = _proto()
        from google.protobuf import json_format
        from google.protobuf.struct_pb2 import Struct

        definitions = []
        for tool in self.config.tools:
            parameters = Struct()
            json_format.ParseDict(tool.parameters or {"type": "object", "properties": {}}, parameters)
            definitions.append(proto.ToolDefinition(name=tool.name, description=tool.description, parameters=parameters))
        return proto.UpdateToolDefinitionsRequest(tool_definitions=definitions)

    def _sent_record(self, request: Any, tools: Any) -> dict[str, Any]:
        """What we configured, for the record -- with the ElevenLabs key removed."""
        from google.protobuf import json_format

        sent = json_format.MessageToDict(request, preserving_proto_field_name=True)
        eleven = sent.get("tts_configuration", {}).get("eleven_labs", {})
        if "api_key" in eleven:
            eleven["api_key"] = "<redacted>"
        sent["tools"] = [d["name"] for d in json_format.MessageToDict(tools).get("toolDefinitions", [])]
        sent["model"] = self.model
        return sent

    async def connect(self) -> None:
        if not self._api_key:
            raise AdapterError(f"{self.name} has no API key")
        proto = _proto()
        missing = [n for n, v in (("DEEPSLATE_VENDOR_ID", self._vendor_id),
                                  ("DEEPSLATE_ORGANIZATION_ID", self._organization_id),
                                  ("ELEVENLABS_API_KEY", self._elevenlabs_key)) if not v]
        if missing:
            raise AdapterError(f"{self.name} needs {', '.join(missing)}")
        url = URL.format(vendor_id=self._vendor_id, organization_id=self._organization_id)
        try:
            self._ws = await self._connector(url, headers={"Authorization": f"Bearer {self._api_key}"})
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"{self.name} connect failed: {exc}") from exc
        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        request, tools = self.initialize_request(), self.tool_definitions()
        self.session_sent = self._sent_record(request, tools)
        self.log.raw({"initialize_session_request": self.session_sent}, direction="out")
        await self._send(proto.ServiceBoundMessage(initialize_session_request=request))
        await self._send(proto.ServiceBoundMessage(update_tool_definitions_request=tools))
        ready = asyncio.ensure_future(self._ready.wait())
        closed = asyncio.ensure_future(self.closed.wait())
        _done, pending = await asyncio.wait({ready, closed}, timeout=READY_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if not self._ready.is_set():
            errors = [e.data for e in self.log.of_kind(ev.SESSION_ERROR)]
            await self.close()
            raise AdapterError(f"{self.name} never became ready: {errors or 'timeout'}")
        self.log.emit(ev.SESSION_OPEN, session_id=None, model=self.model)
        self.log.emit(ev.SESSION_CONFIGURED, turn_detection=self.config.turn_detection.label, model=self.model,
                      resolved_model=None)

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
        state["voicing"] = {"provider": "elevenlabs", "model": ELEVENLABS_MODEL, "voice": self._voice,
                            "location": ELEVENLABS_LOCATION}
        return state

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send(self, message: Any) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        await self._ws.send(message.SerializeToString())

    async def _send_json(self, payload: dict[str, Any]) -> None:
        """Protobuf protocol: a dict is parsed into a ServiceBoundMessage."""
        from google.protobuf import json_format

        self.log.raw(payload, direction="out")
        await self._send(json_format.ParseDict(payload, _proto().ServiceBoundMessage()))

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        if self.closed.is_set():
            raise SessionClosed(f"{self.name} session closed")
        proto = _proto()
        self._packet_id += 1
        await self._send(
            proto.ServiceBoundMessage(
                user_input=proto.UserInput(
                    packet_id=self._packet_id,
                    mode=proto.InferenceTriggerMode.IMMEDIATE,
                    audio_data=proto.AudioData(data=pcm),
                )
            )
        )

    async def _commit(self) -> None:
        raise AdapterError(f"{self.name} has no manual commit")

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        proto = _proto()
        result = output if isinstance(output, str) else json.dumps(output)
        await self._send(proto.ServiceBoundMessage(tool_call_response=proto.ToolCallResponse(id=call_id, result=result)))

    # ── receiving ────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        proto = _proto()
        ws = self._ws
        try:
            async for raw in ws:
                if not isinstance(raw, (bytes, bytearray)):
                    self.log.emit(ev.SESSION_ERROR, error="text frame on a protobuf socket")
                    continue
                message = proto.ClientBoundMessage()
                try:
                    message.ParseFromString(bytes(raw))
                except Exception as exc:  # noqa: BLE001
                    self.log.emit(ev.SESSION_ERROR, error=f"unparseable frame: {exc}")
                    continue
                self._handle(message)
            self.log.emit(ev.SESSION_CLOSED, by="provider", code=getattr(ws, "close_code", None),
                          reason=getattr(ws, "close_reason", None))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.log.emit(ev.SESSION_CLOSED, by="provider", code=getattr(ws, "close_code", None),
                          reason=getattr(ws, "close_reason", None), error=repr(exc))
        self._end_reply("closed")
        self.closed.set()

    def _turn_of(self, part: Any) -> int | None:
        return part.turn_id if part.HasField("turn_id") else self._turn

    def _was_cut(self, part: Any) -> bool:
        turn = self._turn_of(part)
        return turn is not None and turn == self._cut_turn

    def _end_reply(self, status: str) -> None:
        if not self._replying:
            return
        self._replying = False
        self._on_agent_audio_done(reason=status)
        text = "".join(self._reply_text).strip()
        self._reply_text = []
        if text:
            self.agent_text.append(text)
            self.log.emit(ev.AGENT_TRANSCRIPT, text=text)
        self.log.emit(ev.RESPONSE_DONE, status=status, usage=None, turn_id=self._turn)

    def _handle(self, message: Any) -> None:
        proto = _proto()
        from google.protobuf import json_format

        kind = message.WhichOneof("payload")
        if kind == "model_audio_chunk":
            chunk = message.model_audio_chunk
            if chunk.audio.data and not self._was_cut(chunk):
                self._replying = True
                self._on_agent_audio(bytes(chunk.audio.data))
            return
        if kind == "vad_analysis_frame":
            return  # telemetry; not requested, but never let it flood the raw log
        self.log.raw({kind: json_format.MessageToDict(getattr(message, kind), preserving_proto_field_name=True)}
                     if kind else {})

        if kind == "model_text_fragment":
            fragment = message.model_text_fragment
            if fragment.text and not self._was_cut(fragment):
                self._reply_text.append(fragment.text)
        elif kind == "response_begin":
            self._turn = message.response_begin.turn_id
            self._turn_open = True
            self._reply_text = []
        elif kind == "response_end":
            # The end of *generation*: audio is shipped in a burst and keeps
            # playing after this, so the reply can still be talked over.
            self._turn_open = False
            self._end_reply("completed")
        elif kind == "playback_clear_buffer":
            audible, self._audible_at_onset = self._audible_at_onset or self.agent_speaking, False
            if self._turn_open or audible:
                self._cut_turn = self._turn
                self._turn_open = False
                self._on_agent_interrupted()
                self._end_reply("cancelled")
        elif kind == "vad_state_event":
            event = message.vad_state_event
            before, after = proto.VadState.Name(event.from_state), proto.VadState.Name(event.to_state)
            session_ms = round(event.session_time.seconds * 1000 + event.session_time.nanos / 1e6, 1)
            if after == "SPEECH" and before == "SPEECH_STARTING":
                # Read before the onset handler clears playback: the
                # clear-buffer that follows is a barge-in only if this was true.
                self._audible_at_onset = self.agent_speaking or self._turn_open
                self._on_caller_speech_started(audio_ms=session_ms)
            elif after == "SILENCE" and before == "SPEECH_ENDING":
                self.log.emit(ev.VAD_SPEECH_END, audio_ms=session_ms)
        elif kind == "user_transcription_result":
            text = message.user_transcription_result.text
            if text.strip():
                self.caller_text.append(text)
                self.log.emit(ev.CALLER_TRANSCRIPT, text=text)
        elif kind == "tool_call_request":
            request = message.tool_call_request
            arguments = json_format.MessageToDict(request.parameters) if request.HasField("parameters") else {}
            call = {"name": request.name, "call_id": request.id, "arguments": whole_numbers(arguments)}
            self.tool_calls.append(call)
            self.log.emit(ev.TOOL_CALL, **call)
        elif kind == "session_ready":
            self.session_ack = {"session_ready": {}, "echoes_config": False, "applied_as_sent": self.session_sent}
            self._ready.set()
        elif kind == "error":
            error = message.error
            self.log.emit(ev.SESSION_ERROR, error=json_format.MessageToDict(error, preserving_proto_field_name=True))
