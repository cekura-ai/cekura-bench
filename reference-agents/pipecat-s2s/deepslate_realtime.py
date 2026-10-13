"""Deepslate's speech-to-speech model (Opal), spoken directly rather than through its plugin.

Deepslate publishes a Pipecat plugin, and this row does not use it, for the
reason ``phonic_realtime`` exists: the plugin calls a tool's handler itself and
never tells the pipeline a call was made, so no tool call reaches the context or
the run's record, and every tool score on the row would be empty. It also
ignores the context the pipeline opens the call with, which is where every row
receives its greeting and its tools.

The wire format is protobuf, so the message classes come from the vendor's own
SDK, pinned; only the generated classes are used. The SDK's session layer is
not, for two reasons that are each silent when they bite. It configures the
session lazily, at 24 kHz whenever anything other than audio is sent first, and
then reconfigures only the *input* side when 16 kHz audio follows -- so the
agent's replies arrive at one rate labelled as another. And it reconnects on
its own after a drop, into a fresh session that remembers nothing of the call:
a run that should have failed carries on with an agent that has forgotten the
caller. Here the session is configured once, at the pipeline's rate, before any
audio, and a drop is an error.

Three things about this protocol decide how the service is written.

*The service owns the caller's turn.* Its own detector reports each transition,
and it tells the client to discard queued audio the moment the caller starts
speaking. So the row follows the service's boundaries, asks the pipeline not to
broadcast interruptions, and passes the service's own on instead -- only while
the agent is speaking, because the service sends the same message whether or
not anything is playing.

*Speech is text first.* Opal produces text and the service voices it with a
configured text-to-speech step, so a reply's words arrive ahead of its audio,
the way an audio transcript arrives ahead of playback on the other realtime
rows. That step is part of the row and is named on its record.

*Tool arguments travel as a protobuf Struct*, which has one number type, so a
whole number the model wrote comes back as a float -- a ZIP code ``90210``
reaches the tools as ``90210.0``. Whole floats are turned back into integers
before the call is made, so the trace records what the model wrote.

Protocol: https://docs.deepslate.eu/api-reference/realtime
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields
from typing import Any

from deepslate.core.proto import realtime_pb2 as proto
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct
from loguru import logger
from websockets.asyncio.client import connect as websocket_connect

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    AggregationType,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.time import time_now_iso8601

URL = "wss://app.deepslate.eu/api/v1/vendors/{vendor_id}/organizations/{organization_id}/realtime"

#: One rate each way. The session declares both lines, so the service returns
#: audio at the rate it is sent.
SAMPLE_RATE = 16000

#: How long to wait for the service to accept the configuration.
READY_TIMEOUT_S = 10.0

#: What a caller's audio does to a reply in progress. The vendor's SDK sends
#: ``IMMEDIATE`` with every audio frame: a turn the service's detector closes
#: starts a reply at once, replacing any still being generated.
TRIGGER = proto.InferenceTriggerMode.IMMEDIATE


def _duration(ms: int) -> proto.Duration:
    return proto.Duration(seconds=ms // 1000, nanos=(ms % 1000) * 1_000_000)


def whole_numbers(value: Any) -> Any:
    """A Struct's floats that were whole numbers, as integers again; see the module docstring."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: whole_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [whole_numbers(item) for item in value]
    return value


class DeepslateRealtimeLLMService(LLMService):
    """One Deepslate session, for the length of one call.

    The socket is opened with the pipeline; the session is configured with the
    first context frame, which is the one carrying the tools, and the call
    opens once the service reports the session ready.
    """

    def __init__(
        self,
        *,
        api_key: str,
        vendor_id: str,
        organization_id: str,
        instructions: str = "",
        tts: dict[str, Any],
        vad: dict[str, Any],
        temperature: float,
        model: str = "opal",
        tools: ToolsSchema | list | None = None,
        **kwargs,
    ):
        # The framework's settings object, filled in: the model and prompt are
        # what this service runs with, and the other fields have no counterpart.
        unsupported = {f.name: None for f in fields(LLMSettings) if f.name != "extra"}
        super().__init__(
            settings=LLMSettings(**{**unsupported, "model": model, "system_instruction": instructions}), **kwargs
        )
        self._api_key = api_key
        self._url = URL.format(vendor_id=vendor_id, organization_id=organization_id)
        self._instructions = instructions
        self._tts = dict(tts)
        self._vad = dict(vad)
        self._temperature = temperature
        self._tools = tools
        self._websocket = None
        self._closing = False
        self._receive_task = None
        self._open_task = None
        self._ready = asyncio.Event()
        self._context = None
        self._speaking = False
        self._packet_id = 0
        # The reply being delivered, and the last one the caller talked over:
        # audio already in flight for an interrupted reply must not reopen it.
        self._turn: int | None = None
        self._cut_turn: int | None = None
        # Words that arrived ahead of their reply's audio, by turn: the text
        # leads the voice by up to ~1.5 s, and a reply cut off or hung up on
        # before it makes a sound must leave no words in the record.
        self._held: list[tuple[int | None, str]] = []
        self._voiced = False
        # Results already handed over: the context is pushed again for every
        # tool, and each result must be sent exactly once.
        self._delivered: set[str] = set()

    # -- lifecycle --------------------------------------------------------

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        if self._websocket:
            return
        try:
            self._websocket = await websocket_connect(
                uri=self._url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            )
            self._receive_task = self.create_task(self._receive())
        except Exception as exc:
            await self.push_error(error_msg=f"Error connecting: {exc}", exception=exc)
            self._websocket = None

    async def _disconnect(self):
        self._closing = True
        try:
            if self._open_task:
                await self.cancel_task(self._open_task, timeout=1.0)
                self._open_task = None
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
            self._ready.clear()
        except Exception as exc:
            await self.push_error(error_msg=f"Error disconnecting: {exc}", exception=exc)

    # -- sending ----------------------------------------------------------

    async def _send(self, message: proto.ServiceBoundMessage):
        if not self._websocket:
            return
        await self._websocket.send(message.SerializeToString())

    def _tts_configuration(self) -> proto.TtsConfiguration:
        if self._tts["provider"] == "elevenlabs":
            return proto.TtsConfiguration(
                eleven_labs=proto.ElevenLabsTtsConfiguration(
                    api_key=self._tts["api_key"],
                    voice_id=self._tts["voice_id"],
                    model_id=self._tts["model_id"],
                    location=proto.ElevenLabsLocation.Value(self._tts["location"]),
                )
            )
        if self._tts["provider"] == "hosted":
            return proto.TtsConfiguration(
                hosted=proto.HostedTtsConfiguration(
                    voice_ref=proto.HostedVoiceRef(voice_id=self._tts["voice_id"]),
                    mode=proto.HostedTtsMode.Value(self._tts["mode"]),
                )
            )
        raise ValueError(f"unknown Deepslate text-to-speech provider {self._tts['provider']!r}")

    def initialize_request(self) -> proto.InitializeSessionRequest:
        """The one message that configures the session; every field is sent, none left to a default."""
        line = proto.AudioLineConfiguration(
            sample_rate=SAMPLE_RATE, channel_count=1, sample_format=proto.SampleFormat.SIGNED_16_BIT
        )
        return proto.InitializeSessionRequest(
            input_audio_line=line,
            output_audio_line=line,
            vad_configuration=proto.VadConfiguration(
                confidence_threshold=self._vad["confidence_threshold"],
                min_volume=self._vad["min_volume"],
                start_duration=_duration(self._vad["start_duration_ms"]),
                stop_duration=_duration(self._vad["stop_duration_ms"]),
                backbuffer_duration=_duration(self._vad["backbuffer_duration_ms"]),
            ),
            inference_configuration=proto.InferenceConfiguration(
                system_prompt=self._instructions, temperature=self._temperature
            ),
            tts_configuration=self._tts_configuration(),
            # Off: this pipeline does not count the bytes its transport plays, so
            # on a barge-in the service judges for itself what the caller heard.
            supports_playback_reporting=False,
        )

    def tool_definitions(self) -> proto.UpdateToolDefinitionsRequest:
        standard = []
        if self._tools:
            standard = self._tools.standard_tools if isinstance(self._tools, ToolsSchema) else self._tools
        definitions = []
        for tool in standard:
            schema = tool.to_default_dict() if hasattr(tool, "to_default_dict") else dict(tool)
            function = schema.get("function", schema)
            parameters = Struct()
            json_format.ParseDict(function.get("parameters") or {"type": "object", "properties": {}}, parameters)
            definitions.append(
                proto.ToolDefinition(
                    name=function["name"], description=function.get("description", ""), parameters=parameters
                )
            )
        return proto.UpdateToolDefinitionsRequest(tool_definitions=definitions)

    async def _send_audio(self, frame: InputAudioRawFrame):
        # Audio sent before the session is configured would be refused or
        # answered under defaults nobody chose, so it is dropped.
        if not self._ready.is_set():
            return
        self._packet_id += 1
        await self._send(
            proto.ServiceBoundMessage(
                user_input=proto.UserInput(
                    packet_id=self._packet_id, mode=TRIGGER, audio_data=proto.AudioData(data=frame.audio)
                )
            )
        )

    # -- pipeline ---------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_audio(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._end_reply()

        await self.push_frame(frame, direction)

    async def _handle_context(self, context):
        """The first context opens the call; every later one may carry tool results."""
        first = self._context is None
        self._context = context
        if first:
            tools = getattr(context, "tools", None)
            if tools:
                self._tools = tools
            self._delivered.update(self._results(context))
            self._open_task = self.create_task(self._open(self._opening(context)))
            return
        for tool_call_id, output in self._results(context).items():
            if tool_call_id in self._delivered:
                continue
            self._delivered.add(tool_call_id)
            # The service requires an answer to every call it makes, as a string.
            await self._send(
                proto.ServiceBoundMessage(tool_call_response=proto.ToolCallResponse(id=tool_call_id, result=output))
            )

    @staticmethod
    def _results(context) -> dict[str, str]:
        """Completed tool results in the context, by call id, as the strings the service takes."""
        results = {}
        for message in context.get_messages():
            if isinstance(message, LLMSpecificMessage) or message.get("role") != "tool":
                continue
            tool_call_id, content = message.get("tool_call_id"), message.get("content")
            if not tool_call_id or content == "IN_PROGRESS":
                continue
            results[tool_call_id] = content if isinstance(content, str) else json.dumps(content)
        return results

    @staticmethod
    def _opening(context) -> str | None:
        """The opening instruction every row is given."""
        for message in reversed(context.get_messages()):
            if not isinstance(message, LLMSpecificMessage) and message.get("role") == "user":
                content = message.get("content")
                return content if isinstance(content, str) else None
        return None

    async def _open(self, opening: str | None):
        # The tools follow the configuration directly, before any audio, so the
        # session never starts answering without them.
        await self._send(proto.ServiceBoundMessage(initialize_session_request=self.initialize_request()))
        await self._send(proto.ServiceBoundMessage(update_tool_definitions_request=self.tool_definitions()))
        try:
            await asyncio.wait_for(self._ready.wait(), READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self.push_error(error_msg=f"Deepslate did not accept the configuration in {READY_TIMEOUT_S:.0f}s")
            return
        # The opening goes as the reply's instructions, the vendor's documented
        # way to have the agent speak first, rather than as text input, which
        # would sit in the conversation as something the caller said.
        trigger = proto.TriggerInference(extra_instructions=opening) if opening else proto.TriggerInference()
        await self._send(proto.ServiceBoundMessage(trigger_inference=trigger))

    async def _start_reply(self):
        if self._speaking:
            return
        self._speaking = True
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TTSStartedFrame())

    async def _end_reply(self):
        if not self._speaking:
            return
        self._speaking = False
        self._voiced = False
        await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())
        await self.stop_all_metrics()

    # -- receiving --------------------------------------------------------

    async def _receive(self):
        ws = self._websocket
        assert ws is not None
        async for raw in ws:
            # The socket keeps delivering through its close handshake, and the
            # service may begin another reply after the call is torn down; words
            # nobody heard must not reach the record.
            if self._closing:
                break
            if not isinstance(raw, bytes):
                logger.warning(f"{self} received a text frame on a protobuf socket")
                continue
            message = proto.ClientBoundMessage()
            try:
                message.ParseFromString(raw)
            except Exception as exc:  # noqa: BLE001 -- one bad message must not end the call
                logger.warning(f"{self} could not parse a message: {exc}")
                continue
            try:
                await self._dispatch(message)
            except Exception as exc:  # noqa: BLE001 -- one bad event must not end the call
                logger.error(f"{self} failed on {message.WhichOneof('payload')}: {exc}")
        if not self._closing:
            # Closed from the far end while the call was still up. Not retried:
            # a new session would not know the conversation so far.
            await self.push_error(
                error_msg=f"Deepslate closed the session "
                          f"(code {getattr(ws, 'close_code', None)}, {getattr(ws, 'close_reason', None)!r})"
            )

    async def _dispatch(self, message: proto.ClientBoundMessage):
        kind = message.WhichOneof("payload")

        if kind == "model_audio_chunk":
            chunk = message.model_audio_chunk
            if chunk.audio.data and not self._was_cut(chunk):
                await self._start_reply()
                if not self._voiced:
                    # The reply's first sound releases the words held for it.
                    self._voiced = True
                    turn = self._turn_of(chunk)
                    held, self._held = self._held, []
                    # Words that came before any reply had opened belong to this one.
                    for text in (text for held_turn, text in held if held_turn in (turn, None)):
                        await self._push_text(text)
                await self.stop_ttfb_metrics()
                await self.push_frame(TTSAudioRawFrame(audio=chunk.audio.data, sample_rate=SAMPLE_RATE, num_channels=1))

        elif kind == "model_text_fragment":
            fragment = message.model_text_fragment
            if fragment.text and not self._was_cut(fragment):
                await self._start_reply()
                if self._voiced:
                    await self._push_text(fragment.text)
                else:
                    self._held.append((self._turn_of(fragment), fragment.text))

        elif kind == "response_begin":
            self._turn = message.response_begin.turn_id
            await self._start_reply()

        elif kind == "response_end":
            await self._end_reply()

        elif kind == "playback_clear_buffer":
            # Sent whenever the caller starts speaking; it interrupts only a
            # reply that is still being delivered.
            if self._speaking:
                self._cut_turn = self._turn
                self._held = []
                await self._end_reply()
                await self.broadcast_interruption()

        elif kind == "vad_state_event":
            await self._handle_vad(message.vad_state_event)

        elif kind == "user_transcription_result":
            result = message.user_transcription_result
            if result.text.strip():
                # Upstream: the context's user half sits before this service.
                await self.push_frame(
                    TranscriptionFrame(result.text, "", time_now_iso8601(), result={"language": result.language}),
                    FrameDirection.UPSTREAM,
                )

        elif kind == "tool_call_request":
            await self._handle_tool_call(message.tool_call_request)

        elif kind == "session_ready":
            logger.info(f"{self} session ready")
            self._ready.set()

        elif kind == "context_truncated":
            # The model no longer sees these turns: a later lapse may be this.
            truncated = message.context_truncated
            logger.warning(
                f"{self} context truncated: turns {list(truncated.truncated_turn_ids)} "
                f"dropped before turn {truncated.response_turn_id}"
            )

        elif kind == "error":
            error = message.error
            await self.push_error(
                error_msg=f"Deepslate error {proto.SessionErrorCategory.Name(error.category)}: {error.message}"
                          + (f" (trace {error.trace_id})" if error.HasField("trace_id") else "")
            )

    def _turn_of(self, part) -> int | None:
        """Turn ids are 0-based and optional; a piece without one belongs to the latest reply."""
        return part.turn_id if part.HasField("turn_id") else self._turn

    def _was_cut(self, part) -> bool:
        """Whether this piece of a reply belongs to one the caller talked over."""
        turn = self._turn_of(part)
        return turn is not None and turn == self._cut_turn

    async def _push_text(self, text: str):
        frame = TTSTextFrame(text, aggregated_by=AggregationType.TOKEN)
        frame.includes_inter_frame_spaces = True
        await self.push_frame(frame)

    async def _handle_vad(self, event):
        before, after = proto.VadState.Name(event.from_state), proto.VadState.Name(event.to_state)
        if after == "SPEECH" and before == "SPEECH_STARTING":
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        elif after == "SILENCE" and before == "SPEECH_ENDING":
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _handle_tool_call(self, request: proto.ToolCallRequest):
        if not request.name or not request.id:
            logger.warning(f"{self} tool call without a name or id: {request}")
            return
        arguments = json_format.MessageToDict(request.parameters) if request.HasField("parameters") else {}
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    context=self._context,
                    tool_call_id=request.id,
                    function_name=request.name,
                    arguments=whole_numbers(arguments),
                )
            ]
        )
