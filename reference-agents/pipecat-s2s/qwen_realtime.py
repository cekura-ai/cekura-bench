"""Qwen's realtime audio model, spoken directly rather than through a framework service.

Pipecat ships a realtime service for every other provider on the board, and
none for this one. It is written here, in the benchmark, for the same reason
everything else here is one readable file: a row nobody can read is a row
nobody can check.

*Why not subclass the OpenAI Realtime service.* Qwen's wire protocol is
OpenAI-Realtime-shaped but it is the **beta** shape -- ``modalities``,
``input_audio_format`` and ``turn_detection`` at the top level of the session --
while Pipecat's OpenAI service now speaks the GA shape, where those moved inside
a nested ``audio`` object. Subclassing would mean overriding the session
serialisation, the event models and the tool encoding, which is most of the
service, and it would break on the framework's next revision of a shape Qwen
does not follow. The protocol is small enough to speak directly.

*The sample rates are not symmetric.* Qwen takes 16 kHz and returns 24 kHz.
Every other provider here is symmetric, so ``PROVIDERS[...].input_rate`` alone
does not describe this one: the pipeline runs at 16 kHz and outbound frames
declare 24 kHz, which the output transport resamples. Declaring the wrong rate
on those frames does not fail -- it plays the model's voice at the wrong speed,
which reads as a bad model rather than bad wiring.

Protocol: https://www.alibabacloud.com/help/en/model-studio/qwen-audio-realtime-user-guides
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.utils.time import time_now_iso8601

#: Qwen accepts 16 kHz and returns 24 kHz. Both are fixed by the service.
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

#: The region a workspace lives in decides the endpoint *and* the key; a key
#: issued for one region does not authenticate against the other.
ENDPOINTS = {
    "singapore": "wss://{workspace}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime",
    "beijing": "wss://{workspace}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
}


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex[:24]}"


class QwenRealtimeLLMService(LLMService):
    """One Qwen realtime audio session, for the length of one call.

    The session is opened on the first frame and closed with the pipeline. Turn
    detection is the provider's own, which is the same choice every other native
    row on this board makes: a benchmark that replaced each provider's
    endpointer with a shared one would be measuring the shared endpointer.
    """

    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str,
        model: str = "qwen-audio-3.0-realtime-plus",
        voice: str = "longanqian",
        instructions: str = "",
        region: str = "singapore",
        turn_detection: dict[str, Any] | None = None,
        tools: ToolsSchema | list | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if region not in ENDPOINTS:
            raise ValueError(f"region must be one of {sorted(ENDPOINTS)}, not {region!r}")
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._tools = tools
        self._url = f"{ENDPOINTS[region].format(workspace=workspace_id)}?model={model}"
        # semantic_vad is the vendor's own recommendation for this model family.
        # It is recorded on the build record rather than tuned: an endpointer
        # tuned per provider is a configuration difference masquerading as a
        # model difference.
        self._turn_detection = turn_detection or {
            "type": "semantic_vad",
            "silence_duration_ms": 800,
        }
        self._websocket = None
        self._receive_task = None
        self._context = None
        self._session_ready = False
        self._speaking = False
        self._responding = False
        # call_id -> name, populated as the server streams a function call and
        # read when its arguments complete.
        self._pending_calls: dict[str, str] = {}

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
                uri=self._url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
            self._receive_task = self.create_task(self._receive())
            await self._send_session_update()
        except Exception as exc:
            await self.push_error(error_msg=f"Error connecting: {exc}", exception=exc)
            self._websocket = None

    async def _disconnect(self):
        try:
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
            self._session_ready = False
        except Exception as exc:
            await self.push_error(error_msg=f"Error disconnecting: {exc}", exception=exc)

    # -- sending ----------------------------------------------------------

    async def _send(self, event: dict[str, Any]):
        if not self._websocket:
            return
        event.setdefault("event_id", _event_id())
        await self._websocket.send(json.dumps(event))

    def _encoded_tools(self) -> list[dict] | None:
        """Qwen takes OpenAI's *nested* tool shape, not the realtime flat one.

        The realtime APIs this protocol is modelled on put ``name`` and
        ``parameters`` directly on the tool; Qwen keeps them under ``function``,
        as chat completions does. A tool sent in the wrong shape is not
        rejected -- it is ignored, and the model then fails a scenario for
        having no tool rather than for anything about the model.
        """
        if not self._tools:
            return None
        standard = self._tools.standard_tools if isinstance(self._tools, ToolsSchema) else self._tools
        encoded = []
        for tool in standard:
            schema = tool.to_default_dict() if hasattr(tool, "to_default_dict") else dict(tool)
            function = schema.get("function", schema)
            encoded.append(
                {
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters", {}),
                    },
                }
            )
        return encoded

    async def _send_session_update(self):
        # No input-transcription field: the audio model documents the caller's
        # completed transcript as an event it sends on its own, with no setting
        # to ask for it, and a guessed field could fail the session. The event
        # is handled below. Until a real call has shown caller text arriving,
        # the row must not be published.
        session: dict[str, Any] = {
            "modalities": ["text", "audio"],
            "voice": self._voice,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "turn_detection": self._turn_detection,
        }
        if self._instructions:
            session["instructions"] = self._instructions
        tools = self._encoded_tools()
        if tools:
            session["tools"] = tools
        await self._send({"type": "session.update", "session": session})

    async def _send_audio(self, frame: InputAudioRawFrame):
        # Audio sent before the session is configured is accepted and answered
        # under default settings -- wrong voice, wrong prompt, no tools -- so it
        # is dropped rather than allowed to open a call nobody configured.
        if not self._session_ready:
            return
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame.audio).decode(),
            }
        )

    async def _create_response(self):
        await self._send({"type": "response.create"})

    # -- pipeline ---------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_audio(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption()
        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            pass  # the provider's own turn detection owns these

        await self.push_frame(frame, direction)

    async def _handle_context(self, context):
        """The first context frame carries the tools and opens the call.

        A realtime session is configured once and then runs, so the tools have
        to reach it before the first response rather than with each turn. The
        greeting is an instruction in this same first context, which is what
        makes every call on this board open from the agent with the same words.
        """
        first = self._context is None
        self._context = context
        if first:
            tools = getattr(context, "tools", None)
            if tools:
                self._tools = tools
                await self._send_session_update()
            await self._create_response()

    async def _handle_interruption(self):
        await self._send({"type": "response.cancel"})
        if self._speaking:
            self._speaking = False
            await self.push_frame(TTSStoppedFrame())
        if self._responding:
            self._responding = False
            await self.push_frame(LLMFullResponseEndFrame())
        await self.stop_all_metrics()

    # -- receiving --------------------------------------------------------

    async def _receive(self):
        assert self._websocket is not None
        async for message in self._websocket:
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received a non-JSON frame")
                continue
            try:
                await self._dispatch(event)
            except Exception as exc:  # noqa: BLE001 -- one bad event must not end the call
                logger.error(f"{self} failed on {event.get('type')}: {exc}")

    async def _dispatch(self, event: dict[str, Any]):
        kind = event.get("type", "")

        if kind in ("session.created", "session.updated"):
            self._session_ready = True

        elif kind == "response.audio.delta":
            await self.stop_ttfb_metrics()
            if not self._speaking:
                self._speaking = True
                await self.push_frame(TTSStartedFrame())
            await self.push_frame(
                TTSAudioRawFrame(
                    audio=base64.b64decode(event["delta"]),
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    num_channels=1,
                )
            )

        elif kind == "response.audio.done":
            if self._speaking:
                self._speaking = False
                await self.push_frame(TTSStoppedFrame())

        elif kind == "response.created":
            self._responding = True
            await self.push_frame(LLMFullResponseStartFrame())

        elif kind == "response.done":
            if self._speaking:
                self._speaking = False
                await self.push_frame(TTSStoppedFrame())
            if self._responding:
                self._responding = False
                await self.push_frame(LLMFullResponseEndFrame())
            await self.stop_processing_metrics()

        elif kind == "response.audio_transcript.delta":
            # What the model said, as it says it. This is the assistant side of
            # the transcript the run is scored on.
            await self.push_frame(LLMTextFrame(event.get("delta", "")))

        elif kind == "conversation.item.input_audio_transcription.completed":
            await self.push_frame(
                TranscriptionFrame(
                    event.get("transcript", ""), "", time_now_iso8601(), result=event
                )
            )

        elif kind == "input_audio_buffer.speech_started":
            await self.start_ttfb_metrics()
            await self.broadcast_frame(UserStartedSpeakingFrame)

        elif kind == "input_audio_buffer.speech_stopped":
            await self.start_processing_metrics()
            await self.broadcast_frame(UserStoppedSpeakingFrame)

        elif kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                self._pending_calls[item.get("call_id", "")] = item.get("name", "")

        elif kind == "response.function_call_arguments.done":
            await self._handle_function_call(event)

        elif kind == "error":
            error = event.get("error") or {}
            await self.push_error(error_msg=f"Qwen Realtime error: {error}")

    async def _handle_function_call(self, event: dict[str, Any]):
        call_id = event.get("call_id", "")
        name = event.get("name") or self._pending_calls.pop(call_id, "")
        if not name:
            logger.warning(f"{self} function call {call_id} arrived with no name")
            return
        try:
            arguments = json.loads(event.get("arguments") or "{}")
        except json.JSONDecodeError:
            logger.error(f"{self} could not parse arguments for {name}")
            return
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    context=self._context,
                    tool_call_id=call_id,
                    function_name=name,
                    arguments=arguments,
                )
            ]
        )

    async def _send_tool_result(self, tool_call_id: str, result: str | None):
        """Return one tool result and ask for the reply it feeds.

        Qwen does not resume on its own after a tool result, unlike the
        providers whose turn detection restarts the response: the explicit
        ``response.create`` is what keeps the call from stalling silently after
        a lookup succeeds.
        """
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": f"item_{uuid.uuid4().hex[:24]}",
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": result,
                },
            }
        )
        await self._create_response()
