"""Phonic's speech-to-speech model, spoken directly rather than through a framework service.

Pipecat ships no service for this provider, so the protocol is spoken here, in
the benchmark, for the same reason ``qwen_realtime`` is: a row nobody can read is
a row nobody can check.

Three things about this protocol decide how the service is written, and each is
silent when got wrong.

*The output stream never stops.* Audio is paced to real time and keeps coming
between replies as silence. Forwarded as it arrives, the pipeline would see an
agent that never stops speaking -- no reply would ever end, and every latency
and barge-in figure would be measured against that. Speech is therefore taken
from the service's own ``assistant_started_speaking`` / ``assistant_finished_speaking``
events, and only the audio between them is forwarded.

*Being talked over is the service's decision.* It stops a reply only once the
caller has said a set number of words, and says so with ``interrupted_response``.
If the pipeline cut the agent off at the first sound of the caller instead, a
cough would silence a reply the service goes on streaming. So the row asks the
pipeline not to broadcast interruptions, and this service broadcasts one when
the provider reports it.

*Inline tools must be strict.* Every parameter is required, so an optional one
has to be declared nullable, and the model then sends ``null`` for a field it
leaves out. Those nulls are removed before the call reaches the tools, so the
trace records the same arguments another row's would.

Protocol: https://docs.phonic.co/api-reference/conversations/conversations
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
from dataclasses import fields
from typing import Any

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
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.time import time_now_iso8601

URL = "wss://api.phonic.ai/v1/sts/ws"

#: One rate each way: the service takes and returns 16 kHz PCM when asked to.
SAMPLE_RATE = 16000
AUDIO_FORMAT = "pcm_16000"

#: How long the service waits for a tool's output before giving up on it. Its
#: default is 5 s; the mock tools answer in milliseconds, so this only has to be
#: long enough never to be the reason a call failed.
TOOL_OUTPUT_TIMEOUT_MS = 60000

#: How long to wait for the service to accept the configuration.
READY_TIMEOUT_S = 10.0


def strict_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """A tool's parameters in the strict form the service requires.

    Every property becomes required and every object closed. A property that
    was optional stays optional in meaning by accepting ``null``.
    """
    schema = copy.deepcopy(schema)
    if schema.get("type") == "object":
        properties = schema.setdefault("properties", {})
        required = set(schema.get("required") or ())
        for name, prop in properties.items():
            properties[name] = strict_parameters(prop)
            if name not in required:
                _allow_null(properties[name])
        schema["required"] = list(properties)
        schema["additionalProperties"] = False
    elif schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        schema["items"] = strict_parameters(schema["items"])
    return schema


def _allow_null(prop: dict[str, Any]) -> None:
    kind = prop.get("type")
    if isinstance(kind, str):
        prop["type"] = [kind, "null"]
    elif isinstance(kind, list) and "null" not in kind:
        prop["type"] = [*kind, "null"]
    if "enum" in prop and None not in prop["enum"]:
        prop["enum"] = [*prop["enum"], None]


def _spoken(text: str) -> str:
    return " ".join(text.split())


def without_nulls(arguments: dict[str, Any]) -> dict[str, Any]:
    """The arguments a model left out, left out: strict mode sends them as ``null``."""
    return {key: value for key, value in arguments.items() if value is not None}


class PhonicRealtimeLLMService(LLMService):
    """One Phonic conversation, for the length of one call.

    The socket is opened with the pipeline; the configuration is sent with the
    first context frame, which is the one carrying the tools, and the call opens
    once the service has accepted it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        instructions: str = "",
        settings: dict[str, Any] | None = None,
        tools: ToolsSchema | list | None = None,
        **kwargs,
    ):
        # The framework's settings object, filled in: the model and prompt are
        # what this service sends, and the sampling fields have no counterpart.
        unsupported = {f.name: None for f in fields(LLMSettings) if f.name != "extra"}
        super().__init__(
            settings=LLMSettings(**{**unsupported, "model": model, "system_instruction": instructions}), **kwargs
        )
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._instructions = instructions
        # Further configuration keys, sent as they are: the row names every
        # setting it depends on rather than leaving it to a server default.
        self._options = dict(settings or {})
        self._tools = tools
        self._websocket = None
        self._closing = False
        self._receive_task = None
        self._open_task = None
        self._ready = asyncio.Event()
        self._context = None
        self._speaking = False
        self._conversation_id: str | None = None
        # The opening instruction, until the service echoes it back. It is sent
        # as a user message, and the service reports it as ``input_text`` as if
        # the caller had said it.
        self._opening_echo: str | None = None
        # Results already handed over, and calls the service withdrew: a result
        # for a withdrawn call has nobody waiting for it.
        self._delivered: set[str] = set()
        self._withdrawn: set[str] = set()

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
                uri=URL, additional_headers={"Authorization": f"Bearer {self._api_key}"}
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

    async def _send(self, message: dict[str, Any]):
        if not self._websocket:
            return
        await self._websocket.send(json.dumps(message))

    def _encoded_tools(self) -> list[dict[str, Any]]:
        if not self._tools:
            return []
        standard = self._tools.standard_tools if isinstance(self._tools, ToolsSchema) else self._tools
        encoded = []
        for tool in standard:
            schema = tool.to_default_dict() if hasattr(tool, "to_default_dict") else dict(tool)
            function = schema.get("function", schema)
            encoded.append(
                {
                    "type": "custom_websocket",
                    "tool_call_output_timeout_ms": TOOL_OUTPUT_TIMEOUT_MS,
                    "tool_schema": {
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "description": function.get("description", ""),
                            "parameters": strict_parameters(
                                function.get("parameters") or {"type": "object", "properties": {}}
                            ),
                            "strict": True,
                        },
                    },
                }
            )
        return encoded

    def config(self) -> dict[str, Any]:
        """The one message that configures the conversation; nothing is changed after it."""
        return {
            "type": "config",
            "system_prompt": self._instructions,
            "voice_id": self._voice,
            "phonic_model": self._model,
            "input_format": AUDIO_FORMAT,
            "output_format": AUDIO_FORMAT,
            "tools": self._encoded_tools(),
            **self._options,
        }

    async def _send_audio(self, frame: InputAudioRawFrame):
        # Audio sent before the configuration is accepted would be answered
        # under defaults nobody chose, so it is dropped.
        if not self._ready.is_set():
            return
        await self._send({"type": "audio_chunk", "audio": base64.b64encode(frame.audio).decode()})

    # -- pipeline ---------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_audio(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._end_reply()
        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            pass  # the provider's own turn detection owns these

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
            if tool_call_id in self._withdrawn:
                logger.info(f"{self} result for withdrawn tool call {tool_call_id} not sent")
                continue
            await self._send({"type": "tool_call_output", "tool_call_id": tool_call_id, "output": output})

    @staticmethod
    def _results(context) -> dict[str, Any]:
        """Completed tool results in the context, by call id."""
        results = {}
        for message in context.get_messages():
            if isinstance(message, LLMSpecificMessage) or message.get("role") != "tool":
                continue
            tool_call_id, content = message.get("tool_call_id"), message.get("content")
            if not tool_call_id or content == "IN_PROGRESS":
                continue
            try:
                results[tool_call_id] = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError:
                results[tool_call_id] = content
        return results

    @staticmethod
    def _opening(context) -> str | None:
        """The opening instruction every row is given, as the turn the model replies to."""
        for message in reversed(context.get_messages()):
            if not isinstance(message, LLMSpecificMessage) and message.get("role") == "user":
                content = message.get("content")
                return content if isinstance(content, str) else None
        return None

    async def _open(self, opening: str | None):
        await self._send(self.config())
        try:
            await asyncio.wait_for(self._ready.wait(), READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self.push_error(error_msg=f"Phonic did not accept the configuration in {READY_TIMEOUT_S:.0f}s")
            return
        self._opening_echo = _spoken(opening) if opening else None
        await self._send({"type": "generate_reply", **({"user_message": opening} if opening else {})})

    async def _end_reply(self):
        if not self._speaking:
            return
        self._speaking = False
        await self.push_frame(TTSStoppedFrame())
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
        if not self._closing:
            # Closed from the far end while the call was still up.
            await self.push_error(error_msg=f"Phonic closed the conversation {self._conversation_id}")

    async def _dispatch(self, event: dict[str, Any]):
        kind = event.get("type", "")

        if kind == "audio_chunk":
            await self._handle_audio(event)

        elif kind == "assistant_started_speaking":
            await self._start_reply()

        elif kind == "assistant_finished_speaking":
            await self._end_reply()

        elif kind == "interrupted_response":
            # The service has stopped the reply; stop what is already queued here.
            logger.debug(f"{self} reply interrupted after: {event.get('text', '')!r}")
            await self._end_reply()
            await self.broadcast_interruption()

        elif kind == "user_started_speaking":
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

        elif kind == "user_finished_speaking":
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

        elif kind == "input_text":
            if self._opening_echo is not None and _spoken(event.get("text", "")) == self._opening_echo:
                # Our own opening instruction, not the caller: kept out of the transcript.
                self._opening_echo = None
                logger.debug(f"{self} dropped the echo of the opening instruction")
                return
            # Upstream: the context's user half sits before this service.
            await self.push_frame(
                TranscriptionFrame(event.get("text", ""), "", time_now_iso8601(), result=event),
                FrameDirection.UPSTREAM,
            )

        elif kind == "tool_call":
            await self._handle_tool_call(event)

        elif kind == "tool_call_interrupted":
            self._withdrawn.add(event.get("tool_call_id", ""))
            logger.info(f"{self} withdrew tool call {event.get('tool_name')} ({event.get('tool_call_id')})")

        elif kind == "ready_to_start_conversation":
            self._ready.set()

        elif kind == "conversation_created":
            self._conversation_id = event.get("conversation_id")
            logger.info(f"{self} conversation {self._conversation_id}")

        elif kind == "assistant_ended_conversation":
            # No hang-up tool of the service's own is configured, so this is
            # the service ending the conversation by itself.
            logger.warning(f"{self} the service ended conversation {self._conversation_id}")

        elif kind == "warning":
            logger.warning(f"{self} warning: {event.get('warning')}")

        elif kind == "error":
            await self.push_error(
                error_msg=f"Phonic error: {event.get('error')} {event.get('param_errors') or ''}".rstrip()
            )

    async def _start_reply(self):
        if self._speaking:
            return
        self._speaking = True
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TTSStartedFrame())

    async def _handle_audio(self, event: dict[str, Any]):
        text = event.get("text") or ""
        if not self._speaking:
            if not text:
                return  # the silence between replies
            await self._start_reply()
        audio = base64.b64decode(event.get("audio") or "")
        if audio:
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1))
        if text:
            # What the model said, as it says it: the agent's half of the transcript.
            llm_text = LLMTextFrame(text)
            llm_text.append_to_context = False
            await self.push_frame(llm_text)
            tts_text = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
            tts_text.includes_inter_frame_spaces = True
            await self.push_frame(tts_text)

    async def _handle_tool_call(self, event: dict[str, Any]):
        name = event.get("tool_name", "")
        tool_call_id = event.get("tool_call_id", "")
        if not name or not tool_call_id:
            logger.warning(f"{self} tool call without a name or id: {event}")
            return
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    context=self._context,
                    tool_call_id=tool_call_id,
                    function_name=name,
                    arguments=without_nulls(event.get("parameters") or {}),
                )
            ]
        )
