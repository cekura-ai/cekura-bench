"""Deepgram's Voice Agent API, spoken directly rather than through a framework service.

Pipecat ships no service for this socket, so the protocol is spoken here, as it
is for Phonic and Deepslate: a row nobody can read is a row nobody can check.

It is not one speech model but a pipeline Deepgram runs behind one socket.
Flux hears the caller and decides when the turn ended, a text model replies,
and an Aura voice speaks. The text model is GPT-4.1 on our own OpenAI account,
through the bring-your-own endpoint Deepgram documents, not Deepgram's hosted
copy: a hosted model's prompt is capped at 25,000 characters and Deepgram cuts
the rest with only a warning, and the medicare contract's prompt is longer.
Same model, whole prompt.

What this protocol does differently, and how the service follows it:

*Audio is binary.* Caller audio goes out as raw 16 kHz PCM frames and the
agent's comes back as raw 24 kHz PCM frames; every JSON message is a text
frame. Nothing is base64.

*The service yields on the caller's first sound.* ``UserStartedSpeaking``
arrives and the reply in progress ends at once with ``AgentAudioDone``. The
pipeline cuts its own playback on the same announcement, so the row's
barge-in is Deepgram's; audio for the cut reply still in flight is dropped.

*Tools are run here.* A function declared without an endpoint is sent to the
client as a ``FunctionCallRequest`` and answered with a ``FunctionCallResponse``.
A call can be sent before Flux has confirmed the turn; if the caller keeps
talking, ``FunctionCallCancelled`` withdraws it and Deepgram would drop any
reply to it, so none is sent, and the trace marks it withdrawn.

*The opening turn is injected.* Every row opens on the same instruction (say
this greeting). It is sent with ``InjectUserMessage``, and Deepgram reports it
back as the user's words; that echo is kept out of the caller's transcript.

Protocol: https://developers.deepgram.com/docs/voice-agent
"""

from __future__ import annotations

import asyncio
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
)
from pipecat.processors.aggregators.llm_context import LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, FunctionCallRunnerItem, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.time import time_now_iso8601

URL = "wss://agent.deepgram.com/v1/agent/converse"

#: Flux's native rate in; Aura's native rate out, resampled by the output transport.
INPUT_RATE = 16000
OUTPUT_RATE = 24000

#: How the caller is heard: Flux at its own end-of-turn defaults, as shipped.
LISTEN = {"type": "deepgram", "version": "v2", "model": "flux-general-en"}

#: Where each text-model provider type is reached on our own account. Only
#: OpenAI is wired: it is the row's model.
THINK_ENDPOINTS = {"open_ai": "https://api.openai.com/v1/chat/completions"}

#: How long to wait for the service to apply the settings.
READY_TIMEOUT_S = 10.0

#: The socket is closed after a stretch with no message from the client.
#: Audio normally keeps it open; this covers a transport that pauses.
KEEPALIVE_S = 5.0


def split_model(model: str) -> tuple[str, str]:
    """``open_ai/gpt-4.1`` -> (``open_ai``, ``gpt-4.1``). Only the first slash splits."""
    kind, sep, name = model.partition("/")
    if not sep or not kind or not name:
        raise ValueError(f"deepgram-voice-agent model must be '<provider type>/<model>', got {model!r}")
    if kind not in THINK_ENDPOINTS:
        raise ValueError(f"deepgram-voice-agent runs its text model on our own account; no endpoint is wired for {kind!r}")
    return kind, name


def _spoken(text: str) -> str:
    return " ".join(text.split())


class DeepgramVoiceAgentLLMService(LLMService):
    """One Deepgram Voice Agent session, for the length of one call.

    The socket is opened with the pipeline; the settings are sent with the
    first context frame, which is the one carrying the tools, and the call
    opens once the service has applied them.
    """

    def __init__(
        self,
        *,
        api_key: str,
        think_api_key: str,
        model: str,
        voice: str,
        instructions: str = "",
        tools: ToolsSchema | list | None = None,
        **kwargs,
    ):
        # The framework's settings object, filled in: the model and prompt are
        # what this service sends, and the sampling fields have no counterpart.
        unsupported = {f.name: None for f in fields(LLMSettings) if f.name != "extra"}
        super().__init__(
            settings=LLMSettings(**{**unsupported, "model": model, "system_instruction": instructions}), **kwargs
        )
        self._think_kind, self._think_model = split_model(model)
        self._api_key = api_key
        self._think_api_key = think_api_key
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._tools = tools
        self._websocket = None
        self._closing = False
        self._receive_task = None
        self._open_task = None
        self._keepalive_task = None
        self._ready = asyncio.Event()
        self._context = None
        self._speaking = False
        # Set when the caller cut a reply in; audio for that reply still in
        # flight is dropped until the service closes it.
        self._cut = False
        self._session_id: str | None = None
        # The opening instruction, until the service echoes it back as the user's words.
        self._opening_echo: str | None = None
        # Results already handed over, calls the service withdrew, and each call's name.
        self._delivered: set[str] = set()
        self._withdrawn: set[str] = set()
        self._names: dict[str, str] = {}

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
                uri=URL, additional_headers={"Authorization": f"Token {self._api_key}"}, max_size=None
            )
            self._receive_task = self.create_task(self._receive())
        except Exception as exc:
            await self.push_error(error_msg=f"Error connecting: {exc}", exception=exc)
            self._websocket = None

    async def _disconnect(self):
        self._closing = True
        try:
            for task in (self._open_task, self._keepalive_task):
                if task:
                    await self.cancel_task(task, timeout=1.0)
            self._open_task = self._keepalive_task = None
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

    def _declared_functions(self) -> list[dict[str, Any]]:
        """The tools, declared without an endpoint so every call is run here."""
        if not self._tools:
            return []
        standard = self._tools.standard_tools if isinstance(self._tools, ToolsSchema) else self._tools
        functions = []
        for tool in standard:
            schema = tool.to_default_dict() if hasattr(tool, "to_default_dict") else dict(tool)
            function = schema.get("function", schema)
            functions.append({
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            })
        return functions

    def settings(self, *, redacted: bool = False) -> dict[str, Any]:
        """The one message that configures the session; nothing is changed after it.

        ``redacted`` replaces the text-model key, for anything that is logged.
        """
        think: dict[str, Any] = {
            "provider": {"type": self._think_kind, "model": self._think_model},
            "endpoint": {
                "url": THINK_ENDPOINTS[self._think_kind],
                "headers": {"authorization": "<redacted>" if redacted else f"Bearer {self._think_api_key}"},
            },
            "prompt": self._instructions,
        }
        functions = self._declared_functions()
        if functions:
            think["functions"] = functions
        return {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "linear16", "sample_rate": INPUT_RATE},
                "output": {"encoding": "linear16", "sample_rate": OUTPUT_RATE, "container": "none"},
            },
            "agent": {
                "listen": {"provider": dict(LISTEN)},
                "think": think,
                "speak": {"provider": {"type": "deepgram", "model": self._voice}},
            },
        }

    async def _send_audio(self, frame: InputAudioRawFrame):
        # Audio sent before the settings are applied would be heard under
        # defaults nobody chose, so it is dropped.
        if not self._ready.is_set() or not self._websocket:
            return
        await self._websocket.send(frame.audio)

    async def _keep_alive(self):
        while True:
            await asyncio.sleep(KEEPALIVE_S)
            await self._send({"type": "KeepAlive"})

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
            if tool_call_id in self._withdrawn:
                logger.info(f"{self} result for withdrawn tool call {tool_call_id} not sent")
                continue
            await self._send({
                "type": "FunctionCallResponse",
                "id": tool_call_id,
                "name": self._names.get(tool_call_id, ""),
                "content": output if isinstance(output, str) else json.dumps(output),
            })

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
        await self._send(self.settings())
        try:
            await asyncio.wait_for(self._ready.wait(), READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self.push_error(error_msg=f"Deepgram did not apply the settings in {READY_TIMEOUT_S:.0f}s")
            return
        self._keepalive_task = self.create_task(self._keep_alive())
        if opening:
            self._opening_echo = _spoken(opening)
            await self._send({"type": "InjectUserMessage", "content": opening})

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
        await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())
        await self.stop_all_metrics()

    # -- receiving --------------------------------------------------------

    async def _receive(self):
        ws = self._websocket
        assert ws is not None
        async for message in ws:
            try:
                if isinstance(message, bytes):
                    await self._handle_audio(message)
                else:
                    await self._dispatch(json.loads(message))
            except json.JSONDecodeError:
                logger.warning(f"{self} received a text frame that is not JSON")
            except Exception as exc:  # noqa: BLE001 -- one bad event must not end the call
                logger.error(f"{self} failed on a message: {exc}")
        if not self._closing:
            # Closed from the far end while the call was still up.
            await self.push_error(
                error_msg=f"Deepgram closed session {self._session_id} "
                          f"(code {getattr(ws, 'close_code', None)}, {getattr(ws, 'close_reason', None)!r})"
            )

    async def _dispatch(self, event: dict[str, Any]):
        kind = event.get("type", "")

        if kind == "Welcome":
            self._session_id = event.get("request_id")
            logger.info(f"{self} session {self._session_id}")

        elif kind == "SettingsApplied":
            self._ready.set()

        elif kind == "UserStartedSpeaking":
            # The service has already stopped any reply; the pipeline stops its
            # own playback on this proposal, and what is still in flight is dropped.
            self._cut = self._speaking
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

        elif kind == "EndOfTurn":
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

        elif kind == "AgentStartedSpeaking":
            self._cut = False
            await self._start_reply()

        elif kind == "AgentAudioDone":
            self._cut = False
            await self._end_reply()

        elif kind == "ConversationText":
            await self._handle_text(event)

        elif kind == "FunctionCallRequest":
            await self._handle_function_calls(event.get("functions") or [])

        elif kind == "FunctionCallCancelled":
            await self._handle_withdrawn([f.get("id", "") for f in event.get("functions") or []])

        elif kind == "Warning":
            logger.warning(f"{self} warning {event.get('code')}: {event.get('description')}")

        elif kind == "Error":
            await self.push_error(error_msg=f"Deepgram error {event.get('code')}: {event.get('description')}")

    async def _handle_audio(self, audio: bytes):
        if self._cut or not audio:
            return
        if not self._speaking:
            await self._start_reply()
        await self.stop_ttfb_metrics()
        await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=OUTPUT_RATE, num_channels=1))

    async def _handle_text(self, event: dict[str, Any]):
        text = event.get("content") or ""
        if event.get("role") == "user":
            if self._opening_echo is not None and _spoken(text) == self._opening_echo:
                # Our own opening instruction, not the caller: kept out of the transcript.
                self._opening_echo = None
                logger.debug(f"{self} dropped the echo of the opening instruction")
                return
            # Upstream: the context's user half sits before this service.
            await self.push_frame(
                TranscriptionFrame(text, "", time_now_iso8601(), result=event), FrameDirection.UPSTREAM
            )
            return
        if not text or self._cut:
            return
        # One message per sentence of the reply: the agent's half of the transcript.
        await self._start_reply()
        llm_text = LLMTextFrame(f"{text} ")
        llm_text.append_to_context = False
        await self.push_frame(llm_text)
        await self.push_frame(TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE))

    async def _handle_function_calls(self, functions: list[dict[str, Any]]):
        calls = []
        for function in functions:
            if not function.get("client_side", True):
                continue  # one Deepgram runs itself; nothing here to answer
            name, tool_call_id = function.get("name", ""), function.get("id", "")
            if not name or not tool_call_id:
                logger.warning(f"{self} function call without a name or id: {function}")
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                logger.warning(f"{self} function call {name} ({tool_call_id}) arguments are not JSON")
                arguments = {}
            self._names[tool_call_id] = name
            calls.append(FunctionCallFromLLM(
                context=self._context, tool_call_id=tool_call_id, function_name=name, arguments=arguments,
            ))
        if calls:
            await self.run_function_calls(calls)

    async def _handle_withdrawn(self, ids: list[str]):
        """The service withdrew calls it had sent; no result follows them.

        A call still running is cancelled and settled by the framework. One that
        has already returned is settled here, so the trace marks it withdrawn.
        """
        wanted = {i for i in ids if i}
        self._withdrawn.update(wanted)
        cancelled = await self._cancel_function_call_tasks(
            lambda item: item.tool_call_id in wanted, reason="withdrawn by the service (FunctionCallCancelled)"
        )
        running = {item.tool_call_id for item in cancelled}
        settled = []
        for tool_call_id in sorted(wanted - running):
            name = self._names.get(tool_call_id, "unknown")
            logger.info(f"{self} the service withdrew tool call {name} ({tool_call_id}); its result will not be sent")
            settled.append(await self._broadcast_function_call_cancelled(FunctionCallRunnerItem(
                registry_item=self._functions.get(name), function_name=name, tool_call_id=tool_call_id,
                arguments={}, context=self._context,
            )))
        if settled:
            await self._call_event_handler("on_function_calls_cancelled", settled)
