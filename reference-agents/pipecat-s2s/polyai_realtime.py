"""PolyAI's audio-native dialog model (Dialog-RSN-1), on its realtime API.

Dialog-RSN-1 hears the caller's audio directly and decides when they have
finished, then answers in *text* or with a tool call. It does not speak: this
row voices its replies with the cascade rows' text-to-speech, so the row is
"this vendor's listening and reasoning, through one named voice".

The wire format is OpenAI's realtime protocol (GA shape) with a narrower
session: the service refuses a whole ``session.update`` if any field is outside
its schema, so only the fields below are ever sent. No framework service exists
for it, and the framework's OpenAI service sends fields this one rejects.

Three things about this service decide how it is written.

*A turn is text or one tool call, never both -- and speech rides on the call.*
The model calls tools in a loop and finishes with its spoken reply, so an
instruction to "say X, then call Y" cannot be followed as written. The vendor's
answer is a tool argument marked ``poly_spoken``: the model writes a line to say
while the tool runs, and it streams as reply text beside the call. Every tool
here gets one such argument, required, and it is taken off the arguments before
the tool runs, so the tools see exactly the published contract. See
``spoken_tools``.

*The service judges barge-in only while a reply is audible to it.* By default
that is while the reply is being generated, which for a text model is a moment;
the caller then hears it for seconds through the text-to-speech. The service's
playback reports (``poly.output_audio.*``) tell it what the caller is hearing,
which is what lets it tell a backchannel from an interruption during playback.
They are sent from ``ReportsPlayback``, which sits after the output transport,
where frames move at the pace the caller hears them.

*One response at a time.* A ``response.create`` sent while a response is open
cancels it, and a tool's result is answered only when a response is asked for.
One is asked for once every call the model made has its result and the response
that made them has closed -- unless the caller is mid-turn, in which case the
response the service opens for that turn carries the result.

Protocol: https://dialog-rsn-1-eap-docs.pages.dev/dialog-rsn-1/reference/
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import time
import uuid
from dataclasses import fields
from typing import Any

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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
    TTSTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.time import time_now_iso8601

#: The endpoint the vendor's reference names.
DEFAULT_URL = "wss://api.us.poly.ai/v1/realtime"

#: Informational on the wire (the service runs Dialog-RSN-1 whatever is sent),
#: but sent so the session the service echoes names it.
MODEL = "dialog-rsn-1"

#: The model's native input rate; the service resamples anything else.
SAMPLE_RATE = 16000

#: How long to wait for the service to accept the configuration.
READY_TIMEOUT_S = 10.0

#: The argument every tool is given for the line said while it runs.
SPOKEN_ARGUMENT = "utterance"

#: What that argument asks for. The call-control tools end or hand over the
#: call, so their line is the last thing the caller hears from this agent.
SPOKEN_DESCRIPTIONS = {
    "end_call": "Your goodbye to the caller, said as the call ends.",
    "transfer_call": "What you tell the caller as you connect them.",
}
DEFAULT_SPOKEN_DESCRIPTION = "A short sentence to say to the caller while this runs, such as a progress update."

#: Added to the agent's prompt on this row only, in the vendor's words, so a
#: "say X, then call Y" instruction lands in the call's spoken argument.
PROMPT_ADDENDUM = """\
# Speaking while a tool runs

Whenever the instructions say to say something and then call a tool, put that \
sentence in the tool's utterance. Say goodbye only through end_call's \
utterance, and don't speak again after end_call.
"""


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex[:24]}"


def spoken_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each tool with a required spoken argument the service voices while it runs.

    Required on every tool, not only the call-control ones: in a chain of calls
    the model skips an optional line, and the caller then waits in silence while
    the tools run. The ``poly_spoken`` marker is removed by the service before
    the model sees the schema.
    """
    marked = []
    for tool in tools:
        tool = copy.deepcopy(tool)
        parameters = tool.setdefault("parameters", {"type": "object", "properties": {}})
        parameters.setdefault("type", "object")
        properties = parameters.setdefault("properties", {})
        properties[SPOKEN_ARGUMENT] = {
            "type": "string",
            "description": SPOKEN_DESCRIPTIONS.get(tool["name"], DEFAULT_SPOKEN_DESCRIPTION),
            "poly_spoken": True,
        }
        required = list(parameters.get("required") or [])
        if SPOKEN_ARGUMENT not in required:
            required.append(SPOKEN_ARGUMENT)
        parameters["required"] = required
        marked.append(tool)
    return marked


def heard_by_reply(texts: list[str], heard: str, finished: bool) -> list[str]:
    """What the caller heard of each reply played in one stretch of speech.

    A finished stretch was heard whole. A cut one is apportioned word by word,
    in the order the replies were queued, from the words the output transport
    actually played: the transport does not say which reply a word belonged to,
    and replies play in the order they were written.
    """
    if finished:
        return list(texts)
    words = heard.split()
    out = []
    for text in texts:
        count = len(text.split())
        out.append(" ".join(words[:count]))
        words = words[count:]
    return out


class PolyAIRealtimeLLMService(LLMService):
    """One Dialog-RSN-1 session, for the length of one call.

    The socket is opened with the pipeline; the session is configured, tools
    included, with the first context frame, and the call opens once the service
    has accepted that configuration. Replies leave as text frames for the
    text-to-speech that follows this service.
    """

    def __init__(
        self,
        *,
        api_key: str,
        url: str = DEFAULT_URL,
        model: str = MODEL,
        instructions: str = "",
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
        self._url = url
        self._model = model
        self._instructions = instructions
        self._tools = tools
        self._websocket = None
        self._closing = False
        self._receive_task = None
        self._open_task = None
        self._updated = asyncio.Event()
        self._context = None
        # The response the service has open, if any, and whether a new one is
        # owed once it closes.
        self._response_id: str | None = None
        self._reply_owed = False
        self._responding = False
        self._caller_speaking = False
        # Responses the caller talked over: text still arriving for them was
        # never going to be heard.
        self._cut: set[str] = set()
        # Calls the model has made whose results have not been sent yet, and the
        # results already sent: the context is pushed again for every tool, and
        # each result must be sent exactly once.
        self._outstanding: dict[str, str] = {}
        self._delivered: set[str] = set()
        # Spoken arguments by response, kept until the response closes: a line
        # the service did not stream as text is voiced from here.
        self._spoken: dict[str, list[str]] = {}
        self._texted: set[str] = set()
        # Playback: the text each reply sent to the voice, replies written but
        # not yet audible, replies audible now (with when they became so), and
        # the words the caller has heard in the current stretch of speech.
        self._texts: dict[str, list[str]] = {}
        self._unplayed: list[str] = []
        self._playing: list[tuple[str, float]] = []
        self._heard: list[str] = []
        self._bot_speaking = False

    def can_generate_metrics(self) -> bool:
        return True

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
            self._websocket = await websocket_connect(uri=self._url, additional_headers={"X-API-KEY": self._api_key})
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
        except Exception as exc:
            await self.push_error(error_msg=f"Error disconnecting: {exc}", exception=exc)

    # -- sending ----------------------------------------------------------

    async def _send(self, event: dict[str, Any]):
        if not self._websocket:
            return
        event.setdefault("event_id", _event_id())
        await self._websocket.send(json.dumps(event))

    def encoded_tools(self) -> list[dict]:
        """Function tools in the realtime protocol's flat shape, each with its spoken argument."""
        if not self._tools:
            return []
        standard = self._tools.standard_tools if isinstance(self._tools, ToolsSchema) else self._tools
        encoded = []
        for tool in standard:
            schema = tool.to_default_dict() if hasattr(tool, "to_default_dict") else dict(tool)
            function = schema.get("function", schema)
            encoded.append(
                {
                    "type": "function",
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return spoken_tools(encoded)

    def session(self) -> dict[str, Any]:
        """The session configuration: only fields the service's schema accepts."""
        session: dict[str, Any] = {
            "type": "realtime",
            "model": self._model,
            "output_modalities": ["text"],
            "instructions": self._instructions,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    # The only turn-detection field the service reads; its
                    # detector's thresholds are fixed.
                    "turn_detection": {"type": "server_vad", "create_response": True},
                }
            },
        }
        tools = self.encoded_tools()
        if tools:
            session["tools"] = tools
        return session

    async def _send_session_update(self):
        self._updated.clear()
        await self._send({"type": "session.update", "session": self.session()})

    async def _send_audio(self, frame: InputAudioRawFrame):
        # Audio sent before the configuration carrying the tools is accepted
        # would be answered by a session with no tools, so it is dropped.
        if self._open_task is None or not self._open_task.done():
            return
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(frame.audio).decode()})

    async def _request_reply(self):
        """Ask for a response, now or as soon as the open one closes."""
        if self._outstanding:
            return
        if self._caller_speaking:
            # The response the service opens when the caller's turn ends reads
            # the result from the history; asking now would answer mid-turn.
            self._reply_owed = False
            return
        if self._response_id is not None:
            self._reply_owed = True
            return
        self._reply_owed = False
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
            self._open_task = self.create_task(self._open(context))
            return
        for tool_call_id, output in self._results(context).items():
            if tool_call_id in self._delivered:
                continue
            self._delivered.add(tool_call_id)
            self._outstanding.pop(tool_call_id, None)
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": tool_call_id, "output": output},
                }
            )
            await self._request_reply()

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
    def _opening(context) -> list[str]:
        """The opening turn every row is given, as the text of the caller-side messages."""
        texts = []
        for message in context.get_messages():
            if isinstance(message, LLMSpecificMessage) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content)
        return texts

    async def _open(self, context):
        # The tools reach the session before the opening reply is asked for, so
        # the session never answers without them.
        await self._send_session_update()
        try:
            await asyncio.wait_for(self._updated.wait(), READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self.push_error(error_msg=f"Dialog-RSN-1 did not accept the configuration in {READY_TIMEOUT_S:.0f}s")
            return
        for text in self._opening(context):
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
                }
            )
        await self._request_reply()

    async def _handle_interruption(self):
        if self._response_id:
            self._cut.add(self._response_id)
            # The service cancels a reply it judges talked over on its own;
            # this covers one the pipeline judged first, and is a no-op otherwise.
            await self._send({"type": "response.cancel"})
        await self.playback_stopped(finished=False)
        # Replies written but not yet audible were flushed from the voice.
        for response_id in self._unplayed:
            self._texts.pop(response_id, None)
        self._unplayed.clear()
        await self._end_reply()
        await self.stop_all_metrics()

    async def _start_reply(self):
        if self._responding:
            return
        self._responding = True
        await self.push_frame(LLMFullResponseStartFrame())

    async def _end_reply(self):
        if self._responding:
            self._responding = False
            await self.push_frame(LLMFullResponseEndFrame())

    # -- playback reports (see ``ReportsPlayback``) -----------------------

    async def playback_started(self):
        """The caller began hearing the agent: every reply written so far is now audible."""
        self._bot_speaking = True
        await self._report_audible()

    async def _report_audible(self):
        now = time.monotonic()
        for response_id in self._unplayed:
            self._playing.append((response_id, now))
            await self._send({"type": "poly.output_audio.started", "response_id": response_id})
        self._unplayed.clear()

    def playback_heard(self, text: str):
        self._heard.append(text)

    async def playback_stopped(self, *, finished: bool):
        """The agent fell silent: by finishing, or because the caller cut in."""
        self._bot_speaking = False
        if not self._playing:
            self._heard.clear()
            return
        now = time.monotonic()
        texts = ["".join(self._texts.get(response_id, [])) for response_id, _ in self._playing]
        spoken = heard_by_reply(texts, " ".join(self._heard), finished)
        for (response_id, since), text in zip(self._playing, spoken):
            await self._send(
                {
                    "type": "poly.output_audio.stopped",
                    "response_id": response_id,
                    "audio_ms": int((now - since) * 1000),
                    # Not measured: the voice does not report audio it made
                    # but never played.
                    "dropped_ms": 0,
                    "spoken_text": text,
                }
            )
            self._texts.pop(response_id, None)
        self._playing.clear()
        self._heard.clear()

    # -- receiving --------------------------------------------------------

    async def _receive(self):
        ws = self._websocket
        assert ws is not None
        async for message in ws:
            # The socket keeps delivering through its close handshake; words
            # nobody heard must not reach the record.
            if self._closing:
                break
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
            # Closed from the far end while the call was still up. Not retried:
            # the service has no session resumption, so a new session would not
            # know the conversation so far.
            await self.push_error(
                error_msg=f"Dialog-RSN-1 closed the session "
                          f"(code {getattr(ws, 'close_code', None)}, {getattr(ws, 'close_reason', None)!r})"
            )

    async def _dispatch(self, event: dict[str, Any]):
        kind = event.get("type", "")

        if kind == "session.updated":
            self._updated.set()

        elif kind == "response.created":
            self._response_id = (event.get("response") or {}).get("id")
            await self._start_reply()

        elif kind == "response.output_text.delta":
            await self._handle_text(event.get("response_id", "") or self._response_id or "", event.get("delta", ""))

        elif kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                self._outstanding[item.get("call_id", "")] = item.get("name", "")

        elif kind == "response.function_call_arguments.done":
            await self._handle_function_call(event)

        elif kind == "response.done":
            await self._handle_response_done(event.get("response") or {})

        elif kind == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript.strip():
                # Upstream: the context's user half sits before this service.
                await self.push_frame(
                    TranscriptionFrame(transcript, "", time_now_iso8601(), result=event), FrameDirection.UPSTREAM
                )

        elif kind == "input_audio_buffer.speech_started":
            self._caller_speaking = True
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

        elif kind == "input_audio_buffer.speech_stopped":
            self._caller_speaking = False
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

        elif kind == "error":
            await self.push_error(error_msg=f"Dialog-RSN-1 error: {describe_error(event.get('error') or {})}")

    async def _handle_text(self, response_id: str, text: str):
        if not text or response_id in self._cut:
            return
        await self._start_reply()
        await self.stop_ttfb_metrics()
        self._texted.add(response_id)
        if response_id not in self._texts:
            self._texts[response_id] = []
            self._unplayed.append(response_id)
            if self._bot_speaking:
                # Queued behind what is playing, so audible from now on.
                await self._report_audible()
        self._texts[response_id].append(text)
        await self.push_frame(LLMTextFrame(text))

    async def _handle_response_done(self, response: dict[str, Any]):
        response_id = response.get("id", "") or self._response_id or ""
        spoken = self._spoken.pop(response_id, [])
        if spoken and response_id not in self._texted and response_id not in self._cut:
            # The line a tool call carried was not streamed as reply text; voice
            # it from the argument so the caller is not left in silence.
            logger.warning(f"{self} response {response_id} carried a spoken line but no text; voicing the argument")
            for line in spoken:
                await self._handle_text(response_id, line)
        self._texted.discard(response_id)
        self._response_id = None
        await self._end_reply()
        await self.stop_processing_metrics()
        usage = response.get("usage")
        if usage:
            await self.start_llm_usage_metrics(token_usage(usage))
        if response.get("status") == "failed":
            await self.push_error(error_msg=f"Dialog-RSN-1 response failed: {response.get('status_details')}")
        if self._reply_owed:
            await self._request_reply()

    async def _handle_function_call(self, event: dict[str, Any]):
        call_id = event.get("call_id", "")
        name = event.get("name") or self._outstanding.get(call_id, "")
        if not call_id or not name:
            logger.warning(f"{self} function call without a name or id: {event}")
            return
        self._outstanding[call_id] = name
        try:
            arguments = json.loads(event.get("arguments") or "{}")
        except json.JSONDecodeError:
            # Handed on as given: an unparseable argument is the model's answer,
            # and the tool's rejection of it belongs in the record.
            logger.warning(f"{self} could not parse arguments for {name}")
            arguments = {"_unparsed": event.get("arguments")}
        if isinstance(arguments, dict):
            # Speech, not an input: the tool sees exactly the published contract.
            line = arguments.pop(SPOKEN_ARGUMENT, None)
            if isinstance(line, str) and line.strip():
                response_id = event.get("response_id", "") or self._response_id or ""
                self._spoken.setdefault(response_id, []).append(line)
        await self.run_function_calls(
            [FunctionCallFromLLM(context=self._context, tool_call_id=call_id, function_name=name, arguments=arguments)]
        )


class ReportsPlayback(FrameProcessor):
    """Tells the service what the caller is hearing, as they hear it.

    Placed after the output transport: frames there move at the pace of the
    audio the caller hears, so the words passing are the words heard and the
    speaking frames mark when the agent became audible and fell silent.
    """

    def __init__(self, service: PolyAIRealtimeLLMService, **kwargs):
        super().__init__(**kwargs)
        self._service = service

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                await self._service.playback_started()
            elif isinstance(frame, TTSTextFrame):
                self._service.playback_heard(frame.text)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                await self._service.playback_stopped(finished=True)
        await self.push_frame(frame, direction)


def describe_error(error: dict[str, Any]) -> str:
    """One line for a server error, naming the rejected field when there is one."""
    parts = [f"[{error.get('code') or error.get('type') or 'error'}]"]
    if error.get("message"):
        parts.append(str(error["message"]))
    if error.get("param"):
        parts.append(f"(param: {error['param']})")
    return " ".join(parts)


def token_usage(usage: dict[str, Any]) -> LLMTokenUsage:
    """A ``response.done`` usage block in the framework's names."""
    return LLMTokenUsage(
        prompt_tokens=usage.get("input_tokens") or 0,
        completion_tokens=usage.get("output_tokens") or 0,
        total_tokens=usage.get("total_tokens") or 0,
    )
