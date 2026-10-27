"""Zoom's Voice Agent API: a hosted cascade behind a realtime socket.

This row is not a speech-to-speech model. Zoom transcribes the caller with its
own speech-to-text, hands the words to a text model (``gemma-4``, served by
Zoom, unless a customer brings their own), and voices the reply with its own
text-to-speech. The model never hears the call, and every reply's usage counts
text tokens only. It sits on the board because the whole pipeline is one
vendor's service, called the way the speech-to-speech rows are, so a reader can
set a tuned, hosted cascade beside native models and beside this file's own
cascade rows -- and the record says which it is.

*Why it is spoken directly.* The events are OpenAI's later realtime names
(``response.output_audio.delta``), but the session is Zoom's own -- ``listen``,
``think`` and ``speak`` groups rather than OpenAI's audio object -- every server
event arrives inside a ``payload`` envelope, and the socket needs Zoom's
subprotocol. The protocol is small enough to speak here, as ``azure_realtime``
does for Voice Live.

Four things about this service decide how it is written, and each was measured
on the live service rather than taken from its documentation.

*The service owns the caller's turns.* Its detector announces where the caller
starts and stops, and it cancels its own reply when talked over, about 0.4 s
into the caller's speech. None of it can be configured. The row follows those
announcements, and what the caller heard of a cut reply is told back to the
service as a truncation, measured on the audio this pipeline delivered.

*Audio arrives faster than it plays.* A reply streams at about three times
real time, so what was delivered is not what was heard; the truncation takes
the shorter of the two.

*A reply's words lead its sound*, by a quarter to two thirds of a second. Words
are held until their reply's first audio arrives and dropped if it never does.

*A tool result does not resume the reply.* The service waits for every call it
makes to be answered and does not answer on its own; a response is asked for
once every call has its result and the response that made them has closed. A
response asked for while another is open is refused.

Protocol: https://developers.zoom.us/docs/ai-services/voice-agent/live-mode/
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import fields
from typing import Any

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    AggregationType,
    BotStoppedSpeakingFrame,
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
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.time import time_now_iso8601

URL = "wss://api.zoom.us/v2/ws/aiservices/voice-agent/live"
SUBPROTOCOL = "voice-agent.zoom.v2"

#: The service listens at any rate it is told and speaks at 16 kHz only, so
#: both directions run at 16 kHz.
SAMPLE_RATE = 16000

#: Who serves the text model. ``zoom`` is the vendor's own; the others take a
#: customer's key, which this row does not send.
THINK_PROVIDER = "zoom"

#: The only language the service accepts.
LANGUAGE = "en"

#: How long to wait for the service to accept the configuration.
READY_TIMEOUT_S = 10.0

#: The tools that end or hand over the call. The goodbye is said before one is
#: called, so no reply is asked for after it: the text model, given a result
#: with nothing left to say, writes the call as markup, and the voice reads that
#: out for the second before the hang-up lands.
CLOSING_TOOLS = frozenset({"end_call", "transfer_call"})

#: How the service turns down a reply asked for while one is already running.
#: The service starts replies itself when it hears the caller finish, so a reply
#: asked for after a tool result can cross one it has just begun.
REPLY_ALREADY_RUNNING = "Conversation already has an active response in progress"


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex[:24]}"


def unwrap(message: dict[str, Any]) -> dict[str, Any]:
    """The event inside the envelope every server event arrives in, or the message itself if bare."""
    payload = message.get("payload")
    return payload if isinstance(payload, dict) else message


class ZoomVoiceAgentLLMService(LLMService):
    """One Zoom Voice Agent session, for the length of one call.

    The socket is opened with the pipeline; the session is configured, tools
    included, with the first context frame, and the call opens once the service
    has accepted that configuration.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemma-4",
        voice: str = "jenna",
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
        self._model = model
        self._voice = voice
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
        # Set when the service adds a tool result, cleared when it starts a
        # reply: a reply sees only the items added before it started.
        self._result_unanswered = False
        # Set once a closing tool is called: the call is ending, nothing more is said.
        self._ended = False
        # The reply being voiced: its item, when its first audio arrived, and
        # how much audio has been delivered, which is what a truncation reports.
        self._audio_item: tuple[str, int] | None = None
        self._audio_started_ms = 0
        self._audio_bytes = 0
        self._speaking = False
        self._responding = False
        # Words that arrived ahead of their reply's audio, by response: see the
        # module docstring.
        self._held: dict[str, list[str]] = {}
        self._voiced: set[str] = set()
        # Replies the caller talked over: the service cancels them, but audio
        # already in flight must not reopen them.
        self._cut: set[str] = set()
        self._audio_response: str | None = None
        # Calls the model has made whose results have not been sent yet, and the
        # results already sent: the context is pushed again for every tool, and
        # each result must be sent exactly once.
        self._outstanding: dict[str, str] = {}
        self._delivered: set[str] = set()

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
            self._websocket = await websocket_connect(
                uri=URL,
                subprotocols=[SUBPROTOCOL],
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                max_size=None,
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
        except Exception as exc:
            await self.push_error(error_msg=f"Error disconnecting: {exc}", exception=exc)

    # -- sending ----------------------------------------------------------

    async def _send(self, event: dict[str, Any]):
        if not self._websocket:
            return
        event.setdefault("event_id", _event_id())
        await self._websocket.send(json.dumps(event))

    def encoded_tools(self) -> list[dict]:
        """Tools in the realtime protocol's flat shape: name and parameters on the tool itself."""
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
        return encoded

    def session(self) -> dict[str, Any]:
        """The session configuration; every setting the row depends on is sent rather than defaulted."""
        tools = self.encoded_tools()
        return {
            "type": "realtime",
            "listen": {
                "format": {"type": "pcm16", "rate": SAMPLE_RATE},
                "transcription": {"language": LANGUAGE},
            },
            "think": {
                "provider": {"type": THINK_PROVIDER, "model": self._model},
                "instructions": self._instructions,
                "response_token_limit": "inf",
                "tools": tools,
                "tool_choice": "auto" if tools else "none",
            },
            "speak": {"format": {"type": "pcm16", "rate": SAMPLE_RATE}, "voice": self._voice},
        }

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
        if self._ended:
            self._reply_owed = False
            return
        if self._outstanding:
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
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Heard to the end: nothing of it is left to truncate.
            self._audio_item = None

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
            await self.push_error(error_msg=f"Zoom did not accept the configuration in {READY_TIMEOUT_S:.0f}s")
            return
        # The opening instruction goes in as a user message, which is how the
        # framework's realtime services seed a session from its first context.
        for text in self._opening(context):
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
                }
            )
        await self._request_reply()

    async def _handle_interruption(self):
        for response_id in (self._response_id, self._audio_response):
            if response_id:
                self._cut.add(response_id)
                self._held.pop(response_id, None)
        await self._truncate()
        await self._end_reply()
        await self.stop_all_metrics()

    async def _truncate(self):
        """Tell the service how much of the cut reply the caller actually heard.

        The shorter of the time since its first audio and the audio delivered:
        the service otherwise treats the whole reply as heard, and the model
        carries on as if the caller knew what it never said.
        """
        if self._audio_item is None:
            return
        item_id, content_index = self._audio_item
        self._audio_item = None
        delivered_ms = int(self._audio_bytes / 2 / SAMPLE_RATE * 1000)
        elapsed_ms = int(time.time() * 1000) - self._audio_started_ms
        await self._send(
            {
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": content_index,
                "audio_end_ms": max(0, min(elapsed_ms, delivered_ms)),
            }
        )

    async def _start_reply(self):
        if self._responding:
            return
        self._responding = True
        await self.push_frame(LLMFullResponseStartFrame())

    async def _end_reply(self):
        if self._speaking:
            self._speaking = False
            await self.push_frame(TTSStoppedFrame())
        if self._responding:
            self._responding = False
            await self.push_frame(LLMFullResponseEndFrame())

    # -- receiving --------------------------------------------------------

    async def _receive(self):
        ws = self._websocket
        assert ws is not None
        async for message in ws:
            # The socket keeps delivering through its close handshake, and the
            # service may begin another reply after the call is torn down; words
            # nobody heard must not reach the record.
            if self._closing:
                break
            try:
                event = unwrap(json.loads(message))
            except (json.JSONDecodeError, AttributeError):
                logger.warning(f"{self} received a frame that is not a JSON object")
                continue
            try:
                await self._dispatch(event)
            except Exception as exc:  # noqa: BLE001 -- one bad event must not end the call
                logger.error(f"{self} failed on {event.get('type')}: {exc}")
        if not self._closing:
            # Closed from the far end while the call was still up. Not retried:
            # a new session would not know the conversation so far.
            await self.push_error(
                error_msg=f"Zoom closed the session "
                          f"(code {getattr(ws, 'close_code', None)}, {getattr(ws, 'close_reason', None)!r})"
            )

    async def _dispatch(self, event: dict[str, Any]):
        kind = event.get("type", "")

        if kind == "session.created":
            # Zoom's id for the session, which is what its support can look up.
            logger.info(f"{self} Zoom session {(event.get('session') or {}).get('id')}")

        elif kind == "session.updated":
            served = ((event.get("session") or {}).get("think") or {}).get("provider") or {}
            if served.get("model") != self._model:
                logger.warning(f"{self} asked for {self._model!r}, the session reports {served.get('model')!r}")
            self._updated.set()

        elif kind == "conversation.item.created":
            if (event.get("item") or {}).get("type") == "function_call_output":
                self._result_unanswered = True

        elif kind == "response.created":
            self._response_id = (event.get("response") or {}).get("id")
            self._result_unanswered = False
            await self._start_reply()

        elif kind == "response.output_audio.delta":
            await self._handle_audio(event)

        elif kind == "response.output_audio_transcript.delta":
            text, response_id = event.get("delta", ""), event.get("response_id", "")
            if not text or response_id in self._cut:
                return
            if response_id in self._voiced:
                await self._push_text(text)
            else:
                self._held.setdefault(response_id, []).append(text)

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
            await self._truncate()
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

        elif kind == "input_audio_buffer.speech_stopped":
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

        elif kind == "error":
            message = str((event.get("error") or {}).get("message", ""))
            if message.startswith(REPLY_ALREADY_RUNNING):
                await self._handle_reply_refused()
                return
            # Most are recoverable and leave the session open; the message says which.
            await self.push_error(error_msg=f"Zoom error: {event.get('error')}")

    async def _handle_reply_refused(self):
        """A reply asked for after a tool result crossed one the service had begun.

        The running reply answers the result only if it started after the result
        was added; otherwise one more is asked for once it closes.
        """
        if self._result_unanswered:
            self._reply_owed = True
            logger.info(f"{self} reply request crossed a running reply; asking again once it closes")
        else:
            logger.info(f"{self} reply request crossed a running reply that already sees the tool result")

    async def _handle_audio(self, event: dict[str, Any]):
        audio = base64.b64decode(event.get("delta") or "")
        if not audio:
            return
        response_id = event.get("response_id", "")
        if response_id in self._cut:
            return
        self._audio_response = response_id
        item = (event.get("item_id", ""), int(event.get("content_index", 0)))
        if self._audio_item != item:
            # A reply's first audio, or the next item of the same reply: the
            # truncation point is measured within the item being voiced.
            self._audio_item = item
            self._audio_started_ms = int(time.time() * 1000)
            self._audio_bytes = 0
        self._audio_bytes += len(audio)
        await self._start_reply()
        if not self._speaking:
            self._speaking = True
            await self.push_frame(TTSStartedFrame())
        if response_id not in self._voiced:
            # The reply's first sound releases the words held for it.
            self._voiced.add(response_id)
            for text in self._held.pop(response_id, []):
                await self._push_text(text)
        await self.stop_ttfb_metrics()
        await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1))

    async def _handle_response_done(self, response: dict[str, Any]):
        # Words of a reply that never made a sound were never heard.
        self._held.pop(response.get("id", ""), None)
        self._response_id = None
        await self._end_reply()
        await self.stop_processing_metrics()
        usage = response.get("usage")
        if usage:
            await self.start_llm_usage_metrics(token_usage(usage))
        if response.get("status") in ("failed", "incomplete"):
            await self.push_error(error_msg=f"Zoom response {response.get('status')}: {response.get('status_details')}")
        if self._reply_owed:
            await self._request_reply()

    async def _push_text(self, text: str):
        frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        frame.includes_inter_frame_spaces = True
        await self.push_frame(frame)

    async def _handle_function_call(self, event: dict[str, Any]):
        call_id = event.get("call_id", "")
        name = event.get("name") or self._outstanding.get(call_id, "")
        if not call_id or not name:
            logger.warning(f"{self} function call without a name or id: {event}")
            return
        self._outstanding[call_id] = name
        if name in CLOSING_TOOLS:
            self._ended = True
        try:
            arguments = json.loads(event.get("arguments") or "{}")
        except json.JSONDecodeError:
            # Handed on as given: an unparseable argument is the model's answer,
            # and the tool's rejection of it belongs in the record.
            logger.warning(f"{self} could not parse arguments for {name}")
            arguments = {"_unparsed": event.get("arguments")}
        await self.run_function_calls(
            [FunctionCallFromLLM(context=self._context, tool_call_id=call_id, function_name=name, arguments=arguments)]
        )


def token_usage(usage: dict[str, Any]) -> LLMTokenUsage:
    """A ``response.done`` usage block in the framework's names.

    The text model's tokens only: the service reports no audio, because its
    model is given none. The row is priced by the minute, not from these.
    """
    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}
    return LLMTokenUsage(
        prompt_tokens=usage.get("input_tokens") or 0,
        completion_tokens=usage.get("output_tokens") or 0,
        total_tokens=usage.get("total_tokens") or 0,
        cache_read_input_tokens=input_details.get("cached_tokens"),
        input_audio_tokens=input_details.get("audio_tokens"),
        output_audio_tokens=output_details.get("audio_tokens"),
    )
