"""Rime's speech-to-speech model (Prism), on its realtime API.

Prism speaks the event names and flat session of OpenAI's realtime *beta*
protocol, the shape Voice Live and Qwen follow, with its own rules on top. The
framework's OpenAI service speaks the later shape and sends fields Prism
refuses, so the protocol is spoken directly here, mirroring ``azure_realtime``.

*The session is short.* Instructions, voice and tools, and nothing else: Prism
refuses a whole ``session.update`` over any field outside its schema (the GA
``type``, ``audio`` and ``output_modalities``, transcription settings, detector
thresholds). It listens at 16 kHz and speaks at 24 kHz, transcribes the caller
without being asked, and decides the caller's turns with one built-in detector
that takes no parameters beyond its eagerness and end-of-turn timeout, which
are left at the service's defaults; the values in effect are read back from
``session.updated`` onto the record.

*Every reply the client asks for names its cause.* The opening turn is a nudge
(``prism_cause`` ``proactive``) carrying the bench's opening instruction as
``prism_instruction``: the service's own form for "speak first, and say this",
which leaves the instruction out of the caller's side of the conversation. A
nudge is accepted only while the service is idle, which it announces with
``prism.typed_input.ready``; one refused as busy is sent again at the next.

*A tool's result does not resume the reply.* The reply ends after its calls,
and the conversation continues only when a continuation is asked for, naming
the reply that made the calls. The service refuses one that arrives before that
reply's ``response.done`` or before it has acknowledged every result, and keeps
nothing it refused, so the continuation waits for both. A round overtaken by a
newer reply (the caller spoke first) is not continued.

*What the caller heard is reported, from after the output transport.* The
service sends audio faster than it is played. When the caller cuts in, it is
told how much of the reply was heard (``conversation.item.truncate``), which
decides what it keeps of that reply and whether its next one acknowledges the
interruption. When a reply plays out uncut it is told so
(``prism.playback.drained``), which decides when it is next idle and whether
the caller's next words interrupt or open a new turn; without that it assumes
playback ended 1.5 s after the audio would have. Both are measured on the
frames leaving the output transport, through ``ReportsPlayback``.

*A reply's words lead its sound*, so they are held until its first audio and
dropped if it never makes one, as on the Azure row.

Protocol: https://docs.rime.ai/api-reference/prism/websocket
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from dataclasses import fields
from typing import Any

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    AggregationType,
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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.time import time_now_iso8601

URL = "wss://prism.api.rime.ai/v1/realtime"
#: Points the row at another Prism endpoint (Rime's beta host, for instance).
#: Read from the environment only, never from a session; the record names the one used.
URL_ENV = "RIME_PRISM_URL"


def prism_url() -> str:
    return os.environ.get(URL_ENV) or URL

#: The endpoint serves one model and takes no model parameter.
MODEL = "prism"

#: The caller's audio, as the service hears it; anything else is heard too fast or too slow.
SAMPLE_RATE = 16000

#: The reply audio, fixed by the service.
OUTPUT_SAMPLE_RATE = 24000

#: How long to wait for the service to accept the configuration (it answers within 5 s).
READY_TIMEOUT_S = 10.0

#: The ``event_id`` of the opening nudge, so a refusal of it can be recognised.
OPENING_EVENT_ID = "opening"

#: Refusals of a nudge that mean "not idle yet": the service keeps nothing, so
#: the opening is sent again at the next ``prism.typed_input.ready``.
BUSY = frozenset({"instructions_busy", "proactive_unavailable"})

#: Refusals that are the protocol working as documented, not faults: a report
#: about a reply from an earlier stretch of playback, and a requested reply the
#: caller's own speech overtook.
EXPECTED = frozenset({"truncate_not_current", "response_superseded", "nudge_superseded"})

#: Tools that end the call. A reply that carried one and made a sound has said
#: its closing line; continuing it only makes the model speak, or close, again.
CLOSING_TOOLS = frozenset({"end_call", "transfer_call"})


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex[:24]}"


class RimePrismLLMService(LLMService):
    """One Prism session, for the length of one call.

    The socket is opened with the pipeline; the session is configured, tools
    included, with the first context frame, and the call opens with a nudge
    once the service is idle.
    """

    def __init__(
        self,
        *,
        api_key: str,
        url: str | None = None,
        model: str = MODEL,
        voice: str = "astra",
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
        self._url = url or prism_url()
        self._voice = voice
        self._instructions = instructions
        self._tools = tools
        self._websocket = None
        self._closing = False
        self._receive_task = None
        self._open_task = None
        self._updated = asyncio.Event()
        self._context = None
        # The build record, when the pipeline hands it over: the release that
        # answered and the turn settings in effect are known only once connected.
        self.runtime_record: dict[str, Any] | None = None
        self.release: str | None = None
        # Idle as last announced, and the opening instruction while it waits for it.
        self._idle = False
        self._opening: str | None = None
        self._opening_sent: str | None = None
        self._replied = False
        self._responding = False
        self._speaking = False
        # Words that arrived ahead of their reply's audio, by response, and the
        # replies that have made a sound.
        self._held: dict[str, list[str]] = {}
        self._voiced: set[str] = set()
        # Replies the caller talked over: audio still arriving for them is never played.
        self._cut: set[str] = set()
        # Tool rounds: each reply's calls, the results the service has stored,
        # the replies that have ended, and the rounds already continued.
        self._latest_response: str | None = None
        self._calls: dict[str, set[str]] = {}
        self._acked: set[str] = set()
        self._done: set[str] = set()
        self._continued: set[str] = set()
        self._names: dict[str, str] = {}
        # Replies that called a closing tool, and whether one has been continued.
        self._closing_rounds: set[str] = set()
        self._continued_after_closing = False
        # Results already sent: the context is pushed again for every tool, and
        # each result must be sent exactly once.
        self._delivered: set[str] = set()
        # Playback, as heard after the output transport: each reply's message
        # item and milliseconds delivered, replies delivered but not yet
        # audible, replies audible now, and when the current stretch began.
        self._items: dict[str, str] = {}
        self._audio_ms: dict[str, float] = {}
        self._unplayed: list[str] = []
        self._playing: list[str] = []
        self._stretch_started = 0.0
        self._bot_speaking = False
        # The drain token of the latest completed reply, kept until it has played.
        self._drain: tuple[str, str] | None = None
        self._unhandled: set[str] = set()

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
            # The key goes in the header only: the service reads no query parameters.
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
        except Exception as exc:
            await self.push_error(error_msg=f"Error disconnecting: {exc}", exception=exc)

    # -- sending ----------------------------------------------------------

    async def _send(self, event: dict[str, Any]):
        if not self._websocket:
            return
        # Every request carries an id: without one the service suppresses some
        # refusals, and a refusal is how a request that did nothing shows up.
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
        """The session: only fields the service accepts; its turn timing is left at its defaults."""
        session: dict[str, Any] = {"instructions": self._instructions, "voice": self._voice}
        tools = self.encoded_tools()
        if tools:
            session["tools"] = tools
        return session

    async def _send_audio(self, frame: InputAudioRawFrame):
        # Audio sent before the configuration carrying the tools is accepted is
        # refused, and would otherwise be answered by a session with no tools.
        if not self._updated.is_set():
            return
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(frame.audio).decode()})

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
            # The result only: the continuation follows once the service has
            # acknowledged it and the reply that made the call has ended.
            await self._send(
                {
                    "type": "conversation.item.create",
                    "event_id": f"result_{tool_call_id}",
                    "item": {"type": "function_call_output", "call_id": tool_call_id, "output": output},
                }
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
    def opening_instruction(context) -> str:
        """The opening turn every row is given, as one instruction for the first reply."""
        texts = []
        for message in context.get_messages():
            if isinstance(message, LLMSpecificMessage) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
        return "\n".join(texts)

    async def _open(self, context):
        # The tools reach the session before the opening reply is asked for, so
        # the session never answers without them.
        await self._send({"type": "session.update", "event_id": "setup", "session": self.session()})
        try:
            await asyncio.wait_for(self._updated.wait(), READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self.push_error(error_msg=f"Prism did not accept the configuration in {READY_TIMEOUT_S:.0f}s")
            return
        self._opening = self.opening_instruction(context) or None
        await self._send_opening()

    async def _send_opening(self):
        """The nudge that opens the call, once the service is idle; it refuses one otherwise."""
        if self._opening is None or not self._idle:
            return
        self._idle = False
        self._opening_sent, self._opening = self._opening, None
        await self._send(
            {
                "type": "response.create",
                "event_id": OPENING_EVENT_ID,
                "response": {"prism_instruction": self._opening_sent, "metadata": {"prism_cause": "proactive"}},
            }
        )

    async def _maybe_continue(self):
        """Ask for the reply to a tool round once the service will accept it.

        It refuses a continuation that arrives before the parent reply's
        ``response.done`` or before every result's acknowledgement, and keeps
        nothing it refused. A round overtaken by a newer reply is not continued.
        """
        parent = self._latest_response
        if parent is None or parent in self._continued or parent not in self._done:
            return
        calls = self._calls.get(parent)
        if not calls or not calls <= self._acked:
            return
        self._continued.add(parent)
        if parent in self._closing_rounds:
            # A closing reply that sounded has said goodbye; a silent one gets
            # one reply to say it in, and never a second.
            if parent in self._voiced or self._continued_after_closing:
                logger.debug(f"{self} closing round {parent} not continued")
                return
            self._continued_after_closing = True
        await self._send(
            {
                "type": "response.create",
                "event_id": f"continue_{parent}",
                "response": {"metadata": {"prism_cause": "tool_continuation", "prism_parent_response_id": parent}},
            }
        )

    async def _handle_interruption(self):
        for response_id in (self._latest_response, *self._playing, *self._unplayed):
            if response_id:
                self._cut.add(response_id)
                self._held.pop(response_id, None)
        await self.playback_stopped(finished=False)
        await self._end_reply()
        await self.stop_all_metrics()

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

    # -- playback reports (the stage is ``ReportsPlayback`` below) ---

    async def playback_started(self):
        """The caller began hearing the agent: every reply delivered so far is now audible."""
        self._bot_speaking = True
        self._stretch_started = time.monotonic()
        self._playing.extend(self._unplayed)
        self._unplayed.clear()

    def playback_heard(self, text: str):
        """The service is told what was heard in milliseconds, not words."""

    async def playback_stopped(self, *, finished: bool):
        """The agent fell silent: by finishing, or because the caller cut in.

        A finished stretch played every reply in it whole, so the latest
        completed one is reported drained. A cut one is spent on its replies in
        the order they were queued, and the reply the cut fell in is truncated
        at the point reached; replies behind it were never heard, and the
        service counts them so without a report of their own. A reply delivered
        but not yet audible when the cut came was heard not at all.
        """
        self._bot_speaking = False
        cut: list[str] = []
        if finished:
            if self._drain and self._drain[0] in self._playing:
                await self._send_drained()
        else:
            self._drain = None
            heard = self._heard_at_cut()
            if heard is not None:
                item_id, audio_end_ms = heard
                logger.info(f"{self} caller cut in: reporting {audio_end_ms} ms of {item_id} heard")
                await self._send(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": item_id,
                        "content_index": 0,
                        "audio_end_ms": audio_end_ms,
                    }
                )
            self._unplayed, cut = [], self._unplayed
        for response_id in (*cut, *self._playing):
            self._audio_ms.pop(response_id, None)
        self._playing.clear()

    def _heard_at_cut(self) -> tuple[str, int] | None:
        if not self._playing:
            first = next((r for r in self._unplayed if r in self._items), None)
            return (self._items[first], 0) if first else None
        remaining_ms = (time.monotonic() - self._stretch_started) * 1000
        for response_id in self._playing:
            delivered_ms = self._audio_ms.get(response_id, 0.0)
            if remaining_ms < delivered_ms and response_id in self._items:
                return self._items[response_id], max(0, round(remaining_ms))
            remaining_ms -= delivered_ms
        return None  # everything delivered had played; the cut fell in the silence after

    async def _send_drained(self):
        response_id, token = self._drain
        self._drain = None
        logger.debug(f"{self} reply {response_id} played out; reporting it drained")
        await self._send({"type": "prism.playback.drained", "drain_token": token})

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
            # a new session would not know the conversation so far.
            await self.push_error(
                error_msg=f"Prism closed the session "
                          f"(code {getattr(ws, 'close_code', None)}, {getattr(ws, 'close_reason', None)!r})"
            )

    async def _dispatch(self, event: dict[str, Any]):
        kind = event.get("type", "")

        if kind in ("session.created", "session.updated"):
            configured = kind == "session.updated"
            self._note_session(event.get("session") or {}, configured=configured)
            if configured:
                self._updated.set()

        elif kind == "prism.typed_input.ready":
            self._idle = True
            await self._send_opening()

        elif kind == "response.created":
            self._idle = False
            self._replied = True
            if self._opening is not None:
                # The caller spoke before the service was idle, and this reply
                # answers them: a greeting after it would be out of place.
                logger.info(f"{self} the caller spoke first; the opening is not sent")
                self._opening = None
            self._latest_response = (event.get("response") or {}).get("id")
            await self._start_reply()

        elif kind == "response.audio.delta":
            await self._handle_audio(event)

        elif kind == "response.text.delta":
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
                self._names[item.get("call_id", "")] = item.get("name", "")

        elif kind == "response.function_call_arguments.done":
            await self._handle_function_call(event)

        elif kind == "conversation.item.created":
            item = event.get("item") or {}
            if item.get("type") == "function_call_output":
                self._acked.add(item.get("call_id", ""))
                await self._maybe_continue()

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
            self._idle = False
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

        elif kind == "input_audio_buffer.speech_stopped":
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

        elif kind == "conversation.item.truncated":
            kept = event.get("prism_kept_text")
            logger.info(
                f"{self} truncation of {event.get('item_id')} at {event.get('audio_end_ms')} ms accepted"
                + ("" if kept is None else f"; kept {kept[:80]!r}")
            )

        elif kind == "error":
            await self._handle_error(event.get("error") or {})

        elif kind not in self._unhandled and kind not in _READ_ELSEWHERE:
            # Once per kind, so a reply arriving under another event name shows
            # up here rather than as silence.
            self._unhandled.add(kind)
            logger.info(f"{self} ignoring event {kind!r} (first of its kind)")

    def _note_session(self, session: dict[str, Any], *, configured: bool):
        """The release that answered, and once configured, the turn settings in effect."""
        self.release = session.get("version") or self.release
        if self.runtime_record is not None and self.release:
            self.runtime_record["rime_prism_release"] = self.release
        if not configured:
            logger.info(f"{self} session {session.get('id')} on release {self.release}")
            return
        turns = session.get("turn_detection") or {}
        applied = ", ".join(f"{key} {value}" for key, value in sorted(turns.items()))
        if self.runtime_record is not None and applied:
            self.runtime_record["rime_prism_turn_detection"] = f"{applied} (provider defaults, as reported)"
        logger.info(
            f"{self} configured: release {self.release}, voice {session.get('voice')}, "
            f"{len(session.get('tools') or [])} tools, turn detection {applied or 'not reported'}, "
            f"tool waits {session.get('prism_tool_waits')}"
        )

    async def _handle_error(self, error: dict[str, Any]):
        code = error.get("code")
        if error.get("event_id") == OPENING_EVENT_ID and code in BUSY:
            # Not idle after all, and the service kept nothing: it goes again at
            # the next ready, unless a reply to the caller has opened the call.
            if not self._replied:
                self._opening = self._opening_sent
            outcome = "dropped, a reply has opened the call" if self._replied else "resending when idle"
            logger.info(f"{self} opening refused as busy ({code}); {outcome}")
            return
        if code in EXPECTED:
            logger.info(f"{self} {describe_error(error)}")
            return
        await self.push_error(error_msg=f"Prism error: {describe_error(error)}")

    async def _handle_audio(self, event: dict[str, Any]):
        audio = base64.b64decode(event.get("delta") or "")
        response_id = event.get("response_id", "")
        if not audio or response_id in self._cut:
            return
        if event.get("item_id"):
            self._items.setdefault(response_id, event["item_id"])
        await self._start_reply()
        if not self._speaking:
            self._speaking = True
            await self.push_frame(TTSStartedFrame())
        if response_id not in self._voiced:
            # The reply's first sound releases the words held for it.
            self._voiced.add(response_id)
            for text in self._held.pop(response_id, []):
                await self._push_text(text)
        if response_id not in self._unplayed and response_id not in self._playing:
            # Queued behind whatever is playing, so audible as soon as the agent is.
            (self._playing if self._bot_speaking else self._unplayed).append(response_id)
        self._audio_ms[response_id] = self._audio_ms.get(response_id, 0.0) + len(audio) / 2 / OUTPUT_SAMPLE_RATE * 1000
        await self.stop_ttfb_metrics()
        await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=OUTPUT_SAMPLE_RATE, num_channels=1))

    async def _push_text(self, text: str):
        frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        frame.includes_inter_frame_spaces = True
        await self.push_frame(frame)

    async def _handle_response_done(self, response: dict[str, Any]):
        response_id = response.get("id", "")
        # Words of a reply that never made a sound were never heard.
        self._held.pop(response_id, None)
        if response_id:
            self._done.add(response_id)
        await self._end_reply()
        await self.stop_processing_metrics()
        usage = response.get("usage")
        if usage:
            await self.start_llm_usage_metrics(token_usage(usage))
        status = response.get("status")
        if status == "failed":
            await self.push_error(error_msg=f"Prism response failed: {response.get('status_details')}")
        token = (response.get("metadata") or {}).get("prism_drain_token")
        if token and status == "completed" and response_id not in self._cut:
            # Only the latest reply that made a sound has a live token. Reported
            # once its audio has played: now, if it already has.
            self._drain = (response_id, token)
            if response_id not in self._unplayed and response_id not in self._playing:
                await self._send_drained()
        await self._maybe_continue()

    async def _handle_function_call(self, event: dict[str, Any]):
        call_id = event.get("call_id", "")
        name = event.get("name") or self._names.get(call_id, "")
        if not call_id or not name:
            logger.warning(f"{self} function call without a name or id: {event}")
            return
        round_id = event.get("response_id") or self._latest_response or ""
        self._calls.setdefault(round_id, set()).add(call_id)
        if name in CLOSING_TOOLS:
            self._closing_rounds.add(round_id)
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


#: Events that carry nothing this row needs beyond what other events already gave it.
_READ_ELSEWHERE = frozenset({
    "input_audio_buffer.committed",
    "prism.input_audio_buffer.speech_hint",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.failed",
    "response.text.done",
    "response.audio.done",
    "response.output_item.done",
})


def describe_error(error: dict[str, Any]) -> str:
    """One line for a server error: its code, what it affects, and the refused field if any."""
    parts = [f"[{error.get('code') or error.get('type') or 'error'}]"]
    if error.get("message"):
        parts.append(str(error["message"]))
    if error.get("param"):
        parts.append(f"(param: {error['param']})")
    if error.get("scope"):
        parts.append(f"(scope: {error['scope']})")
    return " ".join(parts)


def token_usage(usage: dict[str, Any]) -> LLMTokenUsage:
    """A ``response.done`` usage block in the framework's names; the service reports no splits."""
    return LLMTokenUsage(
        prompt_tokens=usage.get("input_tokens") or 0,
        completion_tokens=usage.get("output_tokens") or 0,
        total_tokens=usage.get("total_tokens") or 0,
    )


class ReportsPlayback(FrameProcessor):
    """Tells the service what the caller is hearing, as they hear it.

    Placed after the output transport: frames there move at the pace of the
    audio the caller hears, so the speaking frames mark when the agent became
    audible and fell silent, and the words passing are the words heard.
    """

    def __init__(self, service: RimePrismLLMService, **kwargs):
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
