"""Smallest AI's speech-to-speech model (Hydra), over its realtime socket.

Hydra's events carry OpenAI Realtime's names for the same things -- audio
deltas, ``speech_started`` / ``speech_stopped``, tool-call arguments,
``response.done`` -- but the session around them is its own, so no framework
service fits and the protocol is spoken directly, as ``azure_realtime`` does.

Five things about this service decide how it is written.

*One configuration, sent once.* The service speaks first (``session.created``)
and then takes exactly one ``session.configure``; ``session.configured`` is
the acknowledgement. A configuration sent before ``session.created`` is refused
and the session never opens, so it waits for it. Tools therefore go in with the first context, and nothing
about the session changes afterwards. The key travels in the URL, so it is
scrubbed from every error this module raises or logs.

*The opening is the session's.* Hydra answers only caller audio and tool
results: a user text message, or a reply asked for with nothing pending, is
ignored without an error. The opening instruction every row is given is
therefore appended to the instructions, and the session is asked to begin with
a reply of its own (``generate_initial_response``). The record names that
addendum and hashes it.

*The service owns the caller's turns.* It announces where the caller starts
and stops and stops its own reply when talked over, and takes no turn-detection
settings at all. The pipeline follows those announcements and stops playback
on them. There is no truncation message to report what was heard.

*Hydra transcribes neither side.* It does stream each reply's words in
``response.output_audio_transcript.delta`` frames (not in its documentation, and
with no ``.done``); those are the agent's transcript, held until the reply's
first audio and dropped if it never sounds, as on the other rows. The caller's
words come from Smallest's own speech-to-text (Pulse Pro, same key), the
vendor's documented answer: what was sent from a second before Hydra heard the
caller start until it heard them stop is transcribed at once, one stretch at a
time so the finals keep their order. (The silence before a turn is left out:
the transcriber invents words in half a minute of it.) Hydra marks the turn complete up to two seconds
after that, too late for the context to place the words before the reply; and
the transcriber's delay is announced to the pipeline the way a speech-to-text
service announces its own, so the context waits for the words that long.

*A tool result does not resume the reply.* A response is asked for once every
call the model made has its result. A reply that called its tools in silence is
waited on until it closes; one that spoke while calling them is not, because
Hydra holds such a reply open until the next one is asked for, and then carries
on in that same reply.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import urllib.parse
import urllib.request
import wave
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
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    STTMetadataFrame,
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

URL = "wss://api.smallest.ai/waves/v1/s2s"
TRANSCRIBE_URL = "https://api.smallest.ai/waves/v1/stt/"
#: The caller's transcriber, pinned by name: Smallest serves more than one.
TRANSCRIPTION_MODEL = "pulse-pro"

#: Caller audio in, at the rate the pipeline runs; reply audio out.
SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

READY_TIMEOUT_S = 10.0
TRANSCRIBE_TIMEOUT_S = 30.0
#: How long after the caller stops their words can take to arrive: Pulse Pro
#: answers a turn in about half a second, and this leaves room for a slow one.
TRANSCRIPT_P99_S = 2.5
#: Caller audio kept from before Hydra announced the start of their speech: the
#: announcement arrives after the first word has begun.
PREROLL_S = 1.0

#: Tools that end the call. A reply that carried one and made a sound has said
#: its closing line; asking for another only makes the model speak, or close, again.
CLOSING_TOOLS = frozenset({"end_call", "transfer_call"})


def session_url(api_key: str, model: str) -> str:
    return f"{URL}?{urllib.parse.urlencode({'model': model, 'api_key': api_key})}"


def redact(text: str, api_key: str) -> str:
    """``text`` with the key removed in every spelling a URL can carry it."""
    if not api_key:
        return text
    for form in {api_key, urllib.parse.quote_plus(api_key), urllib.parse.quote(api_key, safe="")}:
        text = text.replace(form, "<redacted>")
    return text


def opening_addendum(openings: list[str]) -> str:
    """The opening instruction, as a section appended to the system prompt."""
    return "\n".join(text.strip() for text in openings if text.strip())


def wav_bytes(pcm: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buffer.getvalue()


def transcribe(pcm: bytes, api_key: str, rate: int = SAMPLE_RATE) -> str:
    """One caller turn through Pulse Pro; blocking, so it is run in a thread."""
    url = f"{TRANSCRIBE_URL}?{urllib.parse.urlencode({'model': TRANSCRIPTION_MODEL, 'language': 'en'})}"
    request = urllib.request.Request(
        url,
        data=wav_bytes(pcm, rate),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TRANSCRIBE_TIMEOUT_S) as response:
        body = json.loads(response.read())
    return (body.get("transcription") or "").strip()


class SmallestHydraLLMService(LLMService):
    """One Hydra session, for the length of one call.

    The socket is opened with the pipeline; the session is configured, tools and
    opening included, with the first context frame, and caller audio is sent
    only once the service has acknowledged that configuration.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "hydra-v1.1",
        voice: str = "sterling",
        instructions: str = "",
        tools: ToolsSchema | list | None = None,
        **kwargs,
    ):
        unsupported = {f.name: None for f in fields(LLMSettings) if f.name != "extra"}
        super().__init__(
            settings=LLMSettings(**{**unsupported, "model": model, "system_instruction": instructions}), **kwargs
        )
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._tools = tools
        self._opening: list[str] = []
        self._websocket = None
        self._closing = False
        self._receive_task = None
        self._open_task = None
        self._created = asyncio.Event()
        self._configured = asyncio.Event()
        self._context = None
        # The response the service has open, if any, and whether a new one is
        # owed once it closes.
        self._response_id: str | None = None
        self._reply_owed = False
        self._speaking = False
        self._responding = False
        # Words that arrived ahead of their reply's audio, by response.
        self._held: dict[str, list[str]] = {}
        self._voiced: set[str] = set()
        self._worded: set[str] = set()
        # Replies the caller talked over: audio already in flight must not reopen them.
        self._cut: set[str] = set()
        self._audio_response: str | None = None
        # Calls the model made whose results have not been sent, and results
        # already sent: the context is pushed again for every tool.
        self._outstanding: set[str] = set()
        self._delivered: set[str] = set()
        # The reply that called a closing tool, and whether a reply was asked for after it.
        self._closing_round: str | None = None
        self._replied_after_closing = False
        # Caller audio sent since the last completed caller turn, and the turns
        # waiting for their transcript, in order.
        self._caller_pcm = bytearray()
        self._caller_start: int | None = None
        self._transcripts: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue()
        self._transcribe_task = None
        self.silent_replies = 0

    def can_generate_metrics(self) -> bool:
        return True

    # -- lifecycle --------------------------------------------------------

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # The caller's words arrive this long after they stop, as from a
        # speech-to-text service, and the context holds their turn open for them.
        await self.broadcast_frame(STTMetadataFrame, service_name=self.name, ttfs_p99_latency=TRANSCRIPT_P99_S)
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
            self._websocket = await websocket_connect(uri=session_url(self._api_key, self._model), max_size=None)
            self._receive_task = self.create_task(self._receive())
            self._transcribe_task = self.create_task(self._transcribe_turns())
        except Exception as exc:
            # The exception text can carry the URL, and with it the key.
            await self.push_error(error_msg=f"Error connecting: {redact(str(exc), self._api_key)}")
            self._websocket = None

    async def _disconnect(self):
        self._closing = True
        try:
            for task in (self._open_task, self._transcribe_task):
                if task:
                    await self.cancel_task(task, timeout=1.0)
            self._open_task = self._transcribe_task = None
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
        except Exception as exc:
            await self.push_error(error_msg=f"Error disconnecting: {redact(str(exc), self._api_key)}")

    # -- sending ----------------------------------------------------------

    async def _send(self, event: dict[str, Any]):
        if not self._websocket:
            return
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

    def instructions(self) -> str:
        addendum = opening_addendum(self._opening)
        return f"{self._instructions.rstrip()}\n\n{addendum}" if addendum else self._instructions

    def session(self) -> dict[str, Any]:
        """The one configuration the service takes. It has no turn-detection settings to send."""
        session: dict[str, Any] = {
            "instructions": self.instructions(),
            "voice": self._voice,
            "output_audio_sample_rate": OUTPUT_SAMPLE_RATE,
            # The opening reply is the session's own: see the module docstring.
            "generate_initial_response": bool(self._opening),
        }
        tools = self.encoded_tools()
        if tools:
            session["tools"] = tools
        return session

    async def _send_audio(self, frame: InputAudioRawFrame):
        # Audio sent before the configuration carrying the tools is accepted
        # would be answered by a session with no tools, so it is dropped.
        if not self._configured.is_set():
            return
        self._caller_pcm.extend(frame.audio)
        preroll = int(PREROLL_S * SAMPLE_RATE) * 2
        if self._caller_start is None and len(self._caller_pcm) > 2 * preroll:
            # Silent: only the last second can become part of a turn.
            del self._caller_pcm[:-preroll]
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(frame.audio).decode()})

    async def _request_reply(self):
        """Ask for a response, now or as soon as the open one closes."""
        if self._outstanding:
            return
        closing = self._closing_round is not None
        if closing and (self._closing_round in self._voiced or self._replied_after_closing):
            # A closing reply that sounded has said goodbye; a silent one gets
            # one reply to say it in, and never a second.
            self._reply_owed = False
            logger.debug(f"{self} no reply asked for after the closing tool in {self._closing_round}")
            return
        if self._response_id is not None and self._response_id not in self._voiced:
            self._reply_owed = True
            return
        self._reply_owed = False
        self._replied_after_closing = closing
        if self._response_id is not None:
            logger.debug(f"{self} reply asked for with {self._response_id} still open: it spoke while calling its tools")
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
            self._opening = self._openings(context)
            self._open_task = self.create_task(self._open())
            return
        for tool_call_id, output in self._results(context).items():
            if tool_call_id in self._delivered:
                continue
            self._delivered.add(tool_call_id)
            self._outstanding.discard(tool_call_id)
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
    def _openings(context) -> list[str]:
        """The opening turn every row is given, as the text of the caller-side messages."""
        texts = []
        for message in context.get_messages():
            if isinstance(message, LLMSpecificMessage) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content)
        return texts

    async def _open(self):
        try:
            await asyncio.wait_for(self._created.wait(), READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self.push_error(error_msg=f"Hydra did not open the session in {READY_TIMEOUT_S:.0f}s")
            return
        await self._send({"type": "session.configure", "session": self.session()})
        try:
            await asyncio.wait_for(self._configured.wait(), READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self.push_error(error_msg=f"Hydra did not acknowledge the configuration in {READY_TIMEOUT_S:.0f}s")

    async def _handle_interruption(self):
        for response_id in (self._response_id, self._audio_response):
            if response_id:
                self._cut.add(response_id)
                self._held.pop(response_id, None)
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

    # -- the caller's transcript -----------------------------------------

    def _caller_started(self):
        if self._caller_start is None:
            self._caller_start = max(0, len(self._caller_pcm) - int(PREROLL_S * SAMPLE_RATE) * 2)

    def _caller_stopped(self, item_id: str):
        """The caller stopped: their speech, from just before Hydra heard it start, goes to the transcriber."""
        start = self._caller_start or 0
        pcm, self._caller_pcm, self._caller_start = bytes(self._caller_pcm[start:]), bytearray(), None
        if pcm:
            self._transcripts.put_nowait((item_id, pcm))

    async def _transcribe_turns(self):
        """One turn at a time, so the caller's finals keep their order."""
        while True:
            item_id, pcm = await self._transcripts.get()
            started = time.monotonic()
            try:
                text = await asyncio.to_thread(transcribe, pcm, self._api_key)
            except Exception as exc:  # noqa: BLE001 -- a lost turn must not stop the ones after it
                logger.warning(
                    f"{self} caller turn {item_id} ({len(pcm) / 2 / SAMPLE_RATE:.1f}s) not transcribed: "
                    f"{redact(repr(exc), self._api_key)}"
                )
                continue
            logger.debug(
                f"{self} caller turn {item_id}: {len(pcm) / 2 / SAMPLE_RATE:.1f}s transcribed by "
                f"{TRANSCRIPTION_MODEL} in {time.monotonic() - started:.2f}s"
            )
            if text and not self._closing:
                # Upstream: the context's user half sits before this service.
                await self.push_frame(
                    TranscriptionFrame(text, "", time_now_iso8601(), result={"item_id": item_id, "transcriber": TRANSCRIPTION_MODEL}),
                    FrameDirection.UPSTREAM,
                )

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
                error_msg=f"Hydra closed the session "
                          f"(code {getattr(ws, 'close_code', None)}, {getattr(ws, 'close_reason', None)!r})"
            )

    async def _dispatch(self, event: dict[str, Any]):
        kind = event.get("type", "")

        if kind == "session.created":
            logger.info(f"{self} session {event.get('session_id')} open")
            self._created.set()

        elif kind == "session.configured":
            applied = event.get("session") or {}
            logger.info(
                f"{self} configured: voice {applied.get('voice')}, {len(applied.get('tools') or [])} tools, "
                f"opens with its own reply: {applied.get('generate_initial_response')}"
            )
            self._configured.set()

        elif kind == "response.created":
            self._response_id = (event.get("response") or {}).get("id")
            await self._start_reply()

        elif kind == "response.output_audio.delta":
            await self._handle_audio(event)

        elif kind == "response.output_audio_transcript.delta":
            text, response_id = event.get("delta", ""), event.get("response_id", "")
            if not text or response_id in self._cut:
                return
            self._worded.add(response_id)
            if response_id in self._voiced:
                await self._push_text(text)
            else:
                self._held.setdefault(response_id, []).append(text)

        elif kind == "response.function_call_arguments.done":
            await self._handle_function_call(event)

        elif kind == "response.done":
            await self._handle_response_done(event.get("response") or {})

        elif kind == "input_audio_buffer.speech_started":
            self._caller_started()
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

        elif kind == "input_audio_buffer.speech_stopped":
            self._caller_stopped(event.get("item_id", ""))
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

        elif kind in ("error", "warning"):
            message = redact(json.dumps(event.get("error") or event), self._api_key)
            if kind == "error":
                await self.push_error(error_msg=f"Hydra error: {message}")
            else:
                logger.warning(f"{self} Hydra warning: {message}")

    async def _handle_audio(self, event: dict[str, Any]):
        audio = base64.b64decode(event.get("delta") or "")
        if not audio:
            return
        response_id = event.get("response_id", "")
        if response_id in self._cut:
            return
        self._audio_response = response_id
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
        await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=OUTPUT_SAMPLE_RATE, num_channels=1))

    async def _handle_response_done(self, response: dict[str, Any]):
        response_id = response.get("id", "")
        # Words of a reply that never made a sound were never heard.
        self._held.pop(response_id, None)
        if response_id in self._voiced and response_id not in self._worded:
            # The reply text is an undocumented frame; if it stops coming, the
            # agent's half of the transcript goes quiet and this is where it shows.
            self.silent_replies += 1
            logger.warning(f"{self} reply {response_id} was spoken with no text ({self.silent_replies} so far)")
        if self._response_id in (None, response_id):
            # A reply that closes after the next one opened must not end that one.
            self._response_id = None
            await self._end_reply()
            await self.stop_processing_metrics()
        usage = response.get("usage")
        if usage:
            await self.start_llm_usage_metrics(token_usage(usage))
        if response.get("status") == "failed":
            await self.push_error(error_msg=f"Hydra response failed: {response.get('status_details')}")
        if self._reply_owed:
            await self._request_reply()

    async def _push_text(self, text: str):
        frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        # Each piece is a whole phrase with no space at either end ("... with Dr."
        # then "Lee?"), so the context joins them with one.
        frame.includes_inter_frame_spaces = False
        await self.push_frame(frame)

    async def _handle_function_call(self, event: dict[str, Any]):
        call_id, name = event.get("call_id", ""), event.get("name", "")
        if not call_id or not name:
            logger.warning(f"{self} function call without a name or id: {event}")
            return
        self._outstanding.add(call_id)
        if name in CLOSING_TOOLS and self._closing_round is None:
            self._closing_round = event.get("response_id") or self._response_id
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
    """A ``response.done`` usage block in the framework's names."""
    return LLMTokenUsage(
        prompt_tokens=usage.get("input_tokens") or 0,
        completion_tokens=usage.get("output_tokens") or 0,
        total_tokens=usage.get("total_tokens") or 0,
    )
