"""ElevenLabs' speech-to-speech models (Eleven Realtime), through the framework's OpenAI realtime service.

Eleven Realtime speaks OpenAI's realtime protocol -- the GA shape, the same
session, audio and tool events -- on ElevenLabs' own socket, so the framework's
OpenAI service is used unchanged apart from two things it cannot be told:

*Where it connects and how it signs in.* The socket is
``wss://api.elevenlabs.io/v1/convai/realtime`` and the key goes in an
``xi-api-key`` header. The service sends ``Authorization: Bearer``, which this
endpoint refuses (close 4001), so the connection is opened here.

*Which model answers.* The endpoint ignores a ``model`` query parameter and
starts every session on its smaller model; the model is chosen by the
``session.update`` the service sends once the context arrives, so it is set on
the session properties rather than left to the URL.

*Its transcript carries delivery tags.* The text of a reply can hold tags
such as ``[happy]`` that direct how a line is voiced and are never spoken.
They are taken out of the agent's transcript here, as the bench does with
other control tokens, so the record holds what the caller heard. A tag can be
split across deltas, so text from an opening bracket is held until it closes.

*Its events are not quite OpenAI's.* A response arrives without
``status_details``, which the framework's event schema requires, and the
endpoint sends events the framework has no schema for (such as
``output_audio_buffer.cleared``). Either one fails the framework's parse, which
ends its receive loop: the call carries on, and the agent never speaks again.
Each message is therefore normalised before the framework reads it -- the
missing field filled with null, an unknown event dropped and logged -- rather
than letting one event silence the call. A failed response gets an error
message when it arrives without one, since the framework reads it to report the
failure. A finished response also reports no
token usage, which the framework reads unconditionally; such a response is
handled as usual with its usage left unrecorded rather than recorded as zero.

*It refuses half a sample.* An audio append whose bytes do not divide into
16-bit samples closes the session (1008, "Invalid message received"), where
OpenAI's endpoint accepts it. The caller's audio reaches the service after
resampling and can end a chunk mid-sample, so a trailing odd byte is held and
sent at the front of the next chunk: every append is whole samples and the
stream stays aligned.

Everything else -- the caller's turns, barge-in, tool calls and their results
-- is the OpenAI service's own handling of the same events.

Models: https://elevenlabs.io/docs/overview/models#eleven-realtime
Protocol: https://elevenlabs.io/docs/eleven-agents/guides/openai-realtime-compatibility
"""

from __future__ import annotations

import base64
import json
import re
import time
from collections import deque

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

URL = "wss://api.elevenlabs.io/v1/convai/realtime"

#: Both directions, as the endpoint's session reports them.
SAMPLE_RATE = 24000

# A delivery tag: a bracketed lowercase word or phrase of three letters or more
# ("[happy]", "[laughs softly]"), and the space after it. Anything else in
# brackets is left alone as spoken text.
_TAG = re.compile(r"\[[a-z][a-z \-]{2,30}\] ?")


class DeliveryTags:
    """Takes delivery tags out of a reply's transcript as it streams."""

    def __init__(self) -> None:
        self._held = ""

    def feed(self, text: str) -> str:
        text = self._held + text
        self._held = ""
        start = text.rfind("[")
        if start != -1 and "]" not in text[start:] and len(text) - start <= 32:
            text, self._held = text[:start], text[start:]
        return _TAG.sub("", text)

    def flush(self) -> str:
        text, self._held = self._held, ""
        return _TAG.sub("", text)


def normalise(message: str | bytes) -> str | None:
    """One server message, made readable by the framework's schema; None to drop it."""
    event = json.loads(message)
    kind = event.get("type")
    if kind not in events._server_event_types:
        logger.debug(f"Eleven Realtime: dropped {kind!r}, which the framework has no schema for")
        return None
    response = event.get("response")
    if isinstance(response, dict):
        response.setdefault("status_details", None)
        if response.get("status") == "failed":
            # The framework reports a failed response by reading
            # status_details["error"]["message"]; without it that read raises
            # and ends the receive loop, so the call would go silent.
            details = response["status_details"] if isinstance(response["status_details"], dict) else {}
            error = details.get("error") if isinstance(details.get("error"), dict) else {}
            if not error.get("message"):
                error = {**error, "message": "Eleven Realtime reported a failed response without details"}
            response["status_details"] = {**details, "error": error}
    return json.dumps(event)


def _summary(message: str) -> str:
    """A sent message by type and shape, never its audio or text."""
    try:
        event = json.loads(message)
    except ValueError:
        return f"unparseable ({len(message)} chars)"
    kind = event.get("type")
    if kind == "input_audio_buffer.append":
        return f"append {len(base64.b64decode(event.get('audio') or ''))} B"
    item = event.get("item") or {}
    detail = ", ".join(f"{k}={event[k]}" for k in ("item_id", "content_index", "audio_end_ms") if k in event)
    if item:
        detail = f"item {item.get('type')}/{item.get('role', '')}"
    return f"{kind} {len(message)} chars {detail}".strip()


class _NormalisedSocket:
    """The socket, as the framework uses it: send, close, and iteration over normalised messages."""

    def __init__(self, socket):
        self._socket = socket
        self._carry = b""
        # What was last sent, by type and size only, for the log if the endpoint
        # closes the session: it names the message it refused but not which.
        self._sent = deque(maxlen=20)

    async def send(self, message):
        if '"input_audio_buffer.append"' in message:
            event = json.loads(message)
            if event.get("type") == "input_audio_buffer.append":
                pcm = self._carry + base64.b64decode(event.get("audio") or "")
                whole = len(pcm) - len(pcm) % 2
                pcm, self._carry = pcm[:whole], pcm[whole:]
                if not pcm:
                    return
                event["audio"] = base64.b64encode(pcm).decode()
                message = json.dumps(event)
        elif '"input_audio_buffer.clear"' in message:
            self._carry = b""
        self._sent.append((time.monotonic(), _summary(message)))
        await self._socket.send(message)

    async def close(self):
        await self._socket.close()

    async def __aiter__(self):
        try:
            async for message in self._socket:
                normalised = normalise(message)
                if normalised is not None:
                    yield normalised
        except ConnectionClosed as closed:
            now = time.monotonic()
            recent = "; ".join(f"-{now - at:.2f}s {what}" for at, what in self._sent)
            logger.warning(f"Eleven Realtime closed the session ({closed}); last sent: {recent}")
            raise


class ElevenLabsRealtimeLLMService(OpenAIRealtimeLLMService):
    """The OpenAI realtime service, connected to ElevenLabs with ElevenLabs' header."""

    def __init__(self, **kwargs):
        super().__init__(base_url=URL, **kwargs)
        self._tags = DeliveryTags()
        self._unmetered = False

    async def start_llm_usage_metrics(self, tokens):
        if not self._unmetered:
            await super().start_llm_usage_metrics(tokens)

    async def _handle_evt_audio_transcript_delta(self, evt):
        if evt.delta:
            text = self._tags.feed(evt.delta)
            if text:
                await self._push_output_transcript_text_frames(text)

    async def _handle_evt_response_done(self, evt):
        text = self._tags.flush()
        if text:
            await self._push_output_transcript_text_frames(text)
        self._unmetered = evt.response.usage is None
        if self._unmetered:
            evt.response.usage = events.Usage(
                total_tokens=0, input_tokens=0, output_tokens=0,
                input_token_details=events.TokenDetails(), output_token_details=events.TokenDetails(),
            )
        try:
            await super()._handle_evt_response_done(evt)
        finally:
            self._unmetered = False

    async def _connect(self):
        try:
            if self._websocket:
                return
            self._websocket = _NormalisedSocket(await websocket_connect(
                uri=self.base_url, additional_headers={"xi-api-key": self.api_key}
            ))
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            logger.error(f"{self} could not connect: {e}")
            await self.push_error(error_msg=f"Error connecting: {e}", exception=e)
            self._websocket = None
