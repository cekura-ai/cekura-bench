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

Everything else -- the caller's turns, barge-in, tool calls and their results
-- is the OpenAI service's own handling of the same events.

Models: https://elevenlabs.io/docs/overview/models#eleven-realtime
Protocol: https://elevenlabs.io/docs/eleven-agents/guides/openai-realtime-compatibility
"""

from __future__ import annotations

import re

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect

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


class ElevenLabsRealtimeLLMService(OpenAIRealtimeLLMService):
    """The OpenAI realtime service, connected to ElevenLabs with ElevenLabs' header."""

    def __init__(self, **kwargs):
        super().__init__(base_url=URL, **kwargs)
        self._tags = DeliveryTags()

    async def _handle_evt_audio_transcript_delta(self, evt):
        if evt.delta:
            text = self._tags.feed(evt.delta)
            if text:
                await self._push_output_transcript_text_frames(text)

    async def _handle_evt_response_done(self, evt):
        text = self._tags.flush()
        if text:
            await self._push_output_transcript_text_frames(text)
        await super()._handle_evt_response_done(evt)

    async def _connect(self):
        try:
            if self._websocket:
                return
            self._websocket = await websocket_connect(
                uri=self.base_url, additional_headers={"xi-api-key": self.api_key}
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            logger.error(f"{self} could not connect: {e}")
            await self.push_error(error_msg=f"Error connecting: {e}", exception=e)
            self._websocket = None
