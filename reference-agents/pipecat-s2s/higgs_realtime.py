"""Boson AI's speech-to-speech model (Higgs Realtime), on the framework's OpenAI realtime service.

Higgs Realtime speaks the GA shape of OpenAI's realtime protocol, so the
framework's own service carries the call -- its turn handling, barge-in,
truncation, tool calls and transcripts are the ones the OpenAI row runs on.
Five places where Boson's server differs from OpenAI's decide what this
subclass changes, and nothing else is changed.

*The server speaks second.* OpenAI sends ``session.created`` as soon as the
socket opens, and the framework waits for it before configuring the session.
Boson sends nothing until the client's first ``session.update``, which it then
answers with ``session.created`` carrying the applied settings. Left alone,
both sides would wait for the other. So the update is sent as soon as the
socket opens, and that first ``session.created`` is taken as the
acknowledgement the framework otherwise waits for in ``session.updated``.

*A reply request names no modality.* The framework puts ``output_modalities``
in every ``response.create``. Boson takes the session's own modality, and a
request that names one changes what the model does: measured on the vendor's
endpoint, the same caller turn that drew a tool call on 3 of 3 bare requests
drew a spoken answer and no tool call on 3 of 3 requests naming ``["audio"]``.
A row whose model never calls a tool would be scored as one that cannot, so the
request is sent bare, as the vendor's own tool-calling example sends it.

*Boson sends events OpenAI does not.* After each reply's words it sends
``response.output_audio_transcript.length``, and it can send
``conversation.context.summarized``, ``session.idle_timeout`` and
``session.max_duration_reached``. The framework's parser raises on any type it
does not know, which ends its receive loop: on the first live calls the greeting
played and the agent never heard the caller again. Types the framework does not
know are logged and skipped before parsing; every known type is handled as
before.

*Boson's usage has no output breakdown.* Each ``response.done`` carries
token totals and ``input_token_details`` but no ``output_token_details``, which
the framework's model requires, so every reply would fail to parse and end the
loop the same way. An empty breakdown is filled in before parsing; the totals
the record reads are Boson's, unchanged.

*The model is chosen in the session, not the URL.* The framework appends
``?model=`` to the socket URL, which Boson also accepts; the model is named in
the session as well, because the vendor documents that form and the record
should not depend on a query string.

Protocol: https://docs.boson.ai/models/higgs-realtime/migrate-an-existing-integration
"""

from __future__ import annotations

import json

from loguru import logger
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

URL = "wss://api.boson.ai/v1/realtime"
# Boson's own speech-to-text for the caller's words; OpenAI's transcription
# models are not served there, and without one the record has no caller.
TRANSCRIPTION_MODEL = "higgs-stt-3.1"


class HiggsTurnDetection(events.TurnDetection):
    """The framework's server VAD, plus the two flags Boson's server VAD accepts.

    Boson documents ``create_response`` and ``interrupt_response`` as accepted
    and inert -- it always answers at the end of a turn and always stops for a
    barge-in -- but the configuration this row runs on names them, so they are
    sent and the record carries them. The framework's model would drop them.
    """

    create_response: bool | None = None
    interrupt_response: bool | None = None


class _KnownEventsOnly:
    """The socket, with what the framework cannot parse taken out or filled in.

    The framework's receive loop iterates the socket and parses each message;
    an unknown type raises and ends the loop. Wrapping the iteration is the one
    place a message can be dropped before that, without copying the loop.
    """

    def __init__(self, websocket):
        self._websocket = websocket

    def __getattr__(self, name):
        return getattr(self._websocket, name)

    def __aiter__(self):
        return self._known()

    async def _known(self):
        async for message in self._websocket:
            event = json.loads(message)
            kind = event.get("type")
            if kind not in events._server_event_types:
                logger.debug("higgs-realtime: skipping {} (not an OpenAI realtime event)", kind)
                continue
            usage = (event.get("response") or {}).get("usage") if kind == "response.done" else None
            if isinstance(usage, dict) and "output_token_details" not in usage:
                usage["output_token_details"] = {}
                message = json.dumps(event)
            yield message


class HiggsRealtimeLLMService(OpenAIRealtimeLLMService):
    """OpenAI's realtime service, pointed at Boson, for the differences above."""

    def __init__(self, *, api_key: str, **kwargs):
        super().__init__(api_key=api_key, base_url=URL, **kwargs)

    async def _connect(self):
        await super()._connect()
        # The server waits for the client's configuration before it says
        # anything; see the module note.
        if self._websocket:
            await self._send_session_update()

    async def _receive_task_handler(self):
        # Boson's extra event types would end the framework's loop; see the module note.
        self._websocket = _KnownEventsOnly(self._websocket)
        await super()._receive_task_handler()

    async def _handle_evt_session_created(self, evt):
        # The reply to our first session.update, carrying the settings it
        # applied: the acknowledgement OpenAI sends as session.updated.
        await self._handle_evt_session_updated(evt)

    async def send_client_event(self, event: events.ClientEvent):
        if isinstance(event, events.ResponseCreateEvent):
            await self._ws_send({"type": "response.create", "event_id": event.event_id})
            return
        await super().send_client_event(event)
