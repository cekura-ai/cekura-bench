"""The phone leg as a service-bench adapter, so one caller and one set of probes serve both lanes.

Twilio's Media Streams protocol carries a call's audio over a websocket: our
outbound call reaches the agent's number, the TwiML `<Connect><Stream>` points the
media at a socket we run, and from then on the agent's voice arrives as `media`
events and anything we send is played toward it. That is the same shape as a
realtime provider's socket -- audio out, audio in, on a clock -- so it is
implemented as a `RealtimeAdapter` rather than as a second harness. The caller,
the probes, the scenarios and the record format then work over a phone call
unchanged, and a service-bench/the agent bench difference is a difference in the channel rather
than in two implementations of the same idea.

What the phone does not have is declared rather than worked around:

* **no manual commit.** There is no turn-boundary message on a phone call; the
  agent's own endpointer decides. Probes needing an exact commit are excluded.
* **no text modality.** There is no text channel to a phone number at all.
* **no VAD events.** The carrier does not tell us what the agent's endpointer
  decided, so probes reading those events must void rather than infer.

μ-law at 8 kHz in both directions, which is the call, not a setting.

**Timing.** Every inbound frame carries the carrier's own millisecond timestamp
as well as our arrival time, and both are recorded. The carrier clock removes our
receive jitter from the inbound direction; it does not remove carrier latency,
and it shares no origin with our outbound clock. Neither is a substitute for the
injected-tone round trip that gates publishing any the agent bench latency.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, AsyncIterator, Protocol

from service import events as ev
from service.adapters.base import AdapterError, RealtimeAdapter, SessionClosed
from service.audio import pcm_to_ulaw, ulaw_to_pcm
from agent.detector import speech_bounds as _phone_bounds

RATE = 8000


class MediaStream(Protocol):
    """One call's media socket, as this adapter needs it.

    Narrow on purpose: the adapter is then testable against the protocol's own
    message shapes without a network, a carrier or an account, which is where
    a framing mistake would otherwise hide until it cost a real call.
    """

    async def send(self, message: dict[str, Any]) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self) -> AsyncIterator[dict[str, Any]]: ...


class TwilioStreamAdapter(RealtimeAdapter):
    name = "twilio-stream"
    input_rate = RATE
    output_rate = RATE
    supports_manual_commit = False
    supports_text_modality = False
    emits_vad_events = False
    speech_bounds = staticmethod(_phone_bounds)

    def __init__(self, *, stream: MediaStream, model: str = "phone", api_key: str = "", **kw: Any) -> None:
        super().__init__(model=model, api_key=api_key, **kw)
        self._stream = stream
        self._started = asyncio.Event()
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self.media_format: dict[str, Any] | None = None
        # Frames the carrier says it sent vs frames we saw, so a gap in the
        # record is visible as a gap rather than as a quiet agent.
        self.inbound_frames = 0
        self.carrier_first_timestamp_ms: float | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Wait for the carrier to open the stream and announce the call."""
        self._receiver = asyncio.create_task(self._receive())
        try:
            await asyncio.wait_for(self._started.wait(), timeout=30.0)
        except asyncio.TimeoutError as exc:
            raise AdapterError("carrier never started the media stream") from exc

    async def close(self) -> None:
        if self._receiver:
            self._receiver.cancel()
        await self._stream.close()
        self.closed.set()

    async def _send_json(self, payload: dict[str, Any]) -> None:
        await self._stream.send(payload)

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        if self.stream_sid is None:
            raise SessionClosed("media stream is not open")
        await self._stream.send({
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64.b64encode(pcm_to_ulaw(pcm)).decode("ascii")},
        })

    async def discard_outbound(self) -> None:
        """Drop whatever the carrier still holds of our audio but has not played.

        The carrier buffers what we send and plays it at realtime, so abandoning
        a caller utterance by stopping the sender leaves the tail of it still to
        come. Anything measured after that point would be measured against audio
        the agent had not finished hearing.
        """
        if self.stream_sid is not None:
            await self._stream.send({"event": "clear", "streamSid": self.stream_sid})

    async def _commit(self) -> None:
        raise AdapterError("a phone call has no turn-boundary message")

    async def _send_text(self, text: str) -> None:
        raise AdapterError("a phone call has no text channel")

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        raise AdapterError("tools are answered by the agent, not over the call")

    # ── receiving ────────────────────────────────────────────────────────────

    async def _receive(self) -> None:
        try:
            async for message in self._stream:
                self._handle(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                     # noqa: BLE001 - recorded, not swallowed
            self.log.emit(ev.SESSION_ERROR, source=self.name, detail=repr(exc))
            self.closed.set()

    def _handle(self, message: dict[str, Any]) -> None:
        event = message.get("event")
        if event == "start":
            start = message.get("start", {})
            self.stream_sid = start.get("streamSid") or message.get("streamSid")
            self.call_sid = start.get("callSid")
            self.media_format = start.get("mediaFormat")
            self.session_ack = start
            self.log.emit(ev.SESSION_CONFIGURED, stream_sid=self.stream_sid, call_sid=self.call_sid)
            self._started.set()
        elif event == "media":
            media = message.get("media", {})
            if media.get("track") == "outbound":
                return                               # our own audio echoed back
            payload = media.get("payload")
            if not payload:
                return
            stamp = media.get("timestamp")
            if stamp is not None and self.carrier_first_timestamp_ms is None:
                self.carrier_first_timestamp_ms = float(stamp)
            self.inbound_frames += 1
            self._on_agent_audio(ulaw_to_pcm(base64.b64decode(payload)))
        elif event == "stop":
            self._on_agent_audio_done("carrier")
            self.log.emit(ev.SESSION_CLOSED, reason="carrier stopped the stream")
            self.closed.set()
        elif event == "mark":
            self.log.emit(ev.CARRIER_MARK, name=message.get("mark", {}).get("name"))


class WebsocketMediaStream:
    """A ``MediaStream`` over an accepted websocket, decoding the carrier's JSON."""

    def __init__(self, socket: Any) -> None:
        self._socket = socket

    async def send(self, message: dict[str, Any]) -> None:
        await self._socket.send_str(json.dumps(message))

    async def close(self) -> None:
        await self._socket.close()

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        async for raw in self._socket:
            data = getattr(raw, "data", raw)
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            if isinstance(data, str):
                yield json.loads(data)
