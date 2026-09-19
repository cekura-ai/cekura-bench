"""The phone leg adapter, against the carrier's own message shapes.

No network, no account, no call. A framing mistake here -- the wrong envelope,
the echo track counted as the agent, a missing stream id -- would otherwise stay
hidden until it had spent a real call to reveal itself, and a call cannot be
replayed.
"""

from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest

from lane_a import events as ev
from lane_a.adapters.base import AdapterError
from lane_a.audio import pcm_to_ulaw
from lane_b.adapters.twilio_stream import RATE, TwilioStreamAdapter

STREAM_SID = "MZ0000000000000000000000000000000f"
CALL_SID = "CA0000000000000000000000000000000f"


def tone_pcm(ms: float, rate: int = RATE) -> bytes:
    t = np.arange(int(rate * ms / 1000.0)) / rate
    return (np.sin(2 * np.pi * 300 * t) * 12000).astype(np.int16).tobytes()


def frame(payload: bytes, track: str = "inbound", timestamp: str = "0") -> dict:
    return {
        "event": "media",
        "media": {
            "track": track,
            "chunk": "1",
            "timestamp": timestamp,
            "payload": base64.b64encode(pcm_to_ulaw(payload)).decode("ascii"),
        },
    }


class FakeStream:
    """A scripted media socket that records what the adapter sent."""

    def __init__(self, inbound: list[dict]) -> None:
        self.inbound = inbound
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, message: dict) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    async def __aiter__(self):
        for message in self.inbound:
            yield message
            await asyncio.sleep(0)
        while True:                                  # the socket stays open
            await asyncio.sleep(0.01)


def adapter_over(inbound: list[dict]) -> tuple[TwilioStreamAdapter, FakeStream]:
    stream = FakeStream(inbound)
    return TwilioStreamAdapter(stream=stream, log=ev.EventLog(ev.Clock())), stream


START = {"event": "start", "start": {
    "streamSid": STREAM_SID, "callSid": CALL_SID,
    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
}}


class TestTheCallAnnouncesItself:
    @pytest.mark.asyncio
    async def test_connect_waits_for_the_stream_to_start(self):
        """Sending before the carrier announces the stream has nowhere to go.

        Every outbound frame must carry the stream id, so audio sent before the
        start event is silently discarded by the carrier rather than refused.
        """
        adapter, _ = adapter_over([{"event": "connected"}, START])
        await adapter.connect()

        assert adapter.stream_sid == STREAM_SID
        assert adapter.call_sid == CALL_SID
        await adapter.close()


class TestTracks:
    @pytest.mark.asyncio
    async def test_the_agents_voice_is_recorded_and_our_own_echo_is_not(self):
        """The carrier can mirror our audio back on a second track.

        Counting it as the agent would make every caller utterance look like an
        instant reply, which is the one error a latency benchmark must not make.
        """
        adapter, _ = adapter_over([START, frame(tone_pcm(100)), frame(tone_pcm(500), track="outbound")])
        await adapter.connect()
        await asyncio.sleep(0.05)

        assert adapter.inbound_frames == 1
        assert len(adapter.agent_pcm) == len(tone_pcm(100))
        await adapter.close()

    @pytest.mark.asyncio
    async def test_the_carrier_timestamp_is_kept_beside_our_arrival_time(self):
        """Two clocks with no shared origin. Keeping one loses the ability to check."""
        adapter, _ = adapter_over([START, frame(tone_pcm(20), timestamp="1400")])
        await adapter.connect()
        await asyncio.sleep(0.05)

        assert adapter.carrier_first_timestamp_ms == 1400.0
        assert adapter.agent_timeline.chunks[0].t_wall > 0
        await adapter.close()


class TestSending:
    @pytest.mark.asyncio
    async def test_audio_goes_out_mulaw_encoded_under_the_stream_id(self):
        adapter, stream = adapter_over([START])
        await adapter.connect()
        await adapter.send_audio(tone_pcm(20))

        media = [m for m in stream.sent if m["event"] == "media"]
        assert len(media) == 1
        assert media[0]["streamSid"] == STREAM_SID
        assert base64.b64decode(media[0]["media"]["payload"]) == pcm_to_ulaw(tone_pcm(20))
        await adapter.close()

    @pytest.mark.asyncio
    async def test_abandoning_an_utterance_clears_what_the_carrier_still_holds(self):
        """The carrier plays our audio at realtime and buffers the rest.

        Stopping the sender leaves the tail still to come, so anything measured
        after that instant is measured against audio the agent had not yet heard.
        """
        adapter, stream = adapter_over([START])
        await adapter.connect()
        await adapter.discard_outbound()

        assert {"event": "clear", "streamSid": STREAM_SID} in stream.sent
        await adapter.close()


class TestWhatAPhoneCannotDo:
    @pytest.mark.asyncio
    async def test_a_commit_is_refused_rather_than_ignored(self):
        """There is no turn-boundary message on a call. A silent no-op would
        publish provider-endpointed cells under a manual-commit label."""
        adapter, _ = adapter_over([START])
        await adapter.connect()

        with pytest.raises(AdapterError):
            await adapter.commit()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_the_stream_stopping_closes_the_session(self):
        adapter, _ = adapter_over([START, {"event": "stop"}])
        await adapter.connect()
        await asyncio.sleep(0.05)

        assert adapter.closed.is_set()
        await adapter.close()
