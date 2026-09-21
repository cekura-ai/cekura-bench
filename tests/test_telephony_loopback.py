"""The media socket, the loopback and the tone instrument, wired together locally.

This is the whole calibration chain minus the carrier: the caller's audio is
framed exactly as Twilio frames it, encoded as Twilio encodes it, returned by the
endpoint the calibration call will use, and measured by the instrument that will
measure it. What a real call adds on top is the carrier -- which is the quantity
the calibration exists to find, and the only part that cannot be tested here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket

import numpy as np
import pytest
from aiohttp import ClientSession

from service.audio import pcm_to_ulaw, to_array, ulaw_to_pcm
from agent.telephony import StreamServer
from agent.tone import RATE, chirp, round_trip

FRAME_MS = 20.0
FRAME_BYTES = int(RATE * FRAME_MS / 1000.0)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.asyncio
async def test_a_chirp_survives_the_loopback_and_is_found_where_it_was_sent():
    """End to end: framed, encoded, echoed, decoded, located.

    The delay recovered here is our own plumbing rather than a carrier's, so the
    assertion is that it is small and that the instrument found the signal at
    all. A failure means the framing or the codec path is wrong, and on a real
    call that would surface as a transport correction of the wrong size with
    nothing to indicate it.
    """
    server = StreamServer(host="127.0.0.1", port=free_port())
    await server.start()
    try:
        async with ClientSession() as session:
            url = f"http://127.0.0.1:{server.port}/loopback"
            async with session.ws_connect(url) as socket_:
                await socket_.send_str(json.dumps(
                    {"event": "start", "start": {"streamSid": "MZtest", "callSid": "CAtest"}}
                ))

                lead_frames = 10                      # 200 ms of silence before the chirp
                reference = chirp()
                payload = pcm_to_ulaw(reference)
                outbound = b"\xff" * (FRAME_BYTES * lead_frames) + payload
                outbound += b"\xff" * (FRAME_BYTES * 25)

                async def feed() -> None:
                    for at in range(0, len(outbound) - FRAME_BYTES + 1, FRAME_BYTES):
                        await socket_.send_str(json.dumps({
                            "event": "media",
                            "media": {"payload": base64.b64encode(
                                outbound[at : at + FRAME_BYTES]).decode("ascii")},
                        }))
                        await asyncio.sleep(0.002)    # faster than realtime; the echo is not timed

                returned = bytearray()
                feeding = asyncio.create_task(feed())
                expected = len(outbound) // FRAME_BYTES
                for _ in range(expected):
                    message = await asyncio.wait_for(socket_.receive(), timeout=5.0)
                    frame = json.loads(message.data)
                    if frame.get("event") == "media":
                        returned += ulaw_to_pcm(base64.b64decode(frame["media"]["payload"]))
                await feeding

        sent_at_ms = lead_frames * FRAME_MS
        result = round_trip(sent_at_ms=sent_at_ms, recording=bytes(returned), reference=reference)

        assert result is not None, "the chirp did not survive the loopback"
        assert result.round_trip_ms < 5.0            # nothing but our own hop is in this path
        assert result.peak > 0.9
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_the_loopback_returns_the_audio_unchanged():
    """Anything the far end alters is measured as carrier latency or distortion.

    A resample, a re-encode or a buffer here would be invisible in the delay and
    would inflate every transport correction taken from this endpoint.
    """
    server = StreamServer(host="127.0.0.1", port=free_port())
    await server.start()
    try:
        sample = pcm_to_ulaw(to_array(chirp())[:FRAME_BYTES].tobytes())
        async with ClientSession() as session:
            async with session.ws_connect(f"http://127.0.0.1:{server.port}/loopback") as socket_:
                await socket_.send_str(json.dumps(
                    {"event": "start", "start": {"streamSid": "MZtest"}}
                ))
                await socket_.send_str(json.dumps({
                    "event": "media",
                    "media": {"payload": base64.b64encode(sample).decode("ascii")},
                }))
                message = await asyncio.wait_for(socket_.receive(), timeout=5.0)

        frame = json.loads(message.data)
        assert frame["streamSid"] == "MZtest"
        assert base64.b64decode(frame["media"]["payload"]) == sample
    finally:
        await server.stop()
