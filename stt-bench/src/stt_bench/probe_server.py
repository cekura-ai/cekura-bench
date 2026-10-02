"""Independent, single-client loopback receiver for streaming diagnostics.

Launch with ``python -m stt_bench.probe_server``. The first stdout line is JSON
``{"port": N}``; all subsequent communication uses that WebSocket port.

Protocol (all timestamps are absolute ``time.perf_counter()`` seconds):
* Binary audio is exactly 640 bytes, with an unsigned little-endian frame index
  in its first eight bytes. ``audio_ack`` replies carry ``index`` and
  ``received_seconds``.
* Independent ``heartbeat`` messages carry ``sequence``, ``scheduled_seconds``
  and ``sent_seconds``. Missed heartbeat deadlines are recorded, not replayed
  in a burst. They don't depend on the client's audio sender making progress.
* ``{"type": "finish"}`` returns one ``summary`` with the complete index and
  receipt-time inventories, then closes the connection and exits.

A disconnect also exits. A client must connect within 15 seconds. The caller
must still terminate and await its subprocess in its own cleanup block.
``--receiver-stall-ms`` deliberately blocks this process once, just before the
fifth frame's receipt is recorded, to validate independent receive diagnostics.
"""

import argparse
import asyncio
import contextlib
import json
import math
import time

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from .timing import deadline_timer, wait_until


FRAME_BYTES = 640
HEARTBEAT_SECONDS = .020


def summarize(indexes, received_seconds, invalid_frames, emitted_heartbeats=None):
    """Retain raw observations alongside their compact integrity summary."""
    gaps = [(b - a) * 1000 for a, b in zip(received_seconds, received_seconds[1:])]
    return dict(type="summary", frames=len(indexes), indexes=indexes,
                received_seconds=received_seconds, invalid_frames=invalid_frames,
                emitted_heartbeats=emitted_heartbeats if emitted_heartbeats is not None else [],
                frame_order_valid=indexes == list(range(len(indexes))),
                receiver_gap_ms_max=max(gaps) if gaps else None)


async def run_server(receiver_stall_ms=0, receiver_delay_ms=0):
    if any(not math.isfinite(value) or value < 0
           for value in (receiver_stall_ms, receiver_delay_ms)):
        raise ValueError("Receiver delays must be finite nonnegative numbers")
    connected = asyncio.Event()
    finished = asyncio.Event()

    async def handler(ws):
        if connected.is_set():
            await ws.close(code=1008, reason="Only one diagnostic client is supported")
            return
        connected.set()
        indexes, received_seconds, invalid_frames, emitted_heartbeats = [], [], [], []

        async def heartbeats():
            deadline = time.perf_counter() + HEARTBEAT_SECONDS
            sequence = 0
            async with deadline_timer() as timer:
                while True:
                    await wait_until(deadline, timer=timer)
                    sent = time.perf_counter()
                    heartbeat = dict(type="heartbeat", sequence=sequence,
                                     scheduled_seconds=deadline, sent_seconds=sent)
                    await ws.send(json.dumps(heartbeat))
                    emitted_heartbeats.append(heartbeat)
                    skipped = max(1, math.floor((sent - deadline) / HEARTBEAT_SECONDS) + 1)
                    deadline += skipped * HEARTBEAT_SECONDS
                    sequence += skipped

        heartbeat_task = asyncio.create_task(heartbeats())
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    if len(message) != FRAME_BYTES:
                        invalid_frames.append(dict(bytes=len(message),
                                                   received_seconds=time.perf_counter()))
                        await ws.send(json.dumps(dict(type="error", reason="invalid_frame_size",
                                                      bytes=len(message))))
                        continue
                    if len(indexes) == 4 and receiver_stall_ms:
                        time.sleep(receiver_stall_ms / 1000)
                    at = time.perf_counter()
                    index = int.from_bytes(message[:8], "little")
                    indexes.append(index)
                    received_seconds.append(at)
                    if receiver_delay_ms:
                        await asyncio.sleep(receiver_delay_ms / 1000)
                    await ws.send(json.dumps(dict(type="audio_ack", index=index,
                                                  received_seconds=at,
                                                  sent_seconds=time.perf_counter())))
                    continue
                try:
                    command = json.loads(message)
                except json.JSONDecodeError:
                    command = None
                if not isinstance(command, dict) or command.get("type") != "finish":
                    await ws.close(code=1008, reason="Expected binary audio or finish command")
                    return
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
                await ws.send(json.dumps(summarize(indexes, received_seconds, invalid_frames,
                                                  emitted_heartbeats)))
                await ws.close()
                return
        except ConnectionClosed:
            pass
        finally:
            heartbeat_task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                    await heartbeat_task
            finally:
                finished.set()  # Timer failure must not strand run_server's waiter.

    async with serve(handler, "127.0.0.1", 0, compression=None, close_timeout=2) as server:
        print(json.dumps(dict(port=server.sockets[0].getsockname()[1])), flush=True)
        await asyncio.wait_for(connected.wait(), timeout=15)
        await finished.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver-stall-ms", type=float, default=0)
    parser.add_argument("--receiver-delay-ms", type=float, default=0,
                        help="Asynchronous processing delay after each frame receipt")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0
           for value in (args.receiver_stall_ms, args.receiver_delay_ms)):
        parser.error("Receiver delays must be finite and nonnegative")
    try:
        asyncio.run(run_server(args.receiver_stall_ms, args.receiver_delay_ms))
    except TimeoutError:
        parser.exit(1, "No diagnostic client connected within 15 seconds\n")


if __name__ == "__main__":
    main()
