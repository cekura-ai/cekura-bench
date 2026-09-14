"""Protocol and lifecycle tests; these don't assert operating-system timing."""
import asyncio
import json
import sys
import time
import textwrap

from websockets.asyncio.client import connect

from stt_bench.probe_server import summarize


def test_probe_summary_preserves_bad_order_and_invalid_frames():
    result = summarize([0, 2, 1], [10, 10.02, 10.1], [{"bytes": 2}])
    assert result["indexes"] == [0, 2, 1]
    assert not result["frame_order_valid"]
    assert result["frames"] == 3
    assert result["invalid_frames"] == [{"bytes": 2}]
    assert abs(result["receiver_gap_ms_max"] - 80) < 1e-6


def test_probe_summary_retains_complete_heartbeat_inventory():
    heartbeats = [dict(type='heartbeat', sequence=i, scheduled_seconds=10 + i * .02,
                       sent_seconds=10.001 + i * .02) for i in range(3)]
    result = summarize([0], [10.0], [], heartbeats)
    assert result['emitted_heartbeats'] == heartbeats
    assert result['emitted_heartbeats'][-1]['sequence'] == 2


async def with_server(client):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "stt_bench.probe_server",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        line = await asyncio.wait_for(process.stdout.readline(), 10)
        if not line:
            raise AssertionError((await process.stderr.read()).decode())
        port = json.loads(line)["port"]
        async with connect(f"ws://127.0.0.1:{port}", compression=None) as ws:
            await client(ws)
        assert await asyncio.wait_for(process.wait(), 10) == 0
        assert await process.stderr.read() == b""
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()


def test_probe_heartbeats_audio_summary_and_exit():
    async def client(ws):
        # A heartbeat must arrive even before any audio is sent.
        heartbeat = json.loads(await asyncio.wait_for(ws.recv(), 5))
        assert heartbeat["type"] == "heartbeat"
        assert heartbeat["sent_seconds"] >= heartbeat["scheduled_seconds"]
        assert heartbeat["sent_seconds"] <= time.perf_counter()
        for index in range(3):
            await ws.send(index.to_bytes(8, "little") + bytes(632))
        await ws.send(b"bad")
        await ws.send(json.dumps({"type": "finish"}))
        messages = [json.loads(raw) async for raw in ws]
        acknowledgments = [m for m in messages if m["type"] == "audio_ack"]
        assert [m["index"] for m in acknowledgments] == [0, 1, 2]
        summary = next(m for m in messages if m["type"] == "summary")
        assert summary["received_seconds"] == [m["received_seconds"] for m in acknowledgments]
        assert summary["frames"] == 3
        assert summary["frame_order_valid"]
        assert summary["invalid_frames"][0]["bytes"] == 3
        emitted = [heartbeat] + [m for m in messages if m['type'] == 'heartbeat']
        assert summary['emitted_heartbeats'] == emitted
        assert all(m['sent_seconds'] >= m['received_seconds'] for m in acknowledgments)
    asyncio.run(with_server(client))


def test_probe_exits_on_disconnect_without_finish():
    async def client(ws):
        await ws.close()
    asyncio.run(with_server(client))


def test_probe_exits_when_heartbeat_timer_fails():
    bootstrap = textwrap.dedent('''
        import asyncio
        import sys
        import stt_bench.probe_server as server

        class FailedTimer:
            async def __aenter__(self):
                print("timer-failure-injected", file=sys.stderr, flush=True)
                raise RuntimeError("injected timer failure")
            async def __aexit__(self, *exc):
                pass

        server.deadline_timer = FailedTimer
        asyncio.run(server.run_server())
    ''')

    async def scenario():
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-c', bootstrap,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            port = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))['port']
            async with connect(f'ws://127.0.0.1:{port}', compression=None) as ws:
                # Synchronize on the actual injected failure, not a timed sleep.
                assert await asyncio.wait_for(process.stderr.readline(), 5) == b'timer-failure-injected\n'
                await ws.close()
            # Before the fix, heartbeat_task's exception skipped finished.set(),
            # leaving run_server alive here even though the client disconnected.
            assert await asyncio.wait_for(process.wait(), 5) == 0
            assert b'RuntimeError: injected timer failure' in await process.stderr.read()
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
    asyncio.run(scenario())
