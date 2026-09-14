"""Controlled selector/logging experiments; no provider calls or benchmark scores."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import selectors
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from stt_bench.data import FRAME_BYTES, sha256, write_json
from stt_bench.streaming import EventLog, pacing_metrics, read_events, stream_audio
import stt_bench.streaming as streaming

async def baseline_wait(deadline, now, *, timer=None):
    """Frozen pre-fix scheduling method, retained for reproducible comparisons."""
    while (remaining := deadline - now()) > .001:
        await asyncio.sleep(remaining - .001)
    spin_end = time.perf_counter() + .001
    while now() < deadline and time.perf_counter() < spin_end:
        pass


@asynccontextmanager
async def no_production_timer():
    yield None


ORIGINAL_WAIT = baseline_wait
THREAD_WAKES = []
MACH = None
MATCHED_EXECUTOR = None
ASYNC_TIMER = None


async def async_timer_wait(deadline, now, *, timer=None):
    remaining = deadline - now()
    if remaining > .001:
        await ASYNC_TIMER.wait_until(time.perf_counter() + remaining - .001)
    spin_end = time.perf_counter() + .001
    while now() < deadline and time.perf_counter() < spin_end:
        pass


async def async_exact_wait(deadline, now, *, timer=None):
    remaining = deadline - now()
    if remaining > 0:
        await ASYNC_TIMER.wait_until(time.perf_counter() + remaining)


async def timer_trial(path, mode, seconds, critical):
    global ASYNC_TIMER
    from critical_timer import AsyncCriticalTimer
    async with AsyncCriticalTimer(critical=critical) as timer:
        ASYNC_TIMER = timer
        try:
            return await trial(path, mode, seconds)
        finally:
            ASYNC_TIMER = None


async def poll_wait(deadline, now, *, timer=None):
    while (remaining := deadline - now()) > .001:
        await asyncio.sleep(min(.002, remaining - .001))
    spin_end = time.perf_counter() + .001
    while now() < deadline and time.perf_counter() < spin_end:
        pass


async def thread_wait(deadline, now, *, timer=None):
    remaining = deadline - now()
    if remaining > .001:
        coarse_deadline = time.perf_counter() + remaining - .001
        def sleep():
            time.sleep(max(0, coarse_deadline - time.perf_counter()))
            return time.perf_counter()
        woke = await asyncio.to_thread(sleep)
        delivered = time.perf_counter()
        THREAD_WAKES.append(dict(sleep_late_ms=(woke-coarse_deadline)*1000,
                                 callback_delay_ms=(delivered-woke)*1000))
    spin_end = time.perf_counter() + .001
    while now() < deadline and time.perf_counter() < spin_end:
        pass


async def mach_wait(deadline, now, *, timer=None):
    remaining = deadline - now()
    if remaining > .001:
        coarse_deadline = time.perf_counter() + remaining - .001
        woke = await asyncio.to_thread(MACH.wait_until, coarse_deadline)
        delivered = time.perf_counter()
        THREAD_WAKES.append(dict(sleep_late_ms=(woke-coarse_deadline)*1000,
                                 callback_delay_ms=(delivered-woke)*1000))
    spin_end = time.perf_counter() + .001
    while now() < deadline and time.perf_counter() < spin_end:
        pass


async def matched_mach_wait(deadline, now, *, timer=None):
    remaining = deadline - now()
    if remaining > .001:
        coarse_deadline = time.perf_counter() + remaining - .001
        woke = await asyncio.get_running_loop().run_in_executor(MATCHED_EXECUTOR, MACH.wait_until, coarse_deadline)
        delivered = time.perf_counter()
        THREAD_WAKES.append(dict(sleep_late_ms=(woke-coarse_deadline)*1000,
                                 callback_delay_ms=(delivered-woke)*1000))
    spin_end = time.perf_counter() + .001
    while now() < deadline and time.perf_counter() < spin_end:
        pass


class MemoryLog:
    def __init__(self):
        self.origin = time.perf_counter()
        self.events = []

    def now(self):
        return time.perf_counter() - self.origin

    def emit(self, kind, *, at=None, **fields):
        self.events.append(dict(kind=kind, time_seconds=self.now() if at is None else at, **fields))

    def close(self):
        pass


class InstrumentedSelector:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def select(self, timeout=None):
        start = time.perf_counter()
        ready = self.inner.select(timeout)
        elapsed = time.perf_counter() - start
        self.calls.append(dict(timeout=timeout, elapsed=elapsed, ready=len(ready), at=start))
        return ready


async def trial(path, mode, seconds):
    log = EventLog(path) if mode.endswith('disk') else MemoryLog()
    frames = round(seconds / .02)
    pcm = b'\0' * FRAME_BYTES * (frames + 50)
    async def finalize(t0):
        log.emit('local_finalize', t0_seconds=t0)
    try:
        if mode.startswith('noop'):
            async def send(frame):
                pass
            await stream_audio(pcm, frames, send, finalize, log)
        else:
            async def handler(ws):
                async for frame in ws:
                    await ws.send(frame[:8])
            async with serve(handler, '127.0.0.1', 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with connect(f'ws://127.0.0.1:{port}') as ws:
                    received = 0
                    done = asyncio.Event()
                    async def receive():
                        nonlocal received
                        async for frame in ws:
                            received += 1
                            log.emit('reply', count=received)
                            if received == frames + 50:
                                done.set()
                    receiver = asyncio.create_task(receive())
                    await stream_audio(pcm, frames, ws.send, finalize, log)
                    await asyncio.wait_for(done.wait(), 5)
                    await ws.close()
                    await receiver
    finally:
        log.close()
    events = read_events(path) if mode.endswith('disk') else log.events
    if not mode.endswith('disk'):
        path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    return pacing_metrics(events)


def main():
    global MACH, MATCHED_EXECUTOR
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--backends', nargs='+', default=['kqueue', 'select'])
    parser.add_argument('--modes', nargs='+', default=['noop-memory', 'noop-disk', 'websocket-memory', 'websocket-disk'])
    parser.add_argument('--waits', nargs='+', default=['asyncio'])
    args = parser.parse_args()
    streaming.deadline_timer = no_production_timer
    if any(w.startswith('mach') for w in args.waits):
        from native_deadline import MachDeadline
        MACH = MachDeadline()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'experiment-source.py').write_text(Path(__file__).read_text())
    (args.out / 'streaming-source.py').write_text(Path('src/stt_bench/streaming.py').read_text())
    for helper in ('critical_timer.py', 'native_deadline.py'):
        (args.out / helper).write_text(Path(__file__).with_name(helper).read_text())
    streaming_digest = sha256(args.out / 'streaming-source.py')
    results = []
    # Alternate backends, retaining every trial. No retry-until-pass selection.
    for repeat in range(args.repeats):
        for mode in args.modes:
            backends = args.backends if repeat % 2 == 0 else args.backends[::-1]
            choices = [(b, w) for b in backends for w in (args.waits if repeat % 2 == 0 else args.waits[::-1])]
            for backend, wait in choices:
                qos_evidence = []
                if wait in ('mach-matched', 'timer-critical', 'timer-normal'):
                    from native_deadline import capture_qos, match_worker_qos
                    MATCHED_EXECUTOR = ThreadPoolExecutor(max_workers=1, initializer=match_worker_qos,
                                                          initargs=(capture_qos(), qos_evidence))
                    MATCHED_EXECUTOR.submit(lambda: None).result()
                timer = None
                if wait.startswith('timer-'):
                    from critical_timer import CriticalTimer
                    timer = CriticalTimer(critical=wait == 'timer-critical')
                    MACH = timer
                streaming.wait_until = {'thread': thread_wait, 'poll': poll_wait, 'mach': mach_wait,
                                       'mach-matched': matched_mach_wait, 'timer-critical': matched_mach_wait,
                                       'timer-normal': matched_mach_wait, 'async-critical': async_timer_wait,
                                       'async-normal': async_timer_wait, 'async-exact': async_exact_wait, 'asyncio': ORIGINAL_WAIT}[wait]
                THREAD_WAKES.clear()
                selector = InstrumentedSelector(selectors.KqueueSelector() if backend == 'kqueue' else selectors.SelectSelector())
                path = args.out / f'{repeat}-{mode}-{backend}-{wait}.jsonl'
                cpu_start, wall_start = time.process_time(), time.perf_counter()
                with asyncio.Runner(loop_factory=lambda: asyncio.SelectorEventLoop(selector)) as runner:
                    metrics = runner.run(timer_trial(path, mode, args.seconds, wait in ('async-critical', 'async-exact'))
                                         if wait.startswith('async-') else trial(path, mode, args.seconds))
                if MATCHED_EXECUTOR is not None:
                    MATCHED_EXECUTOR.shutdown(wait=True, cancel_futures=True)
                    MATCHED_EXECUTOR = None
                if timer is not None:
                    timer.close()
                cpu_seconds, wall_seconds = time.process_time() - cpu_start, time.perf_counter() - wall_start
                positive = [c for c in selector.calls if c['timeout'] and c['timeout'] > 0]
                overrun = [max(0, c['elapsed'] - c['timeout']) * 1000 for c in positive]
                row = dict(repeat=repeat, mode=mode, backend=backend, wait=wait, **metrics,
                           worker_qos=qos_evidence,
                           process_cpu_seconds=cpu_seconds, wall_seconds=wall_seconds,
                           process_cpu_percent_of_one_core=100 * cpu_seconds / wall_seconds,
                           thread_sleep_late_ms_max=max((x['sleep_late_ms'] for x in THREAD_WAKES), default=None),
                           thread_callback_delay_ms_max=max((x['callback_delay_ms'] for x in THREAD_WAKES), default=None),
                           selector_overrun_ms_max=max(overrun, default=0),
                           selector_over_20ms=sum(x > 20 for x in overrun),
                           selector_worst_calls=sorted(positive, key=lambda c: c['elapsed']-c['timeout'], reverse=True)[:5])
                results.append(row)
                write_json(args.out / 'results.json', dict(mode='scheduler_diagnosis',
                           python=platform.python_version(), system=platform.platform(),
                           streaming_sha256=streaming_digest,
                           seconds=args.seconds, completed_at=datetime.now(timezone.utc).isoformat(), rows=results))
                print(f"{repeat} {mode:18} {backend:6} {wait:7} valid={metrics['valid']} max_gap={metrics['interval_ms_max']:.2f}ms selector_overrun={row['selector_overrun_ms_max']:.2f}ms", flush=True)


if __name__ == '__main__':
    main()
