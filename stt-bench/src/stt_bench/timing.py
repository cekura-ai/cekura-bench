"""Absolute deadline waits for audio sending and local heartbeats.

On macOS, streams own an asynchronous native absolute-deadline timer with no
busy waiting, held inside a scoped process latency activity. Other platforms use cooperative sleeps and at most 1 ms of final
waiting. Neither path promises hard real-time operating-system scheduling.
"""
import asyncio
from contextlib import asynccontextmanager
import sys
import math
import time

COARSE_WAIT_SECONDS = .001
FINAL_WAIT_SECONDS = .001
SCHEDULER_VERSION = 'native-deadline-activity-v4'


def scheduler_backend():
    return 'macos-kqueue-critical-v1+latency-activity-v1' if sys.platform == 'darwin' else 'bounded-cooperative-v2'


def process_activity_identity():
    if sys.platform == 'darwin':
        from .macos_activity import activity_identity
        return activity_identity()
    return None


@asynccontextmanager
async def deadline_timer():
    if sys.platform == 'darwin':
        from .macos_timer import MacOSDeadlineTimer
        from .macos_activity import LatencyActivity
        with LatencyActivity():
            async with MacOSDeadlineTimer() as timer:
                yield timer
    else:
        yield None


async def wait_until(deadline, now=time.perf_counter, *, timer=None):
    if not math.isfinite(deadline):
        raise ValueError('Deadline must be finite')
    if timer is not None:
        # Convert a caller-relative deadline into the same absolute monotonic
        # clock used by the kernel timer. Late deadlines never add a fresh wait.
        if now is time.perf_counter:
            await timer.wait_until(deadline)
            return
        remaining = deadline - now()
        absolute_now = time.perf_counter()
        if remaining > 0:
            await timer.wait_until(absolute_now + remaining)
        return
    while (remaining := deadline - now()) > FINAL_WAIT_SECONDS:
        await asyncio.sleep(min(COARSE_WAIT_SECONDS, remaining - FINAL_WAIT_SECONDS))
    spin_end = time.perf_counter() + FINAL_WAIT_SECONDS
    while now() < deadline and time.perf_counter() < spin_end:
        pass
