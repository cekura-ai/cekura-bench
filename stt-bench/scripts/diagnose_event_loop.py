"""Locate local pauses using timer, selector and per-thread CPU timestamps.

These instrumented probes are diagnostics, not production preflight evidence.
No provider requests. Records are buffered until all trials have finished.
"""
import argparse
import asyncio
import json
import os
import threading
from pathlib import Path
import selectors
import time

from stt_bench.diagnostics import local_probe
import stt_bench.macos_timer as native

SELECTS, TURNS, TIMERS = [], [], []
STOP_MS, OUTPUT_DIR = None, None


def clock_anchor():
    before = time.perf_counter_ns()
    wall = time.time_ns()
    after = time.perf_counter_ns()
    return dict(perf_before_ns=before, wall_ns=wall, perf_after_ns=after)


class MeasuredSelector(selectors.KqueueSelector):
    def select(self, timeout=None):
        start, cpu = time.perf_counter(), time.thread_time()
        ready = super().select(timeout)
        end = time.perf_counter()
        SELECTS.append(dict(start=start, end=end, timeout=timeout,
                            cpu_ms=(time.thread_time()-cpu)*1000,
                            ready_fds=[key.fd for key, mask in ready]))
        return ready


class MeasuredLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(MeasuredSelector())

    def _run_once(self):
        start, cpu, first_select = time.perf_counter(), time.thread_time(), len(SELECTS)
        try:
            return super()._run_once()
        finally:
            TURNS.append(dict(start=start, end=time.perf_counter(),
                              cpu_ms=(time.thread_time()-cpu)*1000,
                              select_indexes=list(range(first_select, len(SELECTS)))))


class MeasuredTimer(native.MacOSDeadlineTimer):
    async def wait_until(self, deadline):
        start, cpu = time.perf_counter(), time.thread_time()
        observed = await super().wait_until(deadline)
        resumed = time.perf_counter()
        TIMERS.append(dict(start=start, deadline=deadline, observed=observed, resumed=resumed,
                           fd=self._fd, cpu_ms=(time.thread_time()-cpu)*1000,
                           observation_late_ms=(observed-deadline)*1000,
                           callback_delivery_ms=(resumed-observed)*1000))
        if STOP_MS is not None and (resumed-deadline)*1000 > STOP_MS:
            # Preserve a small clock-aligned snapshot, then end the profiler's
            # launched target before its one-second recording window rolls out.
            snapshot = dict(reason='diagnostic_stop_after_late_wakeup',
                            pid=os.getpid(), native_thread_id=threading.get_native_id(),
                            anchor=clock_anchor(), timers=TIMERS[-10:],
                            selects=SELECTS[-20:], turns=TURNS[-20:])
            (OUTPUT_DIR/'diagnostic-stop.json').write_text(json.dumps(snapshot))
            os._exit(2)  # Diagnostic only; process exit also closes loopback sockets.
        return observed


def main():
    global STOP_MS, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--stop-on-late-ms', type=float)
    args = parser.parse_args()
    STOP_MS, OUTPUT_DIR = args.stop_on_late_ms, args.out
    native.MacOSDeadlineTimer = MeasuredTimer
    initial_anchor = clock_anchor()
    with asyncio.Runner(loop_factory=MeasuredLoop) as runner:
        report = runner.run(local_probe(args.out, seconds=args.seconds, repeats=args.repeats))
    (args.out/'experiment-source.py').write_text(Path(__file__).read_text())
    (args.out/'loop-evidence.json').write_text(json.dumps(dict(pid=os.getpid(), native_thread_id=threading.get_native_id(),
        anchors=[initial_anchor, clock_anchor()], selects=SELECTS, turns=TURNS, timers=TIMERS)))
    slow = sorted(TIMERS, key=lambda t:t['resumed']-t['deadline'], reverse=True)[:10]
    findings = []
    for timer in slow:
        waits = [s for s in SELECTS if s['start'] <= timer['observed'] and s['end'] >= timer['deadline']]
        turns = [t for t in TURNS if t['start'] <= timer['resumed'] and t['end'] >= timer['deadline']]
        findings.append(dict(timer=timer, selector_waits=waits, event_loop_turns=turns))
    (args.out/'slow-waits.json').write_text(json.dumps(findings, indent=2))
    print(json.dumps(dict(trials=report['trials'], slow_waits=findings[:3]), indent=2), flush=True)


if __name__ == '__main__':
    main()
