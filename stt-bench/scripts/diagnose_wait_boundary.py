"""Locate delays across native wait, Python resumption, and timer callbacks.

Diagnostic overrides only. These outputs cannot qualify a production run.
Build latency_probe_native.c as a dylib first; see --help.
"""
import argparse
import asyncio
import ctypes
import gc
import json
import os
from pathlib import Path
import select
import selectors
import sys
import threading
import time

from stt_bench import diagnostics, macos_timer
from stt_bench.data import sha256
from diagnose_duplex import MemoryLog


class NativeEvent(ctypes.Structure):
    _fields_ = [('ident', ctypes.c_uint64), ('filter', ctypes.c_int32)]


class Wait(ctypes.Structure):
    _fields_ = [(n, ctypes.c_double) for n in ('enter', 'returned', 'cpu_enter', 'cpu_returned')] + [('error', ctypes.c_int32)]


class Sample(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_double) for n in ('before', 'after', 'cpu_seconds')]
               + [('state', ctypes.c_int32), ('error', ctypes.c_int32)]
               + [(n, ctypes.c_uint64) for n in ('pageins', 'bytesread', 'byteswritten')]
               + [('resource_error', ctypes.c_int32)])


class Policy(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in ('role', 'qos', 'relative_priority', 'base_priority', 'current_priority', 'error')]


def record(row):
    return {name: getattr(row, name) for name, _ in row._fields_}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--memory-log', action='store_true')
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--sparse', action='store_true', help='Retain only slow boundaries to limit diagnostic allocation')
    parser.add_argument('--stock-selector', action='store_true', help='Use the production selector implementation')
    parser.add_argument('--interactive-qos', action='store_true', help='Diagnostic change to this process main thread only')
    parser.add_argument('--disable-gc', action='store_true', help='Diagnostic control for cyclic garbage collection')
    parser.add_argument('--inject', choices=['none', 'loop-cpu', 'gil'], default='none')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Choose a new diagnostic output directory')
    lib = ctypes.CDLL(str(args.library.resolve()), use_errno=True)
    lib.initialize.argtypes = []; lib.initialize.restype = ctypes.c_int
    assert lib.initialize() == 0
    lib.measured_wait.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                 ctypes.POINTER(NativeEvent), ctypes.POINTER(Wait)]
    lib.measured_wait.restype = ctypes.c_int
    lib.start_observer.argtypes = [ctypes.POINTER(Sample), ctypes.c_int]
    lib.start_observer.restype = ctypes.c_int
    lib.stop_observer.argtypes = []; lib.stop_observer.restype = ctypes.c_int
    lib.read_policy.argtypes = [ctypes.POINTER(Policy)]; lib.read_policy.restype = None
    lib.set_interactive.argtypes = []; lib.set_interactive.restype = ctypes.c_int
    before_policy, after_policy = Policy(), Policy()
    lib.read_policy(ctypes.byref(before_policy))
    if args.interactive_qos and lib.set_interactive():
        raise RuntimeError('Unable to apply diagnostic thread QoS')
    lib.read_policy(ctypes.byref(after_policy))
    waits, timers, turns, injections, collections, callbacks = [], [], [], [], [], []
    event_buffer, measurement = (NativeEvent * 1024)(), Wait()

    class MeasuredSelector(selectors.KqueueSelector):
        def select(self, timeout=None):
            if args.stock_selector:
                return super().select(timeout)
            maximum = self._max_events or 1
            events = event_buffer
            count = lib.measured_wait(self._selector.fileno(), maximum,
                                      -1 if timeout is None else max(0, timeout), events, ctypes.byref(measurement))
            python_resumed = time.perf_counter()
            if not args.sparse or python_resumed-measurement.enter > .025 or python_resumed-measurement.returned > .002:
                waits.append(dict(**record(measurement), python_resumed=python_resumed, timeout=timeout,
                                  ready_fds=[events[i].ident for i in range(max(0, count))]))
            if count < 0:
                if measurement.error == 4:  # EINTR
                    return []
                raise OSError(measurement.error, os.strerror(measurement.error))
            ready = []
            for event in events[:count]:
                mask = selectors.EVENT_READ if event.filter == select.KQ_FILTER_READ else selectors.EVENT_WRITE
                key = self._key_from_fd(event.ident)
                if key:
                    ready.append((key, mask & key.events))
            return ready

    class MeasuredLoop(asyncio.SelectorEventLoop):
        def __init__(self):
            super().__init__(MeasuredSelector())
        def _run_once(self):
            begin, cpu = time.perf_counter(), time.thread_time()
            super()._run_once()
            end, spent = time.perf_counter(), time.thread_time()-cpu
            if not args.sparse or end-begin > .025 or spent > .002:
                turns.append(dict(start=begin, end=end, cpu_seconds=spent))

    class MeasuredTimer(macos_timer.MacOSDeadlineTimer):
        async def wait_until(self, deadline):
            begin = time.perf_counter()
            observed = await super().wait_until(deadline)
            resumed = time.perf_counter()
            if not args.sparse or resumed-deadline > .003:
                timers.append(dict(start=begin, deadline=deadline, observed=observed, resumed=resumed, fd=self._fd))
            return observed

    original_timer = macos_timer.MacOSDeadlineTimer
    macos_timer.MacOSDeadlineTimer = MeasuredTimer
    original_identity = diagnostics.pacing_identity
    diagnostics.pacing_identity = lambda: dict(**original_identity(),
        diagnostic_only=dict(native_wait=not args.stock_selector, sparse=args.sparse,
                             observer=args.observe, memory_log=args.memory_log, injection=args.inject,
                             interactive_qos=args.interactive_qos, disable_gc=args.disable_gc))
    if args.memory_log:
        diagnostics.EventLog = MemoryLog
    def gc_event(phase, info):
        collections.append(dict(time=time.perf_counter(), phase=phase, generation=info['generation']))
    gc.callbacks.append(gc_event)
    original_handle_run = asyncio.Handle._run
    def measured_callback(handle):
        begin, cpu = time.perf_counter(), time.thread_time()
        original_handle_run(handle)
        end, spent = time.perf_counter(), time.thread_time()-cpu
        if end-begin > .002:
            callback = handle._callback
            owner = getattr(callback, '__self__', None)
            name = getattr(callback, '__qualname__', type(callback).__name__)
            if isinstance(owner, asyncio.Task):
                name += ':' + getattr(owner.get_coro(), '__qualname__', 'task')
            callbacks.append(dict(start=begin, end=end, cpu_seconds=spent, callback=name))
    asyncio.Handle._run = measured_callback
    threads = []
    def busy():
        start, cpu = time.perf_counter(), time.thread_time()
        until = start + .080
        while time.perf_counter() < until:
            pass
        injections.append(dict(kind=args.inject, start=start, end=time.perf_counter(), cpu_seconds=time.thread_time()-cpu))
    def launch_busy():
        thread = threading.Thread(target=busy)
        threads.append(thread)
        thread.start()
    async def execute():
        if args.inject != 'none':
            asyncio.get_running_loop().call_later(.6, busy if args.inject == 'loop-cpu' else launch_busy)
        return await diagnostics.local_probe(args.out, seconds=args.seconds, repeats=args.repeats)
    buffer = (Sample * 120000)()  # Hard bound, about four MB; no disk writes while observing.
    previous_interval = sys.getswitchinterval()
    gc_was_enabled = gc.isenabled()
    if args.disable_gc:
        gc.disable()
    if args.inject == 'gil':
        sys.setswitchinterval(.15)
    count = 0
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        if args.observe and lib.start_observer(buffer, len(buffer)):
            raise RuntimeError('Native observer could not start')
        with asyncio.Runner(loop_factory=MeasuredLoop) as runner:
            report = runner.run(execute())
    finally:
        if args.observe:
            count = lib.stop_observer()
        for thread in threads:
            thread.join()
        sys.setswitchinterval(previous_interval)
        gc.callbacks.remove(gc_event)
        if gc_was_enabled:
            gc.enable()
        asyncio.Handle._run = original_handle_run
        macos_timer.MacOSDeadlineTimer = original_timer
    evidence = dict(waits=waits, timers=timers, turns=turns, thread_samples=[record(r) for r in buffer[:count]],
                    gc=collections, callbacks=callbacks, injections=injections, pid=os.getpid(),
                    cpu_percent=100*(time.process_time()-cpu)/(time.perf_counter()-wall),
                    library_sha256=sha256(args.library), observer_samples=count)
    evidence['policy_before'] = record(before_policy)
    evidence['policy_after'] = record(after_policy)
    (args.out / 'native-wait-evidence.json').write_text(json.dumps(evidence))
    for source in (Path(__file__), Path(__file__).with_name('latency_probe_native.c')):
        (args.out / source.name).write_bytes(source.read_bytes())
    print(json.dumps(dict(out=str(args.out), trials=[{k:t[k] for k in
        ('valid', 'interval_ms_max', 'send_duration_ms_max', 'wakeup_delay_ms_max', 'gate_reasons')}
        for t in report['trials']], cpu_percent=evidence['cpu_percent'])), flush=True)


if __name__ == '__main__':
    main()
