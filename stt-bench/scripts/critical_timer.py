"""Experimental macOS EVFILT_TIMER wake-up with minimal timer coalescing.

Public SDK APIs only: sys/event.h defines kevent64, NOTE_CRITICAL and
NOTE_ABSOLUTE | NOTE_MACHTIME. NOTE_CRITICAL changes this timer's coalescing
policy; it does not change process/thread scheduling priority or grant real-time
execution. CriticalTimer blocks a dedicated worker; AsyncCriticalTimer uses
descriptor readiness to wake the existing asyncio loop without a worker.

Sources: apple-oss-distributions/xnu bsd/sys/event.h and
bsd/kern/kern_event.c (filt_timervalidate, filt_timerarm).
"""

import asyncio
import ctypes
import errno
import math
import os
import threading
import time

try:
    from .native_deadline import MachDeadline
except ImportError:
    from native_deadline import MachDeadline


class _Kevent64(ctypes.Structure):
    _fields_ = [
        ("ident", ctypes.c_uint64), ("filter", ctypes.c_int16),
        ("flags", ctypes.c_uint16), ("fflags", ctypes.c_uint32),
        ("data", ctypes.c_int64), ("udata", ctypes.c_uint64),
        ("ext", ctypes.c_uint64 * 2),
    ]


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


EVFILT_TIMER = -7
EV_ADD = 0x0001
EV_DELETE = 0x0002
EV_ONESHOT = 0x0010
EV_ERROR = 0x4000
NOTE_ABSOLUTE = 0x00000008
NOTE_LEEWAY = 0x00000010
NOTE_CRITICAL = 0x00000020
NOTE_MACHTIME = 0x00000100


class CriticalTimer:
    """One reusable kqueue and one-shot timer; no asynchronous resource owner."""

    def __init__(self, *, critical=True):
        self.clock = MachDeadline()  # Checks platform and actual Python clock.
        if ctypes.sizeof(_Kevent64) != 48 or _Kevent64.data.offset != 16:
            raise RuntimeError("Unexpected kevent64 ABI layout")
        self.lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self.lib.kqueue.argtypes = []
        self.lib.kqueue.restype = ctypes.c_int
        self.lib.kevent64.argtypes = [
            ctypes.c_int, ctypes.POINTER(_Kevent64), ctypes.c_int,
            ctypes.POINTER(_Kevent64), ctypes.c_int, ctypes.c_uint,
            ctypes.POINTER(_Timespec),
        ]
        self.lib.kevent64.restype = ctypes.c_int
        self.critical = bool(critical)
        self._lock = threading.Lock()
        self.fd = self.lib.kqueue()
        if self.fd < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        try:
            os.set_inheritable(self.fd, False)
        except BaseException:
            os.close(self.fd)
            self.fd = -1
            raise

    def wait_until(self, deadline):
        """Wait for a perf_counter absolute deadline and return the wake timestamp.

        Mach absolute ticks are converted using the runtime timebase; wall time
        and continuous-time/sleep epochs are not used. ext[1]=0 supplies zero
        leeway. A one-second safety margin fails if the timer never arrives;
        that generic timeout is only a watchdog, not the intended wake source.
        """
        ticks = self.clock.ticks(deadline)
        if ticks >= 2**63:
            raise OverflowError("EVFILT_TIMER deadline exceeds signed int64")
        with self._lock:
            if self.fd < 0:
                raise RuntimeError("CriticalTimer is closed")
            if time.perf_counter() >= deadline:
                return time.perf_counter()
            flags = NOTE_ABSOLUTE | NOTE_MACHTIME | NOTE_LEEWAY
            if self.critical:
                flags |= NOTE_CRITICAL
            change = _Kevent64(ident=1, filter=EVFILT_TIMER,
                               flags=EV_ADD | EV_ONESHOT, fflags=flags, data=ticks)
            result = _Kevent64()
            safety_deadline = deadline + 1.0
            while True:
                remaining = safety_deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("EVFILT_TIMER did not arrive before watchdog deadline")
                nanos = math.ceil(remaining * 1_000_000_000)
                timeout = _Timespec(*divmod(nanos, 1_000_000_000))
                count = self.lib.kevent64(self.fd, ctypes.byref(change), 1,
                                           ctypes.byref(result), 1, 0, ctypes.byref(timeout))
                if count == -1:
                    code = ctypes.get_errno()
                    if code == errno.EINTR:
                        # Rearm at exactly the same absolute deadline after a
                        # signal; never extend it by a fresh relative interval.
                        continue
                    raise OSError(code, os.strerror(code))
                if count == 0:
                    raise TimeoutError("EVFILT_TIMER wait expired without an event")
                if result.flags & EV_ERROR:
                    raise OSError(result.data, os.strerror(result.data))
                if result.ident != 1 or result.filter != EVFILT_TIMER:
                    raise RuntimeError("Unexpected event from dedicated timer kqueue")
                woke = time.perf_counter()
                if woke < deadline:
                    raise RuntimeError("EVFILT_TIMER returned before its absolute deadline")
                return woke

    def close(self):
        """Close after the worker exits; serialize with any outstanding wait."""
        with self._lock:
            if self.fd >= 0:
                fd, self.fd = self.fd, -1
                os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AsyncCriticalTimer:
    """Wake the current asyncio loop through a readable timer kqueue FD.

    Apple documents monitoring a kqueue descriptor for read activity in its
    CFFileDescriptor reference. asyncio.add_reader supplies that monitoring;
    kevent64 drains the event (ordinary os.read is not the interface).

    Construct and close on the same running loop. Only one wait may be active.
    No worker, priority change, spinning, or periodic polling is used. The
    asyncio timer exists only as a one-second lost-event watchdog.
    """

    def __init__(self, *, critical=True):
        self.loop = asyncio.get_running_loop()
        self._owner = threading.get_ident()
        self._timer = CriticalTimer(critical=critical)
        self._pending = None
        self._active_ident = None
        self._sequence = 0
        self._closed = False
        try:
            self.loop.add_reader(self._timer.fd, self._ready)
        except BaseException:
            self._timer.close()
            raise

    def _check_owner(self):
        if threading.get_ident() != self._owner or asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("AsyncCriticalTimer must be used on its owning event loop")

    def _change(self, event, *, ignore_missing=False):
        zero = _Timespec(0, 0)
        while True:
            count = self._timer.lib.kevent64(self._timer.fd, ctypes.byref(event), 1,
                                             None, 0, 1, ctypes.byref(zero))
            if count >= 0:
                return
            code = ctypes.get_errno()
            if code == errno.EINTR:
                continue
            if ignore_missing and code == errno.ENOENT:
                return
            raise OSError(code, os.strerror(code))

    def _ready(self):
        if self._closed:
            return
        zero = _Timespec(0, 0)
        events = (_Kevent64 * 8)()
        try:
            count = self._timer.lib.kevent64(self._timer.fd, None, 0,
                                             events, len(events), 1, ctypes.byref(zero))
            observed = time.perf_counter()
            if count < 0:
                code = ctypes.get_errno()
                if code == errno.EINTR:
                    return  # Descriptor remains readable; let the loop retry.
                raise OSError(code, os.strerror(code))
            for event in events[:count]:
                # A cancelled generation cannot resolve the next wait.
                if event.ident != self._active_ident:
                    continue
                if event.flags & EV_ERROR:
                    raise OSError(event.data, os.strerror(event.data))
                if event.filter != EVFILT_TIMER:
                    raise RuntimeError("Unexpected event from dedicated async timer")
                if self._pending is not None and not self._pending.done():
                    self._pending.set_result(observed)
        except Exception as exc:
            if self._pending is not None and not self._pending.done():
                self._pending.set_exception(exc)

    async def wait_until(self, deadline):
        """Return the event-observation timestamp after an absolute deadline."""
        self._check_owner()
        if self._closed:
            raise RuntimeError("AsyncCriticalTimer is closed")
        if self._pending is not None:
            raise RuntimeError("Concurrent waits on one AsyncCriticalTimer are unsupported")
        ticks = self._timer.clock.ticks(deadline)
        if ticks >= 2**63:
            raise OverflowError("EVFILT_TIMER deadline exceeds signed int64")
        if time.perf_counter() >= deadline:
            return time.perf_counter()
        self._sequence += 1
        ident = self._sequence
        future = self.loop.create_future()
        self._pending, self._active_ident = future, ident
        flags = NOTE_ABSOLUTE | NOTE_MACHTIME | NOTE_LEEWAY
        if self._timer.critical:
            flags |= NOTE_CRITICAL
        event = _Kevent64(ident=ident, filter=EVFILT_TIMER,
                          flags=EV_ADD | EV_ONESHOT, fflags=flags, data=ticks)
        try:
            self._change(event)
            observed = await asyncio.wait_for(future, max(0, deadline - time.perf_counter()) + 1.0)
            if observed < deadline:
                raise RuntimeError("Async EVFILT_TIMER returned before its absolute deadline")
            return observed
        finally:
            if not future.done():
                future.cancel()
            self._pending, self._active_ident = None, None
            if not self._closed:
                self._change(_Kevent64(ident=ident, filter=EVFILT_TIMER, flags=EV_DELETE),
                             ignore_missing=True)

    def close(self):
        self._check_owner()
        if self._closed:
            return
        self._closed = True
        try:
            self.loop.remove_reader(self._timer.fd)
        finally:
            if self._pending is not None and not self._pending.done():
                self._pending.cancel()
            # Closing removes every filter and disarms an outstanding timer.
            self._timer.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.close()
