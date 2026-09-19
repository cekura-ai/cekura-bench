"""A cancellable macOS deadline timer integrated with asyncio descriptor readiness.

The public ``kevent64`` API exposes EVFILT_TIMER and NOTE_CRITICAL. The latter
requests minimal timer coalescing for this timer; it does not change thread
priority or provide hard real-time scheduling. NOTE_ABSOLUTE | NOTE_MACHTIME
uses the Mach absolute clock, with its runtime tick-to-nanosecond ratio.

A dedicated kqueue descriptor becomes readable when its timer fires. asyncio
monitors that descriptor, so no worker thread, polling loop, or final spin is
needed. The one-second asyncio timeout is a lost-event watchdog only.

Public references:
https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/event.h
https://developer.apple.com/documentation/corefoundation/cffiledescriptor
"""

import asyncio
import ctypes
import errno
import math
import os
import sys
import threading
import time


TIMER_BACKEND = "macos-kqueue-critical-v1"
_EVFILT_TIMER = -7
_EV_ADD = 0x0001
_EV_DELETE = 0x0002
_EV_ONESHOT = 0x0010
_EV_ERROR = 0x4000
_KEVENT_FLAG_IMMEDIATE = 0x000001
_NOTE_ABSOLUTE = 0x00000008
_NOTE_LEEWAY = 0x00000010
_NOTE_CRITICAL = 0x00000020
_NOTE_MACHTIME = 0x00000100


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


class _Kevent64(ctypes.Structure):
    _fields_ = [
        ("ident", ctypes.c_uint64), ("filter", ctypes.c_int16),
        ("flags", ctypes.c_uint16), ("fflags", ctypes.c_uint32),
        ("data", ctypes.c_int64), ("udata", ctypes.c_uint64),
        ("ext", ctypes.c_uint64 * 2),
    ]


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class MacOSDeadlineTimer:
    """One reusable timer owned by one running event loop, with explicit close.

    Use ``async with MacOSDeadlineTimer() as timer`` and await
    ``timer.wait_until(deadline)`` with an absolute ``time.perf_counter()``
    timestamp. Concurrent waits on the same instance are rejected.
    """

    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("MacOSDeadlineTimer requires macOS")
        if time.get_clock_info("perf_counter").implementation != "mach_absolute_time()":
            raise RuntimeError("perf_counter is not the expected Mach absolute clock")
        if (ctypes.sizeof(_Kevent64) != 48 or _Kevent64.data.offset != 16
                or _Kevent64.ext.offset != 32 or ctypes.sizeof(_Timespec) != 16):
            raise RuntimeError("Unexpected macOS timer ABI layout")
        self._loop = asyncio.get_running_loop()
        self._owner = threading.get_ident()
        self._lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self._lib.mach_timebase_info.argtypes = [ctypes.POINTER(_Timebase)]
        self._lib.mach_timebase_info.restype = ctypes.c_int
        self._lib.kqueue.argtypes = []
        self._lib.kqueue.restype = ctypes.c_int
        self._lib.kevent64.argtypes = [
            ctypes.c_int, ctypes.POINTER(_Kevent64), ctypes.c_int,
            ctypes.POINTER(_Kevent64), ctypes.c_int, ctypes.c_uint,
            ctypes.POINTER(_Timespec),
        ]
        self._lib.kevent64.restype = ctypes.c_int
        info = _Timebase()
        rc = self._lib.mach_timebase_info(ctypes.byref(info))
        if rc != 0 or not info.numer or not info.denom:
            raise RuntimeError(f"mach_timebase_info failed: {rc}")
        self._numer, self._denom = info.numer, info.denom
        self._pending = None
        self._active_ident = None
        self._sequence = 0
        self._closed = False
        self._fd = self._lib.kqueue()
        if self._fd < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        try:
            os.set_inheritable(self._fd, False)
            self._loop.add_reader(self._fd, self._ready)
        except BaseException:
            os.close(self._fd)
            self._fd = -1
            self._closed = True
            raise

    def _check_owner(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if threading.get_ident() != self._owner or loop is not self._loop:
            raise RuntimeError("MacOSDeadlineTimer must be used on its owning event loop")

    def _ticks(self, deadline):
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline < 0:
            raise ValueError("Deadline must be finite and nonnegative")
        # Exact upward rounding avoids treating Mach ticks as nanoseconds and
        # does not add a relative delay when event registration itself is late.
        top, bottom = float(deadline).as_integer_ratio()
        numerator = top * 1_000_000_000 * self._denom
        denominator = bottom * self._numer
        ticks = (numerator + denominator - 1) // denominator
        if ticks >= 2**63:  # kevent64.data is signed int64.
            raise OverflowError("EVFILT_TIMER deadline exceeds signed int64")
        return ticks

    def _change(self, event, *, ignore_missing=False):
        zero = _Timespec(0, 0)
        while True:
            count = self._lib.kevent64(self._fd, ctypes.byref(event), 1, None, 0,
                                       _KEVENT_FLAG_IMMEDIATE, ctypes.byref(zero))
            if count >= 0:
                return
            code = ctypes.get_errno()
            if code == errno.EINTR:
                continue
            if ignore_missing and code == errno.ENOENT:
                return  # A delivered one-shot event is already removed.
            raise OSError(code, os.strerror(code))

    def _ready(self):
        if self._closed:
            return
        zero = _Timespec(0, 0)
        events = (_Kevent64 * 8)()
        try:
            count = self._lib.kevent64(self._fd, None, 0, events, len(events),
                                       _KEVENT_FLAG_IMMEDIATE, ctypes.byref(zero))
            observed = time.perf_counter()
            if count < 0:
                code = ctypes.get_errno()
                if code == errno.EINTR:
                    return  # Descriptor remains readable for the next check.
                raise OSError(code, os.strerror(code))
            for event in events[:count]:
                if event.ident != self._active_ident:
                    continue  # A cancelled generation cannot finish a new wait.
                if event.flags & _EV_ERROR:
                    raise OSError(event.data, os.strerror(event.data))
                if event.filter != _EVFILT_TIMER:
                    raise RuntimeError("Unexpected event from dedicated deadline timer")
                if self._pending is not None and not self._pending.done():
                    self._pending.set_result(observed)
        except Exception as exc:
            if self._pending is not None and not self._pending.done():
                self._pending.set_exception(exc)

    async def wait_until(self, deadline):
        """Return the event-observation timestamp; resumption may happen later."""
        self._check_owner()
        if self._closed:
            raise RuntimeError("MacOSDeadlineTimer is closed")
        if self._pending is not None:
            raise RuntimeError("Concurrent waits on one MacOSDeadlineTimer are unsupported")
        ticks = self._ticks(deadline)
        if time.perf_counter() >= deadline:
            return time.perf_counter()
        self._sequence += 1
        ident = self._sequence
        future = self._loop.create_future()
        self._pending, self._active_ident = future, ident
        flags = _NOTE_ABSOLUTE | _NOTE_MACHTIME | _NOTE_LEEWAY | _NOTE_CRITICAL
        event = _Kevent64(ident=ident, filter=_EVFILT_TIMER,
                          flags=_EV_ADD | _EV_ONESHOT, fflags=flags, data=ticks)
        try:
            self._change(event)
            observed = await asyncio.wait_for(future, max(0, deadline - time.perf_counter()) + 1.0)
            if observed < deadline:
                raise RuntimeError("EVFILT_TIMER returned before its absolute deadline")
            return observed
        finally:
            if not future.done():
                future.cancel()
            self._pending, self._active_ident = None, None
            if not self._closed:
                self._change(_Kevent64(ident=ident, filter=_EVFILT_TIMER, flags=_EV_DELETE),
                             ignore_missing=True)

    def close(self):
        """Remove the reader, cancel any waiter, disarm the timer, and close FD."""
        self._check_owner()
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.remove_reader(self._fd)
        finally:
            if self._pending is not None and not self._pending.done():
                self._pending.cancel()
            fd, self._fd = self._fd, -1
            os.close(fd)  # Closing the queue removes all its filters.

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.close()
