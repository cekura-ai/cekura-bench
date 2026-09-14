"""Experimental macOS absolute-deadline wait for scheduler comparisons.

Call ``MachDeadline().wait_until(perf_counter_deadline)`` in a timer worker,
never the asyncio thread. This wait changes neither thread priority nor timer
policy. Optional QoS helpers match only a dedicated worker to its caller for
controlled comparisons. These are hypotheses to measure, not real-time guarantees.

API: apple-oss-distributions/xnu, osfmk/mach/mach_time.h.
Clock conversion: CPython v3.12.8 Python/pytime.c uses mach_absolute_time and
mach_timebase_info for perf_counter on macOS; verify that at runtime.
"""

import ctypes
import math
import os
import sys
import threading
import time


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_QOS_NAMES = {
    0: "UNSPECIFIED", 0x09: "BACKGROUND", 0x11: "UTILITY",
    0x15: "DEFAULT", 0x19: "USER_INITIATED", 0x21: "USER_INTERACTIVE",
}


def _qos_library():
    if sys.platform != "darwin":
        raise RuntimeError("Thread QoS inspection is available only on macOS")
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib.pthread_self.argtypes = []
    lib.pthread_self.restype = ctypes.c_void_p
    lib.pthread_get_qos_class_np.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int),
    ]
    lib.pthread_get_qos_class_np.restype = ctypes.c_int
    lib.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
    lib.pthread_set_qos_class_self_np.restype = ctypes.c_int
    return lib


def capture_qos():
    """Read current thread's requested QoS; runtime overrides are not reported.

    Call on the event-loop thread before creating its dedicated executor.
    API: apple-oss-distributions/libpthread include/pthread/qos.h.
    """
    lib = _qos_library()
    qos, relative = ctypes.c_uint(), ctypes.c_int()
    rc = lib.pthread_get_qos_class_np(lib.pthread_self(), ctypes.byref(qos), ctypes.byref(relative))
    if rc:
        raise OSError(rc, os.strerror(rc))
    return dict(requested_class=qos.value, class_name=_QOS_NAMES.get(qos.value, "UNKNOWN"),
                relative_priority=relative.value, native_thread_id=threading.get_native_id())


def match_worker_qos(source, evidence=None):
    """Executor initializer matching this worker to a captured caller's QoS.

    Example: ThreadPoolExecutor(max_workers=1, initializer=match_worker_qos,
    initargs=(capture_qos(), evidence_list)). Warm the executor before timing.
    This affects only the current dedicated worker. No process-wide priority,
    real-time policy, or main-thread settings are modified. Terminate that
    executor after the trial; do not reuse it for unrelated background work.

    Return before/after evidence and optionally append it to a shared list,
    because ThreadPoolExecutor does not expose initializer return values.
    """
    if threading.current_thread() is threading.main_thread():
        raise RuntimeError("QoS matching must run in a dedicated worker")
    if threading.get_native_id() == source.get("native_thread_id"):
        raise RuntimeError("Refusing to modify the captured source thread")
    requested, relative = source["requested_class"], source["relative_priority"]
    if type(requested) is not int or requested not in _QOS_NAMES or requested == 0:
        raise ValueError("A supported, specified source QoS class is required")
    if type(relative) is not int or not -15 <= relative <= 0:
        raise ValueError("QoS relative priority must be an integer from -15 to 0")
    before = capture_qos()
    rc = _qos_library().pthread_set_qos_class_self_np(requested, relative)
    if rc:
        raise OSError(rc, os.strerror(rc))
    after = capture_qos()
    if (after["requested_class"], after["relative_priority"]) != (requested, relative):
        raise RuntimeError("Worker requested QoS did not match its captured source")
    result = dict(source=dict(source), before=before, after=after,
                  interpretation="Requested thread QoS only; effective runtime overrides are not exposed by this API.")
    if evidence is not None:
        evidence.append(result)
    return result


class MachDeadline:
    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("MachDeadline is available only on macOS")
        if time.get_clock_info("perf_counter").implementation != "mach_absolute_time()":
            raise RuntimeError("perf_counter is not the expected Mach absolute clock")
        # CDLL releases Python's interpreter lock during native calls.
        self.lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        self.lib.mach_timebase_info.argtypes = [ctypes.POINTER(_Timebase)]
        self.lib.mach_timebase_info.restype = ctypes.c_int
        self.lib.mach_wait_until.argtypes = [ctypes.c_uint64]
        self.lib.mach_wait_until.restype = ctypes.c_int
        self.lib.mach_absolute_time.argtypes = []
        self.lib.mach_absolute_time.restype = ctypes.c_uint64
        info = _Timebase()
        rc = self.lib.mach_timebase_info(ctypes.byref(info))
        if rc != 0 or not info.numer or not info.denom:
            raise RuntimeError(f"mach_timebase_info failed: {rc}")
        self.numer, self.denom = info.numer, info.denom

    def ticks(self, deadline):
        """Convert a perf_counter absolute timestamp to ticks, rounding upward."""
        if not math.isfinite(deadline) or deadline < 0:
            raise ValueError("Deadline must be finite and nonnegative")
        top, bottom = float(deadline).as_integer_ratio()
        numerator = top * 1_000_000_000 * self.denom
        denominator = bottom * self.numer
        ticks = (numerator + denominator - 1) // denominator
        if ticks >= 2**64:
            raise OverflowError("Mach deadline exceeds uint64")
        return ticks

    def wait_until(self, deadline):
        """Block this worker until the absolute deadline; return actual wake time.

        The caller owns worker lifetime. Waiting is uninterruptible by asyncio
        cancellation; keep submitted deadlines near (one audio frame) and join
        the worker before closing its event loop. KERN_ABORTED (14) is retried
        against the original deadline, not a fresh relative interval.
        """
        ticks = self.ticks(deadline)
        while self.lib.mach_absolute_time() < ticks:
            rc = self.lib.mach_wait_until(ticks)
            if rc not in (0, 14):
                raise RuntimeError(f"mach_wait_until failed: {rc}")
        return time.perf_counter()
