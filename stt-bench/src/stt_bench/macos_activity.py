"""Scoped public macOS process hint for a user-requested precise streaming run.

Foundation's NSProcessInfo activity API can suppress process-level deferral.
It is a hint, not a real-time guarantee, and user preferences can override it.
The selected options allow idle system sleep and do not request display wake,
set thread priorities, or modify persistent preferences.
"""

import ctypes
import sys


ACTIVITY_BACKEND = "macos-nsprocessinfo-latency-critical-v1"
# Public Foundation/NSProcessInfo.h: UserInitiated is 0x00FFFFFF; remove its
# IdleSystemSleepDisabled bit, then add LatencyCritical (bits 32 through 39).
ACTIVITY_OPTIONS = (0x00FFFFFF & ~(1 << 20)) | 0xFF00000000


def activity_identity():
    return dict(backend=ACTIVITY_BACKEND, options=ACTIVITY_OPTIONS,
                allows_idle_system_sleep=True)


class _ObjectiveC:
    """Typed objc_msgSend calls; each signature matches its Foundation method."""

    def __init__(self):
        self.foundation = ctypes.CDLL(
            "/System/Library/Frameworks/Foundation.framework/Foundation")
        self.objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        self.pointer = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", self.objc))
        self.string = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_char_p)(("objc_msgSend", self.objc))
        self.begin_message = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_uint64, ctypes.c_void_p)(
            ("objc_msgSend", self.objc))
        self.void = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", self.objc))
        self.void_argument = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p)(("objc_msgSend", self.objc))
        self.classes = {name: self.objc.objc_getClass(name.encode())
                        for name in ("NSAutoreleasePool", "NSProcessInfo", "NSString")}
        self.selectors = {name: self.objc.sel_registerName(name.encode()) for name in
                          ("alloc", "init", "drain", "processInfo", "stringWithUTF8String:",
                           "retain", "release", "beginActivityWithOptions:reason:", "endActivity:")}
        if not all(self.classes.values()) or not all(self.selectors.values()):
            raise RuntimeError("Required Foundation activity classes or selectors are unavailable")

    def pool(self):
        allocated = self.pointer(self.classes["NSAutoreleasePool"], self.selectors["alloc"])
        value = self.pointer(allocated, self.selectors["init"])
        if not value:
            raise RuntimeError("Cannot create Foundation autorelease pool")
        return value

    def drain(self, pool):
        self.void(pool, self.selectors["drain"])

    def process_info(self):
        value = self.pointer(self.classes["NSProcessInfo"], self.selectors["processInfo"])
        if not value:
            raise RuntimeError("NSProcessInfo returned nil")
        return value

    def reason(self, text):
        value = self.string(self.classes["NSString"], self.selectors["stringWithUTF8String:"],
                            text.encode("utf-8"))
        if not value:
            raise RuntimeError("Cannot create activity reason string")
        return value

    def retain(self, value):
        if not self.pointer(value, self.selectors["retain"]):
            raise RuntimeError("Cannot retain Foundation activity object")

    def release(self, value):
        self.void(value, self.selectors["release"])

    def begin(self, process, options, reason):
        value = self.begin_message(process, self.selectors["beginActivityWithOptions:reason:"],
                                   options, reason)
        if not value:
            raise RuntimeError("NSProcessInfo did not return an activity token")
        return value

    def end(self, process, activity):
        self.void_argument(process, self.selectors["endActivity:"], activity)


class LatencyActivity:
    """Begin/end one process activity, retaining its token across async work.

    Use ``with LatencyActivity(): ...`` around the complete operation. Temporary
    autoreleased objects are drained during begin/end; no autorelease pool is
    held across asynchronous suspension. This object owns one explicit retain
    on the process object and activity token until close, which is idempotent.
    """

    def __init__(self, reason="STT benchmark streaming timing", *, _api=None):
        if not isinstance(reason, str) or not reason.strip() or "\0" in reason:
            raise ValueError("Activity reason must be a nonempty string without NUL")
        if _api is None and sys.platform != "darwin":
            raise RuntimeError("LatencyActivity requires macOS")
        self._api = _api if _api is not None else _ObjectiveC()
        self.reason = reason
        self._process = self._activity = None
        self._process_retained = self._activity_retained = False
        self._state = "new"

    def _finish_objects(self):
        first_error = None
        actions = []
        if self._activity is not None:
            actions.append(lambda: self._api.end(self._process, self._activity))
        if self._activity_retained:
            actions.append(lambda: self._api.release(self._activity))
        if self._process_retained:
            actions.append(lambda: self._api.release(self._process))
        try:
            for action in actions:
                try:
                    action()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        finally:
            self._process = self._activity = None
            self._process_retained = self._activity_retained = False
        if first_error is not None:
            raise first_error

    def __enter__(self):
        if self._state != "new":
            raise RuntimeError("A LatencyActivity instance can be entered only once")
        self._state = "starting"
        pool = None
        try:
            pool = self._api.pool()
            self._process = self._api.process_info()
            self._api.retain(self._process)
            self._process_retained = True
            reason = self._api.reason(self.reason)
            self._activity = self._api.begin(self._process, ACTIVITY_OPTIONS, reason)
            self._api.retain(self._activity)
            self._activity_retained = True
            self._state = "active"
        except BaseException as error:
            self._state = "closed"
            try:
                self._finish_objects()
            except BaseException as cleanup_error:
                error.add_note(f"Activity cleanup also failed: {cleanup_error}")
            raise
        finally:
            if pool is not None:
                try:
                    self._api.drain(pool)
                except BaseException as error:
                    # __enter__ failed: Python will not call __exit__ for us.
                    try:
                        self.close()
                    except BaseException as cleanup_error:
                        error.add_note(f"Activity cleanup also failed: {cleanup_error}")
                    raise
        return self

    def close(self):
        if self._state in ("new", "closed"):
            self._state = "closed"
            return
        self._state = "closed"
        # Even if pool creation fails, owned objects still need end/release.
        pool = None
        try:
            pool = self._api.pool()
        finally:
            try:
                self._finish_objects()
            finally:
                if pool is not None:
                    self._api.drain(pool)

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"Activity cleanup failed: {error}")
        return False
