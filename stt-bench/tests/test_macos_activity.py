"""Activity ownership and scope checks; no timing benchmark or sleeps."""
import sys

import pytest

from stt_bench.macos_activity import ACTIVITY_OPTIONS, LatencyActivity, activity_identity


class FakeObjectiveC:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []
        self.retains = {"process": 0, "activity": 0}
        self.pools = 0
        self.active = False

    def pool(self):
        self.pools += 1
        return object()

    def drain(self, pool):
        self.pools -= 1

    def process_info(self):
        return "process"

    def reason(self, text):
        return text

    def retain(self, value):
        if self.fail == "retain_activity" and value == "activity":
            raise RuntimeError("retain failed")
        self.retains[value] += 1

    def release(self, value):
        self.retains[value] -= 1
        self.calls.append(("release", value))

    def begin(self, process, options, reason):
        assert process == "process" and options == ACTIVITY_OPTIONS and reason
        if self.fail == "begin":
            raise RuntimeError("begin failed")
        self.active = True
        self.calls.append(("begin", "activity"))
        return "activity"

    def end(self, process, activity):
        assert process == "process" and activity == "activity"
        self.active = False
        self.calls.append(("end", activity))
        if self.fail == "end":
            raise RuntimeError("end failed")


def test_activity_balances_tokens_and_short_autorelease_pools():
    api = FakeObjectiveC()
    activity = LatencyActivity(_api=api)
    with activity:
        assert api.active and api.pools == 0
        assert api.retains == {"process": 1, "activity": 1}
    activity.close()
    assert not api.active and api.pools == 0
    assert api.retains == {"process": 0, "activity": 0}
    assert api.calls.count(("end", "activity")) == 1
    with pytest.raises(RuntimeError, match="only once"):
        activity.__enter__()


@pytest.mark.parametrize("failure", ["begin", "retain_activity"])
def test_failed_enter_ends_any_started_activity_and_releases_owned_objects(failure):
    api = FakeObjectiveC(fail=failure)
    activity = LatencyActivity(_api=api)
    with pytest.raises(RuntimeError, match="failed"):
        activity.__enter__()
    assert not api.active and api.pools == 0
    assert api.retains == {"process": 0, "activity": 0}
    assert activity._state == "closed"


def test_close_error_still_releases_owned_objects():
    api = FakeObjectiveC(fail="end")
    with pytest.raises(RuntimeError, match="end failed"):
        with LatencyActivity(_api=api):
            pass
    assert api.retains == {"process": 0, "activity": 0} and api.pools == 0


def test_body_exception_is_preserved_when_cleanup_also_fails():
    api = FakeObjectiveC(fail="end")
    with pytest.raises(ValueError, match="body failed") as error:
        with LatencyActivity(_api=api):
            raise ValueError("body failed")
    assert any("cleanup failed" in note for note in error.value.__notes__)
    assert api.retains == {"process": 0, "activity": 0} and api.pools == 0


def test_failed_pool_drain_after_begin_does_not_leave_activity_active():
    api = FakeObjectiveC()
    drain = api.drain
    def fail_once(pool):
        drain(pool)
        api.drain = drain
        raise RuntimeError("drain interrupted")
    api.drain = fail_once
    with pytest.raises(RuntimeError, match="drain interrupted"):
        LatencyActivity(_api=api).__enter__()
    assert not api.active and api.pools == 0
    assert api.retains == {"process": 0, "activity": 0}


@pytest.mark.parametrize("reason", ["", " ", "bad\0reason", None])
def test_reason_is_validated_before_native_calls(reason):
    with pytest.raises(ValueError, match="reason"):
        LatencyActivity(reason, _api=FakeObjectiveC())


def test_options_do_not_disable_idle_system_or_display_sleep():
    assert ACTIVITY_OPTIONS == 0xFF00EFFFFF
    assert not ACTIVITY_OPTIONS & (1 << 20)
    assert not ACTIVITY_OPTIONS & (1 << 40)
    assert activity_identity()["allows_idle_system_sleep"]


@pytest.mark.skipif(sys.platform != "darwin", reason="Public Foundation activity API")
def test_actual_foundation_begin_and_end_smoke():
    with LatencyActivity("STT test activity lifecycle") as activity:
        assert activity._activity and activity._process and activity._state == "active"
    assert activity._activity is None and activity._process is None
    assert activity._state == "closed"
