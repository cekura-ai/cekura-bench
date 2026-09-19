"""Offline timing-evidence checks; these never sleep or open a socket."""
import json

import pytest

from stt_bench.data import FRAME_BYTES
from stt_bench.streaming import pacing_metrics


def frames(*, modern=True):
    events = []
    for index in range(53):
        at = (index + 1) * .02
        event = dict(kind="audio_sent", index=index, time_seconds=at,
                     ideal_seconds=at, phase="speech" if index < 3 else "silence",
                     bytes=FRAME_BYTES)
        if modern:
            event.update(send_completed_seconds=at + .0001, scheduled_seconds=at,
                         send_duration_ms=.1, wakeup_delay_ms=0.)
        events.append(event)
    return events


def rejected(events, reason):
    metrics = pacing_metrics(events)
    assert not metrics["valid"]
    assert reason in metrics["gate_reasons"]
    # Even invalid evidence must remain safe to save as strict JSON.
    json.dumps(metrics, allow_nan=False)
    return metrics


@pytest.mark.parametrize("modern", [False, True])
def test_valid_evidence_and_explicit_legacy_completion_status(modern):
    result = pacing_metrics(frames(modern=modern))
    assert result["valid"]
    assert result["send_timing_evidence"] == ("complete" if modern else "legacy_unavailable")


@pytest.mark.parametrize("indexes", [
    [0, 0, 2],  # Duplicate.
    [0, 2, 1],  # Out of order.
    [0, 1, 3],  # Missing index.
    [1, 2, 3],  # Missing first frame.
    [False, 1, 2],  # Booleans must not masquerade as frame indexes.
])
def test_frame_indexes_must_start_at_zero_and_be_contiguous(indexes):
    events = frames()
    for event, index in zip(events, indexes):
        event["index"] = index
    rejected(events, "invalid_frame_indexes")


def test_silence_cannot_precede_speech_even_with_correct_frame_counts():
    events = frames()
    events[0]["phase"], events[-1]["phase"] = "silence", "speech"
    result = rejected(events, "invalid_frame_phase_order")
    assert result["silence_frames"] == 50


def test_long_final_frame_send_is_rejected_without_a_following_gap():
    events = frames()
    events[-1]["send_completed_seconds"] = events[-1]["time_seconds"] + .08
    events[-1]["send_duration_ms"] = 80.
    result = rejected(events, "send_duration_above_40ms")
    assert result["interval_ms_max"] < 21


@pytest.mark.parametrize("field", ["time_seconds", "ideal_seconds", "scheduled_seconds",
                                   "send_completed_seconds", "send_duration_ms", "wakeup_delay_ms"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_timing_never_passes_or_leaks_to_output(field, bad):
    events = frames()
    events[10][field] = bad
    rejected(events, "invalid_timing_fields")


def test_send_completion_cannot_precede_send_start():
    events = frames()
    events[-1]["send_completed_seconds"] = events[-1]["time_seconds"] - .001
    events[-1]["send_duration_ms"] = -1.
    rejected(events, "invalid_timestamp_order")


def test_send_completion_cannot_follow_next_send_start():
    events = frames()
    events[10]["send_completed_seconds"] = events[11]["time_seconds"] + .001
    events[10]["send_duration_ms"] = 21.
    rejected(events, "invalid_timestamp_order")


def test_modern_completion_evidence_cannot_be_partially_missing():
    events = frames()
    del events[10]["send_completed_seconds"]
    result = rejected(events, "incomplete_send_timing")
    assert result["send_timing_evidence"] == "incomplete"


def test_reported_send_duration_must_match_completion_timestamp():
    events = frames()
    events[10]["send_duration_ms"] = 5.
    rejected(events, "inconsistent_send_duration")


def test_ideal_clock_must_be_monotonic_even_when_send_gaps_are_regular():
    events = frames()
    events[10]["ideal_seconds"] = events[9]["ideal_seconds"]
    rejected(events, "invalid_timestamp_order")


def test_missing_required_timestamp_returns_invalid_evidence():
    events = frames()
    del events[10]["time_seconds"]
    rejected(events, "invalid_timing_fields")
