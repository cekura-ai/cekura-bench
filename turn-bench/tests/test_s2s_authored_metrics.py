"""Offline cases on the real service AudioTimeline; no provider calls."""
from types import SimpleNamespace
import numpy as np
import pytest
from turn_bench.author import service_bridge
from turn_bench.s2s.authored_metrics import render_markdown, score_event, summarize
RATE = 16000


def pcm(seconds, loud=True):
    t = np.arange(round(seconds * RATE)) / RATE
    values = 12000 * np.sin(2 * np.pi * 220 * t) if loud else np.zeros(len(t))
    return values.astype(np.int16).tobytes()


def adapter(audio, at=0.0, cut=None):
    timeline = service_bridge.audio().AudioTimeline(RATE)
    timeline.record_pcm(audio, at)
    if cut is not None:
        timeline.cut(cut)
    return SimpleNamespace(agent_pcm=bytearray(audio), agent_timeline=timeline)


def event(kind="barge_in", start=0.5, observed=1.7, conversation="a"):
    return {"conversation_id": conversation, "spec_id": "spec", "event_id": "e",
            "kind": kind, "caller_start_s": start, "caller_end_s": start + 0.3,
            "target_s": start, "anchor_s": 0.0, "void": None,
            "observation_end_s": observed}


@pytest.mark.parametrize("kind,cut,verdict,latency", [
    ("barge_in", 0.7, "pass", 200.0), ("barge_in", None, "fail", None),
    ("backchannel", 0.7, "fail", 200.0), ("backchannel", None, "pass", None),
])
def test_actual_playout_cut_and_buffered_reply(kind, cut, verdict, latency):
    # Entire reply arrived at zero, but is audible for three seconds.
    result = score_event(event(kind), adapter(pcm(3), cut=cut))
    assert result["void"] is None
    assert result["verdict"] == verdict
    assert result["stop_ms"] == latency
    assert result["measurement"]["playback_cut_before_stop"] == (cut is not None)


def test_pcm_quiet_requires_200_ms_and_measures_first_quiet_frame():
    audio = pcm(0.7) + pcm(0.1, False) + pcm(0.2) + pcm(0.3, False)
    result = score_event(event(), adapter(audio))
    assert result["verdict"] == "pass"
    assert result["stop_ms"] == 500.0
    assert result["measurement"]["playback_cut_before_stop"] is False
    assert result["measurement"]["causal_yield_verified"] is False


def test_queued_audio_discarded_at_cut_is_not_measured_as_speech():
    a = adapter(pcm(2))
    second = pcm(2)
    a.agent_timeline.record_pcm(second, 0.1)
    a.agent_pcm.extend(second)
    a.agent_timeline.cut(0.7)
    result = score_event(event(), a)
    assert result["stop_ms"] == 200.0
    assert result["verdict"] == "pass"


@pytest.mark.parametrize("a,record,reason", [
    (lambda: adapter(pcm(0.2)), event(), "no_agent_speech_overlap"),
    (lambda: adapter(pcm(3)), event(observed=0.8), "insufficient_stop_observation"),
    (lambda: adapter(pcm(3), cut=0.7), event(observed=0.8), "insufficient_stop_observation"),
    (lambda: adapter(pcm(3)), {**event(), "observation_end_s": None}, "missing_or_invalid_actual_timing"),
    (lambda: adapter(pcm(3)), {**event(), "void": "end_relative_timing"}, "end_relative_timing"),
])
def test_exclusions_never_infer_a_stop_from_missing_evidence(a, record, reason):
    result = score_event(record, a())
    assert result["void"] == reason
    assert result["verdict"] is None
    assert result["stop_ms"] is None


def test_success_window_boundary_and_early_stop_still_requires_full_observation():
    assert score_event(event(), adapter(pcm(3), cut=1.5))["verdict"] == "pass"
    assert score_event(event(), adapter(pcm(3), cut=1.51))["verdict"] == "fail"
    result = score_event(event(observed=0.9), adapter(pcm(3), cut=0.7))
    assert result["void"] == "insufficient_stop_observation"


def test_summary_pools_counts_resamples_conversations_and_preserves_voids():
    passing = score_event(event(), adapter(pcm(3), cut=0.7))
    failing = score_event(event(conversation="b"), adapter(pcm(3)))
    void = score_event({**event(conversation="c"), "void": "pacing_slip"}, adapter(pcm(3)))
    records = [passing, {**passing, "event_id": "e2"}, failing, void]
    summary = summarize(records)
    assert summary == summarize(records)
    barge = summary["interruption_handling"]
    assert barge["value"] == pytest.approx(2 / 3)
    assert barge["ci95"] == [0.0, 1.0]
    assert barge["draws_defined"] == 2000
    assert barge["n"] == 3 and barge["cells"] == 4 and barge["voids"] == 1
    assert summary["yield_ms"]["p50"] == 200.0
    assert summary["voids"] == {"pacing_slip": 1}
    assert summary["backchannel_handling"]["value"] is None
    assert summarize([passing])["interruption_handling"]["ci95"] is None
    assert "pacing_slip: 1" in render_markdown(summary)
    assert "natural reply ends" in render_markdown(summary)


def test_summary_rejects_unscored_or_unclustered_events():
    with pytest.raises(ValueError, match="explicit void"):
        summarize([event()])
    with pytest.raises(ValueError, match="conversation_id"):
        summarize([{**event(), "conversation_id": None}])


def test_next_caller_within_observation_window_excluded():
    result = score_event({**event(), "next_caller_start_s": 1.6}, adapter(pcm(3), cut=0.7))
    assert result["void"] == "next_caller_contaminates_observation"
