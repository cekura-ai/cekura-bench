"""Track 2 cells -> scorecard rows, and the combined scorecard's rendering."""

import json

import pytest

from turn_bench.s2s.interaction import interaction_summary
from turn_bench.s2s.scorecard import combined_scorecard, render_markdown, write_scorecard


def cell(probe, variant, repeat, verdict=None, void=None, values=None):
    return {
        "provider": "fake", "model": "scripted", "config": "server_vad", "voice": "f-us", "transform": "clean",
        "repeat": repeat, "probe": probe, "variant": variant, "verdict": verdict, "void": void,
        "values": values or {}, "artifacts": {"dir": "x", "slug": f"{variant}/server_vad/f-us/clean/r{repeat}"},
        "started_utc": "", "duration_s": 1.0, "usage": {"input_tokens": 10}, "error": None,
    }


@pytest.fixture
def run_dir(tmp_path):
    cells = [
        cell("barge_in", "barge_in-at400ms", 1, "pass", values={"stop_ms": 410.0, "stopped": True}),
        cell("barge_in", "barge_in-at400ms", 2, "fail", values={"stop_ms": None, "stopped": False}),
        cell("barge_in", "barge_in-at900ms", 1, "pass", values={"stop_ms": 380.0, "stopped": True}),
        cell("barge_in", "barge_in-at900ms", 2, void="agent finished before the barge-in point"),
        cell("backchannel_tolerance", "backchannel_tolerance-at400ms", 1, "pass", values={"kept_talking": True}),
        cell("backchannel_tolerance", "backchannel_tolerance-at400ms", 2, "fail", values={"kept_talking": False}),
    ]
    (tmp_path / "cells.jsonl").write_text("\n".join(json.dumps(c) for c in cells) + "\n")
    (tmp_path / "provenance.json").write_text(json.dumps({"run_id": "r1", "provider": "fake", "model": "scripted", "repeats": 2, "corpus_version": "0.2.0"}))
    return tmp_path


def test_interaction_summary_maps_verdicts_and_voids(run_dir):
    summary = interaction_summary(run_dir)
    assert summary["interruption_handling"]["n"] == 3 and summary["interruption_handling"]["passed"] == 2
    assert summary["interruption_handling"]["value"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["interruption_handling"]["voids"] == 1
    assert summary["yield_ms"]["n"] == 2 and summary["yield_ms"]["p50"] == 395.0
    assert summary["backchannel_handling"]["value"] == 0.5 and summary["backchannel_handling"]["n"] == 2
    assert summary["voids"] == {"agent finished before the barge-in point": 1}
    assert summary["variants"]["barge_in-at900ms"]["voids"] == ["agent finished before the barge-in point"]
    assert summary["usage_totals"] == {"input_tokens": 60}


def track1(conversations, value=0.75):
    row = {"value": value, "ci95": {"lo": 0.5, "hi": 1.0}, "n": 4, "source": "x"}
    none = {"value": None, "ci95": None, "n": 0, "source": "x"}
    return {
        "run_dir": "r", "run_id": "run", "model": "s2s:fake", "config": {}, "provenance": {"model": "scripted", "prompt_sha256": "abc", "readout": "onset"},
        "dataset": {}, "realtime": True, "conversations": conversations, "failures": [],
        "scorecard": {
            "pause_handling": row, "turn_taking": {**row, "latency_ms": {"n": 3, "p50": {"value": 620.0, "ci95": {"lo": 500.0, "hi": 700.0}}, "p90": {"value": 900.0}}},
            "interruption_handling": none, "backchannel_handling": none,
        },
        "eot": {"tp": 3, "fn": 1, "fp": 1, "tn": 3, "recall": 0.75, "fp_rate": 0.25, "latency_ms": {"p10": 1, "p50": 620.0, "p90": 900.0}},
        "int": {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "recall": None, "fp_rate": None, "latency_ms": {"p10": None, "p50": None, "p90": None}},
        "partial": True,
        "diagnostics": {"conversations": 3, "pacing_void": [], "max_slip_ms": 4.0, "responses": 9, "interrupted": 0, "usage": {"input_tokens": 1}, "errors": [], "acknowledged": {"x": 1}, "model": "scripted", "adapter": "fake"},
    }


def test_combined_scorecard_renders_both_tracks_and_checks_reference_rows(run_dir, tmp_path):
    card = combined_scorecard("fake", track1(["1", "2", "3"]), interaction_summary(run_dir), {"rms_vad": track1(["1", "2", "3"], 0.5)})
    text = render_markdown(card)
    assert "s2s:fake" in text and "rms_vad (reference)" in text
    assert "0.750 [0.500, 1.000] (n=4)" in text
    assert "620 [500, 700] / 900 (n=3)" in text
    assert "0.667" in text and "395" in text
    assert "SMOKE" in text
    json_path, md_path = write_scorecard(tmp_path / "cards", card)
    assert json.loads(json_path.read_text())["track2"]["backchannel_handling"]["value"] == 0.5
    assert md_path.read_text() == text
    with pytest.raises(ValueError, match="not the same rows"):
        combined_scorecard("fake", track1(["1", "2", "3"]), None, {"rms_vad": track1(["1", "2"])})
    # Either track alone still renders.
    assert "Track 2" in render_markdown(combined_scorecard("fake", None, interaction_summary(run_dir)))
