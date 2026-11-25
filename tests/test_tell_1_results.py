"""Publication checks for the approved tell-1 snapshot and its five-gate policy."""

import copy
import hashlib
import json
import re
import runpy
from pathlib import Path

import pytest


DIRECTORY = Path(__file__).parents[1] / "results/voice-agent-workflows/retell-tell-1-2026-11-10"
MODULE = runpy.run_path(str(DIRECTORY / "validate.py"))
validate = MODULE["validate"]


@pytest.fixture
def snapshot():
    return json.loads((DIRECTORY / "results.json").read_text())


def test_approved_report_and_all_run_counts(snapshot):
    stats = validate(snapshot)
    assert (stats["Combined"]["passed_calls"], stats["Combined"]["passed_scenarios"]) == (236, 74)
    assert (stats["Appointments"]["passed_calls"], stats["Appointments"]["passed_scenarios"]) == (170, 54)
    assert (stats["Medicare"]["passed_calls"], stats["Medicare"]["passed_scenarios"]) == (66, 20)
    assert hashlib.sha256((DIRECTORY / "report.html").read_bytes()).hexdigest() == snapshot["provenance"]["report_sha256"]
    assert re.findall(r'data-chart="([^"]+)"', (DIRECTORY / "report.html").read_text()) == [
        "Reliability vs. latency", "Task completion", "Infrastructure reliability",
        "Interruption handling", "Voice naturalness", "Repeatable success · Pass³",
        "Average response time",
    ]


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf")])
def test_missing_or_invalid_gate_score_is_not_treated_as_a_pass(snapshot, value):
    snapshot["calls"][0]["metrics"]["Interruption Score"] = value
    with pytest.raises(ValueError, match="Missing/non-numeric applicable metric"):
        validate(snapshot)


def test_no_tool_scenarios_are_the_only_applicability_exceptions(snapshot):
    exempt = [c for c in snapshot["calls"] if c["metrics"]["Tool Call Accuracy"] is None]
    assert len(exempt) == 9
    assert {c["scenario"] for c in exempt} == {"AS12", "AS49", "MS79"}
    applicable = next(c for c in snapshot["calls"] if c["metrics"]["Tool Call Accuracy"] is not None)
    applicable["metrics"]["Tool Call Accuracy"] = None
    with pytest.raises(ValueError, match="Missing/non-numeric applicable metric"):
        validate(snapshot)


def test_duplicate_run_cannot_replace_an_attempt(snapshot):
    snapshot["calls"][0] = copy.deepcopy(snapshot["calls"][1])
    with pytest.raises(ValueError, match="246 unique runs"):
        validate(snapshot)


def test_three_attempts_required_even_when_total_run_count_matches(snapshot):
    snapshot["calls"][0]["scenario"] = snapshot["calls"][1]["scenario"]
    # AS1 and AS2 have identical gate outcomes, so the attempt count must catch this.
    with pytest.raises(ValueError, match="three runs per scenario"):
        validate(snapshot)


def test_stored_verdict_cannot_override_computed_gates(snapshot):
    failed = next(c for c in snapshot["calls"] if not c["passed"])
    failed["passed"] = True
    with pytest.raises(ValueError, match="Verdict disagrees with gates"):
        validate(snapshot)


def test_other_metric_does_not_silently_become_a_gate(snapshot):
    snapshot["scoring"]["gates"].append({"metric": "Transcription Accuracy", "operator": "eq", "threshold": 5})
    with pytest.raises(ValueError, match="Five-gate policy changed"):
        validate(snapshot)


def test_pass_three_counts_actual_triads():
    calls = [{"scenario": s, "passed": p} for s, outcomes in
             [("AS1", [True, True, False]), ("AS2", [True, True, True])]
             for p in outcomes]
    stats = MODULE["summarize"](calls)
    assert stats["pass1"] == 5 / 6
    assert stats["pass3"] == 1 / 2
