"""Publication checks for the ElevenLabs Agents workflow snapshot and its five-gate policy."""

import copy
import json
import runpy
from pathlib import Path

import pytest


DIRECTORY = Path(__file__).parents[1] / "results/voice-agent-workflows/elevenlabs-agents-2026-11-17"
MODULE = runpy.run_path(str(DIRECTORY / "validate.py"))
validate = MODULE["validate"]


@pytest.fixture
def snapshot():
    return json.loads((DIRECTORY / "results.json").read_text())


def test_all_run_counts(snapshot):
    stats = validate(snapshot)
    assert (stats["Combined"]["passed_calls"], stats["Combined"]["passed_scenarios"]) == (224, 67)
    assert (stats["Appointments"]["passed_calls"], stats["Appointments"]["passed_scenarios"]) == (162, 50)
    assert (stats["Medicare"]["passed_calls"], stats["Medicare"]["passed_scenarios"]) == (62, 17)


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf")])
def test_missing_or_invalid_gate_score_is_not_treated_as_a_pass(snapshot, value):
    snapshot["calls"][0]["metrics"]["Interruption Score"] = value
    with pytest.raises(ValueError, match="Missing/non-numeric applicable metric"):
        validate(snapshot)


def test_no_tool_scenarios_are_the_only_applicability_exceptions(snapshot):
    exempt = [c for c in snapshot["calls"] if c["metrics"]["Tool Call Accuracy"] is None]
    assert {c["scenario"] for c in exempt} == {"AS12", "AS49", "MS79"} and len(exempt) == 9
    applicable = next(c for c in snapshot["calls"] if c["metrics"]["Tool Call Accuracy"] is not None)
    applicable["metrics"]["Tool Call Accuracy"] = None
    with pytest.raises(ValueError, match="Missing/non-numeric applicable metric"):
        validate(snapshot)


def test_replaced_run_cannot_be_counted(snapshot):
    replaced = snapshot["replaced_runs"][0]
    call = next(c for c in snapshot["calls"] if c["run_id"] == replaced["replacement_run_id"])
    call["run_id"] = replaced["run_id"]
    with pytest.raises(ValueError, match="Replaced run is still counted"):
        validate(snapshot)


def test_replacement_must_fill_its_own_slot(snapshot):
    replaced = snapshot["replaced_runs"][0]
    call = next(c for c in snapshot["calls"] if c["run_id"] == replaced["replacement_run_id"])
    call["result_id"] = snapshot["workflows"]["Appointments"]["result_id"]
    with pytest.raises(ValueError, match="Replacement does not match its slot"):
        validate(snapshot)


def test_duplicate_run_cannot_replace_an_attempt(snapshot):
    snapshot["calls"][0] = copy.deepcopy(snapshot["calls"][1])
    with pytest.raises(ValueError, match="246 unique runs"):
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
