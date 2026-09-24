"""The same-record Tool Call Accuracy metric, run the way the platform runs it."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent.tool_score import load_schemas, score_run

METRIC = (Path(__file__).resolve().parent.parent / "agent" / "platform_metrics" / "tool_call_accuracy_same_record.py").read_text()
SCHEMAS = load_schemas("medicare")

# A qualification scenario in the platform's generated_mock_tool_entries shape.
PERMISSIONS = {
    "tool_name": "record_medicare_permissions",
    "new_entry": {"input": {
        "caller_name": "Ellen Parker", "caller_relationship": "self", "callback_phone": "2155550144",
        "contact_consent": True, "data_sharing_consent": True,
        "scope_of_appointment_product_types": ["medicare_supplement", "medicare_advantage"],
        "consent_statement": "<freetext>",
    }},
}
QUALIFICATION = {
    "tool_name": "save_medicare_qualification",
    "new_entry": {"input": {
        "consent_id": "perm_1005", "caller_name": "Ellen Parker", "callback_phone": "2155550144",
        "intake_intent": {"$one_of": ["new_shopping", "plan_review"]}, "age_band": "<optional>",
    }},
}
ENTRIES = [PERMISSIONS, QUALIFICATION]


def written(entry, **changes):
    """The call a model makes when it writes ``entry`` exactly, with ``changes`` applied."""
    arguments = {}
    for key, value in entry["new_entry"]["input"].items():
        if value == "<optional>":
            continue
        if value == "<freetext>":
            value = "Caller agreed."
        elif isinstance(value, dict):
            value = value["$one_of"][0]
        arguments[key] = value
    arguments.update(changes)
    return {"name": entry["tool_name"], "arguments": arguments}


def evaluate(calls, entries=ENTRIES):
    transcript = [{"role": "Agent", "content": "One moment."}] + [
        {"role": "Function Call", "content": c["name"], "data": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
        for c in calls
    ]
    namespace = {"data": {"generated_mock_tool_entries": copy.deepcopy(entries), "transcript_json": transcript}}
    exec(METRIC, namespace)  # noqa: S102 -- the platform executes the metric body the same way
    return namespace["_result"], namespace["_explanation"]


def equivalent_passes(calls, entries=ENTRIES):
    expected = [{"name": e["tool_name"], "arguments": e["new_entry"]["input"], "optional": e.get("optional") is True} for e in entries]
    return score_run(expected, calls, "equivalent", SCHEMAS).passed


CASES = {
    "exact": ([written(PERMISSIONS), written(QUALIFICATION)], True),
    "calls in either order": ([written(QUALIFICATION), written(PERMISSIONS)], True),
    "name case and spacing": ([written(PERMISSIONS, caller_name="  ellen   PARKER "), written(QUALIFICATION)], True),
    "name misspelt": ([written(PERMISSIONS, caller_name="Ellen Barker"), written(QUALIFICATION)], False),
    "phone formatted": ([written(PERMISSIONS, callback_phone="+1 (215) 555-0144"), written(QUALIFICATION)], True),
    "phone wrong": ([written(PERMISSIONS, callback_phone="2155550145"), written(QUALIFICATION)], False),
    "list in another order": (
        [written(PERMISSIONS, scope_of_appointment_product_types=["medicare_advantage", "medicare_supplement"]), written(QUALIFICATION)],
        True,
    ),
    "extra argument, null": ([written(PERMISSIONS, beneficiary_name=None), written(QUALIFICATION)], True),
    "extra argument, repeats another": ([written(PERMISSIONS, beneficiary_name="Ellen Parker"), written(QUALIFICATION)], True),
    "extra argument, new information": ([written(PERMISSIONS, beneficiary_name="Rosa Parker"), written(QUALIFICATION)], False),
    "extra argument, the schema's unknown choice": ([written(PERMISSIONS), written(QUALIFICATION, current_coverage="unknown")], True),
    "extra argument, unknown where the schema has no such choice": (
        [written(PERMISSIONS, caller_relationship_note="unknown"), written(QUALIFICATION)],
        False,
    ),
    "one_of choice in another case": ([written(PERMISSIONS), written(QUALIFICATION, intake_intent="Plan_Review")], False),
    "optional field sent": ([written(PERMISSIONS), written(QUALIFICATION, age_band="65_to_69")], True),
    "required call missing": ([written(PERMISSIONS)], False),
    "duplicate call": ([written(PERMISSIONS), written(QUALIFICATION), written(QUALIFICATION)], False),
    "call nobody expected": ([written(PERMISSIONS), written(QUALIFICATION), {"name": "route_medicare_call", "arguments": {}}], False),
}


@pytest.mark.parametrize("name", CASES)
def test_same_record_verdict_matches_the_scorer(name):
    calls, stores_same_record = CASES[name]
    score, explanation = evaluate(calls)
    assert (score == 5) is stores_same_record, explanation
    assert equivalent_passes(calls) is stores_same_record


def test_a_duplicate_costs_what_it_costs_on_tool_call_accuracy():
    # Two required calls matched and one call left over: 5 x 2*2 / (2*2 + 1).
    score, explanation = evaluate([written(PERMISSIONS), written(QUALIFICATION), written(QUALIFICATION)])
    assert score == 4.0
    assert "additional call" in explanation


def test_explanation_names_arguments_never_values():
    _, explanation = evaluate([written(PERMISSIONS, caller_name="Ellen Barker", beneficiary_name="Rosa Parker")])
    assert "value: caller_name" in explanation and "extra: beneficiary_name" in explanation
    assert "save_medicare_qualification: not called" in explanation
    assert "Barker" not in explanation and "Rosa" not in explanation


def test_call_control_is_not_a_tool_call():
    score, _ = evaluate([written(PERMISSIONS), written(QUALIFICATION), {"name": "end_call", "arguments": {}}])
    assert score == 5


def test_an_optional_entry_may_be_left_out():
    entries = ENTRIES + [{**copy.deepcopy(PERMISSIONS), "optional": True, "tool_name": "route_medicare_call"}]
    score, _ = evaluate([written(PERMISSIONS), written(QUALIFICATION)], entries)
    assert score == 5


def test_matching_does_not_depend_on_which_expected_call_comes_first():
    # Two expected lookups: one free-text, one exact. The call that fits only the exact
    # entry must go to it, whichever order the contract lists them in.
    loose = {"tool_name": "save_medicare_qualification", "new_entry": {"input": {"consent_id": "<freetext>"}}}
    exact = {"tool_name": "save_medicare_qualification", "new_entry": {"input": {"consent_id": "perm_1005"}}}
    calls = [
        {"name": "save_medicare_qualification", "arguments": {"consent_id": "perm_1005"}},
        {"name": "save_medicare_qualification", "arguments": {"consent_id": "perm_2000"}},
    ]
    assert evaluate(calls, [loose, exact])[0] == 5
    assert evaluate(calls, [exact, loose])[0] == 5


@pytest.mark.parametrize("calls, expected", [([], None), ([{"name": "lookup_patient", "arguments": {}}], 0.0)])
def test_a_scenario_with_no_expected_calls_scores_only_when_the_agent_called_one(calls, expected):
    assert evaluate(calls, [])[0] == expected
