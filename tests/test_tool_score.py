"""Tool calls scored as written and as stored, against each scenario's expected calls."""

from __future__ import annotations

import json
import shutil

import pytest

from agent.report import ReportError
from agent.tool_score import DEFINITIONS, load_schemas, score_run, summarize

SCHEMAS = load_schemas("medicare")

# A standard qualification scenario, shaped as the suite's contract writes it:
# the caller's name exact, no beneficiary (the caller is the beneficiary), the
# timing and coverage fields optional, the consent statement free text.
PERMISSIONS = {
    "name": "record_medicare_permissions",
    "arguments": {
        "caller_name": "Ellen Parker", "caller_relationship": "self", "callback_phone": "2155550144",
        "contact_consent": True, "data_sharing_consent": True,
        "scope_of_appointment_product_types": ["medicare_supplement"], "consent_statement": "<freetext>",
    },
}
QUALIFICATION = {
    "name": "save_medicare_qualification",
    "arguments": {
        "consent_id": "perm_1005", "caller_name": "Ellen Parker", "callback_phone": "2155550144",
        "product_interest": "medicare_supplement", "intake_intent": {"$one_of": ["new_shopping", "plan_review"]},
        "age_band": "<optional>", "election_window": "<optional>", "current_coverage": "<optional>",
    },
}
EXPECTED = [PERMISSIONS, QUALIFICATION]


def call(entry, **changes):
    """The call a model makes when it writes ``entry`` exactly, with ``changes`` applied."""
    arguments = {k: ("Caller said yes." if v == "<freetext>" else v) for k, v in entry["arguments"].items()}
    arguments = {k: v for k, v in arguments.items() if v != "<optional>"}
    if isinstance(arguments.get("intake_intent"), dict):
        arguments["intake_intent"] = "new_shopping"
    for key, value in changes.items():
        if value is DROP:
            arguments.pop(key, None)
        else:
            arguments[key] = value
    return {"name": entry["name"], "arguments": arguments, "resolution": "exact"}


DROP = object()


def both(calls, expected=EXPECTED):
    return {mode: score_run(expected, calls, mode, SCHEMAS) for mode in ("strict", "equivalent")}


class TestAWrittenContractPassesBoth:
    def test_exact_calls_pass_strict_and_equivalent(self):
        scores = both([call(PERMISSIONS), call(QUALIFICATION)])
        assert all(s.passed and s.matched_calls == 2 for s in scores.values())

    def test_order_does_not_matter(self):
        assert both([call(QUALIFICATION), call(PERMISSIONS)])["strict"].passed

    def test_hanging_up_and_cancelled_calls_are_not_scored(self):
        calls = [call(PERMISSIONS), call(QUALIFICATION), {"name": "end_call", "arguments": {}},
                 {**call(QUALIFICATION, caller_name="x"), "cancelled": True}]
        assert both(calls)["strict"].passed

    def test_optional_fields_may_be_sent_or_left_out(self):
        sent = call(QUALIFICATION, age_band="already_medicare_age", election_window="not_sure")
        assert both([call(PERMISSIONS), sent])["strict"].passed

    def test_one_of_accepts_any_listed_choice_and_nothing_else(self):
        assert both([call(PERMISSIONS), call(QUALIFICATION, intake_intent="plan_review")])["strict"].passed
        wrong = both([call(PERMISSIONS), call(QUALIFICATION, intake_intent="caregiver_inquiry")])["strict"]
        assert not wrong.passed and wrong.defects == {"other_value": 1}

    def test_freetext_must_still_be_said(self):
        empty = both([call(PERMISSIONS, consent_statement=" "), call(QUALIFICATION)])
        assert not empty["strict"].passed and not empty["equivalent"].passed


class TestFormattingFailsStrictButStoresTheSameRecord:
    def test_a_name_in_lower_case(self):
        scores = both([call(PERMISSIONS, caller_name="ellen parker"), call(QUALIFICATION, caller_name="ellen  parker")])
        assert scores["strict"].defects == {"name_value": 2}
        assert scores["equivalent"].passed

    def test_a_phone_number_with_separators_or_a_country_code(self):
        scores = both([call(PERMISSIONS, callback_phone="(215) 555-0144"), call(QUALIFICATION, callback_phone="+1 215 555 0144")])
        assert scores["strict"].defects == {"other_value": 2}
        assert scores["equivalent"].passed

    def test_a_beneficiary_named_as_the_caller_who_is_the_beneficiary(self):
        """The schema says to leave it out; sending it repeats a value the record already holds."""
        scores = both([call(PERMISSIONS, beneficiary_name="Ellen Parker"), call(QUALIFICATION, beneficiary_name="Ellen Parker")])
        assert scores["strict"].defects == {"extra_argument": 2}
        assert scores["equivalent"].passed

    def test_an_unknown_choice_or_a_null_adds_nothing(self):
        scores = both([call(PERMISSIONS), call(QUALIFICATION, has_medicare_part_a="unknown", state=None)])
        assert scores["strict"].defects == {"extra_argument": 1}
        assert scores["equivalent"].passed

    def test_a_list_in_another_order(self):
        expected = [{**PERMISSIONS, "arguments": {**PERMISSIONS["arguments"],
                     "scope_of_appointment_product_types": ["part_d", "medicare_advantage"]}}]
        scores = both([call(PERMISSIONS, scope_of_appointment_product_types=["medicare_advantage", "part_d"])], expected)
        assert not scores["strict"].passed and scores["equivalent"].passed


class TestWrongDataFailsBoth:
    def test_a_misheard_name_is_a_different_person(self):
        scores = both([call(PERMISSIONS, caller_name="Alan Parker"), call(QUALIFICATION)])
        assert scores["strict"].defects == scores["equivalent"].defects == {"name_value": 1}

    def test_an_extra_argument_carrying_new_information(self):
        scores = both([call(PERMISSIONS, beneficiary_name="Margaret Parker"), call(QUALIFICATION)])
        assert scores["equivalent"].defects == {"extra_argument": 1}

    def test_a_placeholder_name(self):
        assert not both([call(PERMISSIONS, caller_name="unknown"), call(QUALIFICATION)])["equivalent"].passed

    def test_an_invented_id(self):
        scores = both([call(PERMISSIONS), call(QUALIFICATION, consent_id="perm_0000")])
        assert scores["equivalent"].defects == {"other_value": 1}

    def test_a_missing_argument(self):
        scores = both([call(PERMISSIONS), call(QUALIFICATION, product_interest=DROP)])
        assert scores["equivalent"].defects == {"missing_argument": 1}

    def test_a_missing_call(self):
        scores = both([call(PERMISSIONS)])
        assert scores["equivalent"].defects == {"missing_call": 1}
        assert scores["equivalent"].matched_calls == 1

    def test_an_optional_expected_call_may_be_skipped_but_not_botched(self):
        expected = [PERMISSIONS, {**QUALIFICATION, "optional": True}]
        assert both([call(PERMISSIONS)], expected)["strict"].passed
        assert not both([call(PERMISSIONS), call(QUALIFICATION, consent_id="perm_0000")], expected)["strict"].passed

    def test_a_call_nobody_expected(self):
        extra = {"name": "route_medicare_call", "arguments": {"route_reason": "sales_or_plan_review"}}
        scores = both([call(PERMISSIONS), call(QUALIFICATION), extra])
        assert scores["equivalent"].defects == {"unmatched_call": 1}


class TestARepeatedCallIsPairedWithTheCallItMeant:
    def test_a_retry_after_a_bad_first_attempt(self):
        """The bad attempt is unmatched, the retry matches: a retry is still a call nobody expected."""
        scores = both([call(PERMISSIONS, caller_name="Unknown caller"), call(PERMISSIONS), call(QUALIFICATION)])
        assert scores["strict"].matched_calls == 2
        assert scores["strict"].defects == {"unmatched_call": 1}

    def test_two_expected_calls_of_one_tool_each_find_their_own(self):
        second = {**QUALIFICATION, "arguments": {**QUALIFICATION["arguments"], "product_interest": "part_d"}}
        calls = [call(second), call(QUALIFICATION)]
        assert both(calls, [QUALIFICATION, second])["strict"].passed


def export(tmp_path, runs):
    path = tmp_path / "runs.jsonl"
    path.write_text("".join(json.dumps(run) + "\n" for run in runs))
    return path


@pytest.fixture
def definitions(tmp_path):
    root = tmp_path / "agent-definitions"
    (root / "medicare").mkdir(parents=True)
    shutil.copy(DEFINITIONS / "medicare" / "tool-definitions.json", root / "medicare")
    (root / "medicare" / "expected-tool-calls.json").write_text(json.dumps({"1": EXPECTED, "2": []}))
    return root


def run(run_id, scenario, calls):
    return {"run_id": run_id, "scenario_id": scenario,
            "custom_metadata": {"config": "nova-sonic", "agent_definition": "medicare", "tool_calls": calls}}


class TestARowCarriesBothScores:
    def test_both_scores_and_the_runs_each_failed(self, definitions):
        report = summarize([
            run(10, 1, [call(PERMISSIONS), call(QUALIFICATION)]),
            run(11, 1, [call(PERMISSIONS, caller_name="ellen parker"), call(QUALIFICATION)]),
            run(12, 1, [call(PERMISSIONS, caller_name="Alan Parker"), call(QUALIFICATION)]),
            run(13, 2, []),
        ], definitions)
        (row,) = report["rows"]
        assert (row["runs"], row["scored_runs"], row["runs_without_expected_calls"]) == (4, 3, 1)
        assert row["strict"]["passed_runs"] == 1 and row["strict"]["failing_run_ids"] == [11, 12]
        assert row["equivalent"]["passed_runs"] == 2 and row["equivalent"]["failing_run_ids"] == [12]
        assert row["strict"]["expected_calls"] == 6 and row["strict"]["matched_calls"] == 4
        assert row["strict"]["defects"] == {"name_value": 2}

    def test_a_scenario_missing_from_the_contract_is_refused(self, definitions):
        with pytest.raises(ReportError, match="scenario 3"):
            summarize([run(10, 3, [])], definitions)

    def test_a_suite_without_a_contract_is_refused(self, definitions):
        with pytest.raises(ReportError, match="expected calls"):
            summarize([{**run(10, 1, []), "custom_metadata": {"config": "x", "agent_definition": "appointments"}}], definitions)
