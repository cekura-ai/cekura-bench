"""The mock-tool server and the trace verifier.

These decide whether a task cell passes, so they are held to the standard the
scoring claim implies: a model that asks for the right record in a different
format has followed the contract, and a model that asks for a record that does
not exist must not be handed one.
"""

from __future__ import annotations

import pytest

from mock_tools.server import MockToolServer
from mock_tools.verifier import ExpectedCall, verify_trace


@pytest.fixture
def server() -> MockToolServer:
    return MockToolServer("appointments")


class TestContract:
    def test_the_published_tools_are_exposed_to_the_model(self, server):
        specs = server.tool_specs()
        assert {spec.name for spec in specs} == set(server.tool_names)
        assert all(spec.description and spec.parameters for spec in specs)

    def test_the_contracts_own_prompt_is_used(self, server):
        assert server.system_prompt and server.first_message


class TestAnswering:
    def test_an_exact_input_returns_the_published_record(self, server):
        assert server.call("lookup_patient", {"phone": "2025550188"})["patient_id"] == "p_1002"

    @pytest.mark.parametrize("spoken", ["(202) 555-0188", "202-555-0188", " 202 555 0188 "])
    def test_formatting_does_not_decide_the_score(self, server, spoken):
        """The contract asks for ten digits; punctuation is the formatter's taste."""
        assert server.call("lookup_patient", {"phone": spoken})["patient_id"] == "p_1002"

    def test_an_unknown_record_is_answered_without_inventing_one(self, server):
        answer = server.call("lookup_patient", {"phone": "4045550000"})
        # The table carries its own "no patient found" row for exactly this, and
        # a number nothing resembles lands on it rather than on a real patient.
        assert "patient_id" not in answer
        assert answer["match"] is False
        assert server.calls[-1].matched is False, "only an exact call counts as a hit"

    def test_an_undeclared_tool_is_reported_as_unknown(self, server):
        assert "unknown tool" in server.call("transfer_to_human", {})["error"]

    def test_calls_are_recorded_in_order(self, server):
        server.call("lookup_patient", {"phone": "2025550188"})
        server.call("check_availability", {"date": "2026-07-08"})
        assert [c.name for c in server.calls] == ["lookup_patient", "check_availability"]
        server.reset()
        assert server.calls == []


class TestVerifier:
    def test_a_correct_trace_passes_on_all_three_axes(self):
        verdict = verify_trace(
            [
                {"name": "lookup_patient", "arguments": {"phone": "2025550188"}},
                {"name": "check_availability", "arguments": {"date": "2026-07-08"}},
                {"name": "book_appointment", "arguments": {"patient_id": "p_1002"}},
            ],
            [
                ExpectedCall("lookup_patient", {"phone": "2025550188"}),
                ExpectedCall("check_availability"),
                ExpectedCall("book_appointment", {"patient_id": "p_1002"}),
            ],
        )
        assert verdict.passed and verdict.presence and verdict.arguments and verdict.order

    def test_a_missing_call_fails_presence_only(self):
        verdict = verify_trace(
            [{"name": "lookup_patient", "arguments": {"phone": "2025550188"}}],
            [ExpectedCall("lookup_patient"), ExpectedCall("book_appointment")],
        )
        assert not verdict.presence and verdict.arguments
        assert verdict.missing == ("book_appointment",)

    def test_booking_before_checking_fails_order_alone(self):
        verdict = verify_trace(
            [{"name": "book_appointment", "arguments": {}}, {"name": "check_availability", "arguments": {}}],
            [ExpectedCall("check_availability"), ExpectedCall("book_appointment")],
        )
        assert verdict.presence and not verdict.order

    def test_a_repeated_lookup_is_not_an_order_violation(self):
        """Looking again after the caller changes their mind is allowed."""
        verdict = verify_trace(
            [
                {"name": "lookup_patient", "arguments": {}},
                {"name": "check_availability", "arguments": {}},
                {"name": "lookup_patient", "arguments": {}},
                {"name": "book_appointment", "arguments": {}},
            ],
            [ExpectedCall("lookup_patient"), ExpectedCall("check_availability"), ExpectedCall("book_appointment")],
        )
        assert verdict.passed

    def test_the_wrong_record_fails_arguments_not_presence(self):
        verdict = verify_trace(
            [{"name": "lookup_patient", "arguments": {"phone": "4155550123"}}],
            [ExpectedCall("lookup_patient", {"phone": "2025550188"})],
        )
        assert verdict.presence and not verdict.arguments
        assert verdict.wrong_arguments == ("lookup_patient",)

    def test_a_forbidden_tool_fails_presence(self):
        verdict = verify_trace(
            [{"name": "cancel_appointment", "arguments": {}}],
            [],
            forbidden=["cancel_appointment"],
        )
        assert not verdict.presence and verdict.forbidden_used == ("cancel_appointment",)

    def test_an_optional_call_may_be_absent(self):
        verdict = verify_trace([], [ExpectedCall("lookup_patient", optional=True)])
        assert verdict.passed


class TestTheMostSpecificRowWins:
    """A table row whose inputs are a subset of another's must not shadow it."""

    @staticmethod
    def _shadowing_pair(server, tool):
        """The general row and the more specific row it could hide.

        Found from the table rather than hard-coded, so the test keeps testing
        the real hazard if the contract is revised.
        """
        rows = server._mocks[tool]["mock_data"]
        for general in rows:
            gi = general["input"]
            for specific in rows:
                si = specific["input"]
                if gi and set(gi) < set(si) and all(si.get(k) == v for k, v in gi.items()):
                    return general, specific
        return None, None

    def test_a_call_naming_more_fields_gets_the_row_that_names_them(self):
        server = MockToolServer("medicare")
        general, specific = self._shadowing_pair(server, "route_medicare_call")
        assert specific is not None, "the contract no longer contains a subset row"
        # Asking with every field the specific row names must answer with it,
        # whichever order the table happens to list the two rows in.
        assert server.call("route_medicare_call", dict(specific["input"])) == specific["output"]
        assert specific["output"] != general["output"], "the pair would not distinguish anything"

    def test_a_call_naming_fewer_fields_still_gets_the_general_row(self):
        server = MockToolServer("medicare")
        general, _ = self._shadowing_pair(server, "route_medicare_call")
        assert server.call("route_medicare_call", dict(general["input"])) == general["output"]


class TestResolvingACallToARecord:
    """How a call finds its row, and what happens when it nearly does.

    The tables are records, not assertions. A row lists the fields that were
    present when it was captured, and many of those are optional in the tool's
    own schema, so a rule that demanded every one of them would make the row
    unreachable for any agent that followed the schema. These tests pin the two
    stages that reach it instead -- and the fact that reaching it is what keeps
    a multi-step task alive, because the identifiers a later call needs only
    exist inside an earlier call's answer.
    """

    @pytest.fixture
    def medicare(self):
        return MockToolServer("medicare")

    # The recorded call that failed a whole cohort of runs: every field the
    # tool's schema requires, one optional field the caller never gave.
    WITHOUT_THE_POSTCODE = {
        "callback_phone": "6025550155",
        "caller_name": "Linda Martinez",
        "consent_id": "perm_1009",
        "has_medicare_part_a": "yes",
        "has_medicare_part_b": "yes",
        "intake_intent": "plan_review",
        "product_interest": "medicare_advantage",
        "state": "AZ",
    }

    def test_an_optional_field_left_out_still_reaches_a_record(self, medicare):
        answer = medicare.call("save_medicare_qualification", self.WITHOUT_THE_POSTCODE)
        # Any answer at all is the point: the next tool in this suite needs an
        # identifier, and an identifier only ever arrives inside an answer.
        assert answer.get("lead_id")

    def test_the_answer_says_what_was_missing(self, medicare):
        answer = medicare.call("save_medicare_qualification", self.WITHOUT_THE_POSTCODE)
        # The table seeds the ordinary ways a call goes wrong, and their outputs
        # name the gap so the agent can go back and close it. An agent told
        # nothing can only tell the caller the system failed -- and then the
        # conversation being scored is a conversation about the harness.
        assert answer["missing_fields"] == ["zip_code"]

    def test_the_complete_call_reaches_the_row_it_names(self, medicare):
        row = next(
            m for m in medicare._mocks["save_medicare_qualification"]["mock_data"]
            if m["input"].get("consent_id") == "perm_1009"
        )
        assert medicare.call("save_medicare_qualification", dict(row["input"])) == row["output"]
        assert medicare.calls[-1].resolution == "exact"

    def test_free_text_never_decides_a_match(self, medicare):
        # No two agents write the same sentence, and the tables store none, so a
        # field the contract declares as free text cannot be part of the lookup.
        row = medicare._mocks["create_handoff_summary"]["mock_data"][0]
        answer = medicare.call(
            "create_handoff_summary",
            {**row["input"], "compliance_notes": "whatever this particular agent chose to write"},
        )
        assert answer == row["output"]
        assert medicare.calls[-1].resolution == "exact"

    def test_a_field_sent_empty_is_a_field_not_sent(self, medicare):
        row = medicare._mocks["record_medicare_permissions"]["mock_data"][0]
        assert medicare.call(
            "record_medicare_permissions", {**row["input"], "beneficiary_name": None}
        ) == row["output"]


class TestWhenSpeechBendsAnArgument:
    """A transcription error is not a task failure, and must not be scored as one."""

    @pytest.fixture
    def server(self):
        return MockToolServer("appointments")

    def test_one_bent_digit_still_finds_the_record(self, server):
        answer = server.call("lookup_patient", {"phone": "2025550189"})  # 2025550188 misheard
        assert answer["patient_id"] == "p_1002"
        assert server.calls[-1].resolution == "fuzzy"
        assert server.calls[-1].matched is False, "near is not exact, and the record must say so"

    def test_a_number_nothing_resembles_gets_no_patient(self, server):
        answer = server.call("lookup_patient", {"phone": "0000000000"})
        assert "patient_id" not in answer

    def test_a_country_code_is_not_a_different_number(self, server):
        assert server.call("lookup_patient", {"phone": "+1 (202) 555-0188"})["patient_id"] == "p_1002"
        assert server.calls[-1].resolution == "exact"

    def test_an_undeclared_tool_is_not_resolved_at_all(self, server):
        server.call("transfer_to_human", {})
        assert server.calls[-1].resolution == "unknown"
