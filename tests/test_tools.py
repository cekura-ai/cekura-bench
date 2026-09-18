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

    def test_an_unknown_record_is_a_miss_not_an_invention(self, server):
        assert server.call("lookup_patient", {"phone": "4045550000"}) == {"result": "no_match"}
        assert server.calls[-1].matched is False

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
