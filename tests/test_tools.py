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

    # Every field the tool's schema requires; the optional postcode left out.
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


class TestAnEnumWrittenTwoWaysIsOneCategory:
    """Several fields are enumerations, and the caller does not speak in tokens.

    The tables store ``original_medicare``; the caller says "Original Medicare".
    A model that echoes the caller and a model that writes the token have chosen
    the same category and differ only in punctuation. Scoring those differently
    would rank a provider on its formatting habits, and providers differ in that
    habit, so the column would tilt for a reason that has nothing to do with
    understanding the caller.
    """

    @staticmethod
    def _qualified(coverage: str):

        server = MockToolServer(suite="medicare")
        consent = server.call("record_medicare_permissions", {
            "caller_name": "Maria Gomez", "callback_phone": "4155550199",
            "contact_consent": True, "data_sharing_consent": True,
            "scope_of_appointment_product_types": ["medicare_advantage"],
            "caller_relationship": "self",
        })["consent_id"]
        output = server.call("save_medicare_qualification", {
            "caller_name": "Maria Gomez", "callback_phone": "4155550199",
            "consent_id": consent, "state": "CA", "zip_code": "94105",
            "age_band": "already_medicare_age", "has_medicare_part_a": "yes",
            "has_medicare_part_b": "yes", "current_coverage": coverage,
            "product_interest": "medicare_advantage", "intake_intent": "new_shopping",
            "caller_relationship": "self",
        })
        return server.calls[-1], output

    def test_the_stored_token_matches(self):
        record, output = self._qualified("original_medicare")
        assert record.resolution == "exact"
        assert output["lead_id"] == "lead_2001"

    def test_the_spoken_form_matches_the_same_record(self):
        record, output = self._qualified("Original Medicare")
        assert record.resolution == "exact"
        assert output["lead_id"] == "lead_2001"

    def test_the_wrong_category_still_does_not(self):
        # The part worth measuring survives: choosing a different category is a
        # real miss, and only the punctuation was ever forgiven.
        record, _ = self._qualified("medicare_advantage")
        assert record.resolution != "exact"


class TestTheOrderOfACategoryListSaysNothing:
    """Both arrays these contracts declare are sets, not sequences.

    A caller who agreed that Medicare Advantage and Part D may be discussed has
    agreed to the same two things whichever order the model lists them in, and
    models differ in the order they emit array members. Scoring the order would
    rank a provider on an arbitrary habit.
    """

    @staticmethod
    def _consent(scope):
        import json


        tools = {t["name"]: t for t in json.load(
            open("agent-definitions/medicare/mock-tools.json")
        )}
        row = next(
            r for r in tools["record_medicare_permissions"]["mock_data"]
            if isinstance(r["input"].get("scope_of_appointment_product_types"), list)
            and len(r["input"]["scope_of_appointment_product_types"]) > 1
        )
        stored = row["input"]["scope_of_appointment_product_types"]
        server = MockToolServer(suite="medicare")
        server.call("record_medicare_permissions", dict(
            row["input"],
            scope_of_appointment_product_types=scope(list(stored)),
        ))
        return server.calls[-1].resolution

    def test_the_stored_order_matches(self):
        assert self._consent(lambda stored: stored) == "exact"

    def test_the_other_order_is_the_same_consent(self):
        assert self._consent(lambda stored: list(reversed(stored))) == "exact"

    def test_a_narrower_consent_is_a_different_record(self):
        assert self._consent(lambda stored: stored[:1]) != "exact"

    def test_a_wider_consent_is_a_different_record(self):
        # Agreeing to discuss one more product category is a different consent,
        # and consent is the one thing here that must not be approximated.
        assert self._consent(
            lambda stored: stored + ["medicare_supplement"]
        ) != "exact"


class TestOneSlotWrittenSeveralWays:
    """Booking is the success action, so the slot must not turn on its spelling.

    The contract asks for ``YYYY-MM-DDTHH:MM:SS`` and the services oblige to
    different degrees -- a space where the T belongs, the seconds left off, a
    trailing Z. Each names the same appointment, and which spelling a provider
    favours says nothing about whether it booked what the caller agreed to.
    """

    @staticmethod
    def _book(change):
        import json


        tools = {t["name"]: t for t in json.load(
            open("agent-definitions/appointments/mock-tools.json")
        )}
        stored = tools["book_appointment"]["mock_data"][0]["input"]
        server = MockToolServer(suite="appointments")
        server.call("book_appointment", dict(stored, datetime=change(stored["datetime"])))
        return server.calls[-1].resolution

    def test_the_contracts_own_spelling(self):
        assert self._book(lambda slot: slot) == "exact"

    def test_a_space_where_the_t_belongs(self):
        assert self._book(lambda slot: slot.replace("T", " ")) == "exact"

    def test_the_seconds_left_off(self):
        assert self._book(lambda slot: slot[:16]) == "exact"

    def test_a_trailing_zulu(self):
        # Nothing in these contracts carries a timezone, so a Z is notation.
        assert self._book(lambda slot: slot + "Z") == "exact"

    def test_a_different_hour_is_a_different_appointment(self):
        assert self._book(lambda slot: slot.replace("T09", "T11")) != "exact"

    def test_a_real_offset_is_not_quietly_accepted(self):
        # A shifted time is a different instant. It is left to miss rather than
        # canonicalised, because guessing a timezone here could move a booking.
        assert self._book(lambda slot: slot + "+05:30") != "exact"


class TestSayingUnknownIsTheSameAsSayingNothing:
    """Several optional fields offer ``unknown`` in their enum while the tool's own
    prose says to omit an argument with no value. Both spellings say the same
    thing, and which one a service reaches for is a habit of that service."""

    @staticmethod
    def _qualify(**overrides):
        server = MockToolServer(suite="medicare")
        stored = server._mocks["save_medicare_qualification"]["mock_data"][0]["input"]
        server.call("save_medicare_qualification", dict(stored, **overrides))
        return server.calls[-1]

    def test_the_contracts_own_record(self):
        assert self._qualify().resolution == "exact"

    def test_an_unknown_age_band_reads_as_omitted(self):
        assert self._qualify(age_band="unknown").resolution == "exact"

    def test_an_unknown_coverage_reads_as_omitted(self):
        assert self._qualify(current_coverage="unknown").resolution == "exact"

    def test_the_record_answered_is_still_the_right_one(self):
        call = self._qualify(age_band="unknown", current_coverage="unknown")
        assert call.output["lead_id"] == self._qualify().output["lead_id"]

    def test_a_wrong_value_is_not_an_abstention(self):
        assert self._qualify(age_band="not_yet_on_medicare").resolution != "exact"

    def test_a_required_field_must_still_be_committed_to(self):
        # product_interest also offers ``unknown``, but it is required: declining
        # to answer there is an answer, not a silence.
        assert self._qualify(product_interest="unknown").resolution != "exact"


class TestAPhraseTheAgentComposesIsNotAnIdentifier:
    """``destination`` is a sentence the agent writes describing where the call is
    going, not a value the caller states. It separates no two records, so
    comparing it could only ever penalise an agent for its choice of words."""

    @staticmethod
    def _handoff(destination):
        server = MockToolServer(suite="medicare")
        rows = server._mocks["create_handoff_summary"]["mock_data"]
        stored = next(r["input"] for r in rows if r["input"].get("route_status") == "closed_no_consent")
        server.call("create_handoff_summary", dict(stored, destination=destination))
        return server.calls[-1]

    def test_the_contracts_own_wording(self):
        assert self._handoff("Medicare.gov, one eight hundred Medicare, or local SHIP").resolution == "exact"

    def test_an_acronym_spelled_out(self):
        spelled = "Medicare.gov, one eight hundred Medicare, or your local State Health Insurance Assistance Program"
        assert self._handoff(spelled).resolution == "exact"

    def test_every_record_is_still_reachable_without_it(self):
        server = MockToolServer(suite="medicare")
        rows = server._mocks["create_handoff_summary"]["mock_data"]
        answered = []
        for row in rows:
            server.calls.clear()
            answered.append(server.call("create_handoff_summary", dict(row["input"], destination="somewhere")))
        assert answered == [row["output"] for row in rows]


class TestDecliningDiffersFromAsserting:
    """An abstention is dropped only where it declines to answer AND no record
    uses it. Where a record does use it, the contract means it as a real answer."""

    @staticmethod
    def _qualify(**overrides):
        server = MockToolServer(suite="medicare")
        stored = server._mocks["save_medicare_qualification"]["mock_data"][0]["input"]
        server.call("save_medicare_qualification", dict(stored, **overrides))
        return server.calls[-1].resolution

    def test_an_election_window_nobody_is_sure_of(self):
        assert self._qualify(election_window="not_sure") == "exact"

    def test_a_relationship_that_names_no_category(self):
        assert self._qualify(caller_relationship="other") == "exact"

    def test_naming_the_wrong_relationship_is_a_claim(self):
        assert self._qualify(caller_relationship="spouse") != "exact"

    def test_a_status_the_contract_keeps_a_record_for_is_compared(self):
        # Callers do say they are unsure of their Part B status, and the tables
        # hold a record for it, so it is an answer rather than a silence.
        assert self._qualify(has_medicare_part_b="unknown") != "exact"


class TestAnIdentifierNamesItsRecord:
    """A record id reaches an agent one way only: an earlier answer handed it over.

    So it is not one field among many to be weighed against the rest. A row
    holding a different one belongs to a different caller, and answering with it
    hands that caller's identifiers to this one -- which the agent then carries
    into every later call, failing each of them for a reason of ours.
    """

    @pytest.fixture
    def medicare(self):
        return MockToolServer("medicare")

    # Every field naming the caller is right; one describing the call is not,
    # and the scenario that authored it accepts more than one value there.
    ONE_FIELD_APART = {
        "consent_id": "perm_1017",
        "caller_name": "Grace Thompson",
        "callback_phone": "9165550194",
        "state": "CA",
        "zip_code": "95814",
        "has_medicare_part_a": "yes",
        "has_medicare_part_b": "yes",
        "product_interest": "medicare_advantage",
        "intake_intent": "new_shopping",
    }

    def test_the_record_the_identifier_names_is_the_one_answered(self, medicare):
        assert medicare.call("save_medicare_qualification", self.ONE_FIELD_APART)["lead_id"] == "lead_2017"

    def test_no_other_callers_record_can_be_reached(self, medicare):
        """Nothing this call could disagree on may move it to another caller."""
        for value in ("plan_review", "switching", "not_sure", ""):
            answer = medicare.call("save_medicare_qualification", {**self.ONE_FIELD_APART, "intake_intent": value})
            assert answer.get("lead_id") in {"lead_2017", None}, value

    def test_an_identifier_no_record_holds_is_not_answered_from_a_neighbour(self, medicare):
        answer = medicare.call("save_medicare_qualification", {**self.ONE_FIELD_APART, "consent_id": "perm_9999"})
        assert "lead_id" not in answer


class TestTheAnswerSaysWhatTheCallSaid:
    """A row remembers a whole caller; a call carries what the agent gathered.

    Returning the row unchanged answers with the part the agent never collected.
    An agent repeating that downstream is then right for a reason it did not
    earn, and the question it skipped stops being visible anywhere.
    """

    @pytest.fixture
    def medicare(self):
        return MockToolServer("medicare")

    NEVER_ASKED_WHERE = {
        "consent_id": "perm_1003",
        "caller_name": "Patricia Lee",
        "callback_phone": "2125550133",
        "product_interest": "multiple",
        "intake_intent": "new_shopping",
        "has_medicare_part_a": "yes",
        "has_medicare_part_b": "yes",
    }

    def test_a_field_the_call_never_carried_is_not_handed_back(self, medicare):
        summary = medicare.call("save_medicare_qualification", self.NEVER_ASKED_WHERE)["safe_summary"]
        assert "zip_code" not in summary and "state" not in summary

    def test_a_field_the_call_carried_comes_back_as_the_call_had_it(self, medicare):
        # Under whatever name the answer files it under: a summary renames as
        # often as it repeats, and a rename would leak just as well.
        summary = medicare.call("save_medicare_qualification", self.NEVER_ASKED_WHERE)["safe_summary"]
        assert summary["intent"] == "new_shopping"
        assert summary["product_interest"] == "multiple"

    def test_what_the_record_computes_is_still_its_own(self, medicare):
        answer = medicare.call("save_medicare_qualification", self.NEVER_ASKED_WHERE)
        assert answer["lead_id"] == "lead_2004"


class TestWhatIsMissingIsAFactAboutTheCall:
    """The one path this contract has for recovering has to be reachable.

    Saying what is absent is how an agent learns to go back and ask. Read off a
    row, that message would only fit the call captured with it, so the gap is
    computed from the call, not copied from the row.
    """

    @pytest.fixture
    def medicare(self):
        return MockToolServer("medicare")

    def test_an_absent_field_is_named_whichever_record_answers(self, medicare):
        for consent, caller in (("perm_1003", "Patricia Lee"), ("perm_1017", "Grace Thompson")):
            answer = medicare.call("save_medicare_qualification", {
                "consent_id": consent, "caller_name": caller, "callback_phone": "2125550133",
                "product_interest": "multiple", "intake_intent": "new_shopping",
                "has_medicare_part_a": "yes", "has_medicare_part_b": "yes",
            })
            assert answer["missing_fields"] == ["zip_code"], consent
            assert answer["routing_ready"] is False, consent

    def test_a_complete_call_is_not_told_to_go_back(self, medicare):
        answer = medicare.call("save_medicare_qualification", {
            "consent_id": "perm_1017", "caller_name": "Grace Thompson",
            "callback_phone": "9165550194", "zip_code": "95814", "state": "CA",
            "product_interest": "medicare_advantage", "intake_intent": "switching",
            "has_medicare_part_a": "yes", "has_medicare_part_b": "yes",
        })
        assert answer["missing_fields"] == [] and answer["routing_ready"] is True

    def test_every_name_it_can_report_is_one_the_agent_can_act_on(self, medicare):
        """A gap named in words no argument matches is one nobody can close."""
        import json as _json
        tools = {t["name"]: t for t in _json.loads(
            (medicare.root / medicare.suite / "mock-tools.json").read_text())}
        schema = {d["name"]: d for d in medicare._definitions}
        for name, tool in tools.items():
            for need, argument in (tool.get("completes_on") or {}).items():
                assert argument in (schema[name].get("parameters") or {}).get("properties", {}), f"{name}.{need}"
