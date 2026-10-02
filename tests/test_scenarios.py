"""Scenarios are data with rules; the rules are tested like code."""

from __future__ import annotations

from service.corpus_v1 import build
from service.scenarios import SCENARIOS, ScenarioSpec, by_id


class TestTheSet:
    def test_every_clip_a_scenario_names_exists_in_the_corpus(self):
        ids = set(build().specs)
        for scenario in SCENARIOS:
            for clip in (scenario.opener, scenario.identify, scenario.fallback):
                assert clip in ids, (scenario.id, clip)
            for _needles, clip in scenario.routes:
                assert clip in ids or clip.startswith("__"), (scenario.id, clip)

    def test_ids_are_unique_and_filesystem_safe(self):
        ids = [s.id for s in SCENARIOS]
        assert len(ids) == len(set(ids))
        assert all("/" not in i and " " not in i for i in ids)

    def test_the_set_covers_both_contracts_and_the_minimum_breadth(self):
        assert sum(1 for s in SCENARIOS if s.contract == "appointments") >= 12
        assert sum(1 for s in SCENARIOS if s.contract == "medicare") >= 3

    def test_every_scenario_can_end(self):
        """A scenario with no end condition runs to max_turns every time."""
        for scenario in SCENARIOS:
            assert scenario.done_when_called or scenario.done_when_said, scenario.id


class TestRouting:
    def test_scenario_routes_win_over_the_shared_ones(self):
        full = by_id("book.fullday")
        assert full.route("What day would you like to come in?") == "task.date.july10"
        assert full.route("I'm sorry, July tenth is fully booked. Is there another date?") == "task.date.july9"

    def test_placeholders_resolve_per_scenario(self):
        cancel = by_id("cancel.pick")
        assert cancel.route("Sure, how can I help you today?") == "open.cancel"
        assert cancel.route("Can I get the phone number on your account?") == "identify.robert"

    def test_finished_by_tool_or_by_words(self):
        assert by_id("book.morning").finished("", [{"name": "book_appointment"}])
        assert by_id("guardrail.emergency").finished("Please hang up and dial 911 right now.", [])
        assert not by_id("guardrail.emergency").finished("What is your phone number?", [])


class TestSerialisation:
    def test_round_trip_through_json_preserves_every_field(self):
        for scenario in SCENARIOS:
            again = ScenarioSpec.from_json(scenario.as_json())
            assert again == scenario
