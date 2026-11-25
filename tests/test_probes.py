"""Probe-level invariants that do not need a provider.

The routing table is a published part of the scenario, so it is tested like code
rather than trusted like a prompt. A caller that improvised its next line with a
language model would put a second model inside the measurement, and two providers
would then be scored partly on how well our caller understood them.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from service.probes import (
    BackchannelTolerance,
    BargeIn,
    EndpointingLadder,
    FalseTrigger,
    ResponseLatency,
    TaskScenario,
    made_calls,
    pump_tools,
)
from service.scenarios import by_id

route_booking_reply = by_id("book.morning").route


class TestRouting:
    @pytest.mark.parametrize(
        "asked,expected",
        [
            ("To get started, what phone number should I use?", "task.identify"),
            ("What's the number on your account?", "task.identify"),
            ("What is the reason for your visit, James?", "task.reason"),
            ("What brings you in?", "task.reason"),
            ("What day would you like to come in?", "task.date"),
            ("Which date works for you?", "task.date"),
            ("Would you like to book that time?", "task.confirm"),
            ("I have Dr. Lee at 9 AM. Does that work?", "task.confirm"),
            ("Shall I go ahead?", "task.confirm"),
        ],
    )
    def test_the_caller_answers_what_was_actually_asked(self, asked, expected):
        assert route_booking_reply(asked) == expected

    def test_an_unrecognised_question_falls_back_to_assent(self):
        """Better to say yes than to stall: a stalled caller scores the script."""
        assert route_booking_reply("Lovely weather today.") == "task.confirm"
        assert route_booking_reply("") == "task.confirm"

    def test_a_greeting_gets_the_request_restated(self):
        """An agent that greets instead of answering has not heard the request yet."""
        assert route_booking_reply("Thanks for calling, how can I help you today?") == "open.book"

    def test_a_date_question_phrased_as_an_offer_still_asks_for_the_date(self):
        """"What day would you like" contains "would you like"; order decides."""
        assert route_booking_reply("What day would you like to come in?") == "task.date"


class TestSlugs:
    def test_every_ladder_rung_gets_its_own_artifact_directory(self):
        slugs = {EndpointingLadder(gap_ms=gap).slug for gap in (400, 600, 800, 1000, 1500, 2000)}
        assert len(slugs) == 6, "rungs sharing a slug would overwrite each other's audio"

    def test_parameterised_probes_encode_their_parameter(self):
        assert BargeIn(after_onset_ms=900).slug != BargeIn(after_onset_ms=400).slug
        assert BackchannelTolerance(after_onset_ms=400).slug != BackchannelTolerance(after_onset_ms=900).slug
        assert FalseTrigger(level_dbfs=-30).slug != FalseTrigger(level_dbfs=-20).slug
        assert ResponseLatency(clip_id="open.book").slug != ResponseLatency(clip_id="open.reschedule").slug

    def test_slugs_are_filesystem_safe(self):
        for probe in (ResponseLatency(), EndpointingLadder(800), BargeIn(), TaskScenario(), FalseTrigger()):
            assert "/" not in probe.slug and " " not in probe.slug


class TestWithdrawnToolCalls:
    """A provider can withdraw a call it already sent (the caller kept talking
    before the turn was confirmed). It stays in the record but is never answered
    and never counts toward a scenario."""

    @staticmethod
    def _ctx():
        answered: list[str] = []

        async def send_tool_result(call_id, _output):
            answered.append(call_id)

        adapter = SimpleNamespace(
            tool_calls=[
                {"name": "book_appointment", "call_id": "fc-1", "arguments": {}, "cancelled": True},
                {"name": "lookup_patient", "call_id": "fc-2", "arguments": {}},
            ],
            send_tool_result=send_tool_result,
        )
        tools = SimpleNamespace(call=lambda _name, _arguments: {"ok": True})
        return SimpleNamespace(adapter=adapter, tools=tools), answered

    async def test_the_pump_does_not_answer_a_withdrawn_call(self):
        ctx, answered = self._ctx()
        stop = asyncio.Event()
        pump = asyncio.create_task(pump_tools(ctx, stop))
        await asyncio.sleep(0.05)
        stop.set()
        await pump
        assert answered == ["fc-2"]

    def test_a_withdrawn_call_does_not_count_toward_the_scenario(self):
        ctx, _ = self._ctx()
        assert [c["call_id"] for c in made_calls(ctx.adapter)] == ["fc-2"]
