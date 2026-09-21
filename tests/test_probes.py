"""Probe-level invariants that do not need a provider.

The routing table is a published part of the scenario, so it is tested like code
rather than trusted like a prompt. A caller that improvised its next line with a
language model would put a second model inside the measurement, and two providers
would then be scored partly on how well our caller understood them.
"""

from __future__ import annotations

import pytest

from service.probes import (
    BackchannelTolerance,
    BargeIn,
    BookingTask,
    EndpointingLadder,
    FalseTrigger,
    ResponseLatency,
    route_booking_reply,
)


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
        for probe in (ResponseLatency(), EndpointingLadder(800), BargeIn(), BookingTask(), FalseTrigger()):
            assert "/" not in probe.slug and " " not in probe.slug
