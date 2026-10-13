"""Count-driven split (``scripted_segments``): the script says how many pauses a
line has; the longest quiet gaps are the pauses, natural gaps stay inside a
segment, and a break that did not render as a pause is a clear error."""

import pytest

from turn_bench.author.bounds import MIN_PAUSE_S, BoundsError, scripted_segments, speech_segments
from turn_bench.author.render import MockRenderer

FRAME_S = 0.01


def render(text):
    return MockRenderer().render(text, "m-us")


def test_natural_pause_longer_than_the_minimum_stays_inside_the_turn():
    # A 700 ms "natural" gap and a scripted 1 s break: one scripted segment boundary.
    text = "First sentence here. <700ms> Second sentence here. <1s> After the break."
    samples, rate = render(text)
    assert len(speech_segments(samples, rate).segments) == 3  # threshold-only would over-split
    bounds = scripted_segments(samples, rate, 2)
    layout = MockRenderer().layout(text, "m-us")
    assert len(bounds.segments) == 2
    assert abs(bounds.segments[0].end_s - layout[1][1] / rate) <= FRAME_S + 1e-9
    assert abs(bounds.segments[1].start_s - layout[2][0] / rate) <= FRAME_S + 1e-9
    assert len(bounds.unscripted_pauses) == 1
    assert bounds.unscripted_pauses[0].end_s - bounds.unscripted_pauses[0].start_s == pytest.approx(0.7, abs=2 * FRAME_S)
    # With no scripted pause at all the whole line is one segment and the gap is recorded.
    one = scripted_segments(samples, rate, 1)
    assert len(one.segments) == 1 and len(one.unscripted_pauses) == 2


def test_break_that_rendered_too_short_is_rejected_with_the_gap_length():
    samples, rate = render("Before the break. <400ms> After the break.")
    with pytest.raises(BoundsError, match=r"only 0 gap\(s\) of >= 0.60s"):
        scripted_segments(samples, rate, 2)


def test_ambiguous_natural_gap_is_rejected():
    samples, rate = render("One. <900ms> Two. <1s> Three.")
    with pytest.raises(BoundsError, match="ambiguous"):
        scripted_segments(samples, rate, 2)
    assert len(scripted_segments(samples, rate, 3).segments) == 3  # both scripted: fine


def test_more_segments_than_runs_is_rejected():
    samples, rate = render("Mm-hmm.")
    with pytest.raises(BoundsError, match="only 1 run"):
        scripted_segments(samples, rate, 2)
    assert MIN_PAUSE_S == 0.6
