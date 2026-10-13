"""Boundary finder: on mock renders every segment boundary (including the ones
``<break/>`` markers create) is recovered within one 10 ms frame."""

import pytest

from turn_bench.author.bounds import speech_segments
from turn_bench.author.render import MockRenderer

FRAME_S = 0.01

LINES = [
    "Good morning. I have a question about my order. <1s> It has not arrived yet.",
    "It was due on Monday. <break time=\"1.5s\"/> The tracking page has not changed since.",
    "That is the main thing. <2.5s> Could you also check the delivery address?",
    "Okay, that's all I needed. Thanks, bye.",
    "Mm-hmm.",
]


@pytest.mark.parametrize("sample_rate", [24000, 48000])
@pytest.mark.parametrize("text", LINES)
def test_segments_match_the_mock_layout_within_one_frame(text, sample_rate):
    renderer = MockRenderer(sample_rate=sample_rate)
    samples, rate = renderer.render(text, "f-us")
    expected = [(s / rate, e / rate) for s, e in renderer.layout(text, "f-us")]
    found = speech_segments(samples, rate).segments
    assert len(found) == len(expected) == text.count("<") + 1
    for (start, end), segment in zip(expected, found):
        assert abs(segment.start_s - start) <= FRAME_S + 1e-9
        assert abs(segment.end_s - end) <= FRAME_S + 1e-9
        assert segment.start_s <= start + 1e-9  # the frame containing the onset, never later
        assert segment.end_s >= end - 1e-9  # the frame containing the last sample, never earlier


def test_short_pauses_do_not_split_and_long_ones_do():
    renderer = MockRenderer()
    text = "One two three. <300ms> Four five. <900ms> Six."
    samples, rate = renderer.render(text, "m-us")
    assert len(speech_segments(samples, rate, min_pause_s=0.6).segments) == 2
    assert len(speech_segments(samples, rate, min_pause_s=0.2).segments) == 3
