"""Composer: placements land on exact samples, a barge-in cuts the agent at
onset + stop_delay, a backchannel leaves the agent alone."""

import numpy as np
import pytest

from turn_bench.author.bounds import speech_segments
from turn_bench.author.compose import OUTPUT_RATE, ComposeError, compose
from turn_bench.author.render import MockRenderer, Render
from turn_bench.author.spec import SpecError, spec_from_json

RATE = OUTPUT_RATE


def make_spec(items, start_s=0.5, default_gap_s=0.7):
    return spec_from_json(
        {
            "schema_version": 2,
            "id": "T",
            "conversation_id": "1",
            "title": "t",
            "voices": {"caller": "f-us", "agent": "m-us"},
            "timing": {"start_s": start_s, "tail_s": 1.0, "default_gap_s": default_gap_s},
            "items": items,
        }
    )


def render_all(spec, renderer):
    """Mock renders at the output rate, so sample offsets are exact and no resampling runs."""
    out = {}
    for item in spec.items:
        samples, rate = renderer.render(item.text, spec.voices[item.speaker])
        out[item.id] = (Render(samples, rate, "sha", {}), speech_segments(samples, rate))
    return out


@pytest.fixture
def renderer():
    return MockRenderer(sample_rate=RATE)


def s(seconds):
    return int(round(seconds * RATE))


def test_turn_items_follow_the_anchor_speech_end_by_the_gap(renderer):
    spec = make_spec(
        [
            {"id": "c1", "speaker": "caller", "text": "Hello there. <1s> Can you help?"},
            {"id": "a1", "speaker": "agent", "text": "Of course I can.", "gap_s": 0.8},
            {"id": "c2", "speaker": "caller", "text": "Thanks a lot."},
        ]
    )
    timeline = compose(spec, render_all(spec, renderer))
    c1, a1, c2 = timeline.items
    assert c1.speech_start == s(0.5)
    assert len(c1.segments) == 2 and c1.segments[1].start - c1.segments[0].end == s(1.0)
    assert a1.speech_start == c1.speech_end + s(0.8)
    assert c2.speech_start == a1.speech_end + s(0.7)  # default gap
    # Audio is where the ground truth says: speech frames loud, the gap silent, on both channels.
    assert np.abs(timeline.channels[1][c1.segments[0].end : c1.segments[1].start]).max() == 0.0
    assert np.abs(timeline.channels[1][c1.segments[1].start : c1.segments[1].start + s(0.05)]).max() > 0.1
    assert np.abs(timeline.channels[2][a1.speech_start : a1.speech_start + s(0.05)]).max() > 0.1
    assert np.abs(timeline.channels[1][a1.speech_start : a1.speech_end]).max() == 0.0
    assert len(timeline.channels[1]) == len(timeline.channels[2]) == c2.speech_end + s(1.0)


def test_backchannel_is_placed_at_the_offset_and_leaves_the_agent_alone(renderer):
    spec = make_spec(
        [
            {"id": "c1", "speaker": "caller", "text": "Tell me everything please."},
            {"id": "a1", "speaker": "agent", "text": "Well there is a great deal to say about all of this and I will go through it slowly."},
            {"id": "b1", "speaker": "caller", "kind": "backchannel", "text": "Mm-hmm.", "into": "a1", "offset_s": 1.5},
            {"id": "c2", "speaker": "caller", "text": "Thanks."},
        ]
    )
    timeline = compose(spec, render_all(spec, renderer))
    a1, b1, c2 = timeline.items[1:]
    assert b1.speech_start == a1.speech_start + s(1.5)
    assert a1.cut_at is None and a1.dropped_segments == []
    assert c2.speech_start == a1.speech_end + s(0.7)  # anchored on the agent, which ends last


def test_barge_in_cuts_the_agent_after_stop_delay_with_a_fade(renderer):
    spec = make_spec(
        [
            {"id": "c1", "speaker": "caller", "text": "Tell me everything please."},
            {"id": "a1", "speaker": "agent", "text": "Well there is a great deal to say about all of this and I will go through it slowly."},
            {"id": "x1", "speaker": "caller", "kind": "barge_in", "text": "Sorry, stop.", "into": "a1", "offset_s": 1.5, "stop_delay_s": 0.4, "fade_s": 0.02},
            {"id": "c2", "speaker": "caller", "text": "One question.", "gap_s": 0.8},
        ]
    )
    renders = render_all(spec, renderer)
    natural_end = renders["a1"][1].segments[0].end_s
    timeline = compose(spec, renders)
    a1, x1, c2 = timeline.items[1:]
    cut = a1.speech_start + s(1.5) + s(0.4)
    assert x1.speech_start == a1.speech_start + s(1.5)
    assert a1.cut_at == cut and a1.speech_end == cut
    assert cut < a1.offset + s(natural_end)  # the agent really was cut short
    agent = timeline.channels[2]
    assert np.abs(agent[cut:]).max() == 0.0
    fade = agent[cut - s(0.02) : cut]
    assert np.abs(fade[-24:]).max() < np.abs(fade[:24]).max()  # fading out towards the cut
    assert np.abs(agent[cut - s(0.2) : cut - s(0.02)]).max() > 0.1  # still speaking before the fade
    assert c2.speech_start == x1.speech_end + s(0.8)  # the barge-in ends last, so it anchors c2


def test_cut_inside_a_pause_drops_the_segments_after_it(renderer):
    spec = make_spec(
        [
            {"id": "c1", "speaker": "caller", "text": "Go ahead."},
            {"id": "a1", "speaker": "agent", "text": "First part here. <1s> Second part here. <1s> Third part."},
            {"id": "x1", "speaker": "caller", "kind": "barge_in", "text": "Stop.", "into": "a1", "offset_s": 0.5, "stop_delay_s": 0.3},
        ]
    )
    timeline = compose(spec, render_all(spec, renderer))
    a1 = timeline.item("a1")
    assert len(a1.segments) == 1 and len(a1.dropped_segments) == 2
    assert a1.segments[0].end == a1.cut_at


def test_same_channel_overlap_and_segment_count_mismatch_are_errors(renderer):
    spec = make_spec(
        [
            {"id": "c1", "speaker": "caller", "text": "Hello there, how are you doing today?"},
            {"id": "c2", "speaker": "caller", "text": "Hi.", "after": "c1", "gap_s": 0.0},
            {"id": "a1", "speaker": "agent", "text": "Fine thanks."},
            {"id": "c3", "speaker": "caller", "kind": "backchannel", "text": "Mm.", "into": "a1", "offset_s": 0.0},
            {"id": "c4", "speaker": "caller", "kind": "backchannel", "text": "Mm.", "into": "a1", "offset_s": 0.05},
        ]
    )
    with pytest.raises(ComposeError, match="overlap on the same channel"):
        compose(spec, render_all(spec, renderer))

    spec = make_spec([{"id": "c1", "speaker": "caller", "text": "One. <1s> Two."}])
    renders = render_all(spec, renderer)
    render, _bounds = renders["c1"]
    renders["c1"] = (render, speech_segments(render.samples, render.sample_rate, min_pause_s=5.0))
    with pytest.raises(ComposeError, match="2 scripted segment\\(s\\) but the render has 1"):
        compose(spec, renders)


def test_barge_in_after_the_agent_has_finished_is_an_error():
    """A barge-in whose cut lands after the agent's speech ends would not overlap
    the agent at all: the build must refuse it rather than label a non-overlap
    as an interruption."""
    from turn_bench.author.bounds import scripted_segments
    from turn_bench.author.build import render_item
    from turn_bench.author.compose import ComposeError, compose
    from turn_bench.author.render import MockRenderer, RenderCache
    from turn_bench.author.spec import SpecError, spec_from_json

    raw = {
        "schema_version": 2, "id": "X", "conversation_id": "1", "title": "t",
        "voices": {"caller": "f-us", "agent": "m-us"},
        "items": [
            {"id": "a1", "speaker": "agent", "text": "Short reply."},
            {"id": "x1", "speaker": "caller", "kind": "barge_in", "text": "Wait.", "into": "a1", "offset_s": 5.0},
        ],
    }
    spec = spec_from_json(raw)
    cache = RenderCache(__import__("tempfile").mkdtemp())
    renders = {}
    for item in spec.items:
        render = render_item(cache, MockRenderer(), spec.voices[item.speaker], item)
        renders[item.id] = (render, scripted_segments(render.samples, render.sample_rate, len(item.segments)))
    with pytest.raises(ComposeError, match="lands after its speech ends"):
        compose(spec, renders)
    raw["items"][1]["offset_s"] = 0.2
    spec = spec_from_json(raw)
    timeline = compose(spec, renders)
    truth = timeline.ground_truth()["items"][0]
    assert truth["cut_at_s"] is not None and truth["uncut_end_s"] > truth["end_s"]
    # Placed from the agent's speech end: the onset is exactly that far before the uncut end.
    del raw["items"][1]["offset_s"]
    raw["items"][1]["offset_from_end_s"] = 0.5
    spec = spec_from_json(raw)
    timeline = compose(spec, renders)
    agent, barge = timeline.ground_truth()["items"]
    assert barge["onset_s"] == pytest.approx(agent["uncut_end_s"] - 0.5, abs=1e-6)
    assert barge["offset_from_end_s"] == pytest.approx(0.5, abs=1e-6) and agent["cut_at_s"] == pytest.approx(barge["onset_s"] + 0.4, abs=1e-6)
    raw["items"][1]["offset_s"] = 1.0
    with pytest.raises(SpecError, match="replaces offset_s"):
        spec_from_json(raw)
    # A backchannel placed after the agent's speech end is refused the same way.
    raw["items"][1] = {"id": "b1", "speaker": "caller", "kind": "backchannel", "text": "Mm-hmm.", "into": "a1", "offset_s": 5.0}
    spec = spec_from_json(raw)
    render = render_item(cache, MockRenderer(), "f-us", spec.items[1])
    renders["b1"] = (render, scripted_segments(render.samples, render.sample_rate, 1))
    with pytest.raises(ComposeError, match="backchannel onset .* lands after"):
        compose(spec, renders)
