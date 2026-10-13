"""Spec schema v2: voice-pairing expansion, seeded gap draws, scripted holds,
per-line speed and the labelled-pause minimum reaching the bounds finder."""

import pytest

from turn_taking_bench.author.build import BuildError, build_conversation
from turn_taking_bench.author.render import MockRenderer, RenderCache
from turn_taking_bench.author.spec import SpecError, spec_from_json, specs_from_json

BASE = {"schema_version": 2, "id": "T", "conversation_id": "77", "title": "t"}
PAIRINGS = [{"caller": "f-us", "agent": "m-us"}, {"caller": "m-us", "agent": "f-gb"}]
ITEMS = [
    {"id": "g", "speaker": "agent", "text": "Hello there."},
    {"id": "c1", "speaker": "caller", "text": "Hi. <1s> One thing."},
    {"id": "a1", "speaker": "agent", "text": "Go ahead."},
    {"id": "c2", "speaker": "caller", "text": "Thanks.", "hold_s": 3.0},
]


def test_pairings_expand_to_numbered_conversations_with_the_same_script():
    specs = specs_from_json({**BASE, "pairings": PAIRINGS, "items": ITEMS})
    assert [s.conversation_id for s in specs] == ["771", "772"]
    assert [s.pairing for s in specs] == ["f-us/m-us", "m-us/f-gb"]
    assert specs[0].items == specs[1].items and specs[0].id == specs[1].id == "T"
    single = spec_from_json({**BASE, "voices": PAIRINGS[0], "items": ITEMS})
    assert single.conversation_id == "77" and single.pairing is None
    with pytest.raises(SpecError, match="exactly one of"):
        specs_from_json({**BASE, "voices": PAIRINGS[0], "pairings": PAIRINGS, "items": ITEMS})
    with pytest.raises(SpecError, match="duplicate pairing"):
        specs_from_json({**BASE, "pairings": [PAIRINGS[0], PAIRINGS[0]], "items": ITEMS})
    with pytest.raises(SpecError, match="describes 2"):
        spec_from_json({**BASE, "pairings": PAIRINGS, "items": ITEMS})
    with pytest.raises(SpecError, match="schema_version"):
        specs_from_json({**BASE, "schema_version": 1, "voices": PAIRINGS[0], "items": ITEMS})


def test_gaps_are_seeded_draws_from_the_range_and_holds_replace_them():
    timing = {"gap_range_s": [0.5, 1.0], "gap_seed": 3}
    specs = specs_from_json({**BASE, "pairings": PAIRINGS, "timing": timing, "items": ITEMS})
    first, second = specs
    gaps = [first.gap_for(first.item(i)) for i in ("c1", "a1")]
    assert all(0.5 <= gap <= 1.0 for gap in gaps) and gaps[0] != gaps[1]
    assert gaps == [first.gap_for(first.item(i)) for i in ("c1", "a1")]  # deterministic
    assert gaps != [second.gap_for(second.item(i)) for i in ("c1", "a1")]  # but per conversation
    other_seed = specs_from_json({**BASE, "pairings": PAIRINGS, "timing": {**timing, "gap_seed": 4}, "items": ITEMS})[0]
    assert gaps != [other_seed.gap_for(other_seed.item(i)) for i in ("c1", "a1")]
    assert first.gap_for(first.item("c2")) == 3.0  # hold_s: exactly the scripted dead air
    plain = spec_from_json({**BASE, "voices": PAIRINGS[0], "timing": {"default_gap_s": 0.7}, "items": ITEMS})
    assert plain.gap_for(plain.item("c1")) == 0.7
    with pytest.raises(SpecError, match="gap_range_s"):
        specs_from_json({**BASE, "pairings": PAIRINGS, "timing": {"gap_range_s": [1.0, 0.5]}, "items": ITEMS})
    with pytest.raises(SpecError, match="hold_s"):
        spec_from_json({**BASE, "voices": PAIRINGS[0], "items": [{"id": "c", "speaker": "caller", "text": "Hi.", "hold_s": 2, "gap_s": 1}]})
    with pytest.raises(SpecError, match="only for turn"):
        spec_from_json({**BASE, "voices": PAIRINGS[0], "items": ITEMS + [{"id": "b", "speaker": "caller", "kind": "backchannel", "text": "Mm.", "into": "a1", "hold_s": 1}]})


def test_speed_and_min_pause_are_validated_and_reach_the_render_and_the_bounds(tmp_path):
    with pytest.raises(SpecError, match="speed"):
        spec_from_json({**BASE, "voices": PAIRINGS[0], "items": [{"id": "c", "speaker": "caller", "text": "Hi.", "speed": 0.5}]})
    with pytest.raises(SpecError, match="min_pause_s"):
        spec_from_json({**BASE, "voices": PAIRINGS[0], "items": [{"id": "c", "speaker": "caller", "text": "Hi.", "min_pause_s": 0}]})
    cache = RenderCache(tmp_path)
    renderer = MockRenderer(sample_rate=48000)
    short = {"id": "c1", "speaker": "caller", "text": "One clause here, <400ms> and another one. <450ms> Done now?"}
    strict = spec_from_json({**BASE, "voices": PAIRINGS[0], "items": [short]})
    with pytest.raises(BuildError, match="below the minimum"):
        build_conversation(strict, renderer, cache)
    lenient = spec_from_json({**BASE, "voices": PAIRINGS[0], "items": [{**short, "min_pause_s": 0.3}]})
    built = build_conversation(lenient, renderer, cache)
    assert built.planted.counts(1) == {"eot_pos": 1, "eot_neg": 2, "int_pos": 0, "int_neg": 0}
    slow = spec_from_json({**BASE, "voices": PAIRINGS[0], "items": [{"id": "c1", "speaker": "caller", "text": "Hello there, slowly.", "speed": 0.8}]})
    normal = spec_from_json({**BASE, "voices": PAIRINGS[0], "items": [{"id": "c1", "speaker": "caller", "text": "Hello there, slowly."}]})
    slow_built, normal_built = build_conversation(slow, renderer, cache), build_conversation(normal, renderer, cache)
    assert slow_built.timeline.item("c1").speech_end - slow_built.timeline.item("c1").speech_start > normal_built.timeline.item("c1").speech_end - normal_built.timeline.item("c1").speech_start
    assert slow_built.timeline.ground_truth()["items"][0]["speed"] == 0.8


def test_spliced_breaks_are_exact_silence_between_separately_rendered_segments(tmp_path):
    cache = RenderCache(tmp_path)
    renderer = MockRenderer(sample_rate=48000)
    text = "Um, <1s> so, <1.2s> I wanted to ask."
    spliced = spec_from_json({**BASE, "voices": PAIRINGS[0], "items": [{"id": "c1", "speaker": "caller", "text": text, "splice_breaks": True}]})
    built = build_conversation(spliced, renderer, cache)
    assert cache.misses == 3  # one render per segment
    reports = built.pauses["c1"]
    breaks = [r for r in reports if r["kind"] == "break"]
    assert [(r["scripted_s"], r["spliced"]) for r in breaks] == [(1.0, True), (1.2, True)]
    assert [r["detected_s"] for r in breaks] == pytest.approx([1.0, 1.2], abs=0.02)  # exact to the frame grid
    assert built.planted.counts(1) == {"eot_pos": 1, "eot_neg": 2, "int_pos": 0, "int_neg": 0}
    truth = built.timeline.ground_truth()["items"][0]
    assert len(truth["render_sha256"]) == 64 and truth["segments"][1]["text"] == "so,"
    with pytest.raises(SpecError, match="splice_breaks"):
        spec_from_json({**BASE, "voices": PAIRINGS[0], "items": [{"id": "c1", "speaker": "caller", "text": text, "splice_breaks": "yes"}]})
