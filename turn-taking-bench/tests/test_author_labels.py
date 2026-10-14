"""Labels -> TurnBench gold: the annotation tracks we write make ``turnbench.gold``
derive exactly the events the script plants, for every seed spec and for the
specific events each family is about."""

import json
import os
from pathlib import Path

import pytest
from turnbench.gold import events_for_conversation

from turn_taking_bench.author.build import build_conversation
from turn_taking_bench.author.labels import LABEL_BACKCHANNEL, LABEL_BARGE_IN, LABEL_TURN, as_conversation, compare
from turn_taking_bench.author.package import encode_audio
from turn_taking_bench.author.render import MockRenderer, RenderCache
from turn_taking_bench.author.spec import load_spec

SPECS = Path(__file__).resolve().parent.parent / "datasets" / "authored" / "specs"
# The leaderboard (test) and holdout templates live outside this repository. Point
# TURN_TAKING_BENCH_PRIVATE_SPECS at their specs directory to run the template-specific
# checks below; without it those checks skip and every public template is still checked.
PRIVATE_SPECS = Path(os.environ["TURN_TAKING_BENCH_PRIVATE_SPECS"]) if os.environ.get("TURN_TAKING_BENCH_PRIVATE_SPECS") else None
SPEC_PATHS = {path.stem: path for d in [SPECS, PRIVATE_SPECS] if d and d.is_dir() for path in sorted(d.glob("*.json"))}
SPEC_NAMES = sorted(SPEC_PATHS)
FULL_SET = len(SPEC_NAMES) == 50


class _Built(dict):
    def __missing__(self, name):
        pytest.skip(f"{name} is a private template; set TURN_TAKING_BENCH_PRIVATE_SPECS to run this check")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every spec file, first voice pairing only (the labels do not depend on the voices)."""
    cache = RenderCache(tmp_path_factory.mktemp("renders"))
    renderer = MockRenderer(sample_rate=48000)  # no resampling: offsets are exact
    built = _Built()
    for path in SPEC_PATHS.values():
        built[path.stem] = build_conversation(load_spec(path)[0], renderer, cache)
        encode_audio(built[path.stem])  # the tests read placements and labels, not audio: keep memory flat
    return built


def derived(conversation):
    conv = as_conversation(conversation.timeline.spec.conversation_id, conversation.tracks, conversation.timeline.duration_s)
    return events_for_conversation(conv)


def test_the_templates_expand_to_one_conversation_per_pairing_with_stable_ids():
    tb = [name for name in SPEC_NAMES if name.startswith("tb-")]
    nb = [name for name in SPEC_NAMES if name.startswith("nb-")]
    if FULL_SET:
        assert len(tb) == 16 and len(nb) == 34
    else:
        public = json.loads((SPECS.parent / "splits.json").read_text())["dev"]
        assert sorted(load_spec(p)[0].id for p in SPEC_PATHS.values()) == sorted(public)
    specs = [spec for path in SPEC_PATHS.values() for spec in load_spec(path)]
    pairings = ["f-us/m-us", "m-us/f-gb", "f-gb/f-us", "m-in/f-us", "f-au/m-us"]
    n_pairings = len({spec.pairing for spec in specs})
    assert set(spec.pairing for spec in specs) == set(pairings[:n_pairings])
    # id scheme: series (9 = TB, 8 = NB) + two-digit template + pairing index, TB ids unchanged from v0.3
    templates = sorted({spec.conversation_id[:-1] for spec in specs})
    if FULL_SET:
        assert templates == sorted([f"9{n:02d}" for n in range(1, 17)] + [f"8{n:02d}" for n in range(1, 35)])
    expected = [f"{t}{i}" for t in templates for i in range(1, n_pairings + 1)]
    assert sorted(spec.conversation_id for spec in specs) == sorted(expected)
    assert {spec.family for spec in specs} == {"pauses", "long_turns", "turn_ends", "backchannels", "barge_in"}
    assert all(spec.difficulty in ("easy", "medium", "hard") for spec in specs)
    assert all(spec.subtype for spec in specs if spec.id.startswith("NB-"))
    assert all(spec.items[0].speaker == "agent" for spec in specs)


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_turnbench_gold_derives_exactly_the_planted_events(built, name):
    conversation = built[name]
    assert compare(conversation.planted, derived(conversation)) == []


def test_pause_ladder_plants_one_negative_per_break_and_one_positive_per_line(built):
    conversation = built["tb-01-pause-ladder"]
    planted, events = conversation.planted, derived(conversation)
    assert planted.counts(1) == {"eot_pos": 5, "eot_neg": 4, "int_pos": 0, "int_neg": 0}
    assert planted.counts(2) == {"eot_pos": 6, "eot_neg": 0, "int_pos": 0, "int_neg": 0}
    pauses = sorted(round(span.end - span.start, 2) for span in events.eot_negative_spans)
    assert pauses == [1.0, 1.5, 2.0, 2.5]
    assert all(label == LABEL_TURN for label in {e["label"] for e in conversation.tracks[1]})


def test_backchannels_are_int_negatives_and_never_turns(built):
    conversation = built["tb-07-backchannels"]
    events = derived(conversation)
    a1, b1 = conversation.timeline.item("a1"), conversation.timeline.item("b1")
    rate = conversation.timeline.sample_rate
    assert [e["label"] for e in conversation.tracks[1] if e["text"] in ("Mm-hmm.", "Okay.")] == [LABEL_BACKCHANNEL] * 2
    assert events.int_positive_events == []
    assert [(span.speaker, round(span.start, 3)) for span in events.int_negative_spans][0] == (1, round(b1.speech_start / rate, 3))
    assert b1.speech_start == a1.speech_start + int(1.5 * rate)
    # The agent's long answer still ends with a real EOT: the backchannel did not take the floor.
    assert any(e.speaker == 2 and abs(e.time_s - a1.speech_end / rate) < 1e-6 for e in events.eot_positive_events)
    assert events.eot_negative_spans == []


def test_barge_in_is_an_int_positive_and_the_cut_agent_turn_ends_there(built):
    conversation = built["tb-09-barge-in-redirect"]
    events = derived(conversation)
    tl = conversation.timeline
    rate = tl.sample_rate
    a1, x1, c2 = tl.item("a1"), tl.item("x1"), tl.item("c2")
    assert [e["label"] for e in conversation.tracks[1] if e["text"].startswith("Sorry, stop")] == [LABEL_BARGE_IN]
    assert [(e.speaker, e.time_s) for e in events.int_positive_events] == [(1, x1.speech_start / rate)]
    assert a1.cut_at == x1.speech_start + int(0.4 * rate)
    assert (2, a1.cut_at / rate) in {(e.speaker, e.time_s) for e in events.eot_positive_events}
    # "...one quick question." then the question: the floor stays with the caller, so that gap is a pause.
    assert [(s.speaker, s.start, s.end) for s in events.eot_negative_spans] == [(1, x1.speech_end / rate, c2.speech_start / rate)]
    assert events.eot_excluded == [] and events.int_excluded == []


def test_hold_after_a_barge_in_is_one_long_mid_turn_pause_with_a_silent_agent(built):
    conversation = built["tb-12-hold-on-silence-go-on"]
    tl, events = conversation.timeline, derived(conversation)
    x1, c2 = tl.item("x1"), tl.item("c2")
    rate = tl.sample_rate
    assert c2.speech_start == x1.speech_end + int(6.0 * rate)  # hold_s replaces the drawn gap exactly
    pauses = [(s.speaker, round(s.end - s.start, 3)) for s in events.eot_negative_spans]
    assert pauses == [(1, 6.0)]
    # Nothing on the agent channel between the cut and the caller's resumption.
    assert all(p.speech_end <= x1.speech_start + int(0.3 * rate) + 1 or p.speech_start >= c2.speech_end for p in tl.items if p.channel == 2)


def test_barge_in_on_the_greeting_cuts_the_agents_first_turn(built):
    conversation = built["tb-11-barge-in-on-greeting"]
    tl, events = conversation.timeline, derived(conversation)
    g, x1 = tl.item("g"), tl.item("x1")
    rate = tl.sample_rate
    assert tl.items[0] is g and g.cut_at == x1.speech_start + int(0.5 * rate) and g.dropped_segments == []
    assert [(e.speaker, e.time_s) for e in events.int_positive_events] == [(1, x1.speech_start / rate)]
    assert (2, g.cut_at / rate) in {(e.speaker, e.time_s) for e in events.eot_positive_events}


def test_same_two_words_label_as_backchannel_then_as_barge_in(built):
    conversation = built["tb-16-two-words-opposite-intent"]
    events = derived(conversation)
    labels = {e["text"]: e["label"] for e in conversation.tracks[1]}
    assert labels["Right, okay."] == LABEL_BACKCHANNEL and labels["No, stop."] == LABEL_BARGE_IN
    assert len(events.int_negative_spans) == 1 and len(events.int_positive_events) == 1
    assert conversation.timeline.item("a1").cut_at is None and conversation.timeline.item("a2").cut_at is not None


def test_back_to_back_barge_ins_each_cut_their_own_agent_turn(built):
    conversation = built["tb-10-back-to-back-barge-ins"]
    tl, events = conversation.timeline, derived(conversation)
    assert tl.item("a1").cut_at is not None and tl.item("a2").cut_at is not None and tl.item("a3").cut_at is None
    assert [e.speaker for e in events.int_positive_events] == [1, 1]
    assert conversation.planted.counts(1)["int_pos"] == 2


def test_short_clause_pauses_are_labelled_with_the_lowered_minimum(built):
    conversation = built["tb-05-long-fact-dense"]
    assert conversation.planted.counts(1) == {"eot_pos": 3, "eot_neg": 5, "int_pos": 0, "int_neg": 0}
    assert sorted(round(r["scripted_s"], 2) for r in conversation.pauses["c1"] if r["kind"] == "break") == [0.5, 0.6, 0.6, 0.7, 0.7]


def test_turn_gaps_are_drawn_from_the_range_and_differ_between_turns(built):
    conversation = built["tb-01-pause-ladder"]
    truth = conversation.timeline.ground_truth()
    gaps = [item["gap_s"] for item in truth["items"][1:] if item["kind"] == "turn"]
    assert gaps and all(0.5 <= gap <= 1.0 for gap in gaps) and len(set(gaps)) > 1
