"""Call flow from word timings: waits, talking over the caller, jumping into a pause, slow stops; and how the call
score takes it in. Synthetic transcripts only."""

import json

import pytest

from conftest import config_dict
from s2s_human_bench import rating
from s2s_human_bench.calls import family_scores
from s2s_human_bench.config import load_config
from s2s_human_bench.flow import measure

SETTINGS = {
    "backchannelMaxWords": 2,
    "backchannelListMaxWords": 4,
    "backchannelWords": ["mhm", "okay", "yeah", "sure", "right"],
    "waits": [[3.0, 0.5], [5.0, 1.0], [8.0, 2.0]],
    "stops": [[2.0, 0.5], [3.0, 1.0]],
    "talkOverSeconds": 0.5,
    "pauseJumpSeconds": 1.0,
    "longWaitSeconds": 5.0,
    "slowStopSeconds": 2.0,
    "holdPhrases": ["hold on", "one moment"],
    "stopPauseSeconds": 0.6,
    "splitGapSeconds": 1.0,
    "stopMinWords": 4,
    "talkOverWeight": 1.0,
    "pauseJumpWeight": 1.0,
    "checkInPhrases": ["are you still there"],
}


def turn(role: str, text: str, start: float, end: float) -> dict:
    words = text.split()
    step = (end - start) / len(words)
    return {
        "role": role,
        "content": text,
        "start_time": start,
        "end_time": end,
        "words": [{"text": w, "start": round(start + i * step, 2), "end": round(start + (i + 1) * step, 2)} for i, w in enumerate(words)],
    }


def caller(text, start, end):
    return turn("Testing Agent", text, start, end)


def agent(text, start, end):
    return turn("Main Agent", text, start, end)


def flow(*turns):
    return measure(list(turns), "Main", "Testing", SETTINGS)


def test_silences_are_weighted_by_length():
    f = flow(caller("I need to move my appointment", 0, 2), agent("Sure, which day works for you", 8.5, 10))
    assert f.replies == 1 and f.waits == [6.5] and f.silences == [6.5] and f.problems == 1.0 and f.clean == 0.0
    assert f.events == [("wait", 2.0, 6.5), ("silence", 2.0, 6.5)]
    f = flow(caller("I need to move my appointment", 0, 2), agent("Sure, which day works for you", 3.0, 4.5))
    assert f.waits == [1.0] and f.silences == [] and f.problems == 0 and f.clean == 100.0


def test_silence_after_an_acknowledgement_counts():
    f = flow(caller("What is my balance please", 0, 2), agent("Okay", 2.5, 3.0), agent("Your balance is one hundred dollars", 9.0, 11.0))
    assert f.waits == [0.5] and f.silences == [6.0] and f.problems == 1.0


def test_a_hold_or_a_question_is_the_callers_silence():
    f = flow(caller("Hold on let me find my card", 0, 2), agent("Are you still there", 14, 15))
    assert f.silences == []
    f = flow(agent("What is your date of birth?", 0, 2), agent("Take your time, I am here when you are ready", 9, 12))
    assert f.silences == []
    f = flow(agent("Let me look that up for you now", 0, 2), agent("I found your booking for Friday", 9, 12))
    assert f.silences == [7.0]  # the caller is waiting on the agent


def test_talking_over_a_caller_mid_turn():
    f = flow(caller("My card number is four one two two nine", 0, 4), agent("Thanks, and your date of birth", 2.0, 4.5))
    assert f.talk_over == [2.0] and f.replies == 0


def test_a_backchannel_is_not_talking_over():
    f = flow(caller("My card number is four one two two nine", 0, 4), agent("Mhm", 2.0, 2.4), agent("Thanks, and your date of birth", 4.6, 6))
    assert f.talk_over == [] and f.replies == 1


def test_jumping_into_a_pause():
    f = flow(
        caller("My date of birth is", 0, 1.5),
        agent("Thanks, can I have the year as well please", 2.0, 4.5),
        caller("March third nineteen eighty", 2.4, 4.0),
    )
    assert f.pause_jumps == 1 and f.problems == 1.0


def test_a_slow_stop_runs_across_turns():
    f = flow(
        agent("Your balance is one hundred and twenty dollars", 0, 4),
        caller("Wait, that is wrong", 2.0, 3.5),
        agent("and the next payment is due on Friday", 4.3, 7.0),
    )
    assert f.stops == [5.0] and f.problems == 1.0


def test_slow_stop_after_the_caller_cuts_in():
    f = flow(agent("Your balance is one hundred and twenty dollars and the next payment is due on Friday", 0, 6), caller("Wait, that is wrong", 2.0, 3.5))
    assert f.stops == [4.0] and f.problems == 1.0 and f.events == [("stop", 2.0, 4.0)]
    f = flow(agent("Your balance is one hundred and twenty dollars and the next payment is due on Friday", 0, 6), caller("Okay sure", 2.0, 2.6))
    assert f.stops == []  # a backchannel, not an interruption


def test_no_turns_no_clean_share():
    assert flow().clean is None


@pytest.fixture
def scored_cfg(tmp_path):
    c = config_dict()
    task = next(f for f in c["families"] if f["key"] == "task")
    task["critical"] = ["2.2"]
    hearing = next(f for f in c["families"] if f["key"] == "hearing")
    hearing["measured"] = 0.5
    c["scoring"]["flow"] = SETTINGS
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(c))
    return load_config(path)


def test_critical_failure_zeroes_its_family(scored_cfg):
    fam = family_scores(scored_cfg, {"1.1": True, "1.2": True, "2.1": True, "2.2": False}, None)
    assert fam["task"] == 0.0 and fam["hearing"] == 100.0


def test_measured_flow_blends_into_its_family(scored_cfg):
    assert family_scores(scored_cfg, {"1.1": True, "1.2": False}, None, clean=40.0)["hearing"] == pytest.approx(45.0)
    assert family_scores(scored_cfg, {}, None, clean=40.0)["hearing"] == 40.0  # nothing judged: measured alone


def test_gap_sized_results():
    assert rating.outcome(80, 80, 5, 12) == 0.5
    assert rating.outcome(92, 80, 5, 12) == pytest.approx(10 / 11)
    assert rating.outcome(80, 92, 5, 12) == pytest.approx(1 / 11)
    assert rating.outcome(84, 80, 5) == 0.5  # without a scale, the tie margin still applies


def test_rank_ranges():
    boots = [{"a": 3.0, "b": 2.0, "c": 1.0}] * 30 + [{"a": 2.0, "b": 3.0, "c": 1.0}] * 10
    places = rating.rank_ranges(boots, ["a", "b", "c"])
    assert places == {"a": [1, 2], "b": [1, 2], "c": [3, 3]}


def test_background_voices_and_markers_are_not_the_caller():
    f = flow(
        agent("Your balance is one hundred and twenty dollars and the next payment is due on Friday", 0, 6),
        caller("[colleague in the background: can you pass me that folder please]", 2.0, 4.5),
    )
    assert f.stops == [] and f.talk_over == []
    f = flow(agent("Let me look that up for you now", 0, 2), caller("[noise:hold-music]", 2.5, 20), agent("I found your booking for Friday", 21, 23))
    assert f.silences == []  # the caller's side was on hold music, not waiting in silence


def test_a_hold_inside_one_caller_line_is_not_talking():
    c = caller("give me a moment okay I am back now", 0, 31)
    starts = [0, 0.4, 0.8, 1.2, 30, 30.4, 30.8, 31.2, 31.6]
    c["words"] = [{"text": w, "start": s, "end": s + 0.3} for w, s in zip("give me a moment okay I am back now".split(), starts, strict=True)]
    f = flow(c, agent("Take your time, I will be right here", 10, 12))
    assert f.talk_over == []


def test_a_short_question_or_a_check_in_is_not_an_interruption():
    f = flow(agent("Yes I am here, would you like me to request the refill", 0, 5), caller("Are you there?", 1.0, 1.8))
    assert f.stops == []
    f = flow(agent("Yes I am here, would you like me to request the refill", 0, 5), caller("Are you still there?", 1.0, 1.8))
    assert f.stops == []


def test_a_recorded_clip_keeps_the_caller_busy_to_its_end():
    turns = (caller("I need to move my appointment please", 0, 2), agent("Sure, which day works for you", 9, 10))
    assert measure(list(turns), "Main", "Testing", SETTINGS).silences == [7.0]
    clips = {"i need to move my appointment please": 5.5}  # the clip plays on for 3.5 s after the line's last word
    assert measure(list(turns), "Main", "Testing", SETTINGS, clips).silences == [3.5]
