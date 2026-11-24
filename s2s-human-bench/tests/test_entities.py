"""Entity capture as first-attempt accuracy, on synthetic details and transcripts. No real scenarios, callers or values."""

import json

import pytest

from conftest import write_call
from s2s_human_bench.calls import load_call
from s2s_human_bench.entities import EntityFileError, agent_text, entity_verdict, load_entities, spoken_attempt

PLATE = {"kind": "code", "true": "bk5132", "keys": "plate", "cue": "plate"}
NAME = {"kind": "name", "true": "okafor", "first": "ada", "keys": "^name$"}
LAST4 = {"kind": "digits", "true": "7731", "keys": "last4", "cue": "ending"}
DAY = {"kind": "date", "true": [3, 9], "keys": "date", "cue": "march|the"}
HOME = {"kind": "address", "true": "27 birch", "keys": "address"}


def caller(text: str, t: float) -> dict:
    return {"role": "Testing Agent", "content": text, "start_time": t}


def agent(text: str, t: float) -> dict:
    return {"role": "Main Agent", "content": text, "start_time": t}


def tool(name: str, t: float, **args) -> dict:
    return {"role": "Function Call", "start_time": t, "data": {"name": name, "arguments": json.dumps(args)}}


def result(text: str, t: float) -> dict:
    return {"role": "Function Call Result", "content": text, "start_time": t}


GAVE_PLATE = caller("my plate is B K five one three two", 1)


# ---- reading one line ---------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, line, expected",
    [
        (PLATE, "I have the plate as B as in Bravo, K, five one three two.", True),
        (PLATE, "So the plate is BK5132, right?", True),
        (PLATE, "The plate is BK5133, correct?", False),
        (PLATE, "Thanks for calling, how can I help?", None),
        (NAME, "Thanks, Ada Okafor.", True),
        (NAME, "Thanks, Ada Okafur.", False),
        (NAME, "I have it as O K A F E R.", False),
        (LAST4, "Paying from the account ending in seven seven three one.", True),
        (LAST4, "Paying from the account ending in seven seven one three.", False),
        (DAY, "That was on March ninth.", True),
        (DAY, "That was on March eighth.", False),
        (HOME, "Updating it to twenty-seven Birch Lane.", True),
        (HOME, "Updating it to twenty-one Birch Lane.", False),
    ],
)
def test_spoken_attempt(spec, line, expected):
    got = spoken_attempt(spec, line)
    assert (got[0] if got else None) is expected


# ---- the first use decides -----------------------------------------------------------


def test_read_back_right_passes_and_no_read_back_is_not_required():
    assert entity_verdict(PLATE, [GAVE_PLATE, agent("Got it, plate BK5132. Sending help.", 5)])[0] is True
    # The tool carried the right value and the agent never read it back: still a pass.
    assert entity_verdict(PLATE, [GAVE_PLATE, tool("send_help", 4, plate="BK5132"), agent("Help is on the way.", 6)])[0] is True


def test_first_use_wins_over_a_later_one():
    later_wrong = [GAVE_PLATE, tool("send_help", 4, plate="BK5132"), agent("The plate is BK5133, correct?", 6)]
    assert entity_verdict(PLATE, later_wrong) == (True, "tool: send_help.plate=BK5132")
    first_wrong = [GAVE_PLATE, agent("The plate is BK5133, correct?", 3), tool("send_help", 6, plate="BK5132")]
    assert entity_verdict(PLATE, first_wrong)[0] is False


def test_a_question_about_the_detail_is_not_a_use():
    asked = [GAVE_PLATE, agent("Sorry, I didn't catch the plate. Could you repeat it?", 3), agent("Plate BK5132, thanks.", 6)]
    assert entity_verdict(PLATE, asked)[0] is True


def test_not_reached_when_unused_or_when_the_detail_came_from_a_tool():
    assert entity_verdict(PLATE, [GAVE_PLATE, agent("How can I help today?", 3)]) == (None, "not reached: no use of the detail")
    from_system = [caller("I need a tow", 1), result('{"plate": "BK5132"}', 2), agent("Your plate is BK5132.", 3)]
    assert entity_verdict(PLATE, from_system)[0] is None


def test_letters_transcribers_swap_are_not_a_fail_on_the_transcript_alone():
    swapped = [GAVE_PLATE, agent("Plate PK5132, sending help.", 3)]
    verdict, how = entity_verdict(PLATE, swapped)
    assert verdict is None and "letters often misheard" in how


# ---- the agent's own text --------------------------------------------------------------


def test_agent_text_decides_over_the_transcript():
    stutter = [GAVE_PLATE, agent("Plate BK5- five, one, five- three two, sending help.", 3)]
    assert entity_verdict(PLATE, stutter)[0] is False
    own = {"from": "model", "turns": [{"text": "Plate BK5132, sending help."}]}
    assert entity_verdict(PLATE, stutter, own) == (True, "agent text: bk5132")
    swapped = [GAVE_PLATE, agent("Plate PK5132, sending help.", 3)]
    assert entity_verdict(PLATE, swapped, own)[0] is True
    wrong = {"from": "model", "turns": [{"text": "Plate BK5133, sending help."}]}
    assert entity_verdict(PLATE, [GAVE_PLATE, agent("Plate BK5132, sending help.", 3)], wrong)[0] is False


def test_a_transcription_is_a_second_reading_not_the_models_text():
    heard_wrong = [GAVE_PLATE, agent("Plate BK5133, sending help.", 3)]
    second = {"from": "transcription", "turns": [{"text": "Plate BK5132, sending help."}]}
    verdict, how = entity_verdict(PLATE, heard_wrong, second)
    assert verdict is None and "readings disagree" in how
    settled = heard_wrong + [tool("send_help", 5, plate="BK5132")]
    assert entity_verdict(PLATE, settled, second)[0] is True
    # Both readings hear the same swapped letters: that is what was said.
    both = {"from": "transcription", "turns": [{"text": "Plate PK5132, sending help."}]}
    assert entity_verdict(PLATE, [GAVE_PLATE, agent("Plate PK5132, sending help.", 3)], both)[0] is False


def test_agent_text_from_the_exported_run_record(tmp_path):
    call_json = tmp_path / "call.json"
    assert agent_text(call_json) is None
    record = {"agent_text": {"from": "model", "turns": [{"at": "00:03", "text": "Plate BK5132."}]}}
    (tmp_path / "call_additional_metadata.json").write_text(json.dumps({"provider_call_details": {"custom_metadata": record}}))
    assert agent_text(call_json) == record["agent_text"]


# ---- the file and the score --------------------------------------------------------------


def test_load_entities_rejects_incomplete_details(tmp_path):
    path = tmp_path / "entities.json"
    path.write_text(json.dumps({"skill": "1.2", "entities": {"B01": {"kind": "code", "true": "bk5132", "keys": "plate"}}}))
    with pytest.raises(EntityFileError):
        load_entities(path)  # a code needs a cue
    path.write_text(json.dumps({"skill": "1.2", "entities": {"B01": PLATE}}))
    assert load_entities(path) == ("1.2", {"B01": PLATE})


def test_entity_file_scores_its_skill(cfg, tmp_path):
    run = tmp_path / "run"
    write_call(run, "B01", {101: True, 102: False}, ["✅ done"])
    write_call(run, "B02", {101: True, 102: True}, ["✅ done"])
    b01 = run / "calls" / "B01" / "1" / "call.json"
    b01.with_name("transcript.json").write_text(json.dumps([GAVE_PLATE, agent("Plate BK5132, sending help.", 3)]))
    assert load_call(cfg, b01).verdicts["1.2"] is False  # the platform metric's verdict
    cfg.entity_skill, cfg.entities = "1.2", {"B01": PLATE}
    call = load_call(cfg, b01)
    assert call.verdicts["1.2"] is True and call.explanations["1.2"] == "transcript: bk5132"
    assert "1.2" in call.tested and "1.2" not in call.misses
    # A scenario the file does not list tests no detail.
    assert load_call(cfg, run / "calls" / "B02" / "1" / "call.json").verdicts["1.2"] is None
