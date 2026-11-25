"""Entity capture as first-attempt accuracy, on synthetic details and transcripts. No real scenarios, callers or values."""

import json

import pytest

from conftest import write_call
from s2s_human_bench.calls import load_call
from s2s_human_bench.entities import (
    EntityFile,
    EntityFileError,
    agent_text,
    agent_text_from_log,
    entity_verdict,
    load_entities,
    spoken_attempt,
)

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
    # The agent's own text shows the first use was wrong: a right tool value later does not change that.
    first_wrong = [GAVE_PLATE, agent("The plate is BK5133, correct?", 3), tool("send_help", 6, plate="BK5132")]
    own = {"from": "model", "turns": ["The plate is BK5133, correct?"]}
    assert entity_verdict(PLATE, first_wrong, own)[0] is False
    # Nor when the caller gave the plate again before the tool call.
    corrected = first_wrong[:2] + [caller("No, B K five one three two.", 4)] + first_wrong[2:]
    assert entity_verdict(PLATE, corrected)[0] is False


def test_a_right_tool_value_with_nothing_between_settles_a_transcript_only_reading():
    misheard = [GAVE_PLATE, agent("The plate is BK5133, sending help.", 3), tool("send_help", 6, plate="BK5132")]
    verdict, how = entity_verdict(PLATE, misheard)
    assert verdict is True and how.startswith("tool right with nothing between")
    asked_again = misheard[:2] + [agent("Sorry, could you repeat the plate?", 4)] + misheard[2:]
    assert entity_verdict(PLATE, asked_again)[0] is False


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
    # The model sent the same wrong letters: the transcript heard it right.
    sent = swapped + [tool("send_help", 5, plate="PK5132")]
    assert entity_verdict(PLATE, sent) == (False, "transcript letters confirmed by tool: send_help.plate=PK5132")
    # A right value after the caller repeated the plate shows nothing about the first use.
    repeated = swapped + [caller("No, it's B K five one three two.", 4), tool("send_help", 5, plate="BK5132")]
    assert entity_verdict(PLATE, repeated)[0] is None
    assert entity_verdict(PLATE, swapped + [tool("send_help", 5, plate="BK5132")])[0] is True


def test_parsing_a_restart_and_a_date_far_from_its_cue():
    restarted = [GAVE_PLATE, agent("Plate BK5- ... 5132, sending help.", 3)]
    assert entity_verdict(PLATE, restarted) == (True, "transcript: bk5132")
    spelled = {"kind": "name", "true": "okkafor", "first": "ada", "keys": "name"}
    assert spoken_attempt(spelled, "Ada O-K-K-A-F-O-R, got it.") == (True, "okkafor")
    birth = {"kind": "date", "true": [3, 9, 1971], "keys": "dob", "cue": "birth|born"}
    slot = "I can verify you with your name and date of birth. One moment. You're verified, Ada. I have Tuesday, October 20th."
    assert spoken_attempt(birth, slot) is None
    assert spoken_attempt(birth, "Your date of birth is March 9th, 1971, correct?") == (True, "[3, 9]")


# ---- the agent's own text --------------------------------------------------------------


def test_agent_text_decides_over_the_transcript():
    stutter = [GAVE_PLATE, agent("Plate BK5- five, one, five- three two, sending help.", 3)]
    assert entity_verdict(PLATE, stutter)[0] is False
    own = {"from": "model", "turns": ["Plate BK5132, sending help."]}
    assert entity_verdict(PLATE, stutter, own) == (True, "agent text: bk5132")
    swapped = [GAVE_PLATE, agent("Plate PK5132, sending help.", 3)]
    assert entity_verdict(PLATE, swapped, own)[0] is True
    wrong = {"from": "model", "turns": ["Plate BK5133, sending help."]}
    assert entity_verdict(PLATE, [GAVE_PLATE, agent("Plate BK5132, sending help.", 3)], wrong)[0] is False


def test_a_transcription_is_a_second_reading_not_the_models_text():
    heard_wrong = [GAVE_PLATE, agent("Plate BK5133, sending help.", 3)]
    second = {"from": "transcription", "turns": ["Plate BK5132, sending help."]}
    verdict, how = entity_verdict(PLATE, heard_wrong, second)
    assert verdict is None and "readings disagree" in how
    settled = heard_wrong + [tool("send_help", 5, plate="BK5132")]
    assert entity_verdict(PLATE, settled, second)[0] is True
    # Both hear the same wrong digits: that is what was said.
    assert entity_verdict(PLATE, heard_wrong, {"from": "transcription", "turns": ["Plate BK5133, sending help."]})[0] is False
    # Both hear the same swapped letters: transcribers share that mistake, so it is no proof.
    both = {"from": "transcription", "turns": ["Plate PK5132, sending help."]}
    assert entity_verdict(PLATE, [GAVE_PLATE, agent("Plate PK5132, sending help.", 3)], both)[0] is None


def test_the_first_sentence_that_uses_the_detail_decides():
    line = [GAVE_PLATE, agent("So the plate is P K 5 1 3 2. Got it, B K 5 1 3 2.", 3)]
    own = {"from": "model", "turns": ["So the plate is P K 5 1 3 2. Got it, B K 5 1 3 2."]}
    assert entity_verdict(PLATE, line, own) == (False, "agent text: pk5132")
    spelled = {"from": "model", "turns": ["So the plate is B. K. 5 1 3 2. Sending help."]}
    assert entity_verdict(PLATE, [GAVE_PLATE, agent("So the plate is BK5132. Sending help.", 3)], spelled)[0] is True


def test_a_blanked_value_that_could_be_the_detail_leaves_the_line_to_the_transcript():
    line = [GAVE_PLATE, agent("Thanks Ada. The plate is BK5132, sending help.", 3)]
    blanked = {"from": "model", "turns": ["Thanks Ada. The plate is [redacted identifying number], sending help."]}
    assert entity_verdict(PLATE, line, blanked) == (True, "transcript: bk5132")
    name_blanked = {"from": "model", "turns": ["Thanks [redacted person]. The plate is BK5133, sending help."]}
    assert entity_verdict(PLATE, line, name_blanked) == (False, "agent text: bk5133")


def test_agent_text_from_an_execution_log():
    log = {
        "entries": [
            {"logger": "s2s_reference_bot", "message": '[00:10] caller said: "my plate is B K five one three two"'},
            {"logger": "s2s_reference_bot", "message": '[00:12] agent said: "So the plate is B K"'},
            {"logger": "s2s_reference_bot", "message": '[00:13] agent said: "5 1 3 2. Sending help."'},
            {"logger": "s2s_reference_bot", "message": '[00:20] caller said: "thanks"'},
            {"logger": "s2s_reference_bot", "message": '[00:22] agent said (interrupted): "You are welcome."'},
        ]
    }
    text = agent_text_from_log(log)
    assert text == {"from": "model", "turns": ["So the plate is B K 5 1 3 2. Sending help.", "You are welcome."]}
    log["entries"].append({"logger": "pipecat.services.google.gemini_live.llm", "message": "connected"})
    assert agent_text_from_log(log)["from"] == "transcription"
    assert agent_text_from_log({"data": {"entries": []}}) is None


def test_agent_text_from_the_exported_call_metadata():
    assert agent_text(None) is None and agent_text({"metadata": {}}) is None
    record = {"agent_text": {"from": "model", "turns": ["Plate BK5132."]}}
    assert agent_text({"provider_call_details": {"custom_metadata": record}}) == record["agent_text"]


# ---- the file and the score --------------------------------------------------------------


def test_load_entities_rejects_incomplete_details(tmp_path):
    path = tmp_path / "entities.json"
    path.write_text(json.dumps({"skill": "1.2", "entities": {"B01": {"kind": "code", "true": "bk5132", "keys": "plate"}}}))
    with pytest.raises(EntityFileError):
        load_entities(path, {"1.2"})  # a code needs a cue
    path.write_text(json.dumps({"skill": "1.2", "entities": {"B01": PLATE}}))
    with pytest.raises(EntityFileError):
        load_entities(path, {"1.1"})  # the skill must be in the config
    assert load_entities(path, {"1.2"}) == EntityFile("1.2", {"B01": PLATE})


def test_entity_file_scores_its_skill(cfg, tmp_path):
    run = tmp_path / "run"
    write_call(run, "B01", {101: True, 102: False}, ["✅ done"])
    write_call(run, "B02", {101: True, 102: True}, ["✅ done"])
    b01 = run / "calls" / "B01" / "1" / "call.json"
    b01.with_name("transcript.json").write_text(json.dumps([GAVE_PLATE, agent("Plate BK5132, sending help.", 3)]))
    assert load_call(cfg, b01).verdicts["1.2"] is False  # the platform metric's verdict
    cfg.entities = EntityFile("1.2", {"B01": PLATE})
    call = load_call(cfg, b01)
    assert call.verdicts["1.2"] is True and call.explanations["1.2"] == "transcript: bk5132"
    assert "1.2" in call.tested and "1.2" not in call.misses
    # The agent's own words, from the exported metadata, decide over the transcript.
    b01.with_name("call_additional_metadata.json").write_text(
        json.dumps({"metadata": {"agent_text": {"from": "model", "turns": ["Plate BK5133, sending help."]}}})
    )
    assert load_call(cfg, b01).verdicts["1.2"] is False
    # A run recorded before agent_text: the call's execution log gives the agent's own text.
    b01.with_name("call_additional_metadata.json").unlink()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / f"{b01.parent.name}.json").write_text(json.dumps({"entries": [{"message": 'agent said: "Plate BK5133, sending help."'}]}))
    assert load_call(cfg, b01).verdicts["1.2"] is True
    cfg.agent_logs = logs
    assert load_call(cfg, b01).verdicts["1.2"] is False
    # A scenario the platform scored for the skill must be in the file.
    with pytest.raises(EntityFileError):
        load_call(cfg, run / "calls" / "B02" / "1" / "call.json")
