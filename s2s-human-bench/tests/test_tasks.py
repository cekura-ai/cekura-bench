"""Task completion from tool calls, on synthetic tasks and transcripts. No real scenarios, callers or values."""

import json

import pytest

from conftest import write_call
from s2s_human_bench.calls import load_call
from s2s_human_bench.tasks import TaskFileError, load_tasks, row_matches, task_verdict, value_matches


def fc(name: str, t: float, **args) -> dict:
    """One tool call as the transcript records it (arguments as a JSON string)."""
    return {"role": "Function Call", "start_time": t, "data": {"name": name, "arguments": json.dumps(args)}}


def task(*requirements, **extra) -> dict:
    """A task whose requirements are (tool, [accepted rows]) pairs."""
    return {"requirements": [{"any_of": [{"tool": tool, "accepted": rows}]} for tool, rows in requirements], **extra}


# ---- values ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected, actual, ok",
    [
        ("555-010-0199", "(555) 010 0199", True),  # phone formatting
        ("A-1001", "a1001", True),  # id formatting
        ("A-1001", "A-1002", False),
        ("250.00", "250", True),
        ("250.00", "205", False),
        ("2026-03-02", "March 2, 2026", True),
        ("2026-03-02", "2026-03-03", False),
        ("2026-03-02T14:30", "2:30 pm", True),  # a bare time: the record fixes the day
        ("2026-03-02T14:30", "Tuesday 2:30 pm", False),  # 2 March 2026 is a Monday
        ("2026-03-02T17:00/19:00", "Monday after 5 pm", True),  # the pinned clock makes Monday one date
        ("2026-03-02 15:00-17:00", "Monday 3-5 pm", True),
        ("2026-03-02 15:00-17:00", "Monday 9 am to 5 pm", False),  # a window must start when asked
        ("12 Elm Street, level 3", "12 Elm St, level three", True),
        ("12 Elm Street", "14 Elm St", False),
        (["Spain", "Portugal"], ["Portugal", "Spain"], True),
        (["Spain", "Portugal"], ["Spain"], False),
        ("true", "yes", True),
        ("A-1001", "None", False),  # "None" is not a value
        ("address on file", None, True),
    ],
)
def test_value_matches(expected, actual, ok):
    assert value_matches(expected, actual) is ok


def test_row_matches_renamed_optional_any_and_nested():
    assert row_matches({"visit_id": "V-77"}, {"visit": "V-77"})  # argument renamed in the deployed tool
    assert not row_matches({"visit_id": "V-77"}, {"visit_id": "V-78", "note": "V-77"})  # present but wrong: no fallback
    assert row_matches({"account": "A-1001", "?date": "2026-03-02"}, {"account": "A-1001"})  # optional, left out
    assert not row_matches({"account": "A-1001", "?date": "2026-03-02"}, {"account": "A-1001", "date": "2026-03-05"})  # optional, wrong
    assert row_matches({"reason": "*"}, {"reason": "anything at all"}) and not row_matches({"reason": "*"}, {})
    assert row_matches({"dob": "1990-01-31"}, {"driver": {"name": "x", "dob": "1990-01-31"}})  # nested argument


# ---- verdicts ----------------------------------------------------------------------

PAY = task(("send_payment", [{"account": "A-1001", "amount": "250.00"}]))


def test_simple_task():
    assert task_verdict(PAY, []) == (False, "send_payment not called")
    assert task_verdict(PAY, [fc("send_payment", 5, account="A-1001", amount="205")]) == (False, "send_payment called with other values")
    assert task_verdict(PAY, [fc("send_payment", 5, account="A-1001", amount="205"), fc("send_payment", 9, account="A-1001", amount="250")]) == (True, "done")


def test_taken_back_before_anything_was_sent():
    lookup = task(("check_balance", [{}]), must_not_run=["send_payment"])
    assert task_verdict(lookup, [fc("check_balance", 2)]) == (True, "done")
    assert task_verdict(lookup, [fc("check_balance", 2), fc("send_payment", 6, account="A-1001")]) == (False, "send_payment ran after the caller took it back")
    assert task_verdict(lookup, []) == (False, "check_balance not called")


def test_undo_after_the_action():
    book = [{"slot": "S2"}]
    cancel_later = task(("book_visit", book), then_undo=["cancel_visit"])
    assert task_verdict(cancel_later, [fc("book_visit", 3, slot="S2"), fc("cancel_visit", 8, visit="V-1")]) == (True, "done")
    assert task_verdict(cancel_later, [fc("book_visit", 3, slot="S2")]) == (False, "cancel_visit not called after the caller took it back")
    assert task_verdict(cancel_later, [fc("cancel_visit", 1), fc("book_visit", 3, slot="S2")])[0] is False

    keep = task(("book_visit", book), undo_tools={"book_visit": ["cancel_visit"]})
    assert task_verdict(keep, [fc("book_visit", 3, slot="S2")]) == (True, "done")
    assert task_verdict(keep, [fc("book_visit", 3, slot="S2"), fc("cancel_visit", 8)]) == (False, "book_visit undone by cancel_visit")


def test_two_bookings_need_two_calls():
    rides = task(("book_ride", [{"rider": "R-5", "?date": "2026-03-02"}]), ("book_ride", [{"rider": "R-5", "?date": "2026-03-04"}]))
    assert task_verdict(rides, [fc("book_ride", 3, rider="R-5")]) == (False, "book_ride called once where 2 separate calls were needed")
    assert task_verdict(rides, [fc("book_ride", 3, rider="R-5"), fc("book_ride", 7, rider="R-5")]) == (True, "done")
    both_monday = [fc("book_ride", 3, rider="R-5", date="2026-03-02"), fc("book_ride", 7, rider="R-5", date="2026-03-02")]
    assert task_verdict(rides, both_monday) == (False, "book_ride called with other values")


# ---- task file and scoring -------------------------------------------------------------


def test_load_tasks_rejects_empty_files(tmp_path):
    (tmp_path / "none.json").write_text(json.dumps({"tasks": {}}))
    (tmp_path / "bare.json").write_text(json.dumps({"tasks": {"B01": {"requirements": []}}}))
    for name in ("none.json", "bare.json"):
        with pytest.raises(TaskFileError):
            load_tasks(tmp_path / name)


def test_task_file_sets_the_outcome(cfg, tmp_path):
    write_call(tmp_path / "r", "B01", {101: True, 102: None, 201: None, 202: 1}, ["✅ a", "❌ b"])
    call_json = next((tmp_path / "r").glob("calls/*/*/call.json"))
    transcript = call_json.with_name("transcript.json")
    transcript.write_text(json.dumps([*json.loads(transcript.read_text()), fc("send_payment", 3.5, account="A-1001", amount="250.00")]))

    cfg.tasks = {"B01": PAY}
    call = load_call(cfg, call_json)
    assert call.outcome == 100.0 and call.task_done  # the task passed although an expected-outcome line was missed
    assert call.score == pytest.approx((100 + 50) / 2)  # Task Handling: half skills (0), half outcome (100)

    cfg.tasks = {"B01": task(("send_payment", [{"account": "A-2002"}]))}
    call = load_call(cfg, call_json)
    assert call.outcome == 0.0 and not call.task_done and call.score == pytest.approx(50)

    cfg.tasks = {"C01": PAY}
    with pytest.raises(TaskFileError):
        load_call(cfg, call_json)
