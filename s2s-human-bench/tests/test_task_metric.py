"""The platform metric is the scorer's checker: built from tasks.py, it gives the same verdicts.
Synthetic tasks and transcripts only."""

import ast
import json
from pathlib import Path

import pytest

from conftest import write_call
from s2s_human_bench.calls import load_call
from s2s_human_bench.task_metric import PROFILE_FIELD, metric_source
from s2s_human_bench.tasks import task_verdict

COPY = Path(__file__).resolve().parents[1] / "metrics" / "task_completion_metric.py"


def fc(name: str, t: float, **args) -> dict:
    return {"role": "Function Call", "start_time": t, "data": {"name": name, "arguments": json.dumps(args)}}


def one(tool: str, rows: list[dict]) -> dict:
    return {"any_of": [{"tool": tool, "accepted": rows}]}


def run_metric(profile: dict, transcript) -> tuple:
    """Run the metric code the way the platform does: a ``data`` dict in, ``_result`` and ``_explanation`` out."""
    env = {"data": {"test_profile": profile, "cekura_transcript_json": transcript}}
    exec(compile(metric_source(), "task_metric", "exec"), env)  # noqa: S102 - the platform runs it the same way
    return env["_result"], env["_explanation"]


def test_committed_copy_is_current():
    # Rebuild with `uv run --locked s2s-human-bench task-metric --out metrics/task_completion_metric.py`
    # and update the platform metric whenever tasks.py changes.
    assert ast.dump(ast.parse(COPY.read_text(encoding="utf-8"))) == ast.dump(ast.parse(metric_source()))


def test_copy_carries_no_annotations_or_loader():
    tree = ast.parse(metric_source())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert {f.name for f in functions} >= {"task_verdict", "row_matches", "value_matches", "tool_calls"}
    assert "load_tasks" not in {f.name for f in functions}
    assert not any(f.returns or any(a.annotation for a in f.args.args) for f in functions)
    assert not any(isinstance(n, ast.AnnAssign) for n in ast.walk(tree))


PAY = {"requirements": [one("send_payment", [{"account": "A-1001", "amount": "250.00"}])]}
RIDES = {"requirements": [one("book_ride", [{"rider": "R-5", "?date": "2026-03-02"}]), one("book_ride", [{"rider": "R-5", "?date": "2026-03-04"}])]}
CASES = [
    (PAY, [fc("send_payment", 5, account="A-1001", amount="250")]),
    (PAY, [fc("send_payment", 5, account="A-1001", amount="205")]),
    (PAY, []),
    ({"requirements": [one("check_balance", [{}])], "must_not_run": ["send_payment"]}, [fc("check_balance", 2), fc("send_payment", 6)]),
    ({"requirements": [one("book_visit", [{"slot": "S2"}])], "then_undo": ["cancel_visit"]}, [fc("book_visit", 3, slot="S2"), fc("cancel_visit", 8)]),
    ({"requirements": [one("book_visit", [{"slot": "S2"}])], "then_undo": ["cancel_visit"]}, [fc("book_visit", 3, slot="S2")]),
    (RIDES, [fc("book_ride", 3, rider="R-5")]),
    (RIDES, [fc("book_ride", 3, rider="R-5", date="Monday"), fc("book_ride", 7, rider="R-5", date="Wednesday")]),
]


@pytest.mark.parametrize("spec, transcript", CASES)
def test_metric_gives_the_scorer_verdict(spec, transcript):
    ok, reason = task_verdict(spec, transcript)
    got, why = run_metric({PROFILE_FIELD: json.dumps(spec), "agent_dir": "x"}, transcript)
    assert got is ok
    assert why.startswith("Main task done") if ok else reason in why


def test_transcript_may_arrive_as_text():
    got, _ = run_metric({PROFILE_FIELD: json.dumps(PAY)}, json.dumps([fc("send_payment", 5, account="A-1001", amount="250")]))
    assert got is True


def test_no_task_or_no_transcript_gives_no_verdict():
    assert run_metric({"agent_dir": "x"}, [])[0] is None  # a scenario without a main task: not reached
    got, why = run_metric({PROFILE_FIELD: json.dumps(PAY)}, "{not json")
    assert got is None and why.startswith("Evaluation error")


def test_scorer_takes_the_task_verdict_from_the_platform_metric(cfg, tmp_path):
    cfg.raw["metrics"]["task"] = 950
    for run, verdict_ in (("done", True), ("missed", False)):
        write_call(tmp_path / run, "B01", {101: True, 102: None, 201: None, 202: 1, 950: verdict_}, ["✅ a", "❌ b"])
    done, missed = (load_call(cfg, next((tmp_path / r).glob("calls/*/*/call.json"))) for r in ("done", "missed"))
    assert (done.outcome, done.task_done) == (100.0, True) and (missed.outcome, missed.task_done) == (0.0, False)

    cfg.tasks = {"B01": PAY}  # an explicit task file wins: this transcript makes no payment
    assert load_call(cfg, next((tmp_path / "done").glob("calls/*/*/call.json"))).outcome == 0.0

    cfg.tasks = None
    write_call(tmp_path / "older", "B01", {101: True, 102: None, 201: None, 202: 1}, ["✅ a", "❌ b"])
    assert load_call(cfg, next((tmp_path / "older").glob("calls/*/*/call.json"))).outcome == 50.0  # no metric: expected-outcome share
