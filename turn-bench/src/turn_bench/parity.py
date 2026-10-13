"""Compare a run's predictions with a reference predictions file.

Per conversation, speaker and task: how many events each side has, how many
times match exactly (same position, bit-identical float), and the largest
absolute time difference. Then both files are scored with Sesame's scorer over
the same conversations and the scores are compared field by field. Scores
that differ make the command fail; the comparison is never rounded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from turnbench.score import TaskScore, score_submission
from turnbench.submission import Submission, load_submission

from turn_bench.data import load_split, split_for, subset
from turn_bench.report import TASKS, task_dict

TASK_KEYS = (("eot", "eot"), ("int", "interruption"))


def compare_lists(ours: list[float], reference: list[float]) -> dict[str, Any]:
    if len(ours) == len(reference):
        return {
            "ours": len(ours),
            "reference": len(reference),
            "exact_matches": sum(a == b for a, b in zip(ours, reference)),
            "max_abs_diff_s": max((abs(a - b) for a, b in zip(ours, reference)), default=0.0),
            "length_mismatch": False,
        }
    # Different counts: positions no longer correspond. Count exact values in
    # common and leave the time difference undefined rather than invent an alignment.
    return {
        "ours": len(ours),
        "reference": len(reference),
        "exact_matches": len(set(ours) & set(reference)),
        "max_abs_diff_s": None,
        "length_mismatch": True,
    }


def compare_predictions(ours: Submission, reference: Submission) -> dict[str, Any]:
    ours_by_id = ours.by_conversation()
    reference_by_id = reference.by_conversation()
    common = [cid for cid in ours_by_id if cid in reference_by_id]
    rows: list[dict[str, Any]] = []
    totals = {
        task: {"ours": 0, "reference": 0, "exact_matches": 0, "max_abs_diff_s": 0.0, "length_mismatches": 0}
        for task, _key in TASK_KEYS
    }
    for cid in common:
        for speaker in (1, 2):
            mine = getattr(ours_by_id[cid], f"speaker_{speaker}")
            theirs = getattr(reference_by_id[cid], f"speaker_{speaker}")
            for task, key in TASK_KEYS:
                cell = compare_lists(getattr(mine, key), getattr(theirs, key))
                rows.append({"conversation_id": cid, "speaker": speaker, "task": task, **cell})
                total = totals[task]
                total["ours"] += cell["ours"]
                total["reference"] += cell["reference"]
                total["exact_matches"] += cell["exact_matches"]
                if cell["length_mismatch"]:
                    total["length_mismatches"] += 1
                    total["max_abs_diff_s"] = None
                elif total["max_abs_diff_s"] is not None:
                    total["max_abs_diff_s"] = max(total["max_abs_diff_s"], cell["max_abs_diff_s"])
    return {
        "conversations_compared": common,
        "only_in_ours": sorted(set(ours_by_id) - set(reference_by_id), key=int),
        "only_in_reference": sorted(set(reference_by_id) - set(ours_by_id), key=int),
        "totals": totals,
        "rows": rows,
    }


def task_scores_equal(ours: TaskScore, reference: TaskScore) -> bool:
    return (
        (ours.tp, ours.fn, ours.fp, ours.tn) == (reference.tp, reference.fn, reference.fp, reference.tn)
        and sorted(ours.latencies_ms) == sorted(reference.latencies_ms)
    )


def restrict(submission: Submission, ids: list[str]) -> Submission:
    wanted = set(ids)
    return Submission(
        schema_version=submission.schema_version,
        predictions=[p for p in submission.predictions if p.conversation_id in wanted],
    )


def run_parity(run_dir: Path, reference_path: Path, split_name: str = "dev") -> int:
    ours = load_submission(run_dir / f"predictions-{split_name}.json")
    reference = load_submission(reference_path)
    comparison = compare_predictions(ours, reference)
    ids = comparison["conversations_compared"]
    if not ids:
        print("no conversations in common between the run and the reference", file=sys.stderr)
        return 2

    dataset = subset(load_split(split_for(split_name)), ids)
    scores_ours = score_submission(restrict(ours, ids), dataset)
    scores_reference = score_submission(restrict(reference, ids), dataset)
    identical = all(
        task_scores_equal(getattr(scores_ours, attr), getattr(scores_reference, attr))
        for _name, attr in TASKS
    )

    result = {
        "run": str(run_dir),
        "reference": str(reference_path),
        "split": split_name,
        "conversations_compared": len(ids),
        "only_in_ours": comparison["only_in_ours"],
        "only_in_reference": comparison["only_in_reference"],
        "events": comparison["totals"],
        "scores": {
            "ours": {name: task_dict(getattr(scores_ours, attr)) for name, attr in TASKS},
            "reference": {name: task_dict(getattr(scores_reference, attr)) for name, attr in TASKS},
        },
        "scores_identical": identical,
        "rows": comparison["rows"],
    }
    (run_dir / "parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(render(result))
    return 0 if identical else 1


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render(result: dict[str, Any]) -> str:
    lines = [
        f"parity: {result['run']} vs {result['reference']} "
        f"({result['conversations_compared']} conversations)",
    ]
    if result["only_in_ours"] or result["only_in_reference"]:
        lines.append(
            f"  not compared — only in ours: {result['only_in_ours']}; "
            f"only in reference: {result['only_in_reference']}"
        )
    lines.append("  events (per task, summed over conversations and speakers):")
    for task, total in result["events"].items():
        diff = "n/a (list lengths differ)" if total["max_abs_diff_s"] is None else f"{total['max_abs_diff_s']:.6f}s"
        lines.append(
            f"    {task.upper()}: ours={total['ours']} reference={total['reference']} "
            f"exact_matches={total['exact_matches']} max_abs_diff={diff} "
            f"length_mismatches={total['length_mismatches']}"
        )
    lines.append("  scores (Sesame scorer, same conversations):")
    lines.append(
        f"    {'task':4}  {'side':9}  {'recall':>6}  {'fp_rate':>7}  {'lat p10/p50/p90 ms':>18}"
        f"  {'tp':>5} {'fn':>5} {'fp':>5} {'tn':>5}"
    )
    for task in ("eot", "int"):
        for side in ("ours", "reference"):
            t = result["scores"][side][task]
            lat = t["latency_ms"]
            latency = f"{_fmt(lat['p10'], 0)}/{_fmt(lat['p50'], 0)}/{_fmt(lat['p90'], 0)}"
            lines.append(
                f"    {task.upper():4}  {side:9}  {_fmt(t['recall']):>6}  {_fmt(t['fp_rate']):>7}  {latency:>18}"
                f"  {t['tp']:5} {t['fn']:5} {t['fp']:5} {t['tn']:5}"
            )
    events_identical = all(
        total["ours"] == total["reference"] == total["exact_matches"] and not total["length_mismatches"]
        for total in result["events"].values()
    )
    lines.append("  events identical: " + ("yes" if events_identical else "NO"))
    lines.append("  scores identical: " + ("yes" if result["scores_identical"] else "NO (exit 1)"))
    return "\n".join(lines)
