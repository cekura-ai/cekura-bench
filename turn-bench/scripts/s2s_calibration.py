"""Calibration of the S2S readouts against Sesame's committed baseline predictions.

Sesame derives events for OpenAI Realtime from the provider's own endpointer
(``baselines/openai_realtime.py``: ``speech_stopped`` -> EOT, ``speech_started``
-> INT, each at the audio position appended when the event arrived, every
channel in its own session with text output and ``create_response: false``).
``turn-bench run --model s2s:openai-realtime --config mode=vad_only --config
readout=vad_events --config model=gpt-realtime --config channels=[1,2]`` is that
method inside this harness. This script compares such a run with the committed
``predictions-dev.json`` on the conversations both have:

* per conversation and overall, Sesame's scorer on both files (``all`` block,
  and the caller / agent blocks of ``turn_bench.report``);
* per channel and task, how many of the reference's events have one of ours
  within ``--tolerance`` seconds (greedy nearest match), the median and p90 of
  the matched time differences, and the counts on each side.

A live provider is not deterministic — its endpointer fires a few tens of
milliseconds earlier or later from run to run, and occasionally not at all —
so the comparison reports closeness, never identity.

    uv run --locked python scripts/s2s_calibration.py \
        --run runs/s2s/calib-openai-vad_only \
        --reference ~/Dev/turnbench/baselines/openai_server_vad/predictions-dev.json \
        --out runs/s2s/calib-openai-vad_only/calibration.md

``--run`` may be given more than once (e.g. the agent-mode run read out by
onsets beside the vad_only run); each is compared with the reference and with
the first run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from turnbench.submission import Submission, load_submission

from turn_bench.data import load_split, split_for, subset
from turn_bench.report import score_conversation_blocks, scores_dict
from turnbench.data import conversation

TASKS = (("eot", "eot"), ("int", "interruption"))


def restrict(submission: Submission, ids: list[str]) -> Submission:
    by_id = submission.by_conversation()
    return Submission(schema_version=submission.schema_version, predictions=[by_id[cid] for cid in ids])


def nearest_matches(ours: list[float], reference: list[float], tolerance: float) -> dict[str, Any]:
    """Greedy one-to-one matching of reference events to our nearest unclaimed event."""
    claimed: set[int] = set()
    deltas: list[float] = []
    for t_ref in reference:
        best, best_index = None, None
        for index, t in enumerate(ours):
            if index in claimed:
                continue
            delta = t - t_ref
            if abs(delta) <= tolerance and (best is None or abs(delta) < abs(best)):
                best, best_index = delta, index
        if best_index is not None:
            claimed.add(best_index)
            deltas.append(best)
    abs_deltas = sorted(abs(d) for d in deltas)
    return {
        "reference": len(reference),
        "ours": len(ours),
        "matched": len(deltas),
        "reference_unmatched": len(reference) - len(deltas),
        "ours_unmatched": len(ours) - len(deltas),
        "delta_median_ms": None if not deltas else round(statistics.median(deltas) * 1000.0, 1),
        "abs_delta_p90_ms": None if not abs_deltas else round(abs_deltas[min(len(abs_deltas) - 1, int(0.9 * len(abs_deltas)))] * 1000.0, 1),
    }


def compare(ours: Submission, reference: Submission, ids: list[str], dataset: Any, tolerance: float) -> dict[str, Any]:
    ours_by, ref_by = ours.by_conversation(), reference.by_conversation()
    per_conversation: dict[str, Any] = {}
    for cid in ids:
        conv = conversation(dataset, cid)
        blocks_ours = score_conversation_blocks(ours_by[cid], conv)
        blocks_ref = score_conversation_blocks(ref_by[cid], conv)
        events: dict[str, Any] = {}
        for speaker in (1, 2):
            mine = getattr(ours_by[cid], f"speaker_{speaker}")
            theirs = getattr(ref_by[cid], f"speaker_{speaker}")
            for task, key in TASKS:
                events[f"speaker_{speaker}.{task}"] = nearest_matches(getattr(mine, key), getattr(theirs, key), tolerance)
        per_conversation[cid] = {
            "scores": {block: {"ours": scores_dict(blocks_ours[block]), "reference": scores_dict(blocks_ref[block])} for block in ("caller", "agent", "all")},
            "events": events,
        }
    return {"conversations": ids, "tolerance_s": tolerance, "per_conversation": per_conversation}


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render(result: dict[str, Any], label_a: str, label_b: str) -> list[str]:
    lines = [f"### {label_a} vs {label_b} ({len(result['conversations'])} conversations: {', '.join(result['conversations'])})", ""]
    lines += ["| conversation | block | task | recall A / B | fp_rate A / B | latency p50 ms A / B | tp A / B | fp A / B |", "|---|---|---|---|---|---|---|---|"]
    for cid, row in result["per_conversation"].items():
        for block in ("caller", "agent", "all"):
            for task in ("eot", "int"):
                a, b = row["scores"][block]["ours"][task], row["scores"][block]["reference"][task]
                if a["tp"] + a["fn"] + a["fp"] + a["tn"] + b["tp"] + b["fn"] + b["fp"] + b["tn"] == 0:
                    continue
                lines.append(
                    f"| {cid} | {block} | {task} | {_fmt(a['recall'])} / {_fmt(b['recall'])} | {_fmt(a['fp_rate'])} / {_fmt(b['fp_rate'])} | "
                    f"{_fmt(a['latency_ms']['p50'], 0)} / {_fmt(b['latency_ms']['p50'], 0)} | {a['tp']} / {b['tp']} | {a['fp']} / {b['fp']} |"
                )
    lines += ["", f"Event matching (B's events with an A event within {result['tolerance_s']} s; Δ = A − B):", "", "| conversation | channel.task | B events | A events | matched | Δ median ms | abs Δ p90 ms |", "|---|---|---|---|---|---|---|"]
    for cid, row in result["per_conversation"].items():
        for key, m in row["events"].items():
            if m["reference"] + m["ours"] == 0:
                continue
            lines.append(f"| {cid} | {key} | {m['reference']} | {m['ours']} | {m['matched']} | {_fmt(m['delta_median_ms'], 1)} | {_fmt(m['abs_delta_p90_ms'], 1)} |")
    return lines + [""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", required=True, help="run directory (predictions-dev.json); repeatable")
    parser.add_argument("--reference", type=Path, required=True, help="Sesame's committed predictions-dev.json")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--tolerance", type=float, default=0.5, help="seconds within which two events count as the same decision")
    parser.add_argument("--out", type=Path, help="markdown report path (calibration.json is written beside it)")
    args = parser.parse_args()

    reference = load_submission(args.reference)
    runs = {run.name: load_submission(run / f"predictions-{args.split}.json") for run in args.run}
    dataset = load_split(split_for(args.split))
    results: dict[str, Any] = {}
    lines = ["# S2S calibration against Sesame's committed predictions", "", f"- Reference: `{args.reference}`", f"- Split: {args.split}; tolerance {args.tolerance} s", ""]
    first_name = next(iter(runs))
    for name, ours in runs.items():
        ids = [cid for cid in ours.by_conversation() if cid in reference.by_conversation()]
        if not ids:
            raise SystemExit(f"{name}: no conversations in common with the reference")
        result = compare(ours, reference, ids, subset(dataset, ids), args.tolerance)
        results[f"{name} vs reference"] = result
        lines += render(result, f"A = {name}", "B = Sesame reference")
        if name != first_name:
            common = [cid for cid in ids if cid in runs[first_name].by_conversation()]
            if common:
                result = compare(ours, runs[first_name], common, subset(dataset, common), args.tolerance)
                results[f"{name} vs {first_name}"] = result
                lines += render(result, f"A = {name}", f"B = {first_name}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        (args.out.parent / "calibration.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
