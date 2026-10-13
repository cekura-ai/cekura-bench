"""Re-report a finished run, and compare two runs' scorecards.

``turn-bench report --run <dir>`` rescores the predictions file a run wrote —
the model is not run again — and rewrites ``report.json`` / ``report.md`` with
the current report layout (scorecard, intervals, breakdowns). The dataset is
the one ``run.json`` names (a local directory, or the Hugging Face split), or
``--dataset`` to point at the same data elsewhere. Sesame's scorer gives the
same per-conversation scores for the same predictions and gold, so every
existing number is reproduced exactly; only the additions are new.
``run.json`` is left untouched; the report records that it was rescored.

``turn-bench compare --run A --run B`` prints both runs' scorecards side by
side with their intervals and flags every row whose intervals do not overlap
(``turn_bench.scorecard.intervals_disjoint`` — the heuristic and its limits
are documented there).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from turnbench.submission import load_submission

from turn_bench import __version__
from turn_bench.data import Split, load_split, local_split, split_for, subset
from turn_bench.report import BLOCK_TITLES, BLOCKS, authored_groups, score_report, write_report
from turn_bench.runner import harness_git_state, log, turnbench_provenance, utc_now
from turn_bench.scorecard import SCORECARD, SCORECARD_TITLES, intervals_disjoint


def load_run(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} is not a run directory (no run.json)")
    return json.loads(path.read_text(encoding="utf-8"))


def split_for_run(run_info: dict[str, Any], dataset_dir: Path | None) -> Split:
    """The split a run was scored on: ``--dataset`` if given, else what
    ``run.json`` recorded (local directory or Hugging Face split)."""
    if dataset_dir is not None:
        return local_split(dataset_dir)
    recorded = run_info["dataset"]
    if recorded.get("kind") == "local" or recorded.get("revision") is None and Path(recorded["source"]).is_dir():
        return local_split(recorded["source"])
    return split_for(recorded["split"])


def check_manifest(run_info: dict[str, Any], split: Split) -> dict[str, Any] | None:
    """For a local dataset: the directory's ``manifest.json`` must hash to what
    the run recorded, so the gold being rescored is the gold that was run."""
    recorded = (run_info.get("dataset") or {}).get("manifest") or {}
    if split.local_dir is None or not recorded.get("sha256"):
        return None
    path = split.local_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{split.local_dir} has no manifest.json but the run recorded one")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != recorded["sha256"]:
        raise RuntimeError(f"{path} sha256 {digest} differs from the run's recorded manifest {recorded['sha256']}")
    return {"path": str(path), "sha256": digest, "matches_run": True}


def rescore_run(run_dir: Path, dataset_dir: Path | None = None, weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Rescore ``run_dir``'s predictions and rewrite its report; returns the report."""
    run_dir = Path(run_dir)
    run_info = load_run(run_dir)
    split = split_for_run(run_info, dataset_dir)
    if not split.labelled:
        raise ValueError(f"split {split.name!r} has no public labels; nothing to score")
    manifest = check_manifest(run_info, split)
    predictions_path = run_dir / run_info["files"]["predictions"]
    submission = load_submission(predictions_path)
    dataset = load_split(split)
    scored_ids = [p.conversation_id for p in submission.predictions]
    log(f"turn-bench report: run={run_dir.name} dataset={split.source} conversations={len(scored_ids)}")
    groups = authored_groups(split.local_dir) if split.local_dir is not None else None
    report = score_report(submission, subset(dataset, scored_ids), groups, weights)
    report["partial"] = len(scored_ids) < len(dataset.index)
    report["conversations_in_split"] = len(dataset.index)
    report["rescored"] = {
        "at": utc_now(),
        "predictions": predictions_path.name,
        "dataset": {"kind": split.kind, "split": split.name, "source": split.source, "revision": split.revision, "manifest": manifest},
        "harness": {"name": "turn-bench", "version": __version__, **harness_git_state()},
        "turnbench": turnbench_provenance(),
    }
    write_report(report, run_dir, run_info)
    log(f"wrote {run_dir}/report.json and report.md")
    return report


# ---- compare ---------------------------------------------------------------------

ROWS = tuple(SCORECARD) + ("overall",)


def compare_reports(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Row-by-row comparison of two reports' scorecards (blocks and breakdowns)."""
    rows: list[dict[str, Any]] = []

    def add(section: str, label: str, card_a: dict[str, Any] | None, card_b: dict[str, Any] | None) -> None:
        for name in ROWS:
            ra = (card_a or {}).get(name)
            rb = (card_b or {}).get(name)
            disjoint = intervals_disjoint(ra and ra["ci95"], rb and rb["ci95"])
            va = None if ra is None else ra["value"]
            vb = None if rb is None else rb["value"]
            rows.append({
                "section": section, "label": label, "row": name,
                "a": ra, "b": rb,
                "ci_disjoint": disjoint,
                "higher": None if not disjoint or va is None or vb is None else ("A" if va > vb else "B"),
            })

    for block in BLOCKS:
        add("blocks", block, (a["blocks"].get(block) or {}).get("scorecard"), (b["blocks"].get(block) or {}).get("scorecard"))
    axes = set(a.get("breakdowns") or {}) & set(b.get("breakdowns") or {})
    for axis in sorted(axes):
        names = [n for n in a["breakdowns"][axis] if n in b["breakdowns"][axis]]
        for name in names:
            for block in BLOCKS:
                add(f"{axis}:{name}", block, (a["breakdowns"][axis][name].get("scorecard") or {}).get(block), (b["breakdowns"][axis][name].get("scorecard") or {}).get(block))
    return {"rows": rows, "flagged": [r for r in rows if r["ci_disjoint"]]}


def _cell(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "—"
    if entry["value"] is None:
        return "n/a"
    ci = entry.get("ci95")
    text = f"{entry['value']:.3f}"
    if ci and ci.get("lo") is not None:
        text += f" [{ci['lo']:.3f}, {ci['hi']:.3f}]"
    return text


def render_compare(label_a: str, label_b: str, a: dict[str, Any], b: dict[str, Any], comparison: dict[str, Any]) -> str:
    lines = [f"compare: A = {label_a}  vs  B = {label_b}"]
    for side, report in (("A", a), ("B", b)):
        boot = report.get("bootstrap") or {}
        lines.append(f"  {side}: {report['conversations_scored']} conversations; bootstrap {boot.get('draws')} draws, seed {boot.get('seed')}")
    if a["conversations_scored"] != b["conversations_scored"]:
        lines.append("  warning: the runs scored different numbers of conversations; they may not be comparable")
    width = max(len(SCORECARD_TITLES[r]) for r in ROWS)
    current = None
    for row in comparison["rows"]:
        section = (row["section"], row["label"])
        if section != current:
            current = section
            title = BLOCK_TITLES.get(row["label"], row["label"]) if row["section"] == "blocks" else f"{row['section']} / {BLOCK_TITLES.get(row['label'], row['label'])}"
            lines += ["", f"  {title}", f"    {'':{width}}  {'A':>24}  {'B':>24}"]
        flag = ""
        if row["ci_disjoint"]:
            flag = f"  <- {row['higher']} higher (CIs disjoint)"
        lines.append(f"    {SCORECARD_TITLES[row['row']]:{width}}  {_cell(row['a']):>24}  {_cell(row['b']):>24}{flag}")
    lines += [
        "",
        f"  flagged {len(comparison['flagged'])} of {len(comparison['rows'])} rows: 95% CIs do not overlap. "
        "Disjoint intervals are strong evidence of a difference (stricter than a 5% test); "
        "overlapping intervals are not evidence of a tie.",
    ]
    return "\n".join(lines)


def run_compare(run_a: Path, run_b: Path) -> int:
    reports = []
    for run_dir in (run_a, run_b):
        path = Path(run_dir) / "report.json"
        if not path.is_file():
            print(f"{run_dir} has no report.json (run `turn-bench report --run {run_dir}` first)", file=sys.stderr)
            return 2
        report = json.loads(path.read_text(encoding="utf-8"))
        if not any("scorecard" in block for block in (report.get("blocks") or {}).values()):
            print(f"{path} has no scorecard; rescore it with `turn-bench report --run {run_dir}`", file=sys.stderr)
            return 2
        reports.append(report)
    comparison = compare_reports(*reports)
    print(render_compare(Path(run_a).name, Path(run_b).name, reports[0], reports[1], comparison))
    return 0
