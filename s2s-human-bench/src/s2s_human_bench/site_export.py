"""Website data files for the real-caller S2S benchmark, from exported runs.

Writes, under the configured names: the benchmark, insights and reports files in the public
S2S page's shapes; per-language ELO and pass³; the skills file (per-skill rates, family
scores, ELO with intervals); and draft example picks. Output depends only on the inputs,
the config and ``date``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date as Date
from pathlib import Path
from typing import Any

from s2s_human_bench.bench import Entry, load_bench
from s2s_human_bench.board import board_doc
from s2s_human_bench.comparisons import comparisons_doc
from s2s_human_bench.config import Config, load_config
from s2s_human_bench.examples import examples_doc
from s2s_human_bench.flow import clip_key
from s2s_human_bench.languages import languages_doc
from s2s_human_bench.s2s import s2s_docs
from s2s_human_bench.scenarios import scenarios_doc
from s2s_human_bench.skills import skills_doc
from s2s_human_bench.tasks import load_tasks

PUBLIC_S2S_PATH = "data/s2s-benchmark.json"


class ExportError(SystemExit):
    """The inputs are missing or incomplete; nothing was written."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_public_s2s(path: Path, ref: str = "origin/main") -> dict[str, Any]:
    """The public S2S benchmark data (model metadata, published prices, build): a file, or a
    website checkout read at ``ref`` so a stale working tree cannot leak in."""
    if path.is_dir():
        out = subprocess.run(["git", "-C", str(path), "show", f"{ref}:{PUBLIC_S2S_PATH}"], capture_output=True, text=True, check=False)
        if out.returncode:
            raise ExportError(f"{path}: cannot read {ref}:{PUBLIC_S2S_PATH}: {out.stderr.strip()}")
        return json.loads(out.stdout)
    return read_json(path)


def human_clips(path: Path | None) -> int:
    """Recorded caller clips, from ``{scenario: [clip, ...]}``; 0 when no file is given."""
    return sum(len(v) for v in read_json(path).values()) if path else 0


def clip_seconds(path: Path | None) -> dict[str, dict[str, float]] | None:
    """Recorded caller clip lengths by scenario and clip text, from ``{scenario: [{text, seconds}, ...]}``; None when no
    file is given or it carries no lengths (the flow measure then ends each caller line at its last word)."""
    if not path:
        return None
    out = {
        code: {clip_key(c["text"]): c["seconds"] for c in clips if isinstance(c, dict) and c.get("text") and c.get("seconds")}
        for code, clips in read_json(path).items()
    }
    return out if any(out.values()) else None


def titles(path: Path | None) -> dict[str, str | None]:
    """Scenario titles from ``{"calls": [{"id", "title"}]}``; none when no file is given."""
    return {c["id"]: c.get("title") for c in read_json(path)["calls"]} if path else {}


def reviewed_calls(path: Path | None) -> dict[str, list[Any]]:
    """Reviewed example calls, ``{site id: [call id | {"call", "hide"}, ...]}``; none when no file is given."""
    return {m: [i if isinstance(i, dict) else str(i) for i in ids] for m, ids in read_json(path).items()} if path else {}


def reviewed_clips(path: Path | None) -> dict[str, Any] | None:
    """Approved clip pairs, ``{skill id: {setup, should, pass, fail}}``; None when no file is given."""
    return read_json(path) if path else None


def build(
    cfg: Config,
    runs: Path,
    public_s2s: dict[str, Any],
    call_titles: dict[str, str | None],
    clips: int,
    day: str,
    exclude_unhealthy: bool = False,
    reviewed: dict[str, list[Any]] | None = None,
    clips_reviewed: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Every output document by kind, and the clip picks."""
    entries = load_bench(cfg, runs, exclude_unhealthy)
    if not entries:
        raise ExportError(f"{runs}: no model has {cfg['inputs']['minRuns']} usable runs")
    if gap := task_metric_gap(cfg, entries):
        print(gap, file=sys.stderr)
    bench, insights, reports = s2s_docs(cfg, entries, public_s2s, day)
    examples, picks = examples_doc(cfg, entries, call_titles, reviewed or {}, clips_reviewed)
    skills = skills_doc(cfg, entries, public_s2s, clips, day)
    docs = dict(
        benchmark=bench,
        insights=insights,
        reports=reports,
        languages=languages_doc(cfg, entries),
        skills=skills,
        examples=examples,
        comparisons=comparisons_doc(cfg, entries),
        board=board_doc(cfg, entries, skills["elo"], day),
        scenarios=scenarios_doc(cfg, entries, call_titles),
    )
    return docs, picks


def task_metric_gap(cfg: Config, entries: list[Entry]) -> str | None:
    """A warning when the config names a task metric but some calls carry no verdict from it (the run was
    not rescored with it, or it gave none): their Task Outcome falls back to the expected-outcome share,
    so the board would mix two scorings."""
    if cfg.tasks is not None or "task" not in cfg["metrics"]:
        return None
    missing = [e.model.name for e in entries for c in e.calls if c.outcome_from != "metric"]
    if not missing:
        return None
    return (
        f"warning: {len(missing)} of {sum(len(e.calls) for e in entries)} calls have no verdict from the task metric"
        f" ({cfg['metrics']['task']}), so their Task Outcome falls back to the expected-outcome share."
        f" Models: {', '.join(dict.fromkeys(missing))}. Rescore their runs with the metric, or pass --tasks."
    )


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def run_site_export(args: Any) -> int:
    cfg = load_config(args.config)
    if args.tasks:
        cfg.tasks = load_tasks(args.tasks)
    names = dict(cfg["outputs"])
    for pair in args.name or []:
        kind, sep, file = pair.partition("=")
        if not sep or kind not in names:
            raise ExportError(f"--name expects KIND=FILE with KIND in {sorted(names)}, got {pair!r}")
        names[kind] = file
    clips = args.clips if args.clips else (args.runs / "scenario_clips.json")
    cfg.clip_seconds = clip_seconds(clips if clips.exists() else None)
    call_titles = args.titles if args.titles else (args.runs / "scenario_titles.json")
    docs, picks = build(
        cfg,
        args.runs,
        read_public_s2s(args.public_s2s),
        titles(call_titles if call_titles.exists() else None),
        human_clips(clips if clips.exists() else None),
        args.date or Date.today().isoformat(),
        args.exclude_unhealthy,
        reviewed_calls(args.reviewed_examples),
        reviewed_clips(args.reviewed_clips),
    )
    for kind, doc in docs.items():
        write_json(args.out / names[kind], doc)
    if args.clip_picks:
        write_json(args.clip_picks, picks)
    models = docs["skills"]["models"]
    print(f"{len(models)} models, {docs['skills']['callSummary']['total']} scenarios -> {args.out}")
    if cfg.tasks is not None:
        overall = [r for r in docs["benchmark"]["results"] if r["suite"] == "all"]
        print(f"  main task done on {sum(r['taskPass'] for r in overall)} of {sum(r['runs'] for r in overall)} calls (task file {args.tasks})")
    for row in docs["skills"]["elo"]:
        print(f"  {row['model']:<24} elo {row['elo']}  score {row['score']}")
    return 0
