"""Website data files for the real-caller S2S benchmark, from exported runs.

Writes, under the configured names: the benchmark, insights and reports files in the public
S2S page's shapes; per-language ELO and pass³; the skills file (per-skill rates, family
scores, ELO with intervals); and draft example picks. Output depends only on the inputs,
the config and ``date``.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date as Date
from pathlib import Path
from typing import Any

from s2s_human_bench.bench import load_bench
from s2s_human_bench.config import Config, load_config
from s2s_human_bench.examples import examples_doc
from s2s_human_bench.languages import languages_doc
from s2s_human_bench.s2s import s2s_docs
from s2s_human_bench.skills import skills_doc

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


def titles(path: Path) -> dict[str, str | None]:
    return {c["id"]: c.get("title") for c in read_json(path)["calls"]}


def build(
    cfg: Config, runs: Path, public_s2s: dict[str, Any], call_titles: dict[str, str | None], clips: int, day: str, exclude_unhealthy: bool = False
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Every output document by kind, and the clip picks."""
    entries = load_bench(cfg, runs, exclude_unhealthy)
    if not entries:
        raise ExportError(f"{runs}: no model has {cfg['inputs']['minRuns']} usable runs")
    bench, insights, reports = s2s_docs(cfg, entries, public_s2s, day)
    examples, picks = examples_doc(cfg, entries, call_titles)
    docs = dict(
        benchmark=bench,
        insights=insights,
        reports=reports,
        languages=languages_doc(cfg, entries),
        skills=skills_doc(cfg, entries, public_s2s, clips, day),
        examples=examples,
    )
    return docs, picks


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def run_site_export(args: Any) -> int:
    cfg = load_config(args.config)
    names = dict(cfg["outputs"])
    for pair in args.name or []:
        kind, sep, file = pair.partition("=")
        if not sep or kind not in names:
            raise ExportError(f"--name expects KIND=FILE with KIND in {sorted(names)}, got {pair!r}")
        names[kind] = file
    clips = args.clips if args.clips else (args.runs / "scenario_clips.json")
    docs, picks = build(
        cfg,
        args.runs,
        read_public_s2s(args.public_s2s),
        titles(args.titles),
        human_clips(clips if clips.exists() else None),
        args.date or Date.today().isoformat(),
        args.exclude_unhealthy,
    )
    for kind, doc in docs.items():
        write_json(args.out / names[kind], doc)
    if args.clip_picks:
        write_json(args.clip_picks, picks)
    models = docs["skills"]["models"]
    print(f"{len(models)} models, {docs['skills']['callSummary']['total']} scenarios -> {args.out}")
    for row in docs["skills"]["elo"]:
        print(f"  {row['model']:<24} elo {row['elo']}  score {row['score']}")
    return 0
