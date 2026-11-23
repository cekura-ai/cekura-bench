"""The scenarios file: every scenario with each model's runs, for a page that shows how each model handled the
same caller. Per scenario, the hard moments it is built around (as the board counts them); per run, whether the
main task got done, the call score, and the skills it missed. No transcripts, call ids or links."""

from __future__ import annotations

from typing import Any

from s2s_human_bench.bench import Entry
from s2s_human_bench.board import DEFAULTS, built_for
from s2s_human_bench.config import Config


def scenarios_doc(cfg: Config, entries: list[Entry], call_titles: dict[str, str | None]) -> dict[str, Any]:
    codes = sorted({c.code for e in entries for c in e.calls})
    use_case = {code: uc["key"] for uc in cfg.raw.get("useCases", []) for code in uc["scenarios"]}
    suite = {code: s["id"] for s in cfg["suites"] for code in codes if code.startswith(s["prefix"])}
    share = {**DEFAULTS, **cfg.scoring.get("board", {})}["momentMinShare"]
    built = {cat["key"]: built_for(entries, set(cat["skills"]), share) for cat in cfg["categories"]}
    out = []
    for code in codes:
        moments = [cat["key"] for cat in cfg["categories"] if code in built[cat["key"]]]
        models = {}
        for e in entries:
            runs = [run.get(code) for run in e.runs]
            models[e.model.site_id] = [
                None if c is None else {"done": c.task_done, "score": None if c.score is None else round(c.score, 1), "missed": c.misses} for c in runs
            ]
        out.append(
            {"code": code, "title": call_titles.get(code), "suite": suite.get(code), "useCase": use_case.get(code), "moments": moments, "models": models}
        )
    return {"skills": {s.id: s.name for s in cfg.skills}, "scenarios": out}
