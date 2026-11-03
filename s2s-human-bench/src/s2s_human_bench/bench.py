"""The loaded benchmark: every published model with its runs, and the pooled views the outputs share."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from s2s_human_bench import rating
from s2s_human_bench.calls import Call, load_run, model_runs
from s2s_human_bench.config import Config, Model

Run = dict[str, Call]


@dataclass
class Entry:
    model: Model
    runs: list[Run]

    @property
    def calls(self) -> list[Call]:
        return [c for run in self.runs for c in run.values()]


def load_bench(cfg: Config, root: Path, exclude_unhealthy: bool = False) -> list[Entry]:
    """Models with at least ``minRuns`` usable runs, in config order: pass³ and ELO need every run."""
    out = []
    for model in cfg.models:
        runs = model_runs(cfg, model, root)
        if len(runs) >= cfg["inputs"]["minRuns"]:
            out.append(Entry(model, [load_run(cfg, r, exclude_unhealthy) for r in runs]))
    return out


def mean_or_none(values: Iterable[float], digits: int | None = 1) -> float | None:
    vals = list(values)
    if not vals:
        return None
    m = statistics.mean(vals)
    return round(m, digits) if digits is not None else m


def scenario_scores(entries: list[Entry], keep: Callable[[str], bool] = lambda code: True) -> rating.Scores:
    """Per scenario, each model's call score averaged over its runs (site ids as keys)."""
    scores: dict[str, dict[str, list[float]]] = {}
    for e in entries:
        for run in e.runs:
            for code, c in run.items():
                if keep(code) and c.score is not None:
                    scores.setdefault(code, {}).setdefault(e.model.site_id, []).append(c.score)
    return {code: {k: statistics.mean(v) for k, v in d.items()} for code, d in sorted(scores.items())}


def elo(cfg: Config, entries: list[Entry], scores: rating.Scores) -> dict[str, float]:
    """Ratings by site id: ranked models by Bradley-Terry, references anchored. Empty with under two ranked models."""
    ranked = [e.model.site_id for e in entries if not e.model.reference]
    if len(ranked) < 2:
        return {}
    refs = [e.model.site_id for e in entries if e.model.reference]
    return rating.ratings(scores, ranked, refs, cfg.scoring["tieMargin"], cfg.scoring["btIterations"])


def passed(cfg: Config, call: Call) -> bool:
    return (call.score or 0) >= cfg.scoring["passScore"]


def pass_all(cfg: Config, runs: list[Run], code: str) -> bool:
    """pass³: the scenario passed in every run (and every run has it)."""
    vals = [r[code].score for r in runs if code in r and r[code].score is not None]
    return len(vals) == len(runs) and all(v >= cfg.scoring["passScore"] for v in vals)


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, as the public S2S page computes it."""
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else None


def published_price(live: dict, site_id: str) -> dict:
    """The public S2S benchmark's overall row for a model, if it publishes a price."""
    return next((r for r in live.get("results", []) if r.get("model") == site_id and r.get("suite") == "all"), {})
