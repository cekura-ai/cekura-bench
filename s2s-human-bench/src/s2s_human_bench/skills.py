"""The skills file: per-skill pass rates, family scores and ELO with intervals, per model."""

from __future__ import annotations

import statistics
from typing import Any

from s2s_human_bench import rating
from s2s_human_bench.bench import Entry, elo, family_means, mean_or_none, published_price, scenario_scores
from s2s_human_bench.config import Config, site_order


def head_to_head(cfg: Config, entries: list[Entry], scores: rating.Scores) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    """Each ranked model's [points, games], and the win/draw/loss count per pair."""
    ranked = [e.model.site_id for e in entries if not e.model.reference]
    won = {k: [0.0, 0] for k in ranked}
    pairs: dict[str, dict[str, Any]] = {}
    for a, b, r in rating.matches_for(scores, ranked, cfg.scoring["tieMargin"]):
        won[a][0] += r
        won[a][1] += 1
        won[b][0] += 1 - r
        won[b][1] += 1
        pr = pairs.setdefault(f"{a}|{b}", {"a": a, "b": b, "a_wins": 0, "b_wins": 0, "draws": 0})
        pr["a_wins" if r == 1 else "b_wins" if r == 0 else "draws"] += 1
    return won, list(pairs.values())


def metric_entries(cfg: Config) -> list[dict[str, Any]]:
    out = [
        {"id": s.id, "name": s.name, "family": cfg.families[s.family], "type": "check", "better": "higher", "unit": "rate", "description": s.description}
        for s in cfg.skills
    ]
    out.sort(key=lambda m: site_order(m["id"]))
    return out + [dict(m) for m in cfg["passiveMetrics"]]


def skills_doc(cfg: Config, entries: list[Entry], live: dict[str, Any], human_clips: int, date: str) -> dict[str, Any]:
    scores = scenario_scores(entries)
    ratings = elo(cfg, entries, scores)
    ranked = [e.model.site_id for e in entries if not e.model.reference]
    ci: dict[str, list[float]] = {}
    won: dict[str, list[float]] = {}
    pairs: list[dict[str, Any]] = []
    if ratings:
        won, pairs = head_to_head(cfg, entries, scores)
        boot = cfg.scoring["bootstrap"]
        ci = rating.bootstrap_ci(scores, ranked, cfg.scoring["tieMargin"], boot["draws"], boot["seed"], boot["btIterations"])

    models, results, passive, elos = [], [], [], []
    for e in entries:
        m, calls = e.model, e.calls
        price = published_price(live, m.site_id)
        cost = price.get("costPerMinuteUsd") if price.get("costPublished") else None
        models.append(
            {
                "id": m.site_id,
                "name": m.name,
                "vendor": m.vendor,
                "model": m.key,
                "recordings": len(calls),
                "baseline": m.reference,
                "runs": len(e.runs),
                "cost_per_min": round(cost, 4) if cost else None,
            }
        )
        for s in cfg.skills:
            vals = [v for c in calls if (v := c.verdicts.get(s.id)) is not None]
            results.append(
                {
                    "model": m.site_id,
                    "metric": s.id,
                    "reached": len(vals),
                    "total": len(calls),
                    "not_reached": len(calls) - len(vals),
                    "skipped": 0,
                    "rate": sum(vals) / len(vals) if vals else None,
                    "ci_lo": None,
                    "ci_hi": None,
                    "pass": sum(vals),
                }
            )
        lats = [c.latency_ms for c in calls if c.latency_ms]
        words = [c.turn_words for c in calls if c.turn_words]
        latency = round(statistics.median(lats)) if lats else None
        outcome = mean_or_none(c.outcome for c in calls if c.outcome is not None)
        passive += [
            {"model": m.site_id, "metric": "EO", "value": (outcome or 0) / 100, "n": len(calls), "unit": "share"},
            {"model": m.site_id, "metric": "L1", "value": latency, "median": latency, "n": len(calls), "unit": "ms"},
            {"model": m.site_id, "metric": "P4", "value": round(statistics.median(words), 1) if words else None, "n": len(calls), "unit": "words"},
        ]
        lo, hi = (round(x) for x in ci[m.site_id]) if m.site_id in ci else (None, None)
        w = won.get(m.site_id)
        elos.append(
            {
                "model": m.site_id,
                "elo": round(ratings[m.site_id]) if m.site_id in ratings else None,
                "win_rate": round(w[0] / w[1], 3) if w and w[1] else None,
                "lo": lo,
                "hi": hi,
                "score": mean_or_none(c.score for c in calls if c.score is not None),
                "families": family_means(cfg, calls),
            }
        )

    codes = sorted({c.code for e in entries for c in e.calls})
    recordings = sum(len(e.calls) for e in entries)
    conf = cfg["skills"]
    return {
        "generatedAt": date,
        "minReached": conf["minReached"],
        "callSummary": {
            "total": len(codes),
            "scenarios": codes,
            "recordings": recordings,
            "calls": recordings,
            "runs": max((len(e.runs) for e in entries), default=0),
            "humanClips": human_clips,
            "domains": conf["domains"],
            "planned": len(codes),
        },
        "weights": dict(cfg.weights),
        "tieMargin": cfg.scoring["tieMargin"],
        "limits": [line.format(scenarios=len(codes)) for line in conf["limits"]],
        "models": models,
        "metrics": metric_entries(cfg),
        "results": results,
        "passive": passive,
        "elo": elos,
        "conditions": [],
        "aggregates": [],
        "pairs": pairs,
    }
