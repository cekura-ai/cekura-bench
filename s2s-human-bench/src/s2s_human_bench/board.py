"""The board file: what a deployer would check before picking a model, with every number's 95% range.

* **Models**: call score, task done, each family (part) of the score and the measured call flow
  (long silences and talking over the caller per 10 calls, slow stops after an interruption), with
  the rating, the shared place (1 plus the models clearly above) and the range of places from the
  skills file.
* **Hard moments**: per call type, over every call of the scenarios built for it: how often the
  moment came up (reach), how often it was handled when it did, and *handled*: the caller's task
  got done and the moment, if it came up, was handled. Counting only the calls that reached the
  moment flatters models whose calls go off track before it.
* **Use cases**: task done and call score on each group of scenarios (cards, appointments, ...).
* **Strengths**: skills and measured call flow where a model is clearly better or worse than the
  typical ranked model: its 95% range clears the median of the others by a set margin.

Ranges treat scenarios as the unit (a scenario's runs are not independent), so they are wider than
call counts alone would suggest. Nothing here is random: output depends only on the inputs.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Any

from s2s_human_bench.bench import Entry
from s2s_human_bench.calls import Call
from s2s_human_bench.comparisons import t_quantile
from s2s_human_bench.config import Config

DEFAULTS = {"level": 0.95, "minGap": 10, "minScenarios": 5, "momentMinShare": 0.25, "flowGaps": {"longWaits": 2.0, "talkOver": 2.0, "slowStops": 10.0}}


def scenario_mean(entry: Entry, value: Callable[[Call], float | None], keep: Callable[[str], bool] = lambda code: True) -> dict[str, float]:
    """Each scenario's value averaged over the runs that have one."""
    vals: dict[str, list[float]] = {}
    for run in entry.runs:
        for code, call in run.items():
            if keep(code) and (v := value(call)) is not None:
                vals.setdefault(code, []).append(v)
    return {code: statistics.mean(v) for code, v in vals.items()}


def mean_range(values: list[float], level: float, digits: int = 1) -> dict[str, Any] | None:
    """Mean over scenarios with its t-interval."""
    if not values:
        return None
    m = statistics.mean(values)
    if len(values) < 2:
        return {"value": round(m, digits), "lo": None, "hi": None, "n": len(values)}
    half = t_quantile(0.5 + level / 2, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return {"value": round(m, digits), "lo": round(m - half, digits), "hi": round(m + half, digits), "n": len(values)}


def ratio_range(parts: list[tuple[float, float]], level: float) -> dict[str, Any] | None:
    """A pooled rate, sum(passed) / sum(judged), over scenarios with a cluster-robust interval (the scenario is the unit)."""
    judged = sum(n for _, n in parts)
    if not judged:
        return None
    r = sum(k for k, _ in parts) / judged
    n = len(parts)
    if n < 2:
        return {"value": round(100 * r, 1), "lo": None, "hi": None, "judged": int(judged), "scenarios": n}
    var = n / (n - 1) * sum((k - r * m) ** 2 for k, m in parts) / judged**2
    half = t_quantile(0.5 + level / 2, n - 1) * math.sqrt(var)
    return {
        "value": round(100 * r, 1),
        "lo": round(100 * max(0.0, r - half), 1),
        "hi": round(100 * min(1.0, r + half), 1),
        "judged": int(judged),
        "scenarios": n,
    }


def flow_summary(cfg: Config, entry: Entry, level: float) -> dict[str, Any]:
    """Measured call flow: long silences while the caller waits and talking over the caller per 10 calls, slow stops per
    interruption, and the typical wait before a reply."""
    settings = cfg.scoring["flow"]
    long_wait, slow_stop = settings["longWaitSeconds"], settings["slowStopSeconds"]
    calls = entry.calls
    waits = [w for c in calls for w in c.flow.waits]
    stops = [s for c in calls for s in c.flow.stops]

    def per10(value: Callable[[Call], float]) -> dict[str, Any] | None:
        r = mean_range(list(scenario_mean(entry, value).values()), level, digits=3)
        return None if r is None else {k: (round(10 * v, 1) if isinstance(v, float) else v) for k, v in r.items()}

    return {
        "calls": len(calls),
        "replies": len(waits),
        "waitP50": round(statistics.median(waits), 2) if waits else None,
        "waitP90": round(sorted(waits)[int(0.9 * len(waits))], 2) if waits else None,
        "longWaits": per10(lambda c: sum(1 for w in c.flow.silences if w >= long_wait)),
        "talkOver": per10(lambda c: len(c.flow.talk_over) + c.flow.pause_jumps),
        "interruptions": len(stops),
        "slowStops": ratio_range(
            [(sum(1 for c in cs for s in c.flow.stops if s >= slow_stop), sum(len(c.flow.stops) for c in cs)) for cs in by_scenario(entry).values()],
            level,
        ),
        "clean": mean_range(list(scenario_mean(entry, lambda c: c.flow.clean).values()), level),
    }


def by_scenario(entry: Entry, keep: Callable[[str], bool] = lambda code: True) -> dict[str, list[Call]]:
    out: dict[str, list[Call]] = {}
    for run in entry.runs:
        for code, call in run.items():
            if keep(code):
                out.setdefault(code, []).append(call)
    return out


def built_for(entries: list[Entry], skills: set[str], min_share: float) -> set[str]:
    """Scenarios built for a set of skills: those where at least ``min_share`` of all models' calls were judged on one
    of them, so a moment that turned up by chance in a few calls does not count."""
    seen: dict[str, list[int]] = {}
    for e in entries:
        for c in e.calls:
            n = seen.setdefault(c.code, [0, 0])
            n[0] += bool(skills & set(c.tested))
            n[1] += 1
    return {code for code, (k, n) in seen.items() if k >= min_share * n}


def moment(call: Call, skills: set[str]) -> tuple[bool, bool]:
    """(the moment came up, it was handled): came up when one of the skills was judged; handled when all judged passed."""
    judged = [s for s in call.tested if s in skills]
    return bool(judged), bool(judged) and not any(s in call.misses for s in judged)


def situations(cfg: Config, entries: list[Entry], level: float, min_share: float) -> list[dict[str, Any]]:
    out = []
    for cat in cfg["categories"]:
        skills = set(cat["skills"])
        codes = built_for(entries, skills, min_share)
        models = {}
        for e in entries:
            groups = by_scenario(e, lambda code, cs=codes: code in cs)
            if not groups:
                continue
            reached = [(sum(moment(c, skills)[0] for c in cs), len(cs)) for cs in groups.values()]
            when = [(sum(moment(c, skills)[1] for c in cs), sum(moment(c, skills)[0] for c in cs)) for cs in groups.values()]
            handled = [statistics.mean(float(c.task_done and (not moment(c, skills)[0] or moment(c, skills)[1])) for c in cs) for cs in groups.values()]
            done = [statistics.mean(float(c.task_done) for c in cs) for cs in groups.values()]
            models[e.model.site_id] = {
                "calls": sum(n for _, n in reached),
                "reached": ratio_range(reached, level),
                "handledWhenReached": ratio_range([x for x in when if x[1]], level),
                "done": share_range(done, level),
                "handled": share_range(handled, level),
            }
        out.append(
            {"key": cat["key"], "label": cat["label"], "description": cat["description"], "skills": cat["skills"], "scenarios": len(codes), "models": models}
        )
    return out


def share_range(values: list[float], level: float) -> dict[str, Any] | None:
    """The mean of 0-1 values over scenarios, with its t-interval, in percent."""
    r = mean_range(values, level, digits=6)
    if r is None:
        return None
    return {k: (round(100 * v, 1) if isinstance(v, float) and k in ("value", "lo", "hi") else v) for k, v in r.items()}


def use_cases(cfg: Config, entries: list[Entry], level: float) -> list[dict[str, Any]]:
    out = []
    for uc in cfg.raw.get("useCases", []):
        codes = set(uc["scenarios"])
        models = {}
        for e in entries:
            keep = lambda code, cs=codes: code in cs  # noqa: E731
            done = scenario_mean(e, lambda c: float(c.task_done), keep)
            if not done:
                continue
            models[e.model.site_id] = {
                "calls": sum(len(cs) for cs in by_scenario(e, keep).values()),
                "done": share_range(list(done.values()), level),
                "score": mean_range(list(scenario_mean(e, lambda c: c.score, keep).values()), level),
            }
        out.append({"key": uc["key"], "label": uc["label"], "suite": uc["suite"], "scenarios": sorted(codes), "models": models})
    return out


def strengths(
    cfg: Config, entries: list[Entry], flows: dict[str, dict[str, Any] | None], settings: dict[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Skills and measured call flow where each model's 95% range clears the median of the other ranked models: by
    ``minGap`` points for a skill, by ``flowGaps`` for long silences and talking over the caller (per 10 calls) and slow
    stops (percent). Fewer flow problems is better."""
    level, gap = settings["level"], settings["minGap"]
    ranked = [e.model.site_id for e in entries if not e.model.reference]
    rates: dict[str, dict[str, dict[str, Any] | None]] = {}
    for e in entries:
        rates[e.model.site_id] = {}
        for s in cfg.skills:
            parts = [(sum(s.id not in c.misses for c in cs if s.id in c.tested), sum(s.id in c.tested for c in cs)) for cs in by_scenario(e).values()]
            parts = [p for p in parts if p[1]]
            rates[e.model.site_id][s.id] = ratio_range(parts, level) if len(parts) >= settings["minScenarios"] else None
    labels = {"longWaits": "Long silences", "talkOver": "Talking over the caller", "slowStops": "Slow stops when cut off"}
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for e in entries:
        me = e.model.site_id
        strong, weak = [], []
        for s in cfg.skills:
            r = rates[me][s.id]
            field = [rates[m][s.id]["value"] for m in ranked if m != me and rates[m][s.id]]
            if r is None or r["lo"] is None or len(field) < 3:
                continue
            med = statistics.median(field)
            row = {"kind": "skill", "id": s.id, "name": s.name, "value": r["value"], "field": round(med, 1)}
            if r["lo"] > med and r["value"] - med >= gap:
                strong.append(row)
            elif r["hi"] < med and med - r["value"] >= gap:
                weak.append(row)
        for key, fgap in settings["flowGaps"].items():
            r = (flows.get(me) or {}).get(key)
            field = [flows[m][key]["value"] for m in ranked if m != me and flows.get(m) and flows[m].get(key)]
            if not r or r.get("lo") is None or len(field) < 3:
                continue
            med = statistics.median(field)
            row = {"kind": "flow", "id": key, "name": labels[key], "value": r["value"], "field": round(med, 1)}
            if r["hi"] < med and med - r["value"] >= fgap:
                strong.append(row)
            elif r["lo"] > med and r["value"] - med >= fgap:
                weak.append(row)
        size = lambda x: abs(x["value"] - x["field"]) / (settings["flowGaps"][x["id"]] if x["kind"] == "flow" else gap)  # noqa: E731
        strong.sort(key=lambda x: -size(x))
        weak.sort(key=lambda x: -size(x))
        out[me] = {"strong": strong, "weak": weak}
    return out


def board_doc(cfg: Config, entries: list[Entry], elo_rows: list[dict[str, Any]], date: str) -> dict[str, Any]:
    settings = {**DEFAULTS, **cfg.scoring.get("board", {})}
    level = settings["level"]
    rating = {r["model"]: r for r in elo_rows}
    models = []
    for e in entries:
        r = rating.get(e.model.site_id, {})
        models.append(
            {
                "id": e.model.site_id,
                "reference": e.model.reference,
                "rating": r.get("elo"),
                "ratingRange": [r.get("lo"), r.get("hi")],
                "place": r.get("place"),
                "placeRange": [r.get("rank_lo"), r.get("rank_hi")],
                "score": mean_range(list(scenario_mean(e, lambda c: c.score).values()), level),
                "done": share_range(list(scenario_mean(e, lambda c: float(c.task_done)).values()), level),
                "parts": {label: mean_range(list(scenario_mean(e, lambda c, k=key: c.families.get(k)).values()), level) for key, label in cfg.families.items()},
                "flow": flow_summary(cfg, e, level) if "flow" in cfg.scoring else None,
            }
        )
    moments = situations(cfg, entries, level, settings["momentMinShare"])
    flow = cfg.scoring.get("flow", {})
    return {
        "generatedAt": date,
        "version": cfg["version"],
        "method": {
            "level": level,
            "passScore": cfg.scoring["passScore"],
            "ratingScale": cfg.scoring.get("ratingScale"),
            "parts": [
                {
                    "key": f["key"],
                    "label": f["label"],
                    "weight": f["weight"],
                    "description": f.get("description", ""),
                    "skills": [s["id"] for s in f["skills"]],
                    "measured": f.get("measured"),
                    "critical": f.get("critical", []),
                    "task": f["key"] == cfg.scoring.get("taskFamily", "task"),
                }
                for f in cfg["families"]
            ],
            "flow": {k: flow[k] for k in ("waits", "stops", "talkOverSeconds", "pauseJumpSeconds", "longWaitSeconds", "slowStopSeconds") if k in flow},
            "strengths": {"minGap": settings["minGap"], "minScenarios": settings["minScenarios"]},
        },
        "models": models,
        "moments": moments,
        "useCases": use_cases(cfg, entries, level),
        "strengths": strengths(cfg, entries, {m["id"]: m["flow"] for m in models}, settings),
    }
