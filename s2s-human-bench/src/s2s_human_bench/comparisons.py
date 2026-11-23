"""Head-to-head comparisons for every pair of published models.

ELO turns small, noisy per-scenario differences into wins and losses, so two close models can
look far apart. A comparison reports what the runs support instead:

* **Score difference**: per scenario, ``b``'s call score averaged over its runs minus ``a``'s,
  then the mean over scenarios with a 95% t-interval. ``separable`` is true when the interval
  excludes zero. The same goes for each family, the task outcome and reply time.
* **Clear skill changes**: a skill counts only when every run of one model beats every run of the
  other on its pass rate *and* the pooled pass rates differ by a two-proportion z-test with a
  Benjamini-Hochberg q below the configured bound (the false-discovery control across a pair's
  skills). Skills that move in every run but stay within chance are counted, not listed: about one
  such skill per pair turns up by chance alone.

Output depends only on the inputs and the config; nothing here is random.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from itertools import combinations
from typing import Any

from s2s_human_bench.bench import Entry
from s2s_human_bench.calls import Call
from s2s_human_bench.config import Config

DEFAULTS = {"level": 0.95, "skillQ": 0.1, "minSkillCallsPerRun": 5}


def t_quantile(p: float, df: int) -> float:
    """Upper quantile of Student's t (Cornish-Fisher from the normal; within 0.002 of exact for df >= 20)."""
    z = statistics.NormalDist().inv_cdf(p)
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def per_scenario(entry: Entry, value: Callable[[Call], float | None]) -> dict[str, float]:
    """Each scenario's value averaged over the runs that have one."""
    vals: dict[str, list[float]] = {}
    for run in entry.runs:
        for code, call in run.items():
            v = value(call)
            if v is not None:
                vals.setdefault(code, []).append(v)
    return {code: statistics.mean(v) for code, v in vals.items()}


def paired(a: Entry, b: Entry, value: Callable[[Call], float | None], level: float, digits: int = 1) -> dict[str, Any] | None:
    """Mean of b minus a over shared scenarios, with its t-interval; None with fewer than three scenarios."""
    sa, sb = per_scenario(a, value), per_scenario(b, value)
    d = [sb[c] - sa[c] for c in sorted(sa) if c in sb]
    if len(d) < 3:
        return None
    m = statistics.mean(d)
    half = t_quantile(0.5 + level / 2, len(d) - 1) * statistics.stdev(d) / math.sqrt(len(d))
    lo, hi = m - half, m + half
    return {"diff": round(m, digits), "lo": round(lo, digits), "hi": round(hi, digits), "scenarios": len(d), "separable": lo > 0 or hi < 0}


def skill_counts(run: dict[str, Call], skill: str) -> tuple[int, int]:
    """(passed, judged) for one skill in one run."""
    judged = [c for c in run.values() if skill in c.tested]
    return sum(skill not in c.misses for c in judged), len(judged)


def bh(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q), in input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    q = [1.0] * len(pvalues)
    running = 1.0
    for rank in range(len(order), 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * len(pvalues) / rank)
        q[i] = running
    return q


def skill_changes(cfg: Config, a: Entry, b: Entry, settings: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Clear skill changes from a to b, largest first, and how many more moved in every run within chance."""
    rows = []
    for skill in cfg.skills:
        ca = [skill_counts(r, skill.id) for r in a.runs]
        cb = [skill_counts(r, skill.id) for r in b.runs]
        if min(n for _, n in ca + cb) < settings["minSkillCallsPerRun"]:
            continue
        ka, na = sum(k for k, _ in ca), sum(n for _, n in ca)
        kb, nb = sum(k for k, _ in cb), sum(n for _, n in cb)
        pa, pb, pooled = ka / na, kb / nb, (ka + kb) / (na + nb)
        se = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
        p = math.erfc(abs(pb - pa) / se / math.sqrt(2)) if se else 1.0
        ra, rb = [k / n for k, n in ca], [k / n for k, n in cb]
        consistent = min(rb) > max(ra) or max(rb) < min(ra)
        rows.append((skill.id, pa, pb, na, nb, ra, rb, consistent, p))
    q = bh([r[-1] for r in rows])
    clear, within = [], 0
    for row, qv in zip(rows, q, strict=True):
        sid, pa, pb, na, nb, ra, rb, consistent, _ = row
        if consistent and qv < settings["skillQ"]:
            clear.append(
                {
                    "skill": sid,
                    "a": round(100 * pa),
                    "b": round(100 * pb),
                    "calls": [na, nb],
                    "runsA": [round(100 * x) for x in ra],
                    "runsB": [round(100 * x) for x in rb],
                    "q": round(qv, 3),
                }
            )
        elif consistent:
            within += 1
    clear.sort(key=lambda s: -abs(s["b"] - s["a"]))
    return clear, within


def comparisons_doc(cfg: Config, entries: list[Entry]) -> dict[str, Any]:
    """Every pair of models, in config order (a before b); a page shows b - a or flips the sign."""
    settings = {**DEFAULTS, **cfg.scoring.get("comparisons", {})}
    level = settings["level"]
    families = dict(cfg.families)
    pairs = []
    for a, b in combinations(entries, 2):
        score = paired(a, b, lambda c: c.score, level)
        if score is None:
            continue
        clear, within = skill_changes(cfg, a, b, settings)
        pairs.append(
            {
                "a": a.model.site_id,
                "b": b.model.site_id,
                "score": score,
                "families": {label: paired(a, b, lambda c, k=key: c.families.get(k), level) for key, label in families.items()},
                "task": paired(a, b, lambda c: c.outcome, level),
                "replySeconds": paired(a, b, lambda c: c.latency_ms / 1000 if c.latency_ms else None, level, digits=2),
                "skills": clear,
                "withinChance": within,
            }
        )
    return {
        "method": {
            "level": level,
            "difference": "per scenario, b's call score averaged over its runs minus a's; mean over scenarios with a t-interval",
            "clearSkill": f"every run of one model beats every run of the other, and Benjamini-Hochberg q < {settings['skillQ']}",
            "minSkillCallsPerRun": settings["minSkillCallsPerRun"],
        },
        "skills": {s.id: s.name for s in cfg.skills},
        "pairs": pairs,
    }
