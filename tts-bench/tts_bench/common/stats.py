"""Statistics shared by every TTS table: percentile latency with a bootstrap CI,
and pass rates.

* Latency is P50 and P90 until a declared minimum count; the tails are withheld
  rather than reported from too few samples.
* Confidence intervals are percentile bootstrap intervals, seeded so they are
  reproducible.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


MIN_TAIL_N = 30              # P95/P99 appear only at or above this many latencies
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 7


def _rng() -> np.random.Generator:
    return np.random.default_rng(BOOTSTRAP_SEED)


def bootstrap_ci(values: Sequence[float], statistic, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float] | None:
    """Percentile bootstrap interval of ``statistic`` over ``values``; None below two samples."""
    data = np.asarray(values, dtype=float)
    if data.size < 2:
        return None
    rng = _rng()
    samples = rng.choice(data, size=(draws, data.size), replace=True)
    stats = np.apply_along_axis(statistic, 1, samples)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(lo), 1), round(float(hi), 1)


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0}
    arr = np.asarray(clean)
    out: dict[str, Any] = {
        "n": int(arr.size),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "min": round(float(arr.min()), 1),
        "max": round(float(arr.max()), 1),
        "p50_ci95": bootstrap_ci(clean, np.median),
    }
    if arr.size >= MIN_TAIL_N:
        out["p95"] = round(float(np.percentile(arr, 95)), 1)
        out["p99"] = round(float(np.percentile(arr, 99)), 1)
    else:
        out["tails_withheld"] = f"P95/P99 need n >= {MIN_TAIL_N}"
    return out


def rate_summary(passes: Sequence[bool]) -> dict[str, Any]:
    flags = [1.0 if p else 0.0 for p in passes]
    if not flags:
        return {"n": 0}
    ci = bootstrap_ci(flags, np.mean)
    return {
        "n": len(flags),
        "passed": int(sum(flags)),
        "rate": round(sum(flags) / len(flags), 4),
        "rate_ci95": None if ci is None else (round(ci[0] / 100.0 if ci[0] > 1 else ci[0], 4), round(ci[1] / 100.0 if ci[1] > 1 else ci[1], 4)),
    }
