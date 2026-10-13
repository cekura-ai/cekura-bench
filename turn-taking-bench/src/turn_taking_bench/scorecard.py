"""Conversational Dynamics scorecard and conversation-level bootstrap intervals.

Nothing here re-scores anything: the inputs are Sesame's per-conversation
``TaskScore`` objects (tp/fn/fp/tn and the TP latencies) exactly as
``turnbench.score`` produced them. This module only (a) maps the pooled EOT /
INT rates onto the four Artificial Analysis "Conversational Dynamics" names
and (b) puts a confidence interval on every number by resampling whole
conversations.

**Scorecard** (per block, per breakdown subset):

    pause handling         = 1 - EOT fp_rate   (mid-turn pauses not cut into)
    turn taking            = EOT recall        (EOT latency p50 / p90 shown beside it)
    interruption handling  = INT recall
    backchannel handling   = 1 - INT fp_rate
    overall                = weighted mean of the values that are *defined*

A value is defined only when events of its kind exist (``n > 0``); otherwise
it is ``None`` ("n/a"), never 0 or 1, and the overall says which components it
is made of. The default weights are equal; they are a presentation choice
(``SCORECARD_WEIGHTS``) and are recorded in the report, so a reader can
recompute the overall under other weights from the four values.

**Confidence intervals** are 95 % percentile cluster-bootstrap intervals, the
convention of ``service/report.py`` in this repository: ``BOOTSTRAP_DRAWS``
draws, ``BOOTSTRAP_SEED`` fixed and recorded in ``report.json``. The cluster
is the conversation: each draw resamples the scored conversations with
replacement, *sums their tp/fn/fp/tn* and takes the pooled rate — it never
averages per-conversation rates. Latency intervals pool the TP latencies of
the resampled conversations and take the percentile. Every statistic of a
block comes from the same draw matrix, so the overall's interval is consistent
with its components'. A draw in which a rate's denominator is zero contributes
no value to that rate's interval (``draws_defined`` records how many did), and
a block scored on fewer than two conversations gets no interval at all.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from turnbench.score import ConversationScores, TaskScore

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 7
CI_LEVEL = 0.95
CI_PERCENTILES = (100 * (1 - CI_LEVEL) / 2, 100 * (1 + CI_LEVEL) / 2)

# Scorecard rows: name -> (task, rate, inverted). ``inverted`` means 1 - rate.
SCORECARD = {
    "pause_handling": ("eot", "fp_rate", True),
    "turn_taking": ("eot", "recall", False),
    "interruption_handling": ("int", "recall", False),
    "backchannel_handling": ("int", "fp_rate", True),
}
SCORECARD_TITLES = {
    "pause_handling": "Pause handling",
    "turn_taking": "Turn taking",
    "interruption_handling": "Interruption handling",
    "backchannel_handling": "Backchannel handling",
    "overall": "Overall",
}
SCORECARD_SOURCES = {
    "pause_handling": "1 - EOT fp_rate",
    "turn_taking": "EOT recall",
    "interruption_handling": "INT recall",
    "backchannel_handling": "1 - INT fp_rate",
}
SCORECARD_WEIGHTS = {name: 1.0 for name in SCORECARD}
TASK_ATTRS = {"eot": "task_eot", "int": "task_int"}


def bootstrap_record() -> dict[str, Any]:
    """What ``report.json`` records about how every interval was computed."""
    return {
        "method": "percentile cluster bootstrap; cluster = conversation; pooled counts per draw",
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "level": CI_LEVEL,
    }


def resolve_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Full weight table from an override of some rows; rejects unknown rows
    and non-positive weights."""
    resolved = dict(SCORECARD_WEIGHTS)
    for name, weight in (weights or {}).items():
        if name not in SCORECARD:
            raise KeyError(f"unknown scorecard row {name!r}; known: {', '.join(SCORECARD)}")
        if not (isinstance(weight, (int, float)) and math.isfinite(weight) and weight > 0):
            raise ValueError(f"weight for {name!r} must be a positive number, got {weight!r}")
        resolved[name] = float(weight)
    return resolved


# ---- pooled counts -------------------------------------------------------------


class Counts:
    """tp/fn/fp/tn of one task for every conversation, as columns, plus the
    pooled TP latencies with the conversation each came from. Everything the
    bootstrap needs, in a shape that resamples with fancy indexing."""

    def __init__(self, scores: Sequence[TaskScore]) -> None:
        self.tp = np.array([s.tp for s in scores], dtype=np.int64)
        self.fn = np.array([s.fn for s in scores], dtype=np.int64)
        self.fp = np.array([s.fp for s in scores], dtype=np.int64)
        self.tn = np.array([s.tn for s in scores], dtype=np.int64)
        self.latencies = np.array([lat for s in scores for lat in s.latencies_ms], dtype=np.float64)
        self.latency_owner = np.array([i for i, s in enumerate(scores) for _ in s.latencies_ms], dtype=np.int64)

    @property
    def conversations(self) -> int:
        return int(self.tp.size)

    def pooled(self, draws: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Summed counts: over all conversations (shape ``()``), or per draw
        row of conversation indices (shape ``(draws,)``)."""
        if draws is None:
            return {k: getattr(self, k).sum() for k in ("tp", "fn", "fp", "tn")}
        return {k: getattr(self, k)[draws].sum(axis=1) for k in ("tp", "fn", "fp", "tn")}

    def latency_percentile(self, q: float, draw: np.ndarray | None = None) -> float:
        """Percentile of the pooled TP latencies — of every conversation, or of
        the conversations in one draw (a conversation drawn k times contributes
        its latencies k times, exactly as summing its counts k times does)."""
        if draw is None:
            pooled = self.latencies
        else:
            multiplicity = np.bincount(draw, minlength=self.conversations)
            pooled = np.repeat(self.latencies, multiplicity[self.latency_owner])
        return float(np.percentile(pooled, q)) if pooled.size else float("nan")


def rate(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise ``numerator / denominator`` with NaN where the denominator is 0."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), np.nan)


def task_rates(pooled: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "recall": rate(pooled["tp"], pooled["tp"] + pooled["fn"]),
        "fp_rate": rate(pooled["fp"], pooled["fp"] + pooled["tn"]),
    }


def scorecard_values(rates: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The four scorecard rows from ``{"eot": {"recall", "fp_rate"}, "int": {...}}``;
    NaN propagates from an undefined rate."""
    return {
        name: (1.0 - rates[task][key]) if inverted else rates[task][key]
        for name, (task, key, inverted) in SCORECARD.items()
    }


def overall_value(values: dict[str, np.ndarray], defined: Sequence[str], weights: dict[str, float]) -> np.ndarray:
    """Weighted mean over ``defined`` rows (the rows the point estimate has);
    NaN for any draw in which one of them is undefined."""
    if not defined:
        return np.asarray(np.nan)
    total = sum(weights[name] for name in defined)
    return sum(weights[name] * np.asarray(values[name], dtype=np.float64) for name in defined) / total


# ---- bootstrap ------------------------------------------------------------------


def draw_matrix(conversations: int, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """``draws`` rows of ``conversations`` indices drawn with replacement, from
    a generator seeded with ``seed`` — the same matrix for the same inputs."""
    return np.random.default_rng(seed).integers(0, conversations, size=(draws, conversations))


def interval(samples: np.ndarray) -> dict[str, Any]:
    """Percentile interval over the defined draws; ``None`` bounds when no
    draw is defined. ``draws_defined`` says how many were."""
    samples = np.asarray(samples, dtype=np.float64)
    defined = samples[np.isfinite(samples)]
    if defined.size == 0:
        return {"lo": None, "hi": None, "draws_defined": 0}
    lo, hi = np.percentile(defined, CI_PERCENTILES)
    return {"lo": float(lo), "hi": float(hi), "draws_defined": int(defined.size)}


def _number(value: Any) -> float | None:
    value = float(value)
    return None if math.isnan(value) else value


def _ci(samples: np.ndarray | None) -> dict[str, Any] | None:
    return None if samples is None else interval(samples)


def bootstrap_block(
    scores: Sequence[ConversationScores],
    weights: dict[str, float] | None = None,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Scorecard and intervals for one set of conversations (a block, or a
    block restricted to a breakdown subset).

    Returns ``{"scorecard": ..., "ci95": ...}`` where ``ci95`` mirrors the
    ``aggregate`` layout (``eot`` / ``int`` → ``recall``, ``fp_rate``,
    ``latency_ms.p50``) and ``scorecard`` holds the four rows plus ``overall``.
    Every row carries its point ``value``, its ``ci95`` and the ``n`` events
    behind it; see the module docstring for the rules."""
    weights = resolve_weights(weights)
    counts = {task: Counts([getattr(s, attr) for s in scores]) for task, attr in TASK_ATTRS.items()}
    n = len(scores)
    matrix = draw_matrix(n, draws, seed) if n >= 2 else None

    point_rates = {task: task_rates(c.pooled()) for task, c in counts.items()}
    draw_rates = {task: task_rates(c.pooled(matrix)) for task, c in counts.items()} if matrix is not None else None
    point_values = scorecard_values(point_rates)
    draw_values = scorecard_values(draw_rates) if draw_rates is not None else None
    defined = [name for name, value in point_values.items() if not math.isnan(float(value))]
    omitted = {name: "no events" for name in SCORECARD if name not in defined}

    def events_behind(name: str) -> int:
        task, key, _inverted = SCORECARD[name]
        pooled = counts[task].pooled()
        return int(pooled["tp"] + pooled["fn"]) if key == "recall" else int(pooled["fp"] + pooled["tn"])

    scorecard: dict[str, Any] = {}
    for name in SCORECARD:
        scorecard[name] = {
            "value": _number(point_values[name]),
            "ci95": _ci(None if draw_values is None else draw_values[name]),
            "n": events_behind(name),
            "source": SCORECARD_SOURCES[name],
        }
    latency = {task: _latency_entry(c, matrix) for task, c in counts.items()}
    scorecard["turn_taking"]["latency_ms"] = latency["eot"]
    scorecard["overall"] = {
        "value": _number(overall_value(point_values, defined, weights)),
        "ci95": _ci(None if draw_values is None else overall_value(draw_values, defined, weights)),
        "defined": defined,
        "omitted": omitted,
        "weights": {name: weights[name] for name in defined},
    }
    ci95 = {
        task: {
            "recall": _ci(None if draw_rates is None else draw_rates[task]["recall"]),
            "fp_rate": _ci(None if draw_rates is None else draw_rates[task]["fp_rate"]),
            "latency_ms": {"p50": latency[task]["p50"]["ci95"]},
        }
        for task in TASK_ATTRS
    }
    return {"conversations": n, "scorecard": scorecard, "ci95": ci95}


def _latency_entry(counts: Counts, matrix: np.ndarray | None) -> dict[str, Any]:
    """p50 (with interval) and p90 (point only) of the pooled TP latencies."""
    entry: dict[str, Any] = {"n": int(counts.latencies.size)}
    p50_draws = None
    if matrix is not None and counts.latencies.size:
        p50_draws = np.array([counts.latency_percentile(50, draw) for draw in matrix])
    entry["p50"] = {"value": _number(counts.latency_percentile(50)), "ci95": _ci(p50_draws)}
    entry["p90"] = {"value": _number(counts.latency_percentile(90))}
    return entry


# ---- comparison -----------------------------------------------------------------


def intervals_disjoint(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool | None:
    """Whether two intervals do not overlap; ``None`` when either is missing.

    A deliberately simple heuristic: disjoint 95 % intervals are strong
    evidence of a difference (stricter than a 5 % test), while overlapping
    intervals say nothing either way — they are *not* evidence of a tie."""
    if not a or not b or a.get("lo") is None or b.get("lo") is None:
        return None
    return a["hi"] < b["lo"] or b["hi"] < a["lo"]
