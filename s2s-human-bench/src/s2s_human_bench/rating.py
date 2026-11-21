"""ELO-scale ratings from head-to-head call scores.

Two models "play" every scenario both completed: the higher mean call score wins, and
scores within the tie margin draw. Ranked models get Bradley-Terry ratings (mean 1000 on a
400-point logistic scale). A reference system is placed on that scale afterwards with the
ranked ratings held fixed, so adding it moves no one else.
"""

from __future__ import annotations

import math
import random
import statistics

Scores = dict[str, dict[str, float]]  # scenario -> model -> mean call score
Match = tuple[str, str, float]  # (a, b, result for a: 1 win, 0.5 draw, 0 loss)


def outcome(a: float, b: float, tie_margin: float) -> float:
    return 0.5 if abs(a - b) <= tie_margin else float(a > b)


def matches_for(scores: Scores, models: list[str], tie_margin: float) -> list[Match]:
    out = []
    for code in scores:
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                sa, sb = scores[code].get(a), scores[code].get(b)
                if sa is not None and sb is not None:
                    out.append((a, b, outcome(sa, sb, tie_margin)))
    return out


def bradley_terry(matches: list[Match], models: list[str], iters: int = 200) -> dict[str, float]:
    """Minorise-maximise Bradley-Terry fit; ratings on the ELO scale with mean 1000."""
    p = {m: 1.0 for m in models}
    wins = {m: 0.0 for m in models}
    pairs: dict[tuple[str, str], int] = {}
    for a, b, s in matches:
        wins[a] += s
        wins[b] += 1 - s
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    for _ in range(iters):
        new = {}
        for m in models:
            den = 0.0
            for (a, b), n in pairs.items():
                if m in (a, b):
                    o = b if m == a else a
                    den += n / (p[m] + p[o])
            new[m] = (wins[m] + 0.5) / (den + 1.0 / p[m])  # weak prior keeps unbeaten models finite
        g = statistics.geometric_mean(new.values())
        p = {m: v / g for m, v in new.items()}
    return {m: 1000 + 400 * math.log10(v) for m, v in p.items()}


def anchored_rating(ref: str, scores: Scores, ratings: dict[str, float], tie_margin: float) -> float | None:
    """The rating at which ``ref``'s expected score against the ranked models (ratings fixed)
    equals its actual score. None when it shared no scenario with them."""
    games = []
    for code in scores:
        sr = scores[code].get(ref)
        if sr is None:
            continue
        for m, r in ratings.items():
            so = scores[code].get(m)
            if so is not None:
                games.append((r, outcome(sr, so, tie_margin)))
    if not games:
        return None
    actual = sum(s for _, s in games)
    lo, hi = 0.0, 2000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        expected = sum(1 / (1 + 10 ** ((r - mid) / 400)) for r, _ in games)
        lo, hi = (mid, hi) if expected < actual else (lo, mid)
    return (lo + hi) / 2


def ratings(scores: Scores, ranked: list[str], references: list[str], tie_margin: float, iters: int = 200) -> dict[str, float]:
    """Ranked models by Bradley-Terry, then each reference anchored on their scale."""
    out = bradley_terry(matches_for(scores, ranked, tie_margin), ranked, iters)
    fixed = dict(out)
    for ref in references:
        r = anchored_rating(ref, scores, fixed, tie_margin)
        if r is not None:
            out[ref] = r
    return out


def bootstrap_ci(scores: Scores, ranked: list[str], tie_margin: float, draws: int, seed: int, iters: int) -> dict[str, list[float]]:
    """95% interval per ranked model from resampling scenarios with replacement."""
    rng = random.Random(seed)
    codes = sorted(scores)
    boots: dict[str, list[float]] = {k: [] for k in ranked}
    for _ in range(draws):
        sample = [rng.choice(codes) for _ in codes]
        r = bradley_terry(matches_for({f"{c}#{i}": scores[c] for i, c in enumerate(sample)}, ranked, tie_margin), ranked, iters)
        for k in ranked:
            boots[k].append(r[k])
    out = {}
    for k, values in boots.items():
        s = sorted(values)
        out[k] = [s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]]
    return out
