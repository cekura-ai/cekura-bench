"""ELO-scale ratings from head-to-head call scores.

Two models "play" every scenario both completed. With a rating scale, the result follows the
size of the gap: ``1 / (1 + 10^(-gap / scale))``, so a near-tie is close to a draw and a wide gap
close to a win. Without one, the higher mean call score wins and scores within the tie margin
draw. Ranked models get Bradley-Terry ratings (mean 1000 on a 400-point logistic scale). A
reference system is placed on that scale afterwards with the ranked ratings held fixed, so
adding it moves no one else.
"""

from __future__ import annotations

import math
import random
import statistics

Scores = dict[str, dict[str, float]]  # scenario -> model -> mean call score
Match = tuple[str, str, float]  # (a, b, result for a: 1 win, 0.5 draw, 0 loss)


def outcome(a: float, b: float, tie_margin: float, scale: float | None = None) -> float:
    """Result for ``a``: by the size of the gap when ``scale`` is given, else win, draw within the margin, or loss."""
    if scale:
        return 1 / (1 + 10 ** (-(a - b) / scale))
    return 0.5 if abs(a - b) <= tie_margin else float(a > b)


def matches_for(scores: Scores, models: list[str], tie_margin: float, scale: float | None = None) -> list[Match]:
    out = []
    for code in scores:
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                sa, sb = scores[code].get(a), scores[code].get(b)
                if sa is not None and sb is not None:
                    out.append((a, b, outcome(sa, sb, tie_margin, scale)))
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


def anchored_rating(ref: str, scores: Scores, ratings: dict[str, float], tie_margin: float, scale: float | None = None) -> float | None:
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
                games.append((r, outcome(sr, so, tie_margin, scale)))
    if not games:
        return None
    actual = sum(s for _, s in games)
    lo, hi = 0.0, 2000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        expected = sum(1 / (1 + 10 ** ((r - mid) / 400)) for r, _ in games)
        lo, hi = (mid, hi) if expected < actual else (lo, mid)
    return (lo + hi) / 2


def ratings(scores: Scores, ranked: list[str], references: list[str], tie_margin: float, iters: int = 200, scale: float | None = None) -> dict[str, float]:
    """Ranked models by Bradley-Terry, then each reference anchored on their scale."""
    out = bradley_terry(matches_for(scores, ranked, tie_margin, scale), ranked, iters)
    fixed = dict(out)
    for ref in references:
        r = anchored_rating(ref, scores, fixed, tie_margin, scale)
        if r is not None:
            out[ref] = r
    return out


def bootstrap_draws(
    scores: Scores, ranked: list[str], tie_margin: float, draws: int, seed: int, iters: int, scale: float | None = None
) -> list[dict[str, float]]:
    """Ratings refitted on ``draws`` resamples of the scenarios (with replacement), seeded."""
    rng = random.Random(seed)
    codes = sorted(scores)
    out = []
    for _ in range(draws):
        sample = [rng.choice(codes) for _ in codes]
        out.append(bradley_terry(matches_for({f"{c}#{i}": scores[c] for i, c in enumerate(sample)}, ranked, tie_margin, scale), ranked, iters))
    return out


def bootstrap_ci(scores: Scores, ranked: list[str], tie_margin: float, draws: int, seed: int, iters: int, scale: float | None = None) -> dict[str, list[float]]:
    """95% interval per ranked model from resampling scenarios with replacement."""
    return interval(bootstrap_draws(scores, ranked, tie_margin, draws, seed, iters, scale), ranked)


def interval(boots: list[dict[str, float]], ranked: list[str]) -> dict[str, list[float]]:
    out = {}
    for k in ranked:
        s = sorted(b[k] for b in boots)
        out[k] = [s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]]
    return out


def rank_ranges(boots: list[dict[str, float]], ranked: list[str]) -> dict[str, list[int]]:
    """Each model's 95% range of places (1 = top) across the resampled fits."""
    places: dict[str, list[int]] = {k: [] for k in ranked}
    for b in boots:
        for place, k in enumerate(sorted(ranked, key=lambda m: -b[m]), start=1):
            places[k].append(place)
    out = {}
    for k, v in places.items():
        s = sorted(v)
        out[k] = [s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]]
    return out
