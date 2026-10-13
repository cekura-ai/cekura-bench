"""Experimental authored Track 2 measurements, independent of TurnBench scoring.

A stop means 200 ms of observed quiet after the exact stimulus injection boundary, on the
reference client's playout clock. It is an acoustic observation, not proof that
the model yielded because of the caller: a natural reply ending can also stop.
Provider event timestamps alone never establish a stop. All exclusions survive
into the summary; conversation-cluster intervals accompany pooled event rates.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from turn_bench.author import service_bridge

SCHEMA = "turn-bench/s2s-authored-interaction/1"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 7
SUCCESS_WINDOW_S = 1.0

NOTES = [
    "Experimental authored Track 2; independent of the unchanged official Sesame Track 1 scorer.",
    "A stop is the first >=200 ms observed silence after stimulus injection, with sustained agent speech overlapping that onset. The success window is 1.0 s.",
    "Agent PCM is measured on the reference client playout clock, including buffer cuts. Caller timing is the exact first transmitted PCM sample, not the scheduled target. Raw clip padding is retained and acoustic onset is diagnostic only.",
    "Acoustic stops can coincide with natural reply ends or pauses: these measurements do not establish that an interruption caused the stop. A playback cut is recorded as supporting evidence, not used as a substitute for observed silence.",
    "Backchannel tolerance means no qualifying stop within 1.0 s; it does not test the semantic content of a subsequent reply.",
    "Voids are counted and excluded from denominators. Missing observations are never inferred from missing provider events or the end of the delivered audio buffer.",
    "95% percentile intervals resample whole conversations (2000 draws, seed 7), pooling event counts and successful barge-in latencies. Fewer than two scored conversations yields no interval.",
]


def _void(record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {**record, "void": reason, "verdict": None, "hit": None, "fp": None,
            "stop_ms": None, "measurement": None}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def score_event(record: dict[str, Any], adapter: Any) -> dict[str, Any]:
    """Score a runner record against its adapter's recorded PCM and timeline.

    ``observation_end_s`` must be supplied by the runner on the same wall clock
    as the timeline. Future queued audio is not evidence of observation. Every event requires
    observation through injection + 1.2 s, including early confirmed stops,
    so exclusions cannot selectively retain early successes.
    """
    if record.get("void"):
        return _void(record, str(record["void"]))
    if record.get("kind") not in {"barge_in", "backchannel"}:
        raise ValueError(f"unknown authored event kind: {record.get('kind')!r}")
    start = record.get("caller_start_s")
    caller_end = record.get("caller_end_s")
    observed = record.get("observation_end_s")
    if not all(_finite(x) for x in (start, caller_end, observed)):
        return _void(record, "missing_or_invalid_actual_timing")
    if caller_end <= start:
        return _void(record, "invalid_caller_timing")
    if observed < start + 1.2 - 1e-8:
        return _void(record, "insufficient_stop_observation")
    next_start = record.get("next_caller_start_s")
    if next_start is not None:
        if not _finite(next_start):
            return _void(record, "missing_or_invalid_actual_timing")
        if start <= next_start < start + 1.2 - 1e-8:
            return _void(record, "next_caller_contaminates_observation")
    timeline = adapter.agent_timeline
    rate = timeline.rate
    raw = np.frombuffer(bytes(adapter.agent_pcm), dtype=np.int16)
    if rate <= 0 or len(raw) != timeline.n_samples:
        return _void(record, "invalid_agent_pcm_timeline")
    if not timeline.chunks:
        return _void(record, "no_agent_audio")

    detector = service_bridge.detector()
    frame_s = detector.FRAME_MS / 1000.0
    confirm_s = detector.OFFSET_FRAMES * frame_s
    # Bound allocation irrespective of conversation length. The pre-roll gives
    # sustained-speech evidence at onset; the tail confirms a stop at 1.0 s.
    window_start = start - 0.1
    window_end = min(observed, start + SUCCESS_WINDOW_S + confirm_s)
    n_frames = max(0, int(math.floor((window_end - window_start) / frame_s + 1e-8)))
    frame_samples = max(1, round(rate * frame_s))
    played = np.zeros(n_frames * frame_samples, dtype=np.int16)
    for chunk in timeline.chunks:
        begins, ends = timeline.playout_span(chunk.seq)
        left, right = max(begins, window_start), min(ends, window_end)
        if right <= left:
            continue
        dest = max(0, round((left - window_start) * rate))
        source = chunk.first_sample + max(0, round((left - begins) * rate))
        length = min(round((right - left) * rate), len(played) - dest,
                     chunk.last_sample - source)
        if length > 0:
            played[dest:dest + length] = raw[source:source + length]
    energy = detector.frame_energy_db(played, rate)
    # The reconstructed client channel is noiseless during empty playback.
    # Include known digital silence only in floor calibration, not time evidence.
    floor = detector.noise_floor_db(np.concatenate([energy, np.full(len(energy), detector.SILENCE_DB)]))
    threshold = floor + detector.MARGIN_DB
    loud = energy > threshold
    onset_idx = round((start - window_start) / frame_s)
    overlap = False
    for first in range(max(0, onset_idx - detector.ONSET_FRAMES + 1), onset_idx + 1):
        last = first + detector.ONSET_FRAMES
        if last <= len(loud) and first <= onset_idx < last and bool(loud[first:last].all()):
            overlap = True
            break
    if not overlap:
        if len(loud) < onset_idx + detector.ONSET_FRAMES:
            return _void(record, "insufficient_overlap_observation")
        return _void(record, "no_agent_speech_overlap")

    stop_s = None
    for i in range(onset_idx, len(loud) - detector.OFFSET_FRAMES + 1):
        if not loud[i:i + detector.OFFSET_FRAMES].any():
            stop_s = window_start + i * frame_s
            break
    within = stop_s is not None and stop_s - start <= SUCCESS_WINDOW_S + 1e-8
    barge = record["kind"] == "barge_in"
    passed = within if barge else not within
    cut = any(start <= at <= (stop_s if stop_s is not None else start + SUCCESS_WINDOW_S) + frame_s
              for at in timeline.cuts)
    return {
        **record, "void": None, "verdict": "pass" if passed else "fail",
        "hit": bool(within) if barge else None,
        "fp": bool(within) if not barge else None,
        "stop_ms": None if not within else round(max(0.0, stop_s - start) * 1000.0, 3),
        "measurement": {"clock": "reference_client_playout", "frame_ms": detector.FRAME_MS,
                        "silence_ms": confirm_s * 1000, "window_ms": SUCCESS_WINDOW_S * 1000,
                        "noise_floor_db": floor, "threshold_db": threshold,
                        "playback_cut_before_stop": cut, "causal_yield_verified": False},
    }


def _interval(values: list[float]) -> list[float] | None:
    return [float(x) for x in np.percentile(values, [2.5, 97.5])] if values else None


def _category(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    selected = [r for r in rows if r["kind"] == kind]
    scored = [r for r in selected if not r.get("void")]
    for row in scored:
        if row.get("verdict") not in {"pass", "fail"}:
            raise ValueError("unscored event must have an explicit void reason")
    clusters = sorted({r["conversation_id"] for r in scored})
    by_cluster = [[r for r in scored if r["conversation_id"] == c] for c in clusters]
    passed = sum(r["verdict"] == "pass" for r in scored)
    latencies = [r["stop_ms"] for r in scored if kind == "barge_in" and r["verdict"] == "pass" and r.get("stop_ms") is not None]
    rates, p50s, p90s = [], [], []
    if len(clusters) >= 2:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for _ in range(BOOTSTRAP_DRAWS):
            draw = [r for i in rng.integers(0, len(clusters), len(clusters)) for r in by_cluster[i]]
            rates.append(sum(r["verdict"] == "pass" for r in draw) / len(draw))
            times = [r["stop_ms"] for r in draw if kind == "barge_in" and r["verdict"] == "pass" and r.get("stop_ms") is not None]
            if times:
                p50s.append(float(np.percentile(times, 50)))
                p90s.append(float(np.percentile(times, 90)))
    return {
        "value": passed / len(scored) if scored else None, "ci95": _interval(rates),
        "n": len(scored), "passed": passed, "failed": len(scored) - passed,
        "cells": len(selected), "voids": len(selected) - len(scored),
        "void_reasons": dict(Counter(r["void"] for r in selected if r.get("void"))),
        "conversations": len(clusters), "draws_defined": len(rates),
        "yield_ms": {"p50": float(np.percentile(latencies, 50)) if latencies else None,
                     "p90": float(np.percentile(latencies, 90)) if latencies else None,
                     "p50_ci95": _interval(p50s), "p90_ci95": _interval(p90s),
                     "n": len(latencies), "draws_defined": len(p50s)},
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool scored events; never average conversation rates or hide exclusions."""
    for row in records:
        if not row.get("conversation_id"):
            raise ValueError("conversation_id is required for cluster bootstrap")
        if row.get("kind") not in {"barge_in", "backchannel"}:
            raise ValueError(f"unknown authored event kind: {row.get('kind')!r}")
    barge = _category(records, "barge_in")
    back = _category(records, "backchannel")
    back["fp"] = back["failed"]
    return {"schema": SCHEMA, "interruption_handling": barge,
            "backchannel_handling": back, "yield_ms": barge["yield_ms"],
            "cells": len(records), "scored": barge["n"] + back["n"],
            "voids": dict(Counter(r["void"] for r in records if r.get("void"))),
            "bootstrap": {"method": "percentile conversation-cluster bootstrap; pooled events",
                          "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "level": 0.95},
            "notes": list(NOTES)}


def render_markdown(summary: dict[str, Any]) -> str:
    def rate(row: dict[str, Any]) -> str:
        if row["value"] is None:
            return f"n/a (n={row['n']})"
        ci = row["ci95"]
        interval = "CI n/a" if ci is None else f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
        return f"{row['value']:.3f} ({interval}; n={row['n']})"
    y = summary["yield_ms"]
    def latency_value(name: str) -> str:
        ci = y[f"{name}_ci95"]
        interval = "CI n/a" if ci is None else f"95% CI [{ci[0]:.0f}, {ci[1]:.0f}]"
        return f"{y[name]:.0f} ({interval})"
    latency = "n/a" if y["p50"] is None else f"{latency_value('p50')} / {latency_value('p90')} ms (n={y['n']})"
    lines = ["# Authored Track 2 — experimental", "",
             "| Interruption handling | Backchannel tolerance | Yield p50 / p90 |", "|---|---|---|",
             f"| {rate(summary['interruption_handling'])} | {rate(summary['backchannel_handling'])} | {latency} |", "",
             f"Events: {summary['cells']}; scored: {summary['scored']}; excluded: {sum(summary['voids'].values())}.", ""]
    lines += [f"- Excluded {reason}: {count}" for reason, count in sorted(summary["voids"].items())]
    lines += [""] + [f"- {note}" for note in summary["notes"]] + [""]
    return "\n".join(lines)
