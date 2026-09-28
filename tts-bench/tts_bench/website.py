"""What the website shows about a TTS campaign, computed from stored runs.

Two steps, so a run never has to move to be published:

* ``summarize(run_dir)`` reduces one run to its published numbers (a few
  kilobytes: no audio, no transcripts), wherever the run is stored.
* ``combine(summaries, models)`` joins the per-run summaries into the one file
  the site reads, with each model's display name and, where a sourced list
  price exists, its cost.

Every number says which cells it came from. Latency is P50/P90 (P95 from 30
cells) with a bootstrap interval on the median; word error is pooled (errors
over reference words); a span or pass count is given as passed/n. A probe the
protocol cannot serve is ``null`` with the reason, never a zero.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tts_bench.common.stats import latency_summary, rate_summary
from tts_bench.store import latest_cells, read_jsonl

SCHEMA = "tts-bench/site-summary/1"
SITE_SCHEMA = 1
RUNAWAY_FACTOR = 2.0      # a recording this many times the model's typical length for the same text...
RUNAWAY_EXCESS_MS = 3000  # ...and at least this much longer is a runaway, not a slower reading


def _exclusion(cell: dict[str, Any]) -> bool:
    return str(cell.get("void") or "").startswith("configuration not supported")


def _values(cells: Iterable[dict[str, Any]], field: str) -> list[float]:
    return [c["values"][field] for c in cells if c["values"].get(field) is not None]


def _lat(values: Sequence[float]) -> dict[str, Any] | None:
    s = latency_summary(values)
    if not s.get("n"):
        return None
    out = {"n": s["n"], "p50": s["p50"], "p90": s["p90"]}
    if "p95" in s:
        out["p95"] = s["p95"]
    if s.get("p50_ci95"):
        out["p50Ci95"] = [round(v, 1) for v in s["p50_ci95"]]
    return out


def _rate(flags: Sequence[bool]) -> dict[str, Any] | None:
    s = rate_summary(flags)
    return None if not s.get("n") else {"n": s["n"], "passed": s["passed"], "rate": s["rate"]}


def _pooled(rows: Sequence[dict[str, Any]], instrument: str) -> dict[str, Any] | None:
    errors = words = 0
    n = 0
    for row in rows:
        score = row["instruments"].get(instrument) or {}
        best = score.get("best")
        if not best:
            continue
        errors += best["errors"]
        words += best["reference_words"]
        n += 1
    return None if not words else {"wer": round(errors / words, 4), "recordings": n, "referenceWords": words}


def _spans(rows: Sequence[dict[str, Any]], instrument: str) -> dict[str, Any]:
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    listener = 0
    for row in rows:
        for span in (row["instruments"].get(instrument) or {}).get("spans") or ():
            if "pass" not in span:
                listener += 1
                continue
            by[span["category"]][0] += 1 if span["pass"] else 0
            by[span["category"]][1] += 1
    total = [sum(v[0] for v in by.values()), sum(v[1] for v in by.values())]
    return {
        "all": None if not total[1] else {"passed": total[0], "n": total[1], "rate": round(total[0] / total[1], 4)},
        "byCategory": {k: {"passed": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)} for k, v in sorted(by.items())},
        "forListener": listener,
    }


def summarize(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    provenance = json.loads((root / "provenance.json").read_text())
    plan = json.loads((root / "plan.json").read_text())["cells"]
    sessions = read_jsonl(root / "sessions.jsonl")
    cells = [c for c in latest_cells(root) if not c.get("sentinel")]
    sentinel = [c for c in latest_cells(root) if c.get("sentinel")]
    scores = read_jsonl(root / "scores-all.jsonl") if (root / "scores-all.jsonl").exists() else []

    by_probe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_probe[cell["probe"]].append(cell)
    measured = {p: [c for c in cs if not c.get("void")] for p, cs in by_probe.items()}
    excluded = {p: next((c["void"].split(": ", 1)[1] for c in cs if _exclusion(c)), None) for p, cs in by_probe.items()}

    one = measured.get("one_shot", [])
    rep = measured.get("repeat", [])
    streamed = measured.get("streamed_input", [])
    load = measured.get("concurrency", [])
    cancel = measured.get("cancel", [])
    cont = measured.get("continuation", [])

    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in one:
        by_cohort[cell["cohort"]].append(cell)

    # Every single synthesis a cell recorded, with its item, for reliability and runaway checks. The load
    # probe's streams are left out: a stall there reflects the account's concurrency limit, not the model,
    # and counting them would make a run with the probe look less reliable than the same model without it.
    singles: list[tuple[str, dict[str, Any]]] = [(c["item"], c["values"]) for c in one + streamed + cancel]
    singles += [(c["item"], c["values"][k]) for c in rep for k in ("first", "second") if isinstance(c["values"].get(k), dict)]
    singles += [(c["item"], c["values"][k]) for c in cont for k in ("whole", "framed") if isinstance(c["values"].get(k), dict)]
    lengths: dict[str, list[float]] = defaultdict(list)
    for item, s in singles:
        if s.get("audio_ms") and s.get("cancel_at_ms") is None:
            lengths[item].append(s["audio_ms"])
    typical = {item: float(np.median(v)) for item, v in lengths.items()}
    runaways = sum(1 for item, s in singles if s.get("cancel_at_ms") is None and s.get("audio_ms")
                   and s["audio_ms"] > RUNAWAY_FACTOR * typical[item] and s["audio_ms"] - typical[item] > RUNAWAY_EXCESS_MS)
    ended_by: dict[str, int] = defaultdict(int)
    for _, s in singles:
        ended_by[str(s.get("ended_by"))] += 1
    timeouts = ended_by.get("timeout", 0)

    instruments = sorted({name for row in scores for name in row["instruments"]})
    def rows(probe: str, cohort: str | None = None) -> list[dict[str, Any]]:
        return [r for r in scores if r["probe"] == probe and (cohort is None or r["cohort"] == cohort)]
    accuracy = {}
    for name in instruments:
        model = next((r["instruments"][name]["model"] for r in scores if name in r["instruments"]), None)
        normalizer = next((r["normalizer"] for r in scores), None)
        accuracy[name] = {
            "model": model, "normalizer": normalizer,
            "oneShot": _pooled(rows("one_shot"), name),
            "streamedInput": _pooled(rows("streamed_input"), name),
            "underLoad": _pooled(rows("concurrency"), name),
            "byCohort": {cohort: _pooled(rows("one_shot", cohort), name) for cohort in sorted(by_cohort)},
            "spans": _spans(rows("one_shot"), name),
            "instrumentErrors": sum(1 for r in scores if "error" in (r["instruments"].get(name) or {})),
        }
    disagreements = sum(1 for r in scores if r.get("instruments_disagree"))

    fails = sum(1 for cs in measured.values() for c in cs if c.get("verdict") == "fail")
    graded = sum(1 for cs in measured.values() for c in cs if c.get("verdict") in ("pass", "fail"))
    refused = sum(1 for c in cells if str(c.get("void") or "").startswith("provider refused"))
    paragraphs = by_cohort.get("paragraph", [])

    return {
        "schema": SCHEMA,
        "run": root.name,
        "provider": provenance["provider"], "model": provenance["model"], "voice": provenance["voice"],
        "adapter": provenance["adapter"], "sampleRateHz": provenance["sample_rate"],
        "capabilities": provenance["capabilities"],
        "site": (sessions[0].get("client") or {}).get("site") if sessions else None,
        "harnessCommit": provenance["harness"]["commit"], "harnessDirty": provenance["harness"]["dirty"],
        "corpusVersion": provenance["corpus_version"], "methodology": provenance["methodology_version"],
        "startedUtc": provenance["started_utc"], "repeats": provenance["repeats"],
        "cells": {
            "planned": len(plan), "completed": len(latest_cells(root)),
            "excludedByProbe": {p: r for p, r in excluded.items() if r},
            "providerRefused": refused, "graded": graded, "failed": fails,
        },
        "sentinel": _lat(_values(sentinel, "ttfa_ms")),
        "latency": {
            "ttfa": _lat(_values(one, "ttfa_ms")),
            "roundtrip": _lat(_values(one, "roundtrip_ms")),
            "leadingSilence": _lat(_values(one, "leading_silence_ms")),
            "warmSecondRequest": _lat([c["values"]["second"]["ttfa_ms"] for c in rep
                                       if isinstance(c["values"].get("second"), dict) and c["values"]["second"].get("ttfa_ms") is not None]),
            "byCohort": {cohort: _lat(_values(cs, "ttfa_ms")) for cohort, cs in sorted(by_cohort.items())},
        },
        "streamedInput": None if excluded.get("streamed_input") else {
            "ttfaFromFirstWord": _lat(_values(streamed, "ttfa_ms")),
            "ttfaFromLastWord": _lat(_values(streamed, "ttfa_from_input_done_ms")),
            "startedBeforeInputDone": _rate([bool(v) for v in _values(streamed, "started_before_input_done")]),
            "wordsPerSecond": 30,
        },
        "underLoad": {
            "streams": 8,
            "perStreamTtfa": _lat([s["ttfa_ms"] for c in load for s in c["values"].get("per_stream") or () if s.get("ttfa_ms") is not None]),
            "slowestStreamTtfa": _lat(_values(load, "ttfa_max_ms")),
            "allStreamsAnswered": _rate([c["values"].get("streams_with_audio") == c["values"].get("streams") for c in load]),
        },
        "throughput": {
            "charsPerSecondParagraph": _lat(_values(paragraphs, "chars_per_s")),
            "charsPerSecond": _lat(_values(one, "chars_per_s")),
            "realtimeFactor": _lat(_values(one, "generation_over_realtime")),
            "audioMsPerChar": _lat(_values(one, "audio_per_char_ms")),
        },
        "playout": {
            "underrunCells": _rate([bool(v) for v in _values(one, "underruns")]),
            "stallMs": _lat(_values(one, "stall_ms")),
            "minMarginMs": _lat(_values(one, "min_margin_ms")),
        },
        "cancel": None if excluded.get("cancel") else {
            "stopped": _rate([c.get("verdict") == "pass" for c in cancel]),
            "toLastChunkMs": _lat(_values(cancel, "cancel_to_last_chunk_ms")),
            "audioAfterCancelMs": _lat(_values(cancel, "audio_after_cancel_ms")),
            "ackMs": _lat(_values(cancel, "cancel_ack_ms")),
        },
        "continuation": None if excluded.get("continuation") else {
            "durationDeltaMs": _lat(_values(cont, "duration_delta_ms")),
            "ttfaDeltaMs": _lat(_values(cont, "ttfa_delta_ms")),
            "framedUnderrunCells": _rate([bool(v) for v in _values(cont, "framed_underruns")]),
        },
        "determinism": {
            "identicalAudio": _rate([bool(v) for v in _values(rep, "identical_pcm")]),
            "durationDeltaAbsMs": _lat([abs(v) for v in _values(rep, "duration_delta_ms")]),
        },
        "reliability": {
            "syntheses": len(singles), "timeouts": timeouts, "runaways": runaways, "endedBy": dict(ended_by),
            "runawayRule": f"a recording over {RUNAWAY_FACTOR:g}x and {RUNAWAY_EXCESS_MS} ms longer than the model's median for the same text",
        },
        "accuracy": accuracy,
        "instrumentDisagreements": disagreements,
    }


def combine(summaries: Sequence[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    """The site file: one entry per model, in the order the model list gives."""
    by_key = {f"{s['provider']}/{s['model']}": s for s in summaries}
    entries, results = [], []
    for key, meta in models["models"].items():
        s = by_key.get(key)
        entries.append({**{k: v for k, v in meta.items() if k != "price"}, "key": key, "measured": s is not None,
                        "price": meta.get("price")})
        if s is not None:
            results.append({"id": meta["id"], **{k: v for k, v in s.items() if k not in ("schema",)}})
    commits = sorted({s["harnessCommit"] for s in summaries})
    sites = sorted({s["site"] for s in summaries if s["site"]})
    return {
        "schemaVersion": SITE_SCHEMA,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "campaign": models.get("campaign"),
        "build": {"harnessCommits": commits, "sites": sites,
                  "corpusVersions": sorted({s["corpusVersion"] for s in summaries}),
                  "methodologies": sorted({s["methodology"] for s in summaries})},
        "models": entries,
        "results": results,
        "methods": {
            "t0": "the first text frame; connection and per-context setup are excluded",
            "ttfa": "first chunk arrival minus t0, plus the leading silence inside the stream",
            "leadingSilence": "first 10 ms window whose DC-removed RMS exceeds 1% of full scale, at 1 ms hops",
            "percentiles": "P50 and P90 over cells, P95 from 30 cells; P50 interval is a 2,000-draw bootstrap",
            "wer": "pooled errors over reference words after whisper-normalizer plus letter/digit token repair; "
                   "the better of the written and the spoken reference",
            "spans": "a labelled hard part passes when an accepted reading appears in the transcript; codes digit by digit",
            "underLoad": "8 simultaneous one-shots on 8 connections under one key",
            "exclusions": "a probe the protocol cannot serve is null with its reason, never zero",
            "location": "every run measured from one cloud region; the round trip is part of every latency",
        },
    }
