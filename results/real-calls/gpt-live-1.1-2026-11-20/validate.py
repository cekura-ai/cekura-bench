"""Check this snapshot: recompute the published aggregates from the per-call rows in results.json.

    python validate.py

Checks the cohort (3 runs x 70 unique scenarios, 210 calls, 30 metric results per call, 6,300 in
total, roles and builds, no superseded attempt selected) and recomputes, overall, per run, per domain
and per language group:
calls, passed calls (score >= 80), pass³ (scenario passed in all three runs), main task done,
mean call score, the four family scores (mean of skill pass rates with at least 3 reaching calls;
Task Handling blended half-and-half with the task outcome) and latency percentiles
(nearest-rank, as the public S2S page computes them). Exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASS = 80.0
MIN_SKILL_CALLS = 3
OUTCOME_IN_TASK = 0.5


def percentile(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else None


def families(calls, family_of):
    seen, ok = defaultdict(int), defaultdict(int)
    for c in calls:
        for sid, v in c["skills"].items():
            seen[sid] += 1
            ok[sid] += bool(v)
    rates = {s: ok[s] / seen[s] for s in seen if seen[s] >= MIN_SKILL_CALLS}
    outcomes = [c["task_outcome"] for c in calls if c["task_outcome"] is not None]
    outcome = statistics.mean(outcomes) if outcomes else None
    out = {}
    for label in dict.fromkeys(family_of.values()):
        vals = [100 * r for s, r in rates.items() if family_of.get(s) == label]
        fam = statistics.mean(vals) if vals else None
        if label == "Task Handling" and outcome is not None:
            fam = outcome if fam is None else (1 - OUTCOME_IN_TASK) * fam + OUTCOME_IN_TASK * outcome
        out[label] = round(fam, 1) if fam is not None else None
    return out


def summary(calls, family_of):
    by_scenario = defaultdict(list)
    for c in calls:
        by_scenario[c["scenario"]].append(c)
    runs = {c["run"] for c in calls}
    lat = [c["latency_ms"] for c in calls if c["latency_ms"] is not None]
    scored = [c["score"] for c in calls if c["score"] is not None]
    return {
        "calls": len(calls),
        "passed": sum(c["passed"] for c in calls),
        "pass3_scenarios": sum(len(v) == len(runs) and all(c["passed"] for c in v) for v in by_scenario.values()),
        "task_done": sum(c["task_done"] for c in calls),
        "mean_call_score": round(statistics.mean(scored), 2) if scored else None,
        "families": families(calls, family_of),
        "latency_ms": {"p50": percentile(lat, 0.5), "p90": percentile(lat, 0.9), "p95": percentile(lat, 0.95)},
    }


def main() -> int:
    data = json.loads((HERE / "results.json").read_text())
    calls = data["calls"]
    view = data["views"]["completed"]
    bad = 0
    by_run = defaultdict(list)
    for c in calls:
        by_run[c["run"]].append(c["scenario"])
    metric_ids = sorted(m["id"] for m in data["metric_definitions"])
    cohort = [
        ("calls", len(calls), 210),
        ("runs", sorted(by_run), ["r1", "r2", "r3"]),
        ("unique scenarios per run", {r: len(set(v)) for r, v in by_run.items()}, {r: 70 for r in ("r1", "r2", "r3")}),
        ("calls per run", {r: len(v) for r, v in by_run.items()}, {r: 70 for r in ("r1", "r2", "r3")}),
        ("metric definitions", len(metric_ids), 30),
        ("metric results", sum(len(c["metrics"]) for c in calls), 6300),
        ("calls with all 30 metrics", sum(sorted(m["id"] for m in c["metrics"]) == metric_ids for c in calls), 210),
        ("metric results not evaluated", sum(m["status"] != "evaluated" for c in calls for m in c["metrics"]), 0),
        ("roles", dict(sorted(Counter(c["role"] for c in calls).items())), {"fixture_rerun": 3, "original": 206, "timeout_rerun": 1}),
        ("roles match cohort header", dict(sorted(Counter(c["role"] for c in calls).items())), data["cohort"]["roles"]),
        ("builds match cohort header", dict(sorted(Counter(c["build"] for c in calls).items())), data["cohort"]["builds"]),
        ("cohort header", (data["cohort"]["calls"], data["cohort"]["metric_entries"]), (210, 6300)),
        ("superseded attempts selected", sorted({(x["run"], x["scenario"], x["attempt"]) for x in data["cohort"]["superseded"]}
                                                & {(c["run"], c["scenario"], c["attempt"]) for c in calls}), []),
        ("replacements name their superseded attempts", sorted((c["run"], c["scenario"]) for c in calls if c["role"] != "original"),
         sorted({(x["run"], x["scenario"]) for x in data["cohort"]["superseded"]})),
    ]
    for name, got, want in cohort:
        if got != want:
            bad += 1
            print(f"MISMATCH cohort {name}: {got} != {want}")
    family_of = {sid: s["family"] for sid, s in view["overall"]["skills"].items()}
    checks = [("overall", calls, view["overall"])]
    for run in sorted({c["run"] for c in calls}):
        checks.append((run, [c for c in calls if c["run"] == run], view["runs"][f"fullrun_gptlive11_final_{run}"]))
    for suite, prefix in (("banking", "F"), ("healthcare", "H")):
        checks.append((suite, [c for c in calls if c["scenario"].startswith(prefix)], view["suites"][suite]))
    for gid, g in view["languages"].items():
        codes = set(g["scenario_codes"])
        checks.append((f"language:{gid}", [c for c in calls if c["scenario"] in codes], g))
    for name, rows, published in checks:
        got = summary(rows, family_of)
        for key, value in got.items():
            want = published[key] if key != "latency_ms" else {k: published["latency_ms"][k] for k in value}
            if value != want:
                bad += 1
                print(f"MISMATCH {name}.{key}: recomputed {value} != published {want}")
    elo = json.loads((HERE / "elo.json").read_text())
    computed = data["elo"]
    model = computed["model"]
    ranked = sorted(elo["ranked_models"], key=lambda m: elo["group_ratings"]["all"][m], reverse=True)
    elo_checks = [
        ("status", computed["status"], "computed"),
        ("source", computed["source"], "elo.json"),
        ("rating", computed["rating"], elo["group_ratings"]["all"][model]),
        ("rounded rating", computed["rounded_rating"], round(elo["group_ratings"]["all"][model])),
        ("rank", computed["rank"], ranked.index(model) + 1),
        ("ranked model count", computed["ranked_models"], len(ranked)),
        ("confidence interval", computed["confidence_interval"], elo["confidence_intervals"][model]),
    ]
    for name, got, want in elo_checks:
        if got != want:
            bad += 1
            print(f"MISMATCH ELO {name}: {got} != {want}")
    definitions = {m["id"]: m for m in data["metric_definitions"]}
    for call in calls:
        normalized = {}
        for metric in call["metrics"]:
            skill = definitions[metric["id"]].get("skill")
            if skill is not None and metric.get("passed") is not None:
                normalized[skill] = metric["passed"]
        if normalized != call["skills"]:
            bad += 1
            print(f"MISMATCH final metric verdicts: {call['run']}/{call['scenario']}")
    reruns = sum(c["role"] != "original" for c in calls)
    print(f"{len(calls)} calls checked ({reruns} replacement calls), "
          f"{sum(len(c['metrics']) for c in calls)} metric results; {len(cohort)} cohort checks, {len(checks)} aggregates, {len(elo_checks)} ELO checks; final metric verdicts checked; "
          f"{'all match' if not bad else f'{bad} mismatches'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
