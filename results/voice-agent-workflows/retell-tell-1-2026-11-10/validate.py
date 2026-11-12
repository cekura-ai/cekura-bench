"""Recompute this frozen workflow result; no API calls or verdict writes."""

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


GATES = [
    {"metric": "Expected Outcome", "operator": "eq", "threshold": 5},
    {"metric": "Infrastructure Issues", "operator": "eq", "threshold": 5},
    {"metric": "Interruption Score", "operator": "gte", "threshold": 1},
    {"metric": "Latency (in ms)", "operator": "lte", "threshold": 10000},
    {"metric": "Tool Call Accuracy", "operator": "eq", "threshold": 5},
]
NO_TOOLS = {"AS12", "AS49", "MS79"}
SCENARIOS = {f"AS{i}" for i in range(1, 60)} | {f"MS{i}" for i in range(60, 83)}
METRICS = {g["metric"] for g in GATES} | {"Transcription Accuracy", "Voice Tone + Clarity"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(calls):
    groups = defaultdict(list)
    for call in calls:
        groups[call["scenario"]].append(call)
    passed = sum(call["passed"] for call in calls)
    triads = sum(all(call["passed"] for call in group) for group in groups.values())
    return {
        "calls": len(calls), "passed_calls": passed, "pass1": passed / len(calls),
        "scenarios": len(groups), "passed_scenarios": triads, "pass3": triads / len(groups),
    }


def validate(data):
    require(data["schema_version"] == 1, "Unknown schema version")
    require(data["scoring"]["gates"] == GATES, "Five-gate policy changed")
    require(data["scoring"]["tool_accuracy_variant"] == "standard", "Tool variant changed")
    require(set(data["scoring"]["tool_accuracy_not_applicable_scenarios"]) == NO_TOOLS,
            "Tool applicability changed")
    require(data["runs_per_scenario"] == 3 and data["concurrency_per_workflow"] == 5,
            "Launch settings changed")
    require(data["review_required_run_ids"] == [], "Review-required runs remain")
    calls = data["calls"]
    require(len(calls) == 246 and len({c["run_id"] for c in calls}) == 246,
            "Expected 246 unique runs")
    require({c["scenario"] for c in calls} == SCENARIOS, "Canonical scenario membership differs")
    groups = defaultdict(list)
    for call in calls:
        scenario = call["scenario"]
        groups[scenario].append(call)
        workflow = "Appointments" if scenario.startswith("AS") else "Medicare"
        result_id = data["workflows"][workflow]["result_id"]
        require(call["result_id"] == result_id, f"Wrong workflow result: {call['run_id']}")
        require(call["url"] == f"https://dashboard.cekura.ai/{data['project_id']}/results/"
                f"{result_id}?call_id={call['run_id']}", f"Wrong call URL: {call['run_id']}")
        require(set(call["metrics"]) == METRICS, f"Metric coverage differs: {call['run_id']}")
        for name, value in call["metrics"].items():
            if name == "Tool Call Accuracy" and scenario in NO_TOOLS:
                require(value is None, f"Unexpected tool score: {call['run_id']}")
                continue
            require(type(value) in (int, float) and math.isfinite(value),
                    f"Missing/non-numeric applicable metric: {call['run_id']} {name}")
            require(value >= 0 and (name == "Latency (in ms)" or value <= 5),
                    f"Out-of-range score: {call['run_id']} {name}")
        failed = []
        for gate in GATES:
            name = gate["metric"]
            operator = gate["operator"]
            threshold = gate["threshold"]
            if name == "Tool Call Accuracy" and scenario in NO_TOOLS:
                continue
            value = call["metrics"][name]
            passed = {"eq": value == threshold, "gte": value >= threshold,
                      "lte": value <= threshold}[operator]
            if not passed:
                failed.append(name)
        require(type(call["passed"]) is bool and call["passed"] == (not failed),
                f"Verdict disagrees with gates: {call['run_id']}")
        require(call["failed_gates"] == failed, f"Failed gates disagree: {call['run_id']}")
    require(all(len(group) == 3 for group in groups.values()), "Expected three runs per scenario")
    stats = {
        "Appointments": summarize([c for c in calls if c["scenario"].startswith("AS")]),
        "Medicare": summarize([c for c in calls if c["scenario"].startswith("MS")]),
        "Combined": summarize(calls),
    }
    require(data["stats"] == stats, "Published counts/rates disagree with runs")
    aggregates = {}
    for name in sorted(METRICS):
        values = [c["metrics"][name] for c in calls if c["metrics"][name] is not None]
        aggregates[name] = {"n": len(values), "mean": statistics.mean(values)}
        if name != "Latency (in ms)":
            aggregates[name]["full_score_rate"] = sum(v == 5 for v in values) / len(values)
    require(data["metrics"] == aggregates, "Published metric aggregates disagree with runs")
    return stats


if __name__ == "__main__":
    directory = Path(__file__).parent
    data = json.loads((directory / "results.json").read_text())
    stats = validate(data)
    report = (directory / "report.html").read_bytes()
    require(hashlib.sha256(report).hexdigest() == data["provenance"]["report_sha256"],
            "Report differs from approved Downloads artifact")
    print(json.dumps(stats, indent=2))
