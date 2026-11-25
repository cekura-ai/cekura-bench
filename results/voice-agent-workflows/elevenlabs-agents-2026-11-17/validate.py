"""Recompute this frozen workflow result; no API calls or verdict writes."""

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
TOOL_METRICS = {"Tool Call Accuracy", "Tool Call Accuracy (list text accepted)", "Tool Call Success"}
SCENARIOS = {f"AS{i}" for i in range(1, 60)} | {f"MS{i}" for i in range(60, 83)}
METRICS = {g["metric"] for g in GATES} | {
    "Transcription Accuracy", "Voice Tone + Clarity", "Unnecessary Repetition Score",
    "Appropriate Call Termination by Main Agent", "Tool Call Success", "Tool Call Accuracy (list text accepted)",
}


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
    counted = {c["run_id"] for c in calls}
    replacements = {}
    for replaced in data["replaced_runs"]:
        require(replaced["run_id"] not in counted, f"Replaced run is still counted: {replaced['run_id']}")
        require(replaced["replacement_run_id"] in counted, f"Replacement is not counted: {replaced['run_id']}")
        replacements[replaced["replacement_run_id"]] = replaced
    groups = defaultdict(list)
    for call in calls:
        scenario = call["scenario"]
        groups[scenario].append(call)
        workflow = data["workflows"]["Appointments" if scenario.startswith("AS") else "Medicare"]
        if call["run_id"] in replacements:
            replaced = replacements[call["run_id"]]
            require(replaced["scenario"] == scenario and call["result_id"] == replaced["replacement_result_id"]
                    and call["result_id"] in workflow["replacement_result_ids"],
                    f"Replacement does not match its slot: {call['run_id']}")
        else:
            require(call["result_id"] == workflow["result_id"], f"Wrong workflow result: {call['run_id']}")
        require(call["url"] == f"https://dashboard.cekura.ai/{data['project_id']}/results/"
                f"{call['result_id']}?call_id={call['run_id']}", f"Wrong call URL: {call['run_id']}")
        require(set(call["metrics"]) == METRICS, f"Metric coverage differs: {call['run_id']}")
        for name, value in call["metrics"].items():
            if name in TOOL_METRICS and scenario in NO_TOOLS:
                require(value is None, f"Unexpected tool score: {call['run_id']}")
                continue
            require(type(value) in (int, float) and math.isfinite(value),
                    f"Missing/non-numeric applicable metric: {call['run_id']} {name}")
            require(value >= 0 and (name == "Latency (in ms)" or value <= 5),
                    f"Out-of-range score: {call['run_id']} {name}")
        failed = []
        for gate in GATES:
            name = gate["metric"]
            if name == "Tool Call Accuracy" and scenario in NO_TOOLS:
                continue
            value = call["metrics"][name]
            passed = {"eq": value == gate["threshold"], "gte": value >= gate["threshold"],
                      "lte": value <= gate["threshold"]}[gate["operator"]]
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
    for name in METRICS:
        values = [c["metrics"][name] for c in calls if c["metrics"][name] is not None]
        aggregates[name] = {"n": len(values), "mean": statistics.fmean(values)}
        if name != "Latency (in ms)":
            aggregates[name]["full_score_rate"] = sum(v == 5 for v in values) / len(values)
    require(data["metrics"] == aggregates, "Published metric aggregates disagree with runs")
    return stats


if __name__ == "__main__":
    print(json.dumps(validate(json.loads((Path(__file__).parent / "results.json").read_text())), indent=2))
