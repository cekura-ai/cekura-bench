"""Synthetic fixtures: a two-family config and tiny exported runs. No real calls, callers or links."""

import json
from pathlib import Path

import pytest

from s2s_human_bench.config import load_config

SKILLS = {  # site id -> (metric id, family)
    "1.1": (101, "hearing"),
    "1.2": (102, "hearing"),
    "2.1": (201, "task"),
    "2.2": (202, "task"),
}


def config_dict() -> dict:
    families = {}
    for sid, (mid, fam) in SKILLS.items():
        skill = {"id": sid, "metric": mid, "name": f"Skill {sid}", "description": f"Synthetic skill {sid}.", "condition": f"Condition {sid}"}
        if mid == 202:
            skill["verdict"] = "count"
        families.setdefault(fam, []).append(skill)
    return {
        "version": "test",
        "inputs": {
            "agentRunPattern": "run_a{agent_id}_r*",
            "callsPerRun": 4,
            "maxSilentCalls": 1,
            "minRuns": 2,
            "healthCheckName": "health",
            "agentRolePrefix": "Main",
            "callerRolePrefix": "Testing",
            "endCallTool": "end_call",
        },
        "scoring": {
            "taskFamily": "task",
            "outcomeInTask": 0.5,
            "passScore": 80.0,
            "tieMargin": 5.0,
            "btIterations": 200,
            "bootstrap": {"draws": 20, "seed": 1, "btIterations": 30},
            "stallSeconds": 10.0,
            "minSkillCalls": 1,
        },
        "metrics": {"latency": 900, "turnLength": 901, "entity": 101},
        "families": [{"key": k, "label": k.title(), "weight": 0.5, "skills": v} for k, v in families.items()],
        "passiveMetrics": [
            {"id": "EO", "name": "Task Outcome", "family": "Every call", "type": "passive", "better": "higher", "unit": "share", "description": "x"}
        ],
        "models": [
            {"key": "alpha", "name": "Alpha", "vendor": "A", "siteId": "alpha-site", "agentId": 1},
            {"key": "beta", "name": "Beta", "vendor": "B", "siteId": "beta-site", "agentId": 2},
            {"key": "gamma", "name": "Gamma", "vendor": "C", "siteId": "gamma-site", "runs": ["gamma_r1", "gamma_r2"]},
            {"key": "ref", "name": "Reference", "vendor": "R", "siteId": "ref-site", "agentId": 3, "reference": True},
        ],
        "suites": [
            {"id": "bank", "prefix": "B", "label": "Bank", "description": "d", "scenarios": 2, "domain": "Banking"},
            {"id": "clinic", "prefix": "C", "label": "Clinic", "description": "d", "scenarios": 2, "domain": "Healthcare"},
        ],
        "categories": [
            {"key": "hear", "label": "Hearing calls", "skills": ["1.1", "1.2"], "description": "d"},
            {"key": "switch", "label": "Switching", "skills": ["2.1"], "description": "d"},
        ],
        "languages": {
            "skill": "2.1",
            "groups": [
                {"id": "all", "label": "All", "note": "n", "scenarios": "all"},
                {"id": "en", "label": "English", "note": "n", "scenarios": "rest"},
                {"id": "xx", "label": "Switches", "note": "n", "scenarios": ["C02"]},
            ],
        },
        "benchmark": {"modelDefaults": {"voice": "-"}, "noPriceNote": "none"},
        "insights": {
            "note": "n",
            "taskNotCompleted": "Task not completed",
            "topArguments": 2,
            "highlights": 1,
            "text": {
                "families": "{strong} {strong_score} {weak} {weak_score}",
                "skill": "{name} {rate} {delta:+d}",
                "strength": "S {skill}",
                "gap": "G {skill}",
                "task": "T {rate}",
            },
        },
        "reports": {"note": "n"},
        "skills": {"minReached": 1, "domains": ["banking", "healthcare"], "limits": ["{scenarios} scenarios."]},
        "examples": {
            "perSuite": 2,
            "minExpectedLines": 1,
            "maxConditions": 2,
            "maxSkills": 2,
            "brokenCall": "never spoke",
            "momentMinTimestamps": 2,
            "momentMinChars": 10,
            "momentMaxChars": 200,
            "clipLeadSeconds": 4,
            "clipTailSeconds": 5,
            "clipMaxSeconds": 30,
        },
        "outputs": {k: f"{k}.json" for k in ("benchmark", "insights", "reports", "languages", "skills", "examples")},
    }


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict()))
    return load_config(path)


def write_call(run: Path, code: str, scores: dict[int, object], lines: list[str], spoke: bool = True, call_id: str = "1") -> None:
    """One synthetic call: metric scores by id (None = not reached), expected-outcome lines."""
    d = run / "calls" / code / call_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "call.json").write_text(
        json.dumps({"code": code, "call_id": call_id, "dashboard_url": f"https://example.test/{run.name}?call_id={call_id}", "duration": "01:30"})
    )
    results = [
        {"id": mid, "name": f"m{mid}", "score": s, "explanation": "Passed at 0:05 and again at 0:20 in the call." if s else "Missed at 0:10 and 0:40."}
        for mid, s in scores.items()
    ]
    results += [{"id": 900, "score": 1200}, {"id": 901, "score": 12}]
    (d / "metrics.json").write_text(json.dumps({"results": results}))
    turns = [{"role": "Testing Agent", "content": "hello", "start_time": 0, "end_time": 2}]
    if spoke:
        turns.append({"role": "Main Agent", "content": "hi there", "start_time": 3, "end_time": 4})
    (d / "transcript.json").write_text(json.dumps(turns))
    (d / "expected_outcome.json").write_text(json.dumps({"explanation": lines}))


GOOD = {101: True, 102: True, 201: None, 202: 0}
BAD = {101: False, 102: False, 201: None, 202: 2}


def write_run(run: Path, quality: dict[str, dict[int, object]], silent: int = 0) -> None:
    for i, code in enumerate(["B01", "B02", "C01", "C02"]):
        scores = dict(quality.get(code, GOOD))
        if code == "C02":
            scores[201] = scores[101]  # the language switch applies only in C02
        write_call(run, code, scores, ["✅ done", "✅ also"] if scores[101] else ["❌ missed", "✅ done"], spoke=i >= silent)
