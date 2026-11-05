"""Exported runs: which run folders count, and each call's score.

A run folder holds ``calls/<code>/<call id>/`` with ``call.json``, ``metrics.json``,
``transcript.json`` and ``expected_outcome.json`` per call.

The call score (0-100) is the weighted mean of the skill families the call reached. A
family's value is the share of its reached skills that passed; Task Handling blends that
with the share of the scenario's expected-outcome lines met (✅ against ❌; 🟡, not
reached, is left out).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from s2s_human_bench.config import Config, Model, Skill

MET, MISSED = "✅", "❌"


@dataclass
class Call:
    code: str
    call_id: str | int | None  # as exported
    url: str
    seconds: int
    verdicts: dict[str, bool | None]  # skill id -> verdict, skills in family order
    explanations: dict[str, str]
    families: dict[str, float | None]
    outcome: float | None  # share of expected-outcome lines met, 0-100
    outcome_lines: list[str]
    score: float | None
    latency_ms: float | None
    turn_words: float | None
    task_done: bool = False
    stalled: bool = False
    agent_ended: bool = False
    spoke: bool = False
    replies: int = 0
    words: int = 0
    unhealthy: bool = False
    tested: list[str] = field(default_factory=list)  # site order
    misses: list[str] = field(default_factory=list)  # site order


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verdict(skill: Skill, result: dict[str, Any]) -> bool | None:
    """True / False, or None when the skill was not reached or does not apply."""
    score = result.get("score")
    if skill.verdict == "explanation":  # enum metric: its value is not exported, the explanation carries it
        text = result.get("explanation") or ""
        if text.startswith("All "):
            return True
        if text.startswith(("N/A", "Not reached", "Excluded")) or not text:
            return None
        return False
    if skill.verdict == "count":
        return None if score is None else score == 0
    return None if score is None else bool(score)


def outcome_share(lines: list[str]) -> float | None:
    met = sum(1 for x in lines if x.startswith(MET))
    missed = sum(1 for x in lines if x.startswith(MISSED))
    return 100 * met / (met + missed) if met + missed else None


def family_scores(cfg: Config, verdicts: dict[str, bool | None], outcome: float | None) -> dict[str, float | None]:
    fam: dict[str, float | None] = {}
    for key in cfg.families:
        vals = [v for s in cfg.skills if s.family == key and (v := verdicts.get(s.id)) is not None]
        fam[key] = 100 * sum(vals) / len(vals) if vals else None
    task = cfg["scoring"].get("taskFamily", "task")
    if outcome is not None:
        w = cfg.scoring["outcomeInTask"]
        fam[task] = outcome if fam[task] is None else (1 - w) * fam[task] + w * outcome
    return fam


def call_score(weights: dict[str, float], families: dict[str, float | None]) -> float | None:
    """Weighted mean of the families the call reached; None when it reached none."""
    parts = [(weights[k], v) for k, v in families.items() if v is not None]
    return sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else None


def _seconds(duration: str | None) -> int:
    mm, ss = ((duration or "0:00").split(":") + ["0"])[:2]
    return int(mm) * 60 + int(ss)


def _speaker(turn: dict[str, Any], prefix: str) -> bool:
    return (turn.get("role") or "").startswith(prefix) and bool((turn.get("content") or "").strip())


def _went_quiet(agent: list[dict[str, Any]], caller: list[dict[str, Any]], limit: float) -> bool:
    """The agent stayed silent longer than ``limit`` after a caller turn, and spoke before the caller did."""
    for ct in caller:
        end = ct.get("end_time") or 0
        nxt = [a.get("start_time") for a in agent if (a.get("start_time") or 0) > end]
        nxt_caller = [x.get("start_time") for x in caller if (x.get("start_time") or 0) > end]
        if nxt and min(nxt) - end > limit and (not nxt_caller or min(nxt_caller) > min(nxt)):
            return True
    return False


def load_call(cfg: Config, call_json: Path) -> Call:
    meta = _read(call_json)
    results = {r["id"]: r for r in _read(call_json.with_name("metrics.json"))["results"]}
    transcript = _read(call_json.with_name("transcript.json"))
    lines = _read(call_json.with_name("expected_outcome.json")).get("explanation") or []
    inputs, metric_ids = cfg["inputs"], cfg["metrics"]

    verdicts = {s.id: verdict(s, results[s.metric]) if s.metric in results else None for s in cfg.skills}
    explanations = {s.id: results[s.metric].get("explanation") or "" for s in cfg.skills if s.metric in results}
    outcome = outcome_share(lines)
    families = family_scores(cfg, verdicts, outcome)
    agent = [t for t in transcript if _speaker(t, inputs["agentRolePrefix"])]
    caller = [t for t in transcript if _speaker(t, inputs["callerRolePrefix"])]
    tools = [(t.get("data") or {}).get("name") for t in transcript if (t.get("data") or {}).get("name")]
    health = inputs["healthCheckName"]
    site = cfg.skills_by_site
    return Call(
        code=meta["code"],
        call_id=meta.get("call_id"),
        url=meta["dashboard_url"],
        seconds=_seconds(meta.get("duration")),
        verdicts=verdicts,
        explanations=explanations,
        families=families,
        outcome=outcome,
        outcome_lines=lines,
        score=call_score(cfg.weights, families),
        latency_ms=results.get(metric_ids["latency"], {}).get("score"),
        turn_words=results.get(metric_ids["turnLength"], {}).get("score"),
        task_done=sum(x.startswith(MISSED) for x in lines) == 0 and sum(x.startswith(MET) for x in lines) > 0,
        stalled=_went_quiet(agent, caller, cfg.scoring["stallSeconds"]),
        agent_ended=inputs["endCallTool"] in tools,
        spoke=bool(agent),
        replies=len(agent),
        words=sum(len((a.get("content") or "").split()) for a in agent),
        unhealthy=any(health in (r.get("name") or "") and r.get("score") is False for r in results.values()),
        tested=[s.id for s in site if verdicts[s.id] is not None],
        misses=[s.id for s in site if verdicts[s.id] is False],
    )


def call_files(run: Path) -> list[Path]:
    return sorted(run.glob("calls/*/*/call.json"))


def load_run(cfg: Config, run: Path, exclude_unhealthy: bool = False) -> dict[str, Call]:
    """A run's calls by scenario code, in code order. With ``exclude_unhealthy``, calls whose
    simulation-health check failed (our setup broke, not the model) are left out."""
    calls = {}
    for path in call_files(run):
        call = load_call(cfg, path)
        if exclude_unhealthy and call.unhealthy:
            continue
        calls[call.code] = call
    return calls


def silent_calls(run: Path, agent_prefix: str) -> int:
    return sum(1 for t in run.glob("calls/*/*/transcript.json") if not any(_speaker(x, agent_prefix) for x in _read(t)))


def usable_run(cfg: Config, run: Path) -> bool:
    """A complete export where the agent spoke on all but a few calls. A run where it never spoke
    on more calls is a throttled or broken run, not model behaviour (the void-run guard)."""
    inputs = cfg["inputs"]
    if len(list(run.glob("calls/*/*/metrics.json"))) < inputs["callsPerRun"]:
        return False
    return silent_calls(run, inputs["agentRolePrefix"]) <= inputs["maxSilentCalls"]


def model_runs(cfg: Config, model: Model, root: Path) -> list[Path]:
    """The model's usable run folders, in name order."""
    if model.runs is not None:
        folders = [root / r for r in model.runs]
    else:
        folders = sorted(root.glob(cfg["inputs"]["agentRunPattern"].format(agent_id=model.agent_id)))
    return [f for f in folders if usable_run(cfg, f)]
