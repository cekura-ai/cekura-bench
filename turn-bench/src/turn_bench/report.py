"""Score a submission with Sesame's scorer and write report.json / report.md.

Nothing here computes a metric: ``turnbench.score`` does, unchanged. This
module only turns its ``TaskScore`` objects into JSON and a Markdown table.

Scores come in three blocks: **caller** (speaker_1 only), **agent** (speaker_2
only) and **all** (both, Sesame's ``score_conversation`` verbatim). The caller
side is the headline for a voice-agent product (it is the user's turns the agent
has to end-point); the agent side is reported separately. A per-role block is
Sesame's ``score_task`` run on the gold restricted to that speaker, with the
other speaker's predictions blanked. ``score_task`` scores each speaker's
channel independently (ledgers, next-event caps and negative spans are all
per speaker), so the caller and agent blocks partition the all block: their
tp/fn/fp/tn add up to it exactly, and their latency lists concatenate to its.

**Breakdowns** (``groups``): the same per-conversation scores merged over
named subsets of the conversations — for an authored dataset, by the spec's
``family`` and ``difficulty`` read from its ``ground_truth.json``
(``authored_groups``). A breakdown row is exactly what scoring that subset on
its own would give; nothing in the scorer changes.

**Scorecard and intervals** (``turn_bench.scorecard``): every block and every
breakdown row also carries the Conversational Dynamics scorecard (pause
handling, turn taking, interruption handling, backchannel handling, overall)
and 95 % conversation-level bootstrap intervals on it and on the EOT / INT
rates and latency p50. They are computed from the same per-conversation
``TaskScore`` objects, so adding them changes no existing number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from turnbench.data import Conversation, Dataset, conversation, conversation_ids
from turnbench.gold import events_for_conversation
from turnbench.score import (
    ConversationScores,
    TaskScore,
    merge,
    score_conversation,
    score_task,
)
from turnbench.submission import ConversationPrediction, Submission, validate_coverage, validate_event_times

from turn_bench.scorecard import (
    SCORECARD,
    SCORECARD_TITLES,
    bootstrap_block,
    bootstrap_record,
    resolve_weights,
)

TASKS = (("eot", "task_eot"), ("int", "task_int"))
ROLES: dict[str, int] = {"caller": 1, "agent": 2}  # Sesame's speaker_1 / speaker_2
BLOCKS = ("caller", "agent", "all")


def score_conversation_for_speaker(prediction: ConversationPrediction, conv: Conversation, speaker: int) -> ConversationScores:
    """Sesame's ``score_task`` on the gold of one speaker only, with the other
    speaker's predictions blanked; see the module docstring."""
    validate_event_times(prediction, conv.duration_s)
    events = events_for_conversation(conv)
    own = {1: prediction.speaker_1, 2: prediction.speaker_2}[speaker]
    return ConversationScores(
        task_eot=score_task(
            [e for e in events.eot_positive_events if e.speaker == speaker],
            [i for i in events.eot_negative_spans if i.speaker == speaker],
            {1: own.eot if speaker == 1 else [], 2: own.eot if speaker == 2 else []},
            events.eot_excluded,
        ),
        task_int=score_task(
            [e for e in events.int_positive_events if e.speaker == speaker],
            [i for i in events.int_negative_spans if i.speaker == speaker],
            {1: own.interruption if speaker == 1 else [], 2: own.interruption if speaker == 2 else []},
            events.int_excluded,
        ),
    )


def score_conversation_blocks(prediction: ConversationPrediction, conv: Conversation) -> dict[str, ConversationScores]:
    """``{"caller": ..., "agent": ..., "all": ...}`` for one conversation; ``all``
    is ``turnbench.score.score_conversation`` unchanged."""
    blocks = {role: score_conversation_for_speaker(prediction, conv, speaker) for role, speaker in ROLES.items()}
    blocks["all"] = score_conversation(prediction, conv)
    return blocks


def _number(value: float) -> float | None:
    return None if math.isnan(value) else value


def task_dict(score: TaskScore) -> dict[str, Any]:
    latency = score.latency()
    return {
        "recall": _number(score.recall),
        "fp_rate": _number(score.fp_rate),
        "latency_ms": {
            "p10": _number(latency.p10),
            "p50": _number(latency.p50),
            "p90": _number(latency.p90),
        },
        "tp": score.tp,
        "fn": score.fn,
        "fp": score.fp,
        "tn": score.tn,
    }


def scores_dict(scores: ConversationScores) -> dict[str, Any]:
    return {name: task_dict(getattr(scores, attr)) for name, attr in TASKS}


Groups = dict[str, dict[str, list[str]]]  # axis -> group name -> conversation ids
# Axes of an authored dataset's ground_truth.json: the spec's family, subtype and
# difficulty, the degradation condition, the voice pairing and the caller voice
# (the accent axis: every pairing's caller is one voice of render.VOICES).
GROUP_AXES = ("family", "subtype", "difficulty", "condition", "pairing", "caller_voice")


def group_name(record: dict[str, Any], axis: str) -> str:
    if axis == "caller_voice":
        return (record.get("voices") or {}).get("caller") or "(none)"
    if axis == "condition":
        return record.get("condition") or record.get("transform") or "(none)"
    return record.get(axis) or "(none)"


def authored_groups(dataset_dir: Path) -> Groups:
    """Conversation ids by every ``GROUP_AXES`` axis from an authored dataset's
    ``ground_truth.json``; empty for a dataset without one. An axis whose only
    value is ``(none)`` is dropped (nothing to break down)."""
    path = Path(dataset_dir) / "ground_truth.json"
    if not path.is_file():
        return {}
    truth = json.loads(path.read_text(encoding="utf-8"))
    groups: Groups = {axis: {} for axis in GROUP_AXES}
    for cid, record in truth.items():
        for axis in GROUP_AXES:
            groups[axis].setdefault(group_name(record, axis), []).append(cid)
    return {axis: names for axis, names in groups.items() if names and set(names) != {"(none)"}}


def _merged(scores: list[ConversationScores]) -> ConversationScores:
    total = ConversationScores(TaskScore(), TaskScore())
    for conversation_scores in scores:
        merge(total.task_eot, conversation_scores.task_eot)
        merge(total.task_int, conversation_scores.task_int)
    return total


def score_report(
    submission: Submission,
    dataset: Dataset,
    groups: Groups | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate and per-conversation scores in the three blocks.

    ``aggregate`` / ``per_conversation`` are the ``all`` block (both speakers,
    Sesame's scoring loop, unchanged shape); ``blocks`` holds every block by
    name with its speaker, aggregate and per-conversation scores, plus its
    ``scorecard`` and ``ci95`` (``turn_bench.scorecard``); ``breakdowns`` the
    per-block aggregates, scorecards and intervals over each named subset in
    ``groups`` (ids outside the scored set are ignored, empty groups dropped).
    ``weights`` overrides the scorecard's overall weights (default equal)."""
    weights = resolve_weights(weights)
    task_ids = conversation_ids(dataset)
    validate_coverage(submission, task_ids)
    by_conversation = submission.by_conversation()
    scored: dict[str, dict[str, ConversationScores]] = {
        cid: score_conversation_blocks(by_conversation[cid], conversation(dataset, cid)) for cid in task_ids
    }
    totals = {block: _merged([scored[cid][block] for cid in task_ids]) for block in BLOCKS}
    per_conversation = {block: {cid: scores_dict(scored[cid][block]) for cid in task_ids} for block in BLOCKS}
    stats = {block: bootstrap_block([scored[cid][block] for cid in task_ids], weights) for block in BLOCKS}
    breakdowns: dict[str, dict[str, Any]] = {}
    for axis, names in (groups or {}).items():
        for name, ids in names.items():
            members = [cid for cid in task_ids if cid in set(ids)]
            if not members:
                continue
            member_stats = {block: bootstrap_block([scored[cid][block] for cid in members], weights) for block in BLOCKS}
            breakdowns.setdefault(axis, {})[name] = {
                "conversations": len(members),
                "blocks": {block: scores_dict(_merged([scored[cid][block] for cid in members])) for block in BLOCKS},
                "scorecard": {block: member_stats[block]["scorecard"] for block in BLOCKS},
                "ci95": {block: member_stats[block]["ci95"] for block in BLOCKS},
            }
    return {
        "conversations_scored": len(task_ids),
        "aggregate": scores_dict(totals["all"]),
        "per_conversation": per_conversation["all"],
        "blocks": {
            block: {
                "speakers": [ROLES[block]] if block in ROLES else sorted(ROLES.values()),
                "aggregate": scores_dict(totals[block]),
                "scorecard": stats[block]["scorecard"],
                "ci95": stats[block]["ci95"],
                "per_conversation": per_conversation[block],
            }
            for block in BLOCKS
        },
        "breakdowns": breakdowns,
        "headline_block": "caller",
        "scorecard_weights": weights,
        "bootstrap": bootstrap_record(),
    }


BLOCK_TITLES = {"caller": "Caller (speaker_1)", "agent": "Agent (speaker_2)", "all": "All speakers"}


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def _fmt_ci(ci: dict[str, Any] | None, digits: int = 3) -> str:
    """``[lo, hi]`` or ``n/a`` (no interval: fewer than two conversations, or
    no draw in which the value was defined)."""
    if not ci or ci.get("lo") is None:
        return "n/a"
    return f"[{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"


def _fmt_value(entry: dict[str, Any] | None, digits: int = 3) -> str:
    if not entry or entry.get("value") is None:
        return "n/a"
    return f"{entry['value']:.{digits}f}"


TASK_HEADER = "| task | recall | recall 95% CI | fp_rate | fp_rate 95% CI | latency ms p10 / p50 / p90 | p50 95% CI | tp | fn | fp | tn |"
TASK_RULE = "|---|---|---|---|---|---|---|---|---|---|---|"


def task_row(label: str, task: dict[str, Any], prefix: str = "", ci: dict[str, Any] | None = None) -> str:
    latency = task["latency_ms"]
    ci = ci or {}
    return (
        f"|{prefix} {label} | {_fmt(task['recall'])} | {_fmt_ci(ci.get('recall'))} | "
        f"{_fmt(task['fp_rate'])} | {_fmt_ci(ci.get('fp_rate'))} | "
        f"{_fmt_ms(latency['p10'])} / {_fmt_ms(latency['p50'])} / {_fmt_ms(latency['p90'])} | "
        f"{_fmt_ci((ci.get('latency_ms') or {}).get('p50'), 0)} | "
        f"{task['tp']} | {task['fn']} | {task['fp']} | {task['tn']} |"
    )


SCORECARD_HEADER = "| | pause handling | turn taking | EOT latency p50 [CI] / p90 ms | interruption handling | backchannel handling | overall |"
SCORECARD_RULE = "|---|---|---|---|---|---|---|"


def scorecard_cell(row: dict[str, Any]) -> str:
    """``value [lo, hi] (n=events)`` or ``n/a (n=0)``."""
    if row["value"] is None:
        return f"n/a (n={row['n']})"
    return f"{row['value']:.3f} {_fmt_ci(row['ci95'])} (n={row['n']})"


def overall_cell(overall: dict[str, Any]) -> str:
    if overall["value"] is None:
        return "n/a"
    text = f"{overall['value']:.3f} {_fmt_ci(overall['ci95'])}"
    if overall["omitted"]:
        text += f" (of {len(overall['defined'])}: " + ", ".join(SCORECARD_TITLES[n].lower() for n in overall["omitted"]) + " n/a)"
    return text


def scorecard_row(label: str, card: dict[str, Any]) -> str:
    lat = card["turn_taking"]["latency_ms"]
    latency = f"{_fmt_value(lat['p50'], 0)} {_fmt_ci(lat['p50']['ci95'], 0)} / {_fmt_value(lat['p90'], 0)} (n={lat['n']})"
    return (
        f"| {label} | {scorecard_cell(card['pause_handling'])} | {scorecard_cell(card['turn_taking'])} | {latency} | "
        f"{scorecard_cell(card['interruption_handling'])} | {scorecard_cell(card['backchannel_handling'])} | "
        f"{overall_cell(card['overall'])} |"
    )


def scorecard_notes(report: dict[str, Any]) -> list[str]:
    boot = report.get("bootstrap") or {}
    weights = report.get("scorecard_weights") or {}
    equal = len(set(weights.values())) <= 1
    return [
        "Pause handling = 1 − EOT fp_rate (mid-turn pauses not cut into); turn taking = EOT recall, with EOT latency "
        "shown beside it, not inside it; interruption handling = INT recall; backchannel handling = 1 − INT fp_rate. "
        "A value with no events behind it is n/a, never 0 or 1; the overall is the "
        + ("unweighted" if equal else "weighted")
        + " mean of the values that are defined (weights are a presentation choice: "
        + ", ".join(f"{SCORECARD_TITLES[n].lower()} {w:g}" for n, w in weights.items())
        + ").",
        f"95% CIs: cluster bootstrap over conversations ({boot.get('draws')} draws, seed {boot.get('seed')}); each draw "
        "resamples whole conversations, sums their tp/fn/fp/tn and takes the pooled rate (latency: the pooled TP "
        "latencies' percentile). n = events behind the value (latency: TPs). No CI below two conversations.",
    ]


def render_markdown(report: dict[str, Any], run_info: dict[str, Any]) -> str:
    model = run_info["model"]
    dataset = run_info["dataset"]
    scorer = run_info["turnbench"]
    scope = f"{report['conversations_scored']} conversations"
    if report.get("partial"):
        scope += f" (subset of {report['conversations_in_split']}; not comparable to a full run)"
    lines = [
        f"# turn-bench report: {model['name']} on {dataset['split']}",
        "",
        f"- Model: `{model['name']}` config `{json.dumps(model['config'])}`",
        f"- Dataset: `{dataset['source']}`" + (f" @ `{dataset['revision']}`" if dataset.get("revision") else " (local directory)"),
        f"- Scorer: Sesame TurnBench `{scorer['commit'] or scorer['version']}` (official scorer, unchanged)",
        f"- Harness: turn-bench {run_info['harness']['version']} @ `{run_info['harness']['git_commit']}`"
        + (" (dirty)" if run_info['harness']['git_dirty'] else ""),
        f"- Scored: {scope}",
    ]
    if report.get("rescored"):
        lines.append(f"- Re-scored from the run's predictions at {report['rescored']['at']} (model not re-run)")
    lines += [
        "",
        "Blocks: **caller** = speaker_1 only (headline), **agent** = speaker_2 only, **all** = both speakers "
        "(Sesame's scorer verbatim). Caller + agent tp/fn/fp/tn add up to all.",
    ]
    blocks = report.get("blocks") or {"all": {"aggregate": report["aggregate"], "per_conversation": report["per_conversation"]}}
    if any("scorecard" in scores for scores in blocks.values()):
        lines += ["", "## Conversational Dynamics scorecard", ""] + scorecard_notes(report) + ["", SCORECARD_HEADER, SCORECARD_RULE]
        for block, scores in blocks.items():
            if "scorecard" in scores:
                lines.append(scorecard_row(BLOCK_TITLES.get(block, block) + (" — headline" if block == report.get("headline_block") else ""), scores["scorecard"]))
        for axis, names in (report.get("breakdowns") or {}).items():
            for block in blocks:
                if not all("scorecard" in group for group in names.values()):
                    continue
                lines += ["", f"### {BLOCK_TITLES.get(block, block)} by {axis}", "", SCORECARD_HEADER.replace("| |", f"| {axis} (n conv) |", 1), SCORECARD_RULE]
                for name, group in names.items():
                    lines.append(scorecard_row(f"{name} ({group['conversations']})", group["scorecard"][block]))
    for block, scores in blocks.items():
        lines += [
            "",
            f"## {BLOCK_TITLES.get(block, block)}",
            "",
            TASK_HEADER,
            TASK_RULE,
            task_row("EOT", scores["aggregate"]["eot"], ci=(scores.get("ci95") or {}).get("eot")),
            task_row("INT", scores["aggregate"]["int"], ci=(scores.get("ci95") or {}).get("int")),
        ]
    for axis, names in (report.get("breakdowns") or {}).items():
        for block in blocks:
            lines += [
                "",
                f"## {BLOCK_TITLES.get(block, block)} by {axis}",
                "",
                TASK_HEADER.replace("| task |", f"| {axis} | n | task |", 1),
                "|---|---" + TASK_RULE,
            ]
            for name, group in names.items():
                for task in ("EOT", "INT"):
                    ci = ((group.get("ci95") or {}).get(block) or {}).get(task.lower())
                    lines.append(task_row(task, group["blocks"][block][task.lower()], prefix=f" {name} | {group['conversations']} |", ci=ci))
    for block, scores in blocks.items():
        lines += [
            "",
            f"## Per conversation: {BLOCK_TITLES.get(block, block)}",
            "",
            "| conversation | EOT recall | EOT fp_rate | EOT p50 ms | INT recall | INT fp_rate | INT p50 ms |",
            "|---|---|---|---|---|---|---|",
        ]
        for cid, conv_scores in scores["per_conversation"].items():
            eot, intr = conv_scores["eot"], conv_scores["int"]
            lines.append(
                f"| {cid} | {_fmt(eot['recall'])} | {_fmt(eot['fp_rate'])} | {_fmt_ms(eot['latency_ms']['p50'])} "
                f"| {_fmt(intr['recall'])} | {_fmt(intr['fp_rate'])} | {_fmt_ms(intr['latency_ms']['p50'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out: Path, run_info: dict[str, Any]) -> None:
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report, run_info), encoding="utf-8")
