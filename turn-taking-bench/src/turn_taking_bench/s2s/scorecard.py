"""One scorecard per S2S provider: Track 1 (end of turn) beside Track 2 (interaction).

Nothing is computed here that the two tracks did not already compute. Track 1
rows come from a ``turn-taking-bench run`` directory's ``report.json`` — the caller
block's Conversational Dynamics scorecard with its conversation-level
bootstrap intervals (``turn_taking_bench.scorecard``) — and Track 2 rows from an
interaction run's ``interaction.json`` (``turn_taking_bench.s2s.interaction``). The
two are not combined into one number: Track 1 scores the model's end-of-turn
decisions on authored conversations with Sesame's scorer, Track 2 scores its
yielding and backchannel tolerance on the service bench's probes, and an
overall across them would average two different experiments.

Reference rows are other Track 1 runs on the same conversations (an energy VAD,
Smart Turn v3), shown so the S2S numbers sit next to something known. They are
refused when their conversation set differs from the S2S run's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "turn-taking-bench/s2s-scorecard/1"


# ---- loading ---------------------------------------------------------------------------------


def load_track1(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"{run_dir} has no report.json (unscored run)")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    block = report["blocks"]["caller"]
    diagnostics = _track1_diagnostics(run_dir)
    conversations = [c["conversation_id"] for c in run_info.get("conversations", []) if c.get("status") == "ok"]
    return {
        "run_dir": str(run_dir),
        "run_id": run_info.get("run_id", run_dir.name),
        "model": run_info["model"]["name"],
        "config": run_info["model"].get("config"),
        "provenance": run_info["model"].get("provenance"),
        "dataset": run_info.get("dataset"),
        "realtime": run_info.get("realtime"),
        "conversations": conversations,
        "failures": run_info.get("failures", []),
        "scorecard": block["scorecard"],
        "eot": block["aggregate"]["eot"],
        "int": block["aggregate"]["int"],
        "partial": report.get("partial"),
        "diagnostics": diagnostics,
    }


def _track1_diagnostics(run_dir: Path) -> dict[str, Any]:
    """Pacing, response counts, usage and session acknowledgements summed over
    the run's ``events.jsonl`` diagnostics (present for S2S models only)."""
    path = Path(run_dir) / "events.jsonl"
    out: dict[str, Any] = {"conversations": 0, "pacing_void": [], "max_slip_ms": 0.0, "responses": 0, "interrupted": 0, "usage": {}, "errors": [], "acknowledged": None, "model": None, "adapter": None}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        diag = row.get("diagnostics") or {}
        if "pacing" not in diag:
            continue
        out["conversations"] += 1
        out["max_slip_ms"] = max(out["max_slip_ms"], diag["pacing"].get("max_slip_ms", 0.0))
        if diag["pacing"].get("pacing_void"):
            out["pacing_void"].append(row["conversation_id"])
        out["responses"] += diag.get("responses", 0)
        out["interrupted"] += diag.get("interrupted", 0)
        for key, value in (diag.get("usage") or {}).items():
            if isinstance(value, int):
                out["usage"][key] = out["usage"].get(key, 0) + value
        if diag.get("error"):
            out["errors"].append({"conversation_id": row["conversation_id"], "error": diag["error"]})
        out["acknowledged"] = out["acknowledged"] or (diag.get("session") or {}).get("acknowledged")
        out["model"] = out["model"] or diag.get("model")
        out["adapter"] = out["adapter"] or diag.get("adapter")
    return out


def load_track2(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "interaction.json"
    if not path.is_file():
        from turn_taking_bench.s2s.interaction import interaction_summary

        return interaction_summary(Path(run_dir))
    return json.loads(path.read_text(encoding="utf-8"))


# ---- combining -------------------------------------------------------------------------------


def combined_scorecard(
    provider: str,
    track1: dict[str, Any] | None,
    track2: dict[str, Any] | None,
    references: dict[str, dict[str, Any]] | None = None,
    *,
    label: str = "SMOKE",
) -> dict[str, Any]:
    references = references or {}
    if track1 is not None:
        for name, reference in references.items():
            if sorted(reference["conversations"]) != sorted(track1["conversations"]):
                raise ValueError(
                    f"reference {name!r} scored conversations {sorted(reference['conversations'])}, "
                    f"the S2S run {sorted(track1['conversations'])}: not the same rows"
                )
    return {
        "schema": SCHEMA,
        "provider": provider,
        "label": label,
        "track1": None if track1 is None else _track1_rows(track1),
        "track1_references": {name: _track1_rows(reference) for name, reference in references.items()},
        "track2": None if track2 is None else _track2_rows(track2),
        "notes": NOTES,
    }


def _track1_rows(track1: dict[str, Any]) -> dict[str, Any]:
    card = track1["scorecard"]
    latency = card["turn_taking"]["latency_ms"]
    return {
        "model": track1["model"],
        "model_id": (track1.get("provenance") or {}).get("model") or track1["diagnostics"].get("model"),
        "run_id": track1["run_id"],
        "conversations": track1["conversations"],
        "failures": track1["failures"],
        "pause_handling": card["pause_handling"],
        "turn_taking": {k: v for k, v in card["turn_taking"].items() if k != "latency_ms"},
        "latency_ms": latency,
        "interruption_handling": card["interruption_handling"],
        "backchannel_handling": card["backchannel_handling"],
        "eot": track1["eot"],
        "pacing": {"max_slip_ms": track1["diagnostics"]["max_slip_ms"], "pacing_void": track1["diagnostics"]["pacing_void"]},
        "responses": track1["diagnostics"]["responses"],
        "usage": track1["diagnostics"]["usage"],
        "session_acknowledged": track1["diagnostics"]["acknowledged"],
        "prompt_sha256": (track1.get("provenance") or {}).get("prompt_sha256"),
        "readout": (track1.get("provenance") or {}).get("readout"),
        "errors": track1["diagnostics"]["errors"],
    }


def _track2_rows(track2: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": track2.get("run_id"),
        "model_id": track2.get("model"),
        "config": track2.get("config"),
        "voices": track2.get("voices"),
        "repeats": track2.get("repeats"),
        "corpus_version": track2.get("corpus_version"),
        "interruption_handling": track2["interruption_handling"],
        "yield_ms": track2["yield_ms"],
        "backchannel_handling": track2["backchannel_handling"],
        "variants": track2.get("variants", {}),
        "voids": track2.get("voids", {}),
        "errors": track2.get("errors", []),
        "usage": track2.get("usage_totals", {}),
        "mapping": track2.get("mapping", {}),
    }


NOTES = [
    "Track 1 (end of turn): Sesame's TurnBench scorer on the caller block (speaker_1) of the named conversations. "
    "Pause handling = 1 - EOT fp_rate (cut-ins during scripted mid-turn pauses); turn taking = EOT recall; latency = "
    "p50 / p90 of the matched EOT latencies. For an S2S model an EOT is an onset of its speech while the caller is quiet "
    "(Sesame's generative protocol); INT is undefined for S2S and shown n/a. 95% CIs are conversation-level cluster "
    "bootstrap intervals and are only as narrow as the number of conversations allows.",
    "Track 2 (interaction): the service bench's barge_in and backchannel_tolerance probes, unchanged, on its caller corpus. "
    "Interruption handling = share of scored barge-in cells in which the model yielded; yield time = caller onset to the "
    "agent falling silent on the reference client (p50 / p90); backchannel handling = share of scored backchannel cells in "
    "which the model kept talking and did not answer it. Voided cells are counted, not scored. CIs are bootstrap intervals "
    "over repeats.",
    "A label of SMOKE means a handful of conversations and repeats: the numbers locate a provider, they do not rank one.",
]


# ---- rendering -------------------------------------------------------------------------------


def _ci(ci: Any, digits: int = 3) -> str:
    if not ci:
        return "n/a"
    if isinstance(ci, dict):
        if ci.get("lo") is None:
            return "n/a"
        return f"[{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"
    lo, hi = ci
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def _value(entry: dict[str, Any] | None, digits: int = 3) -> str:
    if not entry or entry.get("value") is None:
        return f"n/a (n={0 if not entry else entry.get('n', 0)})"
    return f"{entry['value']:.{digits}f} {_ci(entry.get('ci95'), digits)} (n={entry.get('n', 0)})"


def _latency(entry: dict[str, Any]) -> str:
    p50 = entry.get("p50") or {}
    p90 = entry.get("p90") or {}
    if p50.get("value") is None:
        return f"n/a (n={entry.get('n', 0)})"
    return f"{p50['value']:.0f} {_ci(p50.get('ci95'), 0)} / {p90['value']:.0f} (n={entry.get('n', 0)})"


def _track1_line(label: str, rows: dict[str, Any]) -> str:
    eot = rows["eot"]
    return (
        f"| {label} | {_value(rows['pause_handling'])} | {_value(rows['turn_taking'])} | {_latency(rows['latency_ms'])} | "
        f"{eot['tp']} / {eot['fn']} / {eot['fp']} / {eot['tn']} | {len(rows['conversations'])} |"
    )


def render_markdown(card: dict[str, Any]) -> str:
    provider = card["provider"]
    lines = [f"# S2S scorecard — {provider} ({card['label']})", ""]
    track1 = card.get("track1")
    if track1 is not None:
        model_id = track1.get("model_id") or "?"
        lines += [
            f"## Track 1 — end of turn (Sesame method), caller block",
            "",
            f"- Model: `{track1['model']}` = {model_id}; readout `{track1.get('readout')}`; prompt sha256 `{(track1.get('prompt_sha256') or '')[:12]}`",
            f"- Conversations: {', '.join(track1['conversations'])}" + (f"; failed: {len(track1['failures'])}" if track1["failures"] else ""),
            f"- Pacing: max slip {track1['pacing']['max_slip_ms']:.1f} ms" + (f"; pacing void: {', '.join(track1['pacing']['pacing_void'])}" if track1["pacing"]["pacing_void"] else ""),
            f"- Model replies recorded: {track1['responses']}; usage: {json.dumps(track1['usage'])}",
            "",
            "| model | pause handling [CI] (n) | turn taking [CI] (n) | EOT latency p50 [CI] / p90 ms (n) | EOT tp / fn / fp / tn | conversations |",
            "|---|---|---|---|---|---|",
            _track1_line(f"**s2s:{provider}**", track1),
        ]
        for name, reference in (card.get("track1_references") or {}).items():
            lines.append(_track1_line(f"{name} (reference)", reference))
        lines += ["", "INT (interruption handling / backchannel handling in Track 1 terms) is undefined for an S2S model: n/a."]
        if track1.get("errors"):
            lines += ["", "Session errors:"] + [f"- {e['conversation_id']}: {e['error']}" for e in track1["errors"]]
    track2 = card.get("track2")
    if track2 is not None:
        y = track2["yield_ms"]
        yield_text = "n/a (n=0)" if y.get("p50") is None else f"{y['p50']:.0f} {_ci(y.get('p50_ci95'), 0)} / {y['p90']:.0f} (n={y['n']})"
        lines += [
            "",
            "## Track 2 — interaction (service-bench probes)",
            "",
            f"- Model: {track2.get('model_id')}; turn detection {', '.join(track2.get('config') or [])}; caller voice {', '.join(track2.get('voices') or [])}; "
            f"repeats {track2.get('repeats')}; corpus {track2.get('corpus_version')}",
            "",
            "| interruption handling (yielded) [CI] (n) | yield time p50 [CI] / p90 ms (n) | backchannel handling (kept talking) [CI] (n) | voided cells |",
            "|---|---|---|---|",
            f"| {_value(track2['interruption_handling'])} | {yield_text} | {_value(track2['backchannel_handling'])} | "
            f"{track2['interruption_handling']['voids'] + track2['backchannel_handling']['voids']} |",
            "",
            "| probe variant | cells | scored | passed | stop_ms | voids |",
            "|---|---|---|---|---|---|",
        ]
        for variant, row in track2["variants"].items():
            stop = ", ".join(f"{v:.0f}" for v in row["stop_ms"]) or "—"
            voids = "; ".join(row["voids"]) or "—"
            lines.append(f"| {variant} | {row['cells']} | {row['scored']} | {row['passed']} | {stop} | {voids} |")
        if track2.get("errors"):
            lines += ["", "Cell errors:"] + [f"- {e}" for e in track2["errors"]]
    lines += ["", "## Notes", ""] + [f"- {note}" for note in card["notes"]] + [""]
    return "\n".join(lines)


def write_scorecard(out_dir: Path, card: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"s2s-scorecard-{card['provider']}.json"
    md_path = out_dir / f"s2s-scorecard-{card['provider']}.md"
    json_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(card), encoding="utf-8")
    return json_path, md_path
