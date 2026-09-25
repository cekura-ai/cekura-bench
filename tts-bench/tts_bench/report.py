"""Aggregation and rendering for TTS runs. The statistics are ``tts_bench.common.stats``.

Rows are per probe variant × cohort (and per item within a cohort), never
pooled into one number per provider. Latency fields get P50 with a bootstrap
interval and P90; P95/P99 only above the declared minimum n. Voids are tallied
by class and printed, exclusions first.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from tts_bench.common.stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, MIN_TAIL_N, latency_summary, rate_summary
from tts_bench.store import latest_cells, write_manifest

LATENCY_FIELDS = (
    "ttfa_ms", "roundtrip_ms", "leading_silence_ms", "ttfa_from_input_done_ms",
    "cancel_to_last_chunk_ms", "audio_after_cancel_ms", "min_margin_ms", "stall_ms",
    "duration_delta_ms", "ttfa_delta_ms", "ttfa_max_ms", "completion_ms",
)
COUNT_FIELDS = ("underruns", "chunks", "framed_underruns", "chunks_after_cancel")
FLAG_FIELDS = ("identical_pcm", "started_before_input_done")


def load_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    cells = latest_cells(root)
    provenance = json.loads((root / "provenance.json").read_text()) if (root / "provenance.json").exists() else {}
    plan = json.loads((root / "plan.json").read_text())["cells"] if (root / "plan.json").exists() else []
    scores = {}
    if (root / "scores.jsonl").exists():
        for line in (root / "scores.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                scores[row["cell_id"]] = row
    return {"root": str(root), "provenance": provenance, "plan": plan, "cells": cells, "scores": scores}


def _is_exclusion(cell: dict[str, Any]) -> bool:
    return str(cell.get("void") or "").startswith("configuration not supported")


def _void_class(reason: str) -> str:
    if reason.startswith("configuration not supported"):
        return "excluded: " + reason.split(": ", 1)[1]
    if reason.startswith("provider refused"):
        return "provider refused"
    if reason.startswith("harness error"):
        return "harness error"
    return reason


def summarize_group(cells: Sequence[dict[str, Any]], scores: dict[str, Any] | None = None) -> dict[str, Any]:
    scored = [c for c in cells if not c.get("void")]
    voids: dict[str, int] = defaultdict(int)
    for cell in cells:
        if cell.get("void"):
            voids[_void_class(cell["void"])] += 1
    out: dict[str, Any] = {"cells": len(cells), "scored": len(scored), "voids": dict(voids)}
    for field in LATENCY_FIELDS:
        values = [c["values"].get(field) for c in scored if c["values"].get(field) is not None]
        if values:
            out[field] = latency_summary(values)
    for field in COUNT_FIELDS:
        values = [c["values"].get(field) for c in scored if c["values"].get(field) is not None]
        if values:
            out[field] = {"n": len(values), "total": int(sum(values)), "cells_nonzero": sum(1 for v in values if v)}
    for field in FLAG_FIELDS:
        values = [c["values"].get(field) for c in scored if c["values"].get(field) is not None]
        if values:
            out[field] = rate_summary([bool(v) for v in values])
    if any(c.get("verdict") in ("pass", "fail") for c in scored):
        out["success"] = rate_summary([c["verdict"] == "pass" for c in scored if c.get("verdict") in ("pass", "fail")])
    if scores:
        rows = [scores[c["artifacts"]["slug"]] for c in scored if c["artifacts"]["slug"] in scores]
        if rows:
            out["transcription"] = _pool_scores(rows)
    return out


def _pool_scores(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pooled word error over the group: sum of errors over sum of reference words, per instrument."""
    out: dict[str, Any] = {"n": len(rows)}
    instruments: dict[str, dict[str, float]] = defaultdict(lambda: {"errors": 0.0, "words": 0.0, "digits_ok": 0.0, "digits_n": 0.0})
    for row in rows:
        for name, score in row.get("instruments", {}).items():
            if score.get("error"):
                continue
            best = score.get("best") or {}
            instruments[name]["errors"] += best.get("errors", 0)
            instruments[name]["words"] += best.get("reference_words", 0)
            if score.get("digits_expected"):
                instruments[name]["digits_n"] += 1
                instruments[name]["digits_ok"] += 1 if score.get("digits_match") else 0
    for name, acc in instruments.items():
        out[name] = {
            "pooled_wer": None if acc["words"] == 0 else round(acc["errors"] / acc["words"], 4),
            "reference_words": int(acc["words"]),
            "digits_match_rate": None if acc["digits_n"] == 0 else round(acc["digits_ok"] / acc["digits_n"], 4),
            "digits_cells": int(acc["digits_n"]),
        }
    disagreements = [r for r in rows if r.get("instruments_disagree")]
    out["instruments_disagree"] = len(disagreements)
    return out


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    cells, prov, scores = run["cells"], run["provenance"], run.get("scores") or {}
    by_variant_cohort: dict[tuple[str, str], list] = defaultdict(list)
    by_variant_item: dict[tuple[str, str, str], list] = defaultdict(list)
    for cell in cells:
        if cell.get("sentinel"):
            continue
        by_variant_cohort[(cell["variant"], cell["cohort"])].append(cell)
        by_variant_item[(cell["variant"], cell["cohort"], cell["item"])].append(cell)
    return {
        "run_id": prov.get("run_id", Path(run["root"]).name),
        "provider": prov.get("provider"),
        "model": prov.get("model"),
        "voice": prov.get("voice"),
        "sample_rate": prov.get("sample_rate"),
        "capabilities": prov.get("capabilities"),
        "methodology_version": prov.get("methodology_version"),
        "harness_commit": (prov.get("harness") or {}).get("commit"),
        "harness_dirty": (prov.get("harness") or {}).get("dirty"),
        "corpus_version": prov.get("corpus_version"),
        "counts": {
            "planned": len(run["plan"]) or None,
            "completed": len(cells),
            "voids": sum(1 for c in cells if c.get("void")),
            "errors": sum(1 for c in cells if c.get("error") and not _is_exclusion(c)),
            "scored_transcripts": len(scores),
        },
        "rules": {"min_tail_n": MIN_TAIL_N, "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED},
        "sentinel": summarize_group([c for c in cells if c.get("sentinel")]) if any(c.get("sentinel") for c in cells) else None,
        "by_cohort": {f"{v}/{c}": summarize_group(members, scores) for (v, c), members in sorted(by_variant_cohort.items())},
        "by_item": {f"{v}/{c}/{i}": summarize_group(members, scores) for (v, c, i), members in sorted(by_variant_item.items())},
        "voids": _void_tally(cells),
    }


def _void_tally(cells: Sequence[dict[str, Any]]) -> dict[str, int]:
    tally: dict[str, int] = defaultdict(int)
    for cell in cells:
        if cell.get("void"):
            tally[_void_class(cell["void"])] += 1
    return dict(sorted(tally.items()))


def _fmt(summary: dict[str, Any] | None, key: str = "p50") -> str:
    if not summary or summary.get("n", 0) == 0:
        return ""
    value = summary.get(key)
    ci = summary.get("p50_ci95") if key == "p50" else None
    text = "" if value is None else f"{value:.0f}"
    if ci:
        text += f" [{ci[0]:.0f}, {ci[1]:.0f}]"
    return text


def render_markdown(report: dict[str, Any]) -> str:
    caps = report.get("capabilities") or {}
    lines = [
        f"# {report['provider']} / {report['model']} / {report['voice']} @ {report['sample_rate']} Hz",
        "",
        f"run `{report['run_id']}` · methodology `{report['methodology_version']}` · harness "
        f"`{report['harness_commit']}`{' (dirty)' if report['harness_dirty'] else ''} · corpus `{report['corpus_version']}`",
        "",
        f"cells {report['counts']['completed']}/{report['counts']['planned']} · voids {report['counts']['voids']} · "
        f"errors {report['counts']['errors']} · transcripts scored {report['counts']['scored_transcripts']}",
        "",
        f"transport {caps.get('transport')} · streamed input {caps.get('streamed_input')} · cancel {caps.get('cancel')} · "
        f"continuation {caps.get('continuation')} · native rates {caps.get('native_rates')} · 8 kHz mu-law native "
        f"{caps.get('native_mulaw_8k')} · excluded from t0: {caps.get('setup_excluded_from_t0')}",
        "",
    ]
    sentinel = report.get("sentinel")
    if sentinel and sentinel.get("ttfa_ms"):
        s = sentinel["ttfa_ms"]
        lines += [f"**Sentinel** (one fixed cell, n={s['n']}): TTFA P50 {_fmt(s)} ms, spread {s['min']:.0f}–{s['max']:.0f} ms", ""]
    lines += [
        "| probe | cohort | n | TTFA P50 [CI] | round-trip P50 | lead silence P50 | min margin P50 | underrun cells | cancel→last P50 | audio after cancel P50 | Δdur P50 | pooled WER | digits | pass | voids |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, g in report["by_cohort"].items():
        variant, cohort = key.split("/", 1)
        under = g.get("underruns") or {}
        trans = g.get("transcription") or {}
        primary = next((v for v in trans.values() if isinstance(v, dict) and "pooled_wer" in v), {})
        success = g.get("success") or {}
        wer = primary.get("pooled_wer")
        digits = primary.get("digits_match_rate")
        cols = [
            variant, cohort, str(g["scored"]),
            _fmt(g.get("ttfa_ms")), _fmt(g.get("roundtrip_ms")), _fmt(g.get("leading_silence_ms")), _fmt(g.get("min_margin_ms")),
            f"{under['cells_nonzero']}/{under['n']}" if under else "",
            _fmt(g.get("cancel_to_last_chunk_ms")), _fmt(g.get("audio_after_cancel_ms")), _fmt(g.get("duration_delta_ms")),
            "" if wer is None else f"{wer:.3f}",
            "" if digits is None else f"{digits:.2f}",
            f"{success['passed']}/{success['n']}" if success else "",
            str(sum(g["voids"].values())),
        ]
        lines.append("| " + " | ".join(cols) + " |")
    if report["voids"]:
        lines += ["", "## Voids", ""] + [f"- {n} × {k}" for k, n in report["voids"].items()]
    lines += ["", f"Rules: P95/P99 withheld under n={MIN_TAIL_N} · bootstrap {BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED} · "
              "TTFA = first chunk arrival − first text frame + leading silence · playout clock starts at the first audible sample", ""]
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Aggregate TTS runs into reports.")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    for run_dir in args.runs:
        report = summarize_run(load_run(run_dir))
        (Path(run_dir) / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (Path(run_dir) / "report.md").write_text(render_markdown(report))
        write_manifest(Path(run_dir))
        print(f"{run_dir}: report.md written ({report['counts']})")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
