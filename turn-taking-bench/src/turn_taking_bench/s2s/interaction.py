"""Track 2 — interaction: the service bench's barge-in and backchannel probes, as is.

The service bench already measures the two interaction behaviours an EOT
benchmark cannot: whether a provider *yields* when spoken over, and how fast,
and whether a listener's "mm hmm" derails its reply. Those probes are closed
loop (anchored on the agent's own onset), need the service bench's authored
caller corpus and its branching caller, and are not changed here. This module
only builds a ``RunSpec`` for them — the same fixed phone-agent prompt as
Track 1, the provider's native turn detection, one caller voice, a handful of
repeats — runs the service bench's ``Runner`` on it, and reads its cells back
into the scorecard's vocabulary:

    interruption handling = share of scored barge_in probes with verdict "pass"
                            (the listener stopped hearing the agent after the
                            caller's onset); yield time = that probe's
                            ``stop_ms`` (caller onset → agent inaudible on the
                            reference client), p50 / p90
    backchannel handling  = share of scored backchannel_tolerance probes with
                            verdict "pass" (the agent kept talking and did not
                            answer the backchannel as a turn)

A void cell (the agent never spoke, finished before the probe point, or the
provider closed the session) is counted and listed, never scored — the
service bench's own rule. Rates carry the service bench's bootstrap intervals
over repeats (``service.report``), which at a smoke's repeat count are wide
by construction.

The stimuli are the service bench's caller corpus (``corpus/service``, v0.2:
``open.book`` as the opener, ``bargein.stop`` as the barge-in, ``token.backchannel``
as the backchannel), rendered with ElevenLabs on demand; a later version
drives these probes from turn-taking-bench's own scenario specs.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from turn_taking_bench.author import service_bridge
from turn_taking_bench.s2s import SYSTEM_PROMPT, prompt_sha256
from turn_taking_bench.s2s.credentials import credential

SCHEMA = "turn-taking-bench/s2s-interaction/1"
BARGE_IN_MS: tuple[float, ...] = (400.0, 900.0, 1500.0)
BACKCHANNEL_MS: tuple[float, ...] = (400.0, 900.0)
NEEDED_CLIPS: tuple[str, ...] = ("open.book", "bargein.stop", "token.backchannel")

MAPPING = {
    "interruption_handling": "share of scored barge_in cells with verdict pass (service.probes.BargeIn: the listener stopped hearing the agent after the caller's onset)",
    "yield_ms": "barge_in stop_ms of passing cells: caller onset -> agent inaudible on the reference client (service.metrics.barge_in)",
    "backchannel_handling": "share of scored backchannel_tolerance cells with verdict pass (service.probes.BackchannelTolerance: kept talking and did not answer the backchannel)",
    "voids": "cells the service bench voided (agent never spoke, reply over before the probe point, session closed): counted, never scored",
}


def render_missing(corpus: Any, voice_label: str, key: str, clip_ids: tuple[str, ...] = NEEDED_CLIPS) -> list[str]:
    """Render the clips the probes use for one voice if their files are absent,
    with the service bench's own renderer (``service.clips._elevenlabs_pcm``).
    The committed corpus manifest is left alone: a run records the checksums of
    the renders it actually used in its own ``manifest.json``."""
    clips = service_bridge.clips()
    audio = service_bridge.audio()
    voice = corpus.voices[voice_label]
    written: list[str] = []
    for clip_id in clip_ids:
        target = corpus.path_for(clip_id, voice_label)
        if target.exists():
            continue
        audio.write_wav(target, clips._elevenlabs_pcm(key, corpus.specs[clip_id].text, voice), clips.MASTER_RATE)
        written.append(clip_id)
    return written


def missing_clips(corpus: Any, voice_label: str, clip_ids: tuple[str, ...] = NEEDED_CLIPS) -> list[str]:
    return [clip_id for clip_id in clip_ids if not corpus.path_for(clip_id, voice_label).exists()]


def build_spec(
    provider: str,
    out_root: Path,
    *,
    corpus: Any,
    repeats: int = 2,
    voice: str = "f-us",
    model: str | None = None,
    label: str = "s2s-interaction",
    turn_detection: str = "server_vad",
    silence_duration_ms: int | None = None,
    barge_in_ms: tuple[float, ...] = BARGE_IN_MS,
    backchannel_ms: tuple[float, ...] = BACKCHANNEL_MS,
) -> Any:
    base = service_bridge.adapters_base()
    probes = service_bridge.probes()
    runner = service_bridge.runner()
    return runner.RunSpec(
        provider=provider,
        probes=[probes.BargeIn(after_onset_ms=ms) for ms in barge_in_ms]
        + [probes.BackchannelTolerance(after_onset_ms=ms) for ms in backchannel_ms],
        configs=[base.TurnDetection(turn_detection, silence_duration_ms=silence_duration_ms)],
        voices=[voice],
        repeats=repeats,
        model=model,
        instructions=SYSTEM_PROMPT,
        sentinel=False,
        corpus_root=str(corpus.root),
        out_root=str(out_root),
        label=label,
    )


def run_interaction(
    provider: str,
    out_root: Path,
    *,
    repeats: int = 2,
    voice: str = "f-us",
    model: str | None = None,
    label: str = "s2s-interaction",
    render: bool = False,
    env_file: str | None = None,
    turn_detection: str = "server_vad",
    silence_duration_ms: int | None = None,
    on_cell: Callable[[Any], None] | None = None,
) -> Path:
    """Run the probes against ``provider`` and write the service bench's run
    directory plus ``interaction.json`` (this track's summary) inside it."""
    registry = service_bridge.registry()
    entry = registry.PROVIDERS[provider]
    corpus = service_bridge.corpus_v1().build(str(service_bridge.REPO_ROOT / "corpus" / "service"))
    key = "" if provider == "fake" else credential(entry.credential_env, env_file)
    if render:
        rendered = render_missing(corpus, voice, credential("ELEVENLABS_API_KEY", env_file))
        if rendered:
            print(f"rendered {', '.join(rendered)} for {voice}", flush=True)
    absent = missing_clips(corpus, voice)
    if absent:
        raise FileNotFoundError(f"caller corpus clips not rendered for {voice}: {', '.join(absent)} (use --render)")
    spec = build_spec(
        provider, out_root, corpus=corpus, repeats=repeats, voice=voice, model=model, label=label,
        turn_detection=turn_detection, silence_duration_ms=silence_duration_ms,
    )
    runner = service_bridge.runner().Runner(spec, corpus, key)
    print(f"s2s interaction: {provider} {runner.model} repeats={repeats} voice={voice} cells={len(runner.planned)}", flush=True)
    root = asyncio.run(runner.run(on_cell=on_cell))
    report = service_bridge.report()
    summary = report.summarize_run(report.load_run(root))
    (root / "report.md").write_text(report.render_markdown(summary), encoding="utf-8")
    (root / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    interaction = interaction_summary(root)
    (root / "interaction.json").write_text(json.dumps(interaction, indent=2) + "\n", encoding="utf-8")
    return root


# ---- reading the cells back ----------------------------------------------------------------


def load_cells(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "cells.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    report = service_bridge.report()
    scored = [c for c in cells if not c.get("void")]
    summary = report.rate_summary([c.get("verdict") == "pass" for c in scored])
    return {
        "value": summary.get("rate"),
        "ci95": summary.get("rate_ci95"),
        "n": len(scored),
        "passed": summary.get("passed", 0),
        "cells": len(cells),
        "voids": len(cells) - len(scored),
    }


def _variants(cells: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in sorted({c["variant"] for c in cells}):
        rows = [c for c in cells if c["variant"] == variant]
        scored = [c for c in rows if not c.get("void")]
        out[variant] = {
            "cells": len(rows),
            "scored": len(scored),
            "passed": sum(1 for c in scored if c.get("verdict") == "pass"),
            "voids": [c["void"] for c in rows if c.get("void")],
            "stop_ms": [c["values"].get("stop_ms") for c in scored if c["values"].get("stop_ms") is not None],
        }
    return out


def interaction_summary(run_dir: Path) -> dict[str, Any]:
    """The scorecard rows of one interaction run, read from its ``cells.jsonl``."""
    run_dir = Path(run_dir)
    cells = load_cells(run_dir)
    report = service_bridge.report()
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8")) if (run_dir / "provenance.json").is_file() else {}
    barge = [c for c in cells if c["probe"] == "barge_in"]
    back = [c for c in cells if c["probe"] == "backchannel_tolerance"]
    yields = [c["values"]["stop_ms"] for c in barge if not c.get("void") and c.get("verdict") == "pass" and c["values"].get("stop_ms") is not None]
    latency = report.latency_summary(yields)
    voids = Counter(c["void"] for c in cells if c.get("void"))
    return {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "run_id": provenance.get("run_id", run_dir.name),
        "provider": provenance.get("provider") or (cells[0]["provider"] if cells else None),
        "model": provenance.get("model") or (cells[0]["model"] if cells else None),
        "adapter": provenance.get("adapter"),
        "config": sorted({c["config"] for c in cells}),
        "voices": sorted({c["voice"] for c in cells}),
        "repeats": provenance.get("repeats"),
        "corpus_version": provenance.get("corpus_version"),
        "instructions_sha256": prompt_sha256(SYSTEM_PROMPT),
        "interruption_handling": {**_rate(barge), "source": MAPPING["interruption_handling"]},
        "yield_ms": {
            "p50": latency.get("p50"),
            "p50_ci95": latency.get("p50_ci95"),
            "p90": latency.get("p90"),
            "n": latency.get("n", 0),
            "source": MAPPING["yield_ms"],
        },
        "backchannel_handling": {**_rate(back), "source": MAPPING["backchannel_handling"]},
        "variants": _variants(barge + back),
        "voids": dict(voids),
        "errors": [c["error"] for c in cells if c.get("error")],
        "usage_totals": _usage_totals(cells),
        "mapping": MAPPING,
    }


def _usage_totals(cells: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for cell in cells:
        for key, value in (cell.get("usage") or {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals
