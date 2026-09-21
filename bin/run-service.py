#!/usr/bin/env python
"""Run a service-bench campaign against one provider configuration.

    python bin/run-service.py --provider openai-realtime --suite latency --repeats 5 --voice f-us --voice m-us

Writes a run directory under data/service/ containing, per cell, the caller and
agent audio, the normalized event log, the raw provider frames and a
self-contained cell record, then audits the directory and writes the
aggregated report beside it. Every published number is recomputable from those
files.

A holdout set authored outside this repository runs through the same probes:

    python bin/run-service.py --suite task --holdout /path/to/holdout
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import corpus_spec  # noqa: E402
from service.adapters.base import TurnDetection  # noqa: E402
from service.audit import audit_run  # noqa: E402
from service.corpus_v1 import build  # noqa: E402
from service.probes import (  # noqa: E402
    BackchannelTolerance,
    BargeIn,
    CallerTranscription,
    EndpointingLadder,
    FalseTrigger,
    FilledPause,
    ResponseLatency,
    task_probes,
)
from service.registry import PROVIDERS  # noqa: E402
from service.report import load_run, render_markdown, summarize_run  # noqa: E402
from service.runner import DEFAULT_INSTRUCTIONS, RunSpec, Runner  # noqa: E402
from service.scenarios import SCENARIOS  # noqa: E402
from service.transforms import TRANSFORMS  # noqa: E402

INSTRUCTIONS = DEFAULT_INSTRUCTIONS

MANUAL = TurnDetection("manual")
VAD500 = TurnDetection("server_vad", silence_duration_ms=500)
VAD200 = TurnDetection("server_vad", silence_duration_ms=200)
SEMANTIC = TurnDetection("semantic_vad")

DEFAULT_PARAMS = {
    "ladder_ms": [400, 600, 800, 1000, 1500, 2000],
    "filled_pause_ms": [1500, 2000],
    "barge_in_ms": [400, 900, 1500],
    "backchannel_ms": [400, 900],
    "false_trigger_dbfs": [-30, -20],
    "latency_clips": ["open.book", "open.digits", "open.question"],
    "transcription_clips": ["task.identify", "identify.alt", "task.cancel", "open.digits", "open.book"],
}


def suites(params: dict, scenarios) -> dict[str, tuple[list, list]]:
    p = {**DEFAULT_PARAMS, **params}
    return {
        "smoke": ([ResponseLatency()], [MANUAL, VAD500]),
        "latency": ([ResponseLatency(clip_id=c) for c in p["latency_clips"]], [MANUAL, VAD200, VAD500, SEMANTIC]),
        "endpointing": (
            [EndpointingLadder(gap_ms=g) for g in p["ladder_ms"]]
            + [EndpointingLadder(gap_ms=g, first="date.part1", second="date.part2", name="endpointing_ladder_date") for g in p["ladder_ms"]]
            + [FilledPause(gap_ms=g) for g in p["filled_pause_ms"]],
            [VAD500],
        ),
        "interaction": (
            [BargeIn(after_onset_ms=ms) for ms in p["barge_in_ms"]]
            + [BargeIn(after_onset_ms=0, name="simultaneous_start"), BargeIn(clip_id="correct.midanswer", name="barge_in_correction")]
            + [BackchannelTolerance(after_onset_ms=ms) for ms in p["backchannel_ms"]],
            [VAD500],
        ),
        "noise": ([FalseTrigger(duration_ms=20000, level_dbfs=db) for db in p["false_trigger_dbfs"]], [VAD500]),
        "transcription": ([CallerTranscription(clip_id=c) for c in p["transcription_clips"]], [VAD500]),
        # Tools run in the service bench, not only in the agent lane. The text arms are the
        # same scenarios with the same tool server and only the modality changed,
        # which is what makes a voice failure attributable to the speech pathway.
        "task": (task_probes("appointments", scenarios), [VAD500]),
        "task-text": (task_probes("appointments", scenarios), [VAD500]),
        "task-medicare": (task_probes("medicare", scenarios), [VAD500]),
        "task-medicare-text": (task_probes("medicare", scenarios), [VAD500]),
        # Robustness: the same two probes under every published transform.
        "robustness": ([ResponseLatency(), CallerTranscription(clip_id="task.identify")], [VAD500]),
    }


TOOL_SUITES = {"task": "appointments", "task-text": "appointments", "task-medicare": "medicare", "task-medicare-text": "medicare"}
TEXT_SUITES = {"task-text", "task-medicare-text"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai-realtime", choices=sorted(PROVIDERS))
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--model")
    parser.add_argument("--voice", action="append", help="caller voice label; repeatable")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--transform", action="append", help="degradation by name; repeatable; 'all' for every one")
    parser.add_argument("--scenario", action="append", help="restrict a task suite to these scenario ids")
    parser.add_argument("--no-sentinel", action="store_true")
    parser.add_argument("--holdout", help="directory holding corpus.json, audio/, scenarios.json, params.json")
    parser.add_argument("--env", help="dotenv file holding the provider credential")
    parser.add_argument("--out", default="data/service")
    parser.add_argument("--label")
    args = parser.parse_args()

    entry = PROVIDERS[args.provider]
    key = os.environ.get(entry.credential_env)
    if not key and args.env:
        from dotenv import dotenv_values

        key = dotenv_values(args.env).get(entry.credential_env)
    if not key:
        print(f"{entry.credential_env} is not set", file=sys.stderr)
        return 2

    if args.holdout:
        corpus = corpus_spec.load_corpus(args.holdout)
        scenarios = corpus_spec.load_scenarios(args.holdout) or SCENARIOS
        params = corpus_spec.load_params(args.holdout)
    else:
        corpus, scenarios, params = build(), SCENARIOS, {}
    if args.scenario:
        scenarios = [s for s in scenarios if s.id in set(args.scenario)]

    table = suites(params, scenarios)
    if args.suite not in table:
        print(f"unknown suite {args.suite}; known: {sorted(table)}", file=sys.stderr)
        return 2
    probes, configs = table[args.suite]
    transforms = args.transform or ["clean"]
    if transforms == ["all"]:
        transforms = list(TRANSFORMS)
    if args.suite == "robustness" and not args.transform:
        transforms = list(TRANSFORMS)

    suite_tools = TOOL_SUITES.get(args.suite)
    label = args.label or args.suite + ("-holdout" if args.holdout else "")
    spec = RunSpec(
        provider=args.provider,
        probes=probes,
        configs=configs,
        voices=args.voice or ["f-us"],
        repeats=args.repeats,
        model=args.model,
        # A tool suite uses the contract's own published system prompt, so the
        # model under test is given exactly what a third party would give it.
        instructions="" if suite_tools else INSTRUCTIONS,
        suite=suite_tools,
        modality="text" if args.suite in TEXT_SUITES else "audio",
        transforms=transforms,
        sentinel=not args.no_sentinel,
        corpus_root=str(corpus.root),
        out_root=args.out,
        label=label,
    )

    def show(cell) -> None:
        for field in ("latency_ms", "per_minute", "stop_ms", "interrupted_in_gap", "wer", "ended_by"):
            if field in cell.values:
                detail = f"{field}={cell.values[field]}"
                break
        else:
            detail = ""
        print(
            f"  {cell.variant:38} {cell.config:18} {cell.voice:5} {cell.transform:10} r{cell.repeat}  "
            f"{cell.verdict or '-':5} {cell.void or detail}",
            flush=True,
        )

    runner = Runner(spec, corpus, key)
    print(f"service bench: {args.provider} {runner.model} suite={args.suite} repeats={args.repeats} cells={len(runner.planned)}")
    out = asyncio.run(runner.run(on_cell=show))
    print(f"\nrun -> {out}")

    report = summarize_run(load_run(out))
    (out / "report.md").write_text(render_markdown(report))
    import json

    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    # Audited here rather than on demand. A record gap found now costs one rerun;
    # found later it costs a rerun against a provider that has changed underneath
    # the result, which is not the same measurement.
    audit = audit_run(out)
    if audit["ok"]:
        print(f"record complete: {audit['found']}/{audit['planned']} cells, recomputable; report.md written")
        return 0
    print(f"record INCOMPLETE: {audit['found']}/{audit['planned']} cells", file=sys.stderr)
    for problem in audit["problems"]:
        print(f"  ! {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
