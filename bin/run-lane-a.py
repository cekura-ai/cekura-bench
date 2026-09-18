#!/usr/bin/env python
"""Run a Lane A campaign against one provider configuration.

    python bin/run-lane-a.py --provider openai-realtime --suite smoke --repeats 3

Writes a run directory under data/lane-a/ containing, per cell, the caller and
agent audio, the normalized event log, the raw provider frames, and a cells.jsonl
of results with provenance. Every published number is recomputable from those.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lane_a.adapters.base import TurnDetection  # noqa: E402
from lane_a.corpus_v1 import build  # noqa: E402
from lane_a.probes import (  # noqa: E402
    BackchannelTolerance,
    BargeIn,
    BookingTask,
    EndpointingLadder,
    FalseTrigger,
    ResponseLatency,
)
from lane_a.registry import PROVIDERS  # noqa: E402
from lane_a.runner import RunSpec, Runner  # noqa: E402

INSTRUCTIONS = (
    "You are Riley, the receptionist at Cedar Valley Family Practice. "
    "Answer in one or two short sentences. Never mention that you are an AI."
)

LADDER_MS = (400, 600, 800, 1000, 1500, 2000)

SUITES = {
    "smoke": (
        [ResponseLatency()],
        [TurnDetection("manual"), TurnDetection("server_vad", silence_duration_ms=500)],
    ),
    "latency": (
        [ResponseLatency()],
        [
            TurnDetection("manual"),
            TurnDetection("server_vad", silence_duration_ms=200),
            TurnDetection("server_vad", silence_duration_ms=500),
            TurnDetection("semantic_vad"),
        ],
    ),
    "endpointing": (
        [EndpointingLadder(gap_ms=gap) for gap in LADDER_MS],
        [TurnDetection("server_vad", silence_duration_ms=500)],
    ),
    "interaction": (
        [BargeIn(), BackchannelTolerance()],
        [TurnDetection("server_vad", silence_duration_ms=500)],
    ),
    "noise": ([FalseTrigger(duration_ms=20000)], [TurnDetection("server_vad", silence_duration_ms=500)]),
    # Tools run in Lane A, not only in the agent lane. The text arm below is the
    # same probe with the same tool server and only the modality changed, which
    # is what makes a voice failure attributable to the speech pathway.
    "task": ([BookingTask()], [TurnDetection("server_vad", silence_duration_ms=500)]),
    "task-text": ([BookingTask()], [TurnDetection("server_vad", silence_duration_ms=500)]),
}

# Suites that score tool calls need the published contract served to the model.
TOOL_SUITES = {"task": "appointments", "task-text": "appointments"}
TEXT_SUITES = {"task-text"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai-realtime", choices=sorted(PROVIDERS))
    parser.add_argument("--suite", default="smoke", choices=sorted(SUITES))
    parser.add_argument("--model")
    parser.add_argument("--voice", action="append", help="caller voice label; repeatable")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--env", help="dotenv file holding the provider credential")
    parser.add_argument("--out", default="data/lane-a")
    args = parser.parse_args()

    entry = PROVIDERS[args.provider]
    key = os.environ.get(entry.credential_env)
    if not key and args.env:
        from dotenv import dotenv_values

        key = dotenv_values(args.env).get(entry.credential_env)
    if not key:
        print(f"{entry.credential_env} is not set", file=sys.stderr)
        return 2

    corpus = build()
    probes, configs = SUITES[args.suite]
    suite_tools = TOOL_SUITES.get(args.suite)
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
        out_root=args.out,
        label=args.suite,
    )

    def show(cell) -> None:
        for key in ("latency_ms", "per_minute", "stop_ms", "interrupted_in_gap", "continued"):
            if key in cell.values:
                detail = f"{key}={cell.values[key]}"
                break
        else:
            detail = ""
        print(
            f"  {cell.variant:34} {cell.config:18} {cell.voice:5} r{cell.repeat}  "
            f"{cell.verdict or '-':5} {cell.void or detail}",
            flush=True,
        )

    runner = Runner(spec, corpus, key)
    print(f"lane A: {args.provider} {runner.model} suite={args.suite} repeats={args.repeats}")
    out = asyncio.run(runner.run(on_cell=show))
    print(f"\nrun -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
