#!/usr/bin/env python
"""Check the harness against public audio with known answers.

    python bin/validate-harness.py --provider openai-realtime --per-category 10

Runs spoken reasoning questions through the ordinary Lane A runner and grades the
answers by exact match. A score far below what the same model scores elsewhere
means the fault is ours -- a wrong sample rate, a truncated send, a bad resample
-- and it is far cheaper to find that here than in a published ranking.

The run writes the same artifacts as any other, so the validation is itself
recomputable rather than a number in a terminal.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lane_a.adapters.base import TurnDetection  # noqa: E402
from lane_a.audit import audit_run  # noqa: E402
from lane_a.corpus_v1 import build  # noqa: E402
from lane_a.external import (  # noqa: E402
    INSTRUCTIONS,
    SpokenQuestion,
    load_clip,
    metadata,
    sample,
    summarize,
)
from lane_a.registry import PROVIDERS  # noqa: E402
from lane_a.runner import RunSpec, Runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai-realtime", choices=sorted(PROVIDERS))
    parser.add_argument("--model")
    parser.add_argument("--per-category", type=int, default=10, help="questions drawn from each of the four")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--env", help="dotenv file holding the provider credential")
    parser.add_argument("--cache", default="data/external/big-bench-audio")
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

    cache = Path(args.cache)
    questions = sample(metadata(cache), args.per_category, seed=args.seed)
    rate = entry.adapter.input_rate
    probes = [SpokenQuestion(question=q, clip=load_clip(q, cache, rate)) for q in questions]
    print(f"validation: {args.provider} on {len(probes)} questions at {rate} Hz")

    spec = RunSpec(
        provider=args.provider,
        probes=probes,
        configs=[TurnDetection("server_vad", silence_duration_ms=500)],
        voices=["dataset"],          # the audio is the dataset's, not our corpus's
        repeats=1,
        model=args.model,
        instructions=INSTRUCTIONS,
        out_root=args.out,
        label="validation",
    )

    def show(cell) -> None:
        mark = {"pass": "ok  ", "fail": "WRONG"}.get(cell.verdict or "", "void")
        print(
            f"  {mark} {cell.values.get('category','?'):18} "
            f"said={str(cell.values.get('extracted')):8} want={cell.values.get('official_answer')}",
            flush=True,
        )

    runner = Runner(spec, build(), key)
    out = asyncio.run(runner.run(on_cell=show))

    report = summarize(runner.cells)
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{json.dumps(report, indent=2)}")
    print(f"run -> {out}")

    audit = audit_run(out)
    if not audit["ok"]:
        for problem in audit["problems"]:
            print(f"  ! {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
