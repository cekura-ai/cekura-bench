#!/usr/bin/env python
"""Run the TTS component bench against one provider.

    python bin/run-tts.py --provider elevenlabs --suite latency --repeats 3
    python bin/run-tts.py --provider cartesia --probe cancel --probe streamed_input
    python bin/run-tts.py --provider deepgram --suite full --corpus /path/to/corpus.json

Writes a run directory under data/tts/ holding, per cell, every synthesised
audio file, the per-chunk arrival timeline, the event log and the raw provider
frames, then the cell record. Round-trip transcription is a separate, offline
pass (bin/score-tts.py) so the instrument can be changed without re-running
the provider.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_bench import corpus as corpus_mod  # noqa: E402
from tts_bench.probes import Cancel, Concurrency, Continuation, OneShot, Repeat, StreamedInput  # noqa: E402
from tts_bench.registry import PROVIDERS  # noqa: E402
from tts_bench.report import load_run, render_markdown, summarize_run  # noqa: E402
from tts_bench.runner import RunSpec, Runner  # noqa: E402

PROBES = {
    "one_shot": OneShot,
    "streamed_input": StreamedInput,
    "cancel": Cancel,
    "continuation": Continuation,
    "repeat": Repeat,
    "concurrency": Concurrency,
}

SUITES = {
    "smoke": lambda: [OneShot()],
    "latency": lambda: [OneShot(), Repeat()],
    "streaming": lambda: [StreamedInput(words_per_s=30.0), StreamedInput(words_per_s=10.0), Continuation()],
    "interaction": lambda: [Cancel(after_first_audio_ms=300.0), Cancel(after_first_audio_ms=1000.0)],
    "load": lambda: [Concurrency(streams=8)],
    "full": lambda: [OneShot(), Repeat(), StreamedInput(words_per_s=30.0), Continuation(),
                     Cancel(after_first_audio_ms=300.0), Concurrency(streams=8)],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="fake", choices=sorted(PROVIDERS))
    parser.add_argument("--suite", default="smoke", choices=sorted(SUITES))
    parser.add_argument("--probe", action="append", choices=sorted(PROBES), help="run these probes instead of a suite")
    parser.add_argument("--model")
    parser.add_argument("--voice")
    parser.add_argument("--rate", type=int, default=24000)
    parser.add_argument("--option", action="append", default=[], help="adapter option key=value; repeatable, recorded per cell")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cohort", action="append", help="restrict to these cohorts")
    parser.add_argument("--item", action="append", help="restrict to these item ids")
    parser.add_argument("--corpus", help="a corpus.json in the published shape, instead of the built-in set")
    parser.add_argument("--no-sentinel", action="store_true")
    parser.add_argument("--env", help="dotenv file holding the provider credential")
    parser.add_argument("--out", default="data/tts")
    parser.add_argument("--label")
    args = parser.parse_args()

    entry = PROVIDERS[args.provider]
    key = os.environ.get(entry.credential_env) or "unused"
    if key == "unused" and args.env and args.provider != "fake":
        from dotenv import dotenv_values

        key = dotenv_values(args.env).get(entry.credential_env) or "unused"
    if key == "unused" and args.provider != "fake":
        print(f"{entry.credential_env} is not set", file=sys.stderr)
        return 2

    version, items = (corpus_mod.load(args.corpus) if args.corpus else (corpus_mod.CORPUS_VERSION, list(corpus_mod.ITEMS)))
    if args.cohort:
        items = [i for i in items if i.cohort in set(args.cohort)]
    if args.item:
        items = [i for i in items if i.id in set(args.item)]
    if not items:
        print("no corpus items selected", file=sys.stderr)
        return 2

    probes = [PROBES[name]() for name in args.probe] if args.probe else SUITES[args.suite]()
    spec = RunSpec(
        provider=args.provider, probes=probes, items=items, repeats=args.repeats,
        model=args.model, voice=args.voice, sample_rate=args.rate, sentinel=not args.no_sentinel,
        options=tuple(tuple(opt.split("=", 1)) for opt in args.option),
        corpus_version=version, out_root=args.out,
        label=args.label or ("-".join(args.probe) if args.probe else args.suite),
    )

    def show(cell) -> None:
        for name in ("ttfa_ms", "cancel_to_last_chunk_ms", "duration_delta_ms", "ttfa_max_ms"):
            if name in cell.values:
                detail = f"{name}={cell.values[name]}"
                break
        else:
            detail = ""
        print(f"  {cell.variant:24} {cell.item:22} r{cell.repeat}  {cell.verdict or '-':5} {cell.void or detail}", flush=True)

    runner = Runner(spec, key)
    print(f"tts bench: {args.provider} {runner.config.label} suite={args.label or args.suite} cells={len(runner.planned)}")
    out = asyncio.run(runner.run(on_cell=show))
    print(f"\nrun -> {out}")
    report = summarize_run(load_run(out))
    (out / "report.md").write_text(render_markdown(report))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"report.md written: {report['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
