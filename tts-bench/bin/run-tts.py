#!/usr/bin/env python
"""Run the TTS component bench against one provider.

    python bin/run-tts.py --provider elevenlabs --suite latency --repeats 3
    python bin/run-tts.py --provider cartesia --probe cancel --probe streamed_input
    python bin/run-tts.py --provider deepgram --suite full --corpus /path/to/corpus.json
    python bin/run-tts.py --resume <run directory> --env .env

Writes a run directory in the store ($TTS_BENCH_STORE, else data/runs) holding,
per cell, every synthesised audio file, the per-chunk arrival timeline, the
event log and the raw provider frames, then the cell record, plus run.log and
progress.jsonl for the run as a whole. An interrupted run is completed with
--resume, which re-runs only the cells that have no usable record. Round-trip
transcription is a separate, offline pass (bin/score-tts.py) so the instrument
can be changed without re-running the provider.
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
from tts_bench.probes import PROBES, Cancel, Concurrency, Continuation, OneShot, Repeat, StreamedInput  # noqa: E402
from tts_bench.registry import PROVIDERS  # noqa: E402
from tts_bench.report import load_run, render_markdown, summarize_run  # noqa: E402
from tts_bench.runner import ResumeRefused, RunSpec, Runner  # noqa: E402
from tts_bench.store import STORE_ENV, write_manifest  # noqa: E402

SITE_ENV = "TTS_BENCH_SITE"

SUITES = {
    "smoke": lambda: [OneShot()],
    "latency": lambda: [OneShot(), Repeat()],
    "streaming": lambda: [StreamedInput(words_per_s=30.0), StreamedInput(words_per_s=10.0), Continuation()],
    "interaction": lambda: [Cancel(after_first_audio_ms=300.0), Cancel(after_first_audio_ms=1000.0)],
    "load": lambda: [Concurrency(streams=8)],
    "full": lambda: [OneShot(), Repeat(), StreamedInput(words_per_s=30.0), Continuation(),
                     Cancel(after_first_audio_ms=300.0), Concurrency(streams=8)],
}


def credential(provider: str, env_file: str | None) -> str | None:
    if provider == "fake":
        return "unused"
    name = PROVIDERS[provider].credential_env
    key = os.environ.get(name)
    if not key and env_file:
        from dotenv import dotenv_values

        key = dotenv_values(env_file).get(name)
    return key or None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="fake", choices=sorted(PROVIDERS))
    parser.add_argument("--suite", default="smoke", choices=sorted(SUITES))
    parser.add_argument("--probe", action="append", choices=sorted(PROBES), help="run these probes instead of a suite")
    parser.add_argument("--model")
    parser.add_argument("--all-models", action="store_true", help="one run per model in the provider's lineup, in order")
    parser.add_argument("--voice")
    parser.add_argument("--rate", type=int, default=24000)
    parser.add_argument("--option", action="append", default=[], help="adapter option key=value; repeatable, recorded per cell")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cohort", action="append", help="restrict to these cohorts")
    parser.add_argument("--item", action="append", help="restrict to these item ids")
    parser.add_argument("--corpus", help="a corpus.json in the published shape, instead of the built-in set")
    parser.add_argument("--no-sentinel", action="store_true")
    parser.add_argument("--env", help="dotenv file holding the provider credential")
    parser.add_argument("--store", help=f"directory runs are written to (default ${STORE_ENV}, else data/runs)")
    parser.add_argument("--label")
    parser.add_argument("--resume", help="a run directory to complete; its own plan, corpus and configuration are used")
    parser.add_argument("--site", default=os.environ.get(SITE_ENV),
                        help=f"where this client runs, e.g. aws-us-east-1 (default ${SITE_ENV}); required for a real provider")
    parser.add_argument("--allow-harness-change", action="store_true",
                        help="resume although the code differs from the run's first session (recorded in sessions.jsonl)")
    args = parser.parse_args()

    run_dir = Path(args.resume) if args.resume else None
    if run_dir is not None:
        spec = RunSpec.from_run(run_dir)
    else:
        version, items = (corpus_mod.load(args.corpus) if args.corpus else (corpus_mod.CORPUS_VERSION, list(corpus_mod.ITEMS)))
        if args.cohort:
            items = [i for i in items if i.cohort in set(args.cohort)]
        if args.item:
            items = [i for i in items if i.id in set(args.item)]
        if not items:
            print("no corpus items selected", file=sys.stderr)
            return 2
        models = [m.model for m in PROVIDERS[args.provider].models] if args.all_models else [args.model]
        specs = [RunSpec(
            provider=args.provider, probes=[PROBES[name]() for name in args.probe] if args.probe else SUITES[args.suite](),
            items=items, repeats=args.repeats,
            model=model, voice=args.voice, sample_rate=args.rate, sentinel=not args.no_sentinel,
            options=tuple(tuple(opt.split("=", 1)) for opt in args.option),
            corpus_version=version, store=args.store,
            label=args.label or ("-".join(args.probe) if args.probe else args.suite),
        ) for model in models]
        status = 0
        for spec in specs:
            status = run_one(spec, None, args) or status
            if status == 130:                    # interrupted: stop, do not start the next model
                break
        return status
    return run_one(spec, run_dir, args)


def run_one(spec: RunSpec, run_dir: Path | None, args: argparse.Namespace) -> int:
    if spec.provider != "fake" and not args.site:
        print(f"--site (or ${SITE_ENV}) is required: every latency includes the network from here to the provider", file=sys.stderr)
        return 2
    key = credential(spec.provider, args.env)
    if key is None:
        print(f"{PROVIDERS[spec.provider].credential_env} is not set", file=sys.stderr)
        return 2

    runner = Runner(spec, key, resume=run_dir, allow_harness_change=args.allow_harness_change, echo=True, site=args.site)
    print(f"tts bench: {spec.provider} {runner.config.label} cells={len(runner.planned)} -> {runner.root}")
    try:
        out = asyncio.run(runner.run())
    except ResumeRefused as exc:
        print(f"resume refused: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"\ninterrupted; every finished cell is kept. Continue with:\n  bin/run-tts.py --resume {runner.root}"
              f"{' --env ' + args.env if args.env else ''}", file=sys.stderr)
        return 130
    report = summarize_run(load_run(out))
    (out / "report.md").write_text(render_markdown(report))
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_manifest(out)
    print(f"\nrun -> {out}\nreport.md written: {report['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
