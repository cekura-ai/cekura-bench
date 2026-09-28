#!/usr/bin/env python
"""Export a TTS campaign for the website.

    python bin/export-website.py summarize <run>... --out <dir>        # one small JSON per run, next to nothing else
    python bin/export-website.py combine <summary.json>... --models config/site-models.json --out tts-benchmark.json

``summarize`` needs only the run's records (it reads no audio), so it can run
wherever the run is stored and only its output has to move.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_bench import website  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("summarize", "combine"))
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", help="combine: display names, order and sourced prices")
    args = parser.parse_args(argv)

    if args.command == "summarize":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for run in args.paths:
            summary = website.summarize(run)
            path = out / f"{summary['run']}.site.json"
            path.write_text(json.dumps(summary, indent=1) + "\n")
            print(f"{path}")
        return 0
    if not args.models:
        print("--models is required for combine", file=sys.stderr)
        return 2
    summaries = [json.loads(Path(p).read_text()) for p in args.paths]
    site = website.combine(summaries, json.loads(Path(args.models).read_text()))
    Path(args.out).write_text(json.dumps(site, indent=1) + "\n")
    print(f"{args.out}: {len(site['results'])} of {len(site['models'])} models measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
