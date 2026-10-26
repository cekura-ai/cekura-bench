#!/usr/bin/env python
"""Export a TTS campaign for the website.

    python bin/export-website.py summarize <run>... --out <dir>         # one small JSON per run, next to nothing else
    python bin/export-website.py summarize <run>... --amend --out <dir> # one JSON per model; later runs amend earlier
    python bin/export-website.py combine <summary.json>... --models config/site-models.json --out tts-benchmark.json

``summarize`` needs only the run's records (it reads no audio), so it can run
wherever the run is stored and only its output has to move. With ``--amend``
the runs are grouped by provider, model and voice: the earliest is the base,
and each later one re-measured some of its cells (a reworded or an added
sentence), so the newer reading of a cell replaces the older.
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
    parser.add_argument("--amend", action="store_true", help="summarize: later runs of a model amend its earliest run")
    args = parser.parse_args(argv)

    if args.command == "summarize":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        groups: dict[tuple[str, ...], list[str]] = {}
        for run in args.paths:
            p = json.loads((Path(run) / "provenance.json").read_text())
            groups.setdefault((p["provider"], p["model"], p["voice"]) if args.amend else (run,), []).append(run)
        for runs in groups.values():
            summary = website.summarize(runs[0], runs[1:])
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
