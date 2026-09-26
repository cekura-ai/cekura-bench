#!/usr/bin/env python
"""Look after stored TTS runs: list them, check them, archive them, index them.

    python bin/tts-store.py list                 # every run in the store with its state
    python bin/tts-store.py verify <run>...      # recheck MANIFEST.sha256; non-zero exit on any problem
    python bin/tts-store.py archive <run>...     # WAV -> FLAC, sample-exact or not at all
    python bin/tts-store.py manifest <run>...    # rewrite MANIFEST.sha256 (after a deliberate change)
    python bin/tts-store.py index <run>...       # markdown rows for a run index kept outside the store
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_bench import store  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("list", "verify", "archive", "manifest", "index"))
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--store", help=f"store to list (default ${store.STORE_ENV}, else data/runs)")
    args = parser.parse_args(argv)

    if args.command == "list":
        root = Path(args.store) if args.store else store.default_store()
        for s in store.list_runs(root):
            state = "finished" if s["finished"] else "INCOMPLETE"
            flags = " ".join(name for name in ("scored", "archived") if s[name])
            print(f"{s['run_id']:48} {state:10} {s['completed']:>4}/{s['planned']:<4} voids {s['voids']:<3} "
                  f"errors {s['errors']:<3} {s['model']} {flags}")
        return 0
    if not args.runs:
        parser.error(f"{args.command} needs at least one run directory")
    status = 0
    if args.command == "index":
        print(store.INDEX_HEADER)
    for run in map(Path, args.runs):
        if args.command == "verify":
            problems = store.verify(run)
            print(f"{run.name}: {'intact' if not problems else f'{len(problems)} problem(s)'}")
            for problem in problems:
                print(f"  {problem}")
            status = status or (1 if problems else 0)
        elif args.command == "archive":
            before = store.verify(run)
            if before:
                print(f"{run.name}: not archived, the run does not match its manifest: {before[:3]}")
                status = 1
                continue
            result = store.archive(run)
            print(f"{run.name}: {result['converted']} files to FLAC, {result['bytes_saved'] / 1e6:.1f} MB saved, "
                  f"manifest {result['manifest_sha256'][:16]}")
        elif args.command == "manifest":
            print(f"{run.name}: manifest {store.write_manifest(run)[:16]}")
        elif args.command == "index":
            print(store.index_row(run))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
