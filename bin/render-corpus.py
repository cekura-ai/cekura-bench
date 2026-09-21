#!/usr/bin/env python
"""Render the service bench caller corpus. Idempotent: existing renders are left alone.

    python bin/render-corpus.py [--overwrite] [--voice f-us]
    python bin/render-corpus.py --dir /path/to/holdout     # a corpus.json directory

Needs ELEVENLABS_API_KEY in the environment or a .env beside this repo.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import corpus_spec  # noqa: E402
from service.corpus_v1 import build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--voice", action="append", help="voice label; repeatable, default all")
    parser.add_argument("--env", help="dotenv file to read ELEVENLABS_API_KEY from")
    parser.add_argument("--dir", help="render a corpus described by <dir>/corpus.json instead of the public set")
    args = parser.parse_args()

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key and args.env:
        from dotenv import dotenv_values

        key = dotenv_values(args.env).get("ELEVENLABS_API_KEY")
    if not key:
        print("ELEVENLABS_API_KEY is not set", file=sys.stderr)
        return 2

    corpus = corpus_spec.load_corpus(args.dir) if args.dir else build()
    wanted = args.voice or list(corpus.voices)
    for label in wanted:
        written = corpus.render(key, corpus.voices[label], overwrite=args.overwrite)
        print(f"{label}: {len(written)} rendered, {len(corpus.specs) - len(written)} already present")
    print("manifest ->", corpus.write_manifest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
