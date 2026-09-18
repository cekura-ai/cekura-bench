#!/usr/bin/env python
"""Render the Lane A caller corpus. Idempotent: existing renders are left alone.

    python bin/render-corpus.py [--overwrite] [--voice f-us]

Needs ELEVENLABS_API_KEY in the environment or a .env beside this repo.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lane_a.corpus_v1 import build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--voice", action="append", help="voice label; repeatable, default all")
    parser.add_argument("--env", help="dotenv file to read ELEVENLABS_API_KEY from")
    args = parser.parse_args()

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key and args.env:
        from dotenv import dotenv_values

        key = dotenv_values(args.env).get("ELEVENLABS_API_KEY")
    if not key:
        print("ELEVENLABS_API_KEY is not set", file=sys.stderr)
        return 2

    corpus = build()
    wanted = args.voice or list(corpus.voices)
    for label in wanted:
        written = corpus.render(key, corpus.voices[label], overwrite=args.overwrite)
        print(f"{label}: {len(written)} rendered, {len(corpus.specs) - len(written)} already present")
    print("manifest ->", corpus.write_manifest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
