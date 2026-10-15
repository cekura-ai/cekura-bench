#!/usr/bin/env python
"""Transcribe a TTS run's audio offline and score it. See tts_bench/score.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_bench.score import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
