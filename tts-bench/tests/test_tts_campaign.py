"""The per-model campaign driver: one command that runs, packs and verifies a model, and is safe to repeat."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def drive(tmp_path: Path) -> tuple[int, dict]:
    env = {**os.environ, "TTS_BENCH_STORE": str(tmp_path / "store")}
    proc = subprocess.run(
        [sys.executable, "bin/campaign-model.py", "--provider", "fake", "--model", "fake-1", "--label", "camp",
         "--suite", "smoke", "--repeats", "1", "--out", str(tmp_path / "out"), "--skip-scoring", "--allow-dirty"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    line = [l for l in proc.stdout.splitlines() if l.startswith("CAMPAIGN_RESULT ")]
    return proc.returncode, (json.loads(line[-1].split(" ", 1)[1]) if line else {"stderr": proc.stderr[-2000:]})


def test_a_model_is_run_packed_and_checksummed_and_a_repeat_runs_nothing_again(tmp_path):
    code, first = drive(tmp_path)
    assert code == 0, first
    packed = tmp_path / "out" / first["packed"]
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == first["sha256"]
    assert (tmp_path / "out" / f"{first['packed']}.sha256").read_text().split()[0] == first["sha256"]
    with tarfile.open(packed) as tar:
        names = tar.getnames()
    assert f"{first['run']}/MANIFEST.sha256" in names and f"{first['run']}/archive.json" in names
    assert first["completed"] == first["planned"] and first["voids"] == 0

    code, second = drive(tmp_path)
    assert code == 0 and second["run"] == first["run"]
    sessions = (tmp_path / "store" / first["run"] / "sessions.jsonl").read_text().splitlines()
    assert len(sessions) == 1                       # the finished run was not opened again
