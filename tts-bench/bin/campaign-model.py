#!/usr/bin/env python
"""One model's campaign run, end to end, safe to start again at any point.

    python bin/campaign-model.py --provider cartesia --model sonic-3.6 --label campaign-a --out /tmp/out

Runs the suite (or resumes the run of the same provider, model and label
already in the store), retries account refusals once, transcribes and scores
every recording, writes the report, archives the audio to FLAC, verifies the
manifest and packs the run into ``<out>/<run>.tar.gz`` with a ``.sha256``
beside it. Every step is idempotent: a finished cell is never re-run and a
returned transcript is never requested again, so the whole command can simply
be repeated after an interruption. The last line of output is a JSON result.

The provider and transcriber keys, ``TTS_BENCH_SITE`` and ``TTS_BENCH_STORE``
come from the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tts_bench import store  # noqa: E402
from tts_bench.common import provenance as prov  # noqa: E402

RESULT = "CAMPAIGN_RESULT "


def step(*args: str) -> int:
    print(f"\n$ {' '.join(args)}", flush=True)
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def find_run(root: Path, provider: str, model: str, label: str) -> Path | None:
    suffix = f"-{provider}-{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}-{label}"
    found = [p for p in sorted(root.iterdir()) if p.name.endswith(suffix) and (p / "plan.json").exists()] if root.exists() else []
    if len(found) > 1:
        raise SystemExit(f"more than one run for {provider}/{model}/{label}: {[p.name for p in found]}")
    return found[0] if found else None


def retryable(run_dir: Path) -> int:
    return sum(1 for c in store.latest_cells(run_dir) if c.get("void") and not c["void"].startswith("configuration not supported"))


def instrument_errors(run_dir: Path) -> int:
    path = run_dir / "scores-all.jsonl"
    return sum(1 for row in store.read_jsonl(path) for v in row["instruments"].values() if "error" in v) if path.exists() else -1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--item", action="append", default=[],
                        help="run only these corpus items: an amendment to an earlier campaign, joined at export")
    parser.add_argument("--out", required=True, help="directory the packed run is written to")
    parser.add_argument("--score-concurrency", type=int, default=8)
    parser.add_argument("--skip-scoring", action="store_true", help="pack the run unscored; score it later with bin/score-tts.py")
    parser.add_argument("--allow-dirty", action="store_true", help="for testing the driver itself; a campaign never uses it")
    args = parser.parse_args()

    harness = prov.harness_state()
    if not args.allow_dirty and (harness.get("dirty") or harness.get("commit") in (None, "unknown")):
        print(f"refusing: a campaign runs on a clean, committed tree (harness {harness})", file=sys.stderr)
        return 2
    root = store.default_store()
    root.mkdir(parents=True, exist_ok=True)

    run_dir = find_run(root, args.provider, args.model, args.label)
    if run_dir is None:
        status = step("bin/run-tts.py", "--provider", args.provider, "--model", args.model, "--suite", args.suite,
                      "--repeats", str(args.repeats), "--label", args.label,
                      *(flag for item in args.item for flag in ("--item", item)))
        run_dir = find_run(root, args.provider, args.model, args.label)
    else:
        status = step("bin/run-tts.py", "--resume", str(run_dir)) if not store.run_status(run_dir)["finished"] else 0
    if run_dir is None or status not in (0,):
        print(f"run failed with exit {status}", file=sys.stderr)
        return status or 1
    if retryable(run_dir):                         # a rate limit or a refused request: one more pass
        status = step("bin/run-tts.py", "--resume", str(run_dir))
        if status:
            return status

    for _ in range(0 if args.skip_scoring else 2):  # a transient transcriber error is asked for again, once
        if step("bin/score-tts.py", str(run_dir), "--concurrency", str(args.score_concurrency)):
            return 3
        if instrument_errors(run_dir) == 0:
            break
    if step("-m", "tts_bench.report", str(run_dir)) or step("bin/tts-store.py", "archive", str(run_dir)):
        return 4
    if step("bin/tts-store.py", "verify", str(run_dir)):
        return 5

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    packed = out / f"{run_dir.name}.tar.gz"
    partial = packed.with_suffix(".partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as tar:     # FLAC does not compress further; stay fast
        tar.add(run_dir, arcname=run_dir.name)
    os.replace(partial, packed)
    digest = sha256(packed)
    (out / f"{run_dir.name}.tar.gz.sha256").write_text(f"{digest}  {packed.name}\n")
    s = store.run_status(run_dir)
    print(RESULT + json.dumps({
        "run": run_dir.name, "packed": packed.name, "sha256": digest, "bytes": packed.stat().st_size,
        "planned": s["planned"], "completed": s["completed"], "voids": s["voids"], "errors": s["errors"],
        "retryable_voids": retryable(run_dir), "instrument_errors": instrument_errors(run_dir),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
