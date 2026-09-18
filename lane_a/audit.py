"""Check a run directory is complete, before the chance to redo it is gone.

The point of recording everything is to never repeat a campaign. That only holds
if a gap is noticed while repeating is still cheap -- a missing timeline found in
a week, against a provider that has since shipped a new model, is a result that
cannot be recovered at any price. So this runs over a finished directory and says
plainly what is absent, truncated or inconsistent.

It checks four things and nothing else:

* every planned cell exists, and every existing cell was planned;
* every cell carries the files its probe needs, non-empty;
* every artifact still matches the checksum written beside it;
* the published latency still recomputes from the files.

It reads only the run directory, so it is equally usable by a reader who has the
artifacts and does not have us.

    python -m lane_a.audit data/lane-a/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from lane_a.provenance import sha256_file
from lane_a.recompute import recompute_latency

# What must be present is decided by what the cell says happened, not by a fixed
# list. The text arm ships no waveform at all, and a false-trigger cell that
# passes ships no agent audio because silence was the correct outcome -- so the
# check is that the files and the record agree, which is the failure that would
# actually mislead a reader.
ALWAYS = ("events.jsonl", "timelines.json")


def audit_cell(directory: Path) -> list[str]:
    problems: list[str] = []
    record_path = directory / "cell.json"
    if not record_path.exists():
        return [f"{directory}: no cell.json"]
    record = json.loads(record_path.read_text())

    for name in ALWAYS:
        path = directory / name
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"{directory}: {name} missing or empty")

    voided = record.get("result", {}).get("void")
    audio = record.get("audio", {})
    for name, samples in (("caller.wav", audio.get("caller_samples")), ("agent.wav", audio.get("agent_samples"))):
        present = (directory / name).exists() and (directory / name).stat().st_size > 0
        if samples and not present:
            problems.append(f"{directory}: {name} missing, but the record has {samples} samples")
        if present and not samples:
            problems.append(f"{directory}: {name} exists, but the record has no samples")

    for name, expected in record.get("artifacts", {}).items():
        path = directory / name
        if not path.exists():
            problems.append(f"{directory}: {name} was recorded but is gone")
            continue
        if path.stat().st_size != expected.get("bytes"):
            problems.append(f"{directory}: {name} size changed since the run")
        elif sha256_file(path) != expected.get("sha256"):
            problems.append(f"{directory}: {name} checksum does not match the record")

    published = (record.get("result", {}).get("values") or {}).get("latency_ms")
    if published is not None and not voided:
        recomputed = recompute_latency(directory).get("latency_ms")
        if recomputed is None:
            problems.append(f"{directory}: latency {published} ms cannot be recomputed from the files")
        elif abs(recomputed - published) > 0.5:
            problems.append(f"{directory}: published {published} ms, files give {recomputed} ms")
    return problems


def audit_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    problems: list[str] = []

    for name in ("provenance.json", "plan.json", "manifest.json", "cells.jsonl"):
        if not (root / name).exists():
            problems.append(f"{root}: {name} missing")

    planned: list[str] = []
    if (root / "plan.json").exists():
        planned = [cell["cell_id"] for cell in json.loads((root / "plan.json").read_text())["cells"]]

    directories = sorted(path.parent for path in root.rglob("cell.json"))
    found = {str(directory.relative_to(root)) for directory in directories}
    problems += [f"planned but not run: {cell}" for cell in planned if cell not in found]
    if planned:
        problems += [f"present but not planned: {cell}" for cell in sorted(found - set(planned))]

    for directory in directories:
        problems += audit_cell(directory)

    return {
        "run": str(root),
        "planned": len(planned),
        "found": len(found),
        "problems": problems,
        "ok": not problems,
    }


def main(argv: list[str]) -> int:
    worst = 0
    for path in argv:
        report = audit_run(path)
        print(f"{report['run']}: {report['found']}/{report['planned']} cells")
        for problem in report["problems"]:
            print(f"  ! {problem}")
        if not report["ok"]:
            worst = 1
        else:
            print("  complete and self-consistent")
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
