"""Everything a cell must carry to never need running again.

A benchmark result is only as durable as the record around it. A number whose
exact code, configuration, corpus, detector settings and provider session cannot
be recovered is not a measurement that can be defended later -- and recovering it
by rerunning is not recovery at all, because the provider has moved on. So the
rule here is that a cell directory is self-describing: hand someone that one
directory, with no run root, no repository and no access to us, and they can say
what was measured, under what, with what code, and check the number themselves.

Two details are easy to skip and expensive to skip. A dirty working tree makes a
commit hash a lie, so the uncommitted patch travels with the run. And the
detector constants decide where a boundary lands, so they are stamped too: change
``MARGIN_DB`` and every published latency moves, silently, with an unchanged
commit still attached.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from lane_a import detector

CELL_SCHEMA = "lane-a-cell/1"
RUN_SCHEMA = "lane-a-run/1"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Recorded, not merely imported: these constants place the agent-side boundary,
# so a result is only reproducible alongside the values that produced it.
DETECTOR_SETTINGS = {
    "frame_ms": detector.FRAME_MS,
    "silence_db": detector.SILENCE_DB,
    "onset_frames": detector.ONSET_FRAMES,
    "offset_frames": detector.OFFSET_FRAMES,
    "margin_db": detector.MARGIN_DB,
    "bounds_pad_ms": detector.BOUNDS_PAD_MS,
}

_PACKAGES = ("numpy", "websockets", "python-dotenv")


def _git(*args: str, strip: bool = True) -> str:
    """``strip=False`` for output whose leading column is data.

    ``git status --porcelain`` puts the status flags in the first two columns, so
    stripping the output silently eats the first entry's flags and shifts its
    path by a character.
    """
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout
        return out.strip() if strip else out
    except Exception:  # noqa: BLE001 -- an unversioned checkout is a caveat, not a crash
        return ""


def harness_state() -> dict[str, Any]:
    """Which code ran, including whether the commit alone identifies it."""
    status = _git("status", "--porcelain", strip=False)
    entries = [line[2:].strip() for line in status.splitlines() if len(line) > 2]
    return {
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(entries),
        "dirty_files": sorted(entry for entry in entries if entry),
    }


def harness_patch() -> str:
    """The uncommitted diff, so a dirty run stays reconstructable.

    Tracked changes only: an untracked file shows up in ``dirty_files`` but not
    here, which the reader can see for themselves rather than being told a patch
    is complete when it is not.
    """
    return _git("diff", "HEAD")


def environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            from importlib.metadata import version

            packages[name] = version(name)
        except Exception:  # noqa: BLE001 -- a missing optional package is not a failure
            packages[name] = "absent"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
        "packages": packages,
        "detector": dict(DETECTOR_SETTINGS),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    """Size and checksum of every artifact, so a truncated cell is detectable.

    A half-written wav from a killed run reads as a short one, and a short one
    reads as a fast agent. Publishing the checksum makes that a caught error
    rather than a quiet result.
    """
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "cell.json":
            inventory[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return inventory


def describe(obj: Any) -> dict[str, Any]:
    """A probe's own parameters, whatever they are.

    Read off the dataclass rather than listed by hand, so a new knob on a probe
    cannot be added without appearing in the record. A ladder rung, a noise level
    and a barge-in offset are all the configuration under test; a cell that
    omitted them would not say what question it answered.
    """
    if not is_dataclass(obj):
        return {}
    out: dict[str, Any] = {}
    for spec in fields(obj):
        value = getattr(obj, spec.name, None)
        out[spec.name] = value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
    return out


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def session_snapshot(config: Any) -> dict[str, Any]:
    """The session as requested: instructions, tools, modality, boundary policy.

    The full instruction text is kept, not just its hash. A prompt is part of the
    measured configuration, and a cell that stored only a digest would let two
    runs be told apart without letting either be understood.
    """
    detection = config.turn_detection
    return {
        "instructions": config.instructions,
        "instructions_sha256": text_digest(config.instructions),
        "voice": config.voice,
        "modality": config.modality,
        "transcribe_input": config.transcribe_input,
        "temperature": config.temperature,
        "turn_detection": {
            "label": detection.label,
            "mode": detection.mode,
            "silence_duration_ms": detection.silence_duration_ms,
            "threshold": detection.threshold,
            "prefix_padding_ms": detection.prefix_padding_ms,
        },
        "tools": [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in config.tools
        ],
    }
