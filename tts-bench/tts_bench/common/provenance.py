"""What a run must carry so that it never needs running again.

A number whose code, configuration and detector settings cannot be recovered is not
a measurement that can be defended later, and recovering it by rerunning is not
recovery: the provider has moved on. A dirty working tree makes a commit hash a lie,
so the uncommitted patch travels with the run. Git state is read for this
subproject only, so unrelated changes elsewhere in the repository do not mark a run
dirty.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tts_bench.common import detector


SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]

# Recorded, not merely imported: a change to these moves the adaptive onset
# published beside every TTFA, with an unchanged commit still attached.
DETECTOR_SETTINGS = {
    "frame_ms": detector.FRAME_MS,
    "silence_db": detector.SILENCE_DB,
    "onset_frames": detector.ONSET_FRAMES,
    "margin_db": detector.MARGIN_DB,
}
_PACKAGES = ("numpy", "websockets", "aiohttp", "python-dotenv")


def _git(*args: str, strip: bool = True) -> str:
    """``strip=False`` for output whose leading column is data.

    ``git status --porcelain`` puts the status flags in the first two columns, so
    stripping the output silently eats the first entry's flags and shifts its
    path by a character.
    """
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=SUBPROJECT_ROOT
        ).stdout
        return out.strip() if strip else out
    except Exception:  # noqa: BLE001 -- an unversioned checkout is a caveat, not a crash
        return ""


def harness_state() -> dict[str, Any]:
    """Which code ran, including whether the commit alone identifies it."""
    status = _git("status", "--porcelain", "--", ".", strip=False)
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
    return _git("diff", "HEAD", "--", ".")


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


def write_json(path: Path, payload: Any, indent: int | None = 2) -> None:
    path.write_text(json.dumps(payload, indent=indent) + "\n", encoding="utf-8")


def client_network(site: str | None, endpoint: str | None, samples: int = 5) -> dict[str, Any]:
    """Where the client measured from, and how far away the provider's front door was.

    Every first-audio number contains one network round trip, so a row is only
    comparable with rows measured from the same place. ``site`` is the
    operator's label for that place (a cloud region, an office); the endpoint's
    resolved addresses and the TCP connect time to it show which of the
    provider's points of presence answered and how far it was. Only the host is
    recorded: some endpoints carry a key in the query string.
    """
    out: dict[str, Any] = {"site": site, "host": None, "addresses": [], "tcp_connect_ms": [], "tcp_connect_median_ms": None}
    if not endpoint:
        return out
    parts = urlsplit(endpoint)
    host, port = parts.hostname, parts.port or (80 if parts.scheme in ("ws", "http") else 443)
    out["host"] = host
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        out["error"] = f"resolve: {exc}"
        return out
    out["addresses"] = sorted({info[4][0] for info in infos})
    for _ in range(samples):
        started = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=5):
                out["tcp_connect_ms"].append(round((time.perf_counter() - started) * 1000, 1))
        except OSError as exc:
            out["error"] = f"connect: {exc}"
            break
    if out["tcp_connect_ms"]:
        out["tcp_connect_median_ms"] = statistics.median(out["tcp_connect_ms"])
    return out
