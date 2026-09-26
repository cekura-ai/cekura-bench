"""Where runs live, how they are logged, and how a stored run is proven intact.

A provider call cannot be repeated: the service changes underneath, and paying
for a campaign twice is the cost of losing one. So a run directory is the unit
of record and everything else serves it:

* **Location.** Runs go to one store outside any checkout (``TTS_BENCH_STORE``),
  so deleting a working copy never deletes a measurement. Without the variable
  they fall back to ``data/runs`` inside the subproject.
* **Relative paths.** Every path a run records is relative to its own
  directory, so a run can be copied, archived and rescored anywhere.
* **Cells land whole.** A cell is written under ``.partial/`` and renamed into
  place when complete, and its row is appended to ``cells.jsonl`` only after.
  A killed run leaves complete cells and at most one discarded partial one,
  which is what makes resume safe.
* **Two logs.** ``run.log`` for a person, ``progress.jsonl`` for a program: one
  line per cell start and end with status and error class, the provider frames
  that preceded a failure, and a heartbeat with counts and an ETA.
* **Integrity.** ``MANIFEST.sha256`` (``sha256sum -c`` format) covers every file;
  ``verify`` rechecks it. ``archive`` replaces WAV with FLAC only after the FLAC
  decodes to the identical samples, and keeps the original hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

from tts_bench.common.provenance import SUBPROJECT_ROOT, sha256_file

STORE_ENV = "TTS_BENCH_STORE"
MANIFEST = "MANIFEST.sha256"
PARTIAL = ".partial"
SUPERSEDED = "superseded"
RAW_TAIL = 8          # provider frames kept in the progress log when a cell fails


def default_store() -> Path:
    configured = os.environ.get(STORE_ENV)
    return Path(configured).expanduser() if configured else SUBPROJECT_ROOT / "data" / "runs"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ── records ───────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every complete line. A final line cut short by a kill is skipped, not fatal."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def latest_cells(run_dir: Path) -> list[dict[str, Any]]:
    """One row per cell id, the last attempt winning, in first-seen order.

    A resumed run appends a new row for every cell it re-ran; the earlier row
    stays in ``cells.jsonl`` as history and its files move to ``superseded/``.
    """
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(run_dir / "cells.jsonl"):
        slug = (row.get("artifacts") or {}).get("slug")
        if slug:
            rows[slug] = row
    return list(rows.values())


def cell_dir(run_dir: Path, cell: dict[str, Any]) -> Path:
    """A cell's directory, whatever launch directory the run was started from."""
    return run_dir / cell["artifacts"]["slug"]


def append_line(handle: TextIO, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


# ── logs ──────────────────────────────────────────────────────────────────────

class RunLog:
    """``run.log`` and ``progress.jsonl``, appended across every session of a run."""

    def __init__(self, run_dir: Path, planned: int, echo: bool = False) -> None:
        self.planned = planned
        self.echo = echo
        self._text = open(run_dir / "run.log", "a", encoding="utf-8")
        self._jsonl = open(run_dir / "progress.jsonl", "a", encoding="utf-8")
        self._started = time.monotonic()
        self._ran = 0
        self.done = 0
        self.voids = 0
        self.errors = 0

    def event(self, kind: str, message: str, **data: Any) -> None:
        at = _utc()
        append_line(self._jsonl, {"utc": at, "kind": kind, **data})
        self._text.write(f"{at} {kind:10} {message}\n")
        self._text.flush()
        if self.echo:
            print(message, flush=True)

    def counts(self) -> dict[str, Any]:
        remaining = self.planned - self.done
        rate = self._ran / (time.monotonic() - self._started) if self._ran else None
        return {"done": self.done, "planned": self.planned, "voids": self.voids, "errors": self.errors,
                "eta_s": None if not rate else round(remaining / rate)}

    def cell_finished(self, cell_id: str, status: str, duration_s: float, void: str | None,
                      error: dict[str, Any] | None, raw_tail: list[Any] | None, detail: str = "") -> None:
        self._ran += 1
        self.done += 1
        self.voids += bool(void)
        self.errors += bool(error)
        extra: dict[str, Any] = {"cell_id": cell_id, "status": status, "duration_s": duration_s, **self.counts()}
        if void:
            extra["void"] = void
        if error:
            extra["error_class"] = error["type"]
            extra["error"] = error["message"]
            extra["raw_tail"] = raw_tail or []
        c = self.counts()
        eta = "" if c["eta_s"] is None else f" eta {c['eta_s']}s"
        self.event("cell_end", f"{cell_id} {status}{': ' + void if void else ''}{' ' + detail if detail else ''} "
                   f"[{c['done']}/{c['planned']}, voids {c['voids']}, errors {c['errors']}{eta}]", **extra)

    def heartbeat(self, current: str | None) -> None:
        c = self.counts()
        self.event("heartbeat", f"alive, in {current or '-'} [{c['done']}/{c['planned']}]", current=current, **c)

    def close(self) -> None:
        self._text.close()
        self._jsonl.close()


def raw_tail(path: Path, n: int = RAW_TAIL) -> list[Any]:
    return [row.get("payload") for row in read_jsonl(path)[-n:]]


# ── integrity ─────────────────────────────────────────────────────────────────

def _files(run_dir: Path) -> Iterable[Path]:
    for path in sorted(run_dir.rglob("*")):
        rel = path.relative_to(run_dir)
        if path.is_file() and rel.parts[0] != PARTIAL and path.name != MANIFEST:
            yield path


def write_manifest(run_dir: Path) -> str:
    """Checksum every file; returns the manifest's own hash, which the run index quotes."""
    lines = [f"{sha256_file(path)}  {path.relative_to(run_dir).as_posix()}" for path in _files(run_dir)]
    (run_dir / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(run_dir / MANIFEST)


def verify(run_dir: Path) -> list[str]:
    """Every problem found, empty when the run is intact."""
    manifest = run_dir / MANIFEST
    if not manifest.exists():
        return [f"no {MANIFEST}"]
    problems: list[str] = []
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        path = run_dir / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"changed: {rel}")
    for path in _files(run_dir):
        rel = path.relative_to(run_dir).as_posix()
        if rel not in listed:
            problems.append(f"not in manifest: {rel}")
    return problems


# ── archive ───────────────────────────────────────────────────────────────────

def _pcm_sha256(samples: Any) -> str:
    return hashlib.sha256(samples.tobytes()).hexdigest()


def archive(run_dir: Path) -> dict[str, Any]:
    """WAV to FLAC in place, each one kept only if it decodes to the same samples.

    ``archive.json`` maps every converted file to its original hash and the
    hash of its samples, so the cell records' WAV checksums stay checkable.
    """
    import soundfile as sf

    record_path = run_dir / "archive.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {"files": {}}
    converted = 0
    saved = 0
    for wav in sorted(run_dir.rglob("audio-*.wav")):
        rel = wav.relative_to(run_dir)
        if rel.parts[0] == PARTIAL:
            continue
        samples, rate = sf.read(wav, dtype="int16", always_2d=True)
        flac = wav.with_suffix(".flac")
        sf.write(flac, samples, rate, format="FLAC", subtype="PCM_16")
        back, back_rate = sf.read(flac, dtype="int16", always_2d=True)
        if back_rate != rate or back.shape != samples.shape or _pcm_sha256(back) != _pcm_sha256(samples):
            flac.unlink()
            raise RuntimeError(f"FLAC round trip changed the samples of {rel}; the WAV is kept")
        record["files"][rel.with_suffix(".flac").as_posix()] = {
            "from": rel.as_posix(), "wav_sha256": sha256_file(wav), "wav_bytes": wav.stat().st_size,
            "pcm_sha256": _pcm_sha256(samples), "frames": len(samples), "rate": rate,
        }
        saved += wav.stat().st_size - flac.stat().st_size
        wav.unlink()
        converted += 1
    record["archived_utc"] = _utc()
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest_sha = write_manifest(run_dir)
    return {"converted": converted, "bytes_saved": saved, "manifest_sha256": manifest_sha}


# ── the store ─────────────────────────────────────────────────────────────────

def run_status(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    provenance = json.loads((run_dir / "provenance.json").read_text()) if (run_dir / "provenance.json").exists() else {}
    plan = json.loads((run_dir / "plan.json").read_text()).get("cells", []) if (run_dir / "plan.json").exists() else []
    cells = latest_cells(run_dir)
    return {
        "run_id": run_dir.name,
        "provider": provenance.get("provider"),
        "model": provenance.get("model"),
        "voice": provenance.get("voice"),
        "options": (provenance.get("config") or {}).get("options"),
        "harness_commit": (provenance.get("harness") or {}).get("commit"),
        "harness_dirty": (provenance.get("harness") or {}).get("dirty"),
        "started_utc": provenance.get("started_utc"),
        "planned": len(plan),
        "completed": len(cells),
        "voids": sum(1 for c in cells if c.get("void")),
        "errors": sum(1 for c in cells if c.get("error")),
        "finished": bool(summary.get("finished")) and len(cells) >= len(plan),
        "scored": (run_dir / "scores.jsonl").exists(),
        "archived": (run_dir / "archive.json").exists(),
        "manifest": (run_dir / MANIFEST).exists(),
    }


def list_runs(store: Path) -> list[dict[str, Any]]:
    if not store.exists():
        return []
    return [run_status(path) for path in sorted(store.iterdir()) if (path / "plan.json").exists()]


def index_row(run_dir: Path) -> str:
    """One markdown row for the run index kept outside the store."""
    s = run_status(run_dir)
    manifest = run_dir / MANIFEST
    digest = sha256_file(manifest)[:16] if manifest.exists() else "-"
    options = ",".join(f"{k}={v}" for k, v in (s["options"] or {}).items()) or "-"
    commit = (s["harness_commit"] or "")[:9] + (" dirty" if s["harness_dirty"] else "")
    return (f"| {s['run_id']} | {s['provider']} | {s['model']} | {s['voice']} | {options} | {commit} | "
            f"{s['completed']}/{s['planned']} | {s['voids']} | {s['errors']} | {'yes' if s['scored'] else 'no'} | {digest} |")


INDEX_HEADER = ("| run | provider | model | voice | options | harness | cells | voids | errors | scored | manifest |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|")
