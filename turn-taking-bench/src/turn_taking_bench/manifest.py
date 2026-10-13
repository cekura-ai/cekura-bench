"""Dataset manifests: the pinned snapshot's identity, committed without its data.

A manifest (``datasets/<name>/manifest.json``) records which Hugging Face
dataset revision a split is run from, the licence that governs it, the
conversation ids, and for every file of that snapshot its repo path, size and
sha256. No audio, transcript, annotation or duration leaves the dataset: the
manifest is identifiers and hashes only, so it can live in git while the data
(Licensed Material under a gated, non-commercial licence) stays in the local
Hugging Face cache.

``turn-taking-bench manifest build`` writes one from the local cache; ``turn-taking-bench
run`` verifies the local snapshot against it before streaming any audio and
records the result in ``run.json``. Hashing ~4 GB once is fine (files are
streamed in 1 MiB chunks); afterwards a file whose size and mtime are
unchanged since it was last hashed is trusted from ``.cache/`` so a run does
not pay for it again.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turn_taking_bench.assets import TURN_TAKING_BENCH_ROOT, sha256_of

SCHEMA_VERSION = 1
MANIFESTS_DIR = TURN_TAKING_BENCH_ROOT / "datasets"
VERIFY_CACHE = TURN_TAKING_BENCH_ROOT / ".cache" / "manifest-verify.json"
HF_DATASET_URL = "https://huggingface.co/datasets/{source}"


class ManifestError(RuntimeError):
    """The local data does not match the committed manifest (or the manifest
    does not match the pinned revision). A run must not proceed."""


@dataclass(frozen=True)
class Verification:
    files: int
    bytes: int
    hashed: int  # files hashed in this call (the rest were trusted from the cache)
    seconds: float


# ---- building -------------------------------------------------------------------


def snapshot_files(snapshot_dir: Path) -> list[Path]:
    """Every regular file under the snapshot (symlinks to cache blobs followed),
    as paths relative to it, sorted."""
    files = [
        path.relative_to(snapshot_dir)
        for path in snapshot_dir.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(snapshot_dir).parts)
    ]
    return sorted(files, key=str)


def build_manifest(
    *,
    name: str,
    source: str,
    revision: str,
    split: str,
    snapshot_dir: Path,
    conversation_ids: list[str],
    license: dict[str, Any],
    gated: bool,
    source_urls: dict[str, str],
    generated_by: dict[str, Any],
) -> dict[str, Any]:
    files = []
    for relative in snapshot_files(snapshot_dir):
        path = snapshot_dir / relative
        files.append({"path": relative.as_posix(), "size": path.stat().st_size, "sha256": sha256_of(path)})
    if not files:
        raise ManifestError(f"no files under {snapshot_dir}")
    by_path = {entry["path"]: entry for entry in files}
    license = dict(license)
    if "LICENSE" in by_path:
        license["sha256"] = by_path["LICENSE"]["sha256"]
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "dataset": source,
        "repo_type": "dataset",
        "revision": revision,
        "split": split,
        "gated": gated,
        "source_urls": source_urls,
        "license": license,
        "conversations": {"count": len(conversation_ids), "ids": list(conversation_ids)},
        "files": files,
        "total_bytes": sum(entry["size"] for entry in files),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": generated_by,
    }


def build_local_manifest(
    *,
    name: str,
    dataset_dir: Path,
    conversation_ids: list[str],
    generated_by: dict[str, Any],
    authored: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A manifest for a dataset that lives in a local directory (an authored
    dataset built by ``turn-taking-bench author build``). Same ``files`` entries as an
    HF manifest, so ``verify_snapshot`` checks it the same way; ``kind`` is
    ``"local"``, there is no revision, and the data itself sits next to it
    (``manifest.json`` lists every file in the directory but itself)."""
    files = []
    for relative in snapshot_files(dataset_dir):
        if relative.as_posix() == "manifest.json":
            continue
        path = dataset_dir / relative
        files.append({"path": relative.as_posix(), "size": path.stat().st_size, "sha256": sha256_of(path)})
    if not files:
        raise ManifestError(f"no files under {dataset_dir}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "local",
        "name": name,
        "dataset": name,
        "repo_type": None,
        "revision": None,
        "split": name,
        "gated": False,
        "source_urls": {},
        "license": {},
        "conversations": {"count": len(conversation_ids), "ids": list(conversation_ids)},
        "files": files,
        "total_bytes": sum(entry["size"] for entry in files),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": generated_by,
        "authored": authored or {},
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# ---- finding ---------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"{path}: schema_version {manifest.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    return manifest


def find_manifest(source: str, revision: str | None, manifests_dir: Path = MANIFESTS_DIR) -> tuple[Path, dict[str, Any]] | None:
    """The committed manifest for this dataset, or None when there is none.
    A manifest for the dataset at another revision is an error: the pin moved
    (turnbench was bumped) and the manifest must be rebuilt."""
    for path in sorted(manifests_dir.glob("*/manifest.json")):
        manifest = load_manifest(path)
        if manifest["dataset"] != source:
            continue
        if manifest["revision"] != revision:
            raise ManifestError(
                f"{path} pins {source} at revision {manifest['revision']}, but the harness is set to "
                f"revision {revision}; rebuild the manifest (`turn-taking-bench manifest build`) or fix the pin"
            )
        return path, manifest
    return None


# ---- verifying -------------------------------------------------------------------


def _load_cache(cache_path: Path) -> dict[str, Any]:
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(f".{os.getpid()}.tmp")  # per process: parallel runs share the cache
    tmp.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    os.replace(tmp, cache_path)


def verify_snapshot(
    manifest: dict[str, Any], snapshot_dir: Path, cache_path: Path = VERIFY_CACHE, log=None
) -> Verification:
    """Check every file the manifest lists against the local snapshot: present,
    same size, same sha256. Raises ``ManifestError`` listing every problem.

    A file is hashed when its real path has no cache entry or its size or
    mtime changed since the cached hash; otherwise the cached hash is used.
    """
    started = time.monotonic()
    cache = _load_cache(cache_path)
    problems: list[str] = []
    hashed = 0
    total = 0
    for entry in manifest["files"]:
        path = snapshot_dir / entry["path"]
        if not path.is_file():
            problems.append(f"{entry['path']}: missing")
            continue
        stat = path.stat()
        total += stat.st_size
        if stat.st_size != entry["size"]:
            problems.append(f"{entry['path']}: size {stat.st_size}, manifest {entry['size']}")
            continue
        key = os.path.realpath(path)
        cached = cache.get(key)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            actual = cached["sha256"]
        else:
            if log:
                log(f"hashing {entry['path']} ({stat.st_size / 1e6:.0f} MB)")
            actual = sha256_of(path)
            hashed += 1
            cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": actual}
        if actual != entry["sha256"]:
            problems.append(f"{entry['path']}: sha256 {actual}, manifest {entry['sha256']}")
    if hashed:
        _save_cache(cache_path, cache)
    if problems:
        raise ManifestError(
            f"local snapshot {snapshot_dir} does not match the manifest for {manifest['dataset']} "
            f"@ {manifest['revision']}:\n  " + "\n  ".join(problems)
            + "\nThe data on disk is not the pinned dataset; refusing to run. Re-download the snapshot "
            "(or, if the pin changed deliberately, rebuild the manifest with `turn-taking-bench manifest build`)."
        )
    return Verification(files=len(manifest["files"]), bytes=total, hashed=hashed, seconds=round(time.monotonic() - started, 3))


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verification_record(path: Path, verification: Verification, root: Path = TURN_TAKING_BENCH_ROOT) -> dict[str, Any]:
    """What ``run.json`` keeps: which manifest (path + its own sha256) was
    verified and how much work that took."""
    try:
        shown = path.resolve().relative_to(root).as_posix()
    except ValueError:
        shown = str(path)
    return {
        "path": shown,
        "sha256": manifest_sha256(path),
        "verified": True,
        "files": verification.files,
        "bytes": verification.bytes,
        "hashed_now": verification.hashed,
        "verify_s": verification.seconds,
    }
