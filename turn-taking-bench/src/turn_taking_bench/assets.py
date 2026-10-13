"""Model assets that are too large for git: pinned, fetched once, checked by hash.

An adapter that needs weights declares them as ``Asset`` specs (a Hugging Face
file at a pinned revision, a plain URL, or a git checkout at a pinned commit)
and ``turn-taking-bench setup <model>`` materialises them into a gitignored directory.
Every file is verified against its declared sha256 after download and again
before a run (``verify``), and ``assets.json`` in that directory records what
was fetched so ``run.json`` can carry the exact model provenance.

The directory is ``<turn-taking-bench>/.assets/<model>/`` unless ``TURN_TAKING_BENCH_ASSETS_DIR``
or the adapter's ``assets_dir`` config says otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

TURN_TAKING_BENCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = TURN_TAKING_BENCH_ROOT / ".assets"
ASSETS_RECORD = "assets.json"


@dataclass(frozen=True)
class HfFile:
    """One file of a Hugging Face repo at a pinned revision."""

    name: str  # file name inside the assets directory
    repo_id: str
    filename: str
    revision: str
    sha256: str
    repo_type: str = "model"


@dataclass(frozen=True)
class UrlFile:
    """A file at a fixed URL (pin the URL to a commit, the hash does the rest)."""

    name: str
    url: str
    sha256: str


@dataclass(frozen=True)
class GitCheckout:
    """A git repository at a pinned commit, fetched shallowly."""

    name: str  # directory name inside the assets directory
    url: str
    commit: str


Asset = HfFile | UrlFile | GitCheckout


def assets_dir(model_name: str, override: str | os.PathLike[str] | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("TURN_TAKING_BENCH_ASSETS_DIR")
    root = Path(env).expanduser().resolve() if env else DEFAULT_ASSETS_ROOT
    return root / model_name


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _checked_file(target: Path, expected_sha256: str) -> dict:
    actual = sha256_of(target)
    if actual != expected_sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"{target.name}: sha256 {actual} does not match the pinned {expected_sha256}; "
            "the download was discarded"
        )
    return {"path": target.name, "sha256": actual, "size": target.stat().st_size}


def _fetch_hf(asset: HfFile, target: Path) -> dict:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        asset.repo_id, filename=asset.filename, revision=asset.revision, repo_type=asset.repo_type
    )
    shutil.copyfile(cached, target)
    return {
        "source": "huggingface",
        "repo_id": asset.repo_id,
        "repo_type": asset.repo_type,
        "filename": asset.filename,
        "revision": asset.revision,
        **_checked_file(target, asset.sha256),
    }


def _fetch_url(asset: UrlFile, target: Path) -> dict:
    with urllib.request.urlopen(asset.url, timeout=60) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    return {"source": "url", "url": asset.url, **_checked_file(target, asset.sha256)}


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _fetch_git(asset: GitCheckout, target: Path) -> dict:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    _git(["init", "-q"], target)
    _git(["remote", "add", "origin", asset.url], target)
    _git(["fetch", "-q", "--depth", "1", "origin", asset.commit], target)
    _git(["checkout", "-q", "--detach", "FETCH_HEAD"], target)
    head = _git(["rev-parse", "HEAD"], target)
    if head != asset.commit:
        raise RuntimeError(f"{asset.name}: checked out {head}, expected {asset.commit}")
    return {"source": "git", "url": asset.url, "path": asset.name, "commit": head}


def load_record(directory: Path) -> dict:
    path = directory / ASSETS_RECORD
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _target(directory: Path, asset: Asset) -> Path:
    return directory / asset.name


def is_present(directory: Path, asset: Asset) -> bool:
    target = _target(directory, asset)
    if isinstance(asset, GitCheckout):
        return (target / ".git").exists()
    return target.is_file()


def verify(directory: Path, assets: list[Asset]) -> list[str]:
    """Problems with the installed assets (empty list = all present and matching).
    Files are re-hashed; a git checkout is checked for its pinned commit."""
    problems: list[str] = []
    for asset in assets:
        target = _target(directory, asset)
        if not is_present(directory, asset):
            problems.append(f"{asset.name}: missing")
            continue
        if isinstance(asset, GitCheckout):
            try:
                head = _git(["rev-parse", "HEAD"], target)
            except (OSError, subprocess.CalledProcessError) as exc:
                problems.append(f"{asset.name}: not a git checkout ({exc})")
                continue
            if head != asset.commit:
                problems.append(f"{asset.name}: at commit {head}, pinned {asset.commit}")
        else:
            actual = sha256_of(target)
            if actual != asset.sha256:
                problems.append(f"{asset.name}: sha256 {actual}, pinned {asset.sha256}")
    return problems


def install(directory: Path, assets: list[Asset], *, force: bool = False, log=print) -> dict:
    """Fetch every asset that is missing (or all with ``force``), verify, and
    write ``assets.json``. Returns the record."""
    directory.mkdir(parents=True, exist_ok=True)
    record = load_record(directory)
    entries = record.get("assets", {})
    for asset in assets:
        target = _target(directory, asset)
        if is_present(directory, asset) and not force and not verify(directory, [asset]):
            log(f"  {asset.name}: present")
            continue
        log(f"  {asset.name}: fetching")
        if isinstance(asset, HfFile):
            entry = _fetch_hf(asset, target)
        elif isinstance(asset, UrlFile):
            entry = _fetch_url(asset, target)
        else:
            entry = _fetch_git(asset, target)
        entry["fetched_at"] = _utc_now()
        entries[asset.name] = entry
        log(f"  {asset.name}: ok ({entry.get('sha256') or entry.get('commit')})")
    record = {"directory": str(directory), "assets": entries, "updated_at": _utc_now()}
    (directory / ASSETS_RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def require(directory: Path, assets: list[Asset], model_name: str) -> None:
    problems = verify(directory, assets)
    if problems:
        raise FileNotFoundError(
            f"{model_name} assets not ready in {directory}: " + "; ".join(problems)
            + f". Run `turn-taking-bench setup {model_name}`."
        )
