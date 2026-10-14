"""Dataset manifest: build from a snapshot directory, verify against it, cache
hashes, and refuse to run on any difference."""

import hashlib

import pytest

from turn_taking_bench import manifest
from turn_taking_bench.manifest import ManifestError, build_manifest, find_manifest, load_manifest, verify_snapshot, write_manifest


def fake_snapshot(root):
    snapshot = root / "snapshot"
    (snapshot / "data").mkdir(parents=True)
    (snapshot / "data" / "dev-00000-of-00002.parquet").write_bytes(b"\x00" * 3000 + b"shard zero")
    (snapshot / "data" / "dev-00001-of-00002.parquet").write_bytes(b"shard one")
    (snapshot / "LICENSE").write_text("licence text")
    (snapshot / ".gitattributes").write_text("ignored")  # dotfiles are not part of the manifest
    return snapshot


def build(snapshot):
    return build_manifest(
        name="fake", source="org/fake", revision="abc123", split="dev", snapshot_dir=snapshot,
        conversation_ids=["1", "2"], license={"name": "L"}, gated=True, source_urls={}, generated_by={},
    )


def test_build_lists_every_file_with_size_and_sha256(tmp_path):
    snapshot = fake_snapshot(tmp_path)
    built = build(snapshot)
    assert [f["path"] for f in built["files"]] == ["LICENSE", "data/dev-00000-of-00002.parquet", "data/dev-00001-of-00002.parquet"]
    shard = built["files"][1]
    assert shard["size"] == 3010
    assert shard["sha256"] == hashlib.sha256(b"\x00" * 3000 + b"shard zero").hexdigest()
    assert built["license"]["sha256"] == hashlib.sha256(b"licence text").hexdigest()
    assert built["conversations"] == {"count": 2, "ids": ["1", "2"]}
    assert built["total_bytes"] == 3010 + 9 + 12


def test_verify_passes_then_trusts_cached_hashes_and_fails_on_any_change(tmp_path):
    snapshot = fake_snapshot(tmp_path)
    built = build(snapshot)
    cache = tmp_path / "cache" / "verify.json"

    first = verify_snapshot(built, snapshot, cache_path=cache)
    assert (first.files, first.hashed, first.bytes) == (3, 3, 3031)
    second = verify_snapshot(built, snapshot, cache_path=cache)
    assert second.hashed == 0  # size + mtime unchanged: no re-hash

    # Same size, new content: mtime moves, the file is re-hashed, the mismatch is reported.
    (snapshot / "data" / "dev-00001-of-00002.parquet").write_bytes(b"shard ONE")
    with pytest.raises(ManifestError, match=r"dev-00001-of-00002.parquet: sha256"):
        verify_snapshot(built, snapshot, cache_path=cache)

    (snapshot / "data" / "dev-00001-of-00002.parquet").unlink()
    (snapshot / "LICENSE").write_text("licence text, longer")
    with pytest.raises(ManifestError) as excinfo:
        verify_snapshot(built, snapshot, cache_path=cache)
    assert "dev-00001-of-00002.parquet: missing" in str(excinfo.value)
    assert "LICENSE: size 20, manifest 12" in str(excinfo.value)


def test_find_manifest_matches_dataset_and_rejects_a_moved_pin(tmp_path):
    snapshot = fake_snapshot(tmp_path)
    manifests = tmp_path / "datasets"
    write_manifest(build(snapshot), manifests / "fake" / "manifest.json")
    path, found = find_manifest("org/fake", "abc123", manifests_dir=manifests)
    assert path == manifests / "fake" / "manifest.json" and found["revision"] == "abc123"
    assert find_manifest("org/other", "abc123", manifests_dir=manifests) is None
    with pytest.raises(ManifestError, match="rebuild the manifest"):
        find_manifest("org/fake", "def456", manifests_dir=manifests)
    assert load_manifest(path)["schema_version"] == manifest.SCHEMA_VERSION
