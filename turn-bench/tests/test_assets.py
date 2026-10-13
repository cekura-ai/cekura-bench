import hashlib

import pytest

from turn_bench import assets
from turn_bench.assets import UrlFile


def test_asset_verification_reports_missing_and_mismatched_files(tmp_path):
    spec = UrlFile(name="weights.bin", url="https://example.invalid/weights.bin", sha256="0" * 64)
    assert assets.verify(tmp_path, [spec]) == ["weights.bin: missing"]
    (tmp_path / "weights.bin").write_bytes(b"not the pinned bytes")
    [problem] = assets.verify(tmp_path, [spec])
    assert problem.startswith("weights.bin: sha256 " + hashlib.sha256(b"not the pinned bytes").hexdigest())
    with pytest.raises(FileNotFoundError, match="turn-bench setup"):
        assets.require(tmp_path, [spec], "demo")
