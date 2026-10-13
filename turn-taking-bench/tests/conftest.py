"""Shared fixtures: a subset of the authored specs, reduced to one voice pairing,
so the end-to-end tests build a handful of conversations rather than all 48."""

import json
from pathlib import Path

import pytest

SPECS = Path(__file__).resolve().parent.parent / "datasets" / "authored" / "specs"
SEED_SPECS = ("tb-04-slow-speaker", "nb-29-two-word-acks", "tb-09-barge-in-redirect")


def subset_specs(out: Path, names=SEED_SPECS, pairings: int = 1) -> Path:
    """Copy the named specs into ``out`` keeping only the first ``pairings`` pairings."""
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        raw = json.loads((SPECS / f"{name}.json").read_text(encoding="utf-8"))
        if "pairings" in raw:
            raw["pairings"] = raw["pairings"][:pairings]
        (out / f"{name}.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return out


@pytest.fixture(scope="session")
def seed_specs_dir(tmp_path_factory) -> Path:
    return subset_specs(tmp_path_factory.mktemp("specs"))

