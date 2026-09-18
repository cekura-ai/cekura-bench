"""A run must produce files a published number can be recomputed from.

This is the claim the lane rests on: not "trust our harness" but "here is the
audio, here is when each sample became audible, recompute it yourself". A test
that only checked the in-memory result would pass happily while the artifacts
were missing the one thing that makes them measurable, which is exactly what
happened before per-chunk timestamps were written to disk.

Runs against the scripted agent, so it needs no API key and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lane_a.adapters.base import TurnDetection
from lane_a.clips import Corpus
from lane_a.corpus_v1 import build
from lane_a.probes import ResponseLatency
from lane_a.recompute import recompute_latency
from lane_a.runner import RunSpec, Runner

pytestmark = pytest.mark.asyncio

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "lane-a"


def corpus_or_skip() -> Corpus:
    corpus = build(root=str(CORPUS_ROOT))
    if not corpus.path_for("open.book", "f-us").exists():
        pytest.skip("caller corpus not rendered; run bin/render-corpus.py")
    return corpus


async def run_one(tmp_path: Path):
    spec = RunSpec(
        provider="fake",
        probes=[ResponseLatency()],
        configs=[TurnDetection("server_vad", silence_duration_ms=300)],
        voices=["f-us"],
        repeats=1,
        out_root=str(tmp_path),
    )
    runner = Runner(spec, corpus_or_skip(), api_key="unused")
    root = await runner.run()
    return runner, root


class TestArtifacts:
    async def test_a_cell_ships_everything_needed_to_recheck_it(self, tmp_path):
        runner, root = await run_one(tmp_path)
        cell = runner.cells[0]
        assert cell.void is None, cell.void
        directory = Path(cell.artifacts["dir"])
        for name in ("agent.wav", "caller.wav", "events.jsonl", "raw.jsonl", "timelines.json"):
            assert (directory / name).exists(), f"missing {name}"

    async def test_the_published_number_is_recomputable_from_the_files(self, tmp_path):
        runner, root = await run_one(tmp_path)
        cell = runner.cells[0]
        published = cell.values["latency_ms"]
        recomputed = recompute_latency(cell.artifacts["dir"])["latency_ms"]
        assert recomputed == pytest.approx(published, abs=0.2), (published, recomputed)

    async def test_provenance_names_what_was_measured(self, tmp_path):
        _runner, root = await run_one(tmp_path)
        provenance = json.loads((root / "provenance.json").read_text())
        for key in ("methodology_version", "harness_commit", "corpus_version", "model", "configs", "repeats"):
            assert provenance.get(key), f"provenance is missing {key}"

    async def test_the_corpus_manifest_travels_with_the_run(self, tmp_path):
        _runner, root = await run_one(tmp_path)
        manifest = json.loads((root / "manifest.json").read_text())
        rendered = [c for c in manifest["clips"] if c["renders"]]
        assert rendered and all(r["sha256"] for c in rendered for r in c["renders"].values())

    async def test_timelines_round_trip_through_disk(self, tmp_path):
        runner, _root = await run_one(tmp_path)
        payload = json.loads((Path(runner.cells[0].artifacts["dir"]) / "timelines.json").read_text())
        from lane_a.audio import AudioTimeline

        agent = AudioTimeline.from_json(payload["agent"])
        assert agent.rate == payload["agent"]["rate"]
        assert agent.n_samples == sum(row[2] for row in payload["agent"]["chunks"])
        assert agent.time_of_sample(0) == pytest.approx(payload["agent"]["chunks"][0][3])
