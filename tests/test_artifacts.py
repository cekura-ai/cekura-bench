"""A run must produce files a published number can be recomputed from.

This is the claim the lane rests on: not "trust our harness" but "here is the
audio, here is when each sample became audible, recompute it yourself". A test
that only checked the in-memory result would pass happily while the artifacts
were missing the one thing that makes them measurable, which is exactly what
happened before per-chunk timestamps were written to disk.

Runs against the scripted agent, so it needs no API key and no network.
"""

from __future__ import annotations

import asyncio
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
        sentinel=False,   # the sentinel has its own test; here it would only slow the rest
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
        for key in ("methodology_version", "corpus_version", "model", "configs", "repeats"):
            assert provenance.get(key), f"provenance is missing {key}"
        assert provenance["harness"]["commit"]

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


class TestTheRecordIsComplete:
    """What a cell must carry so the campaign never has to be repeated.

    A provider run cannot be reproduced later -- the model behind the endpoint
    changes -- so anything missing from the record is not merely inconvenient,
    it is unrecoverable. These tests pin down the fields that were learned the
    hard way rather than the whole schema.
    """

    async def test_the_cell_says_what_was_measured_without_the_run_root(self, tmp_path):
        runner, _root = await run_one(tmp_path)
        record = json.loads((Path(runner.cells[0].artifacts["dir"]) / "cell.json").read_text())
        assert record["identity"]["model"] and record["identity"]["probe"]
        # The label alone loses the parameters: two ladders configured differently
        # would publish under the same name.
        assert record["session"]["requested"]["turn_detection"]["silence_duration_ms"] == 300
        assert record["probe_params"]["clip_id"] == "open.book"
        # Change a detector constant and every latency moves; the commit would not.
        assert record["environment"]["detector"]["margin_db"]
        assert record["harness"]["commit"]
        assert record["corpus"]["utterances"][0]["speech_end_sample"] > 0
        assert record["artifacts"]["agent.wav"]["sha256"]

    async def test_a_dirty_tree_is_disclosed_rather_than_implied_clean(self, tmp_path):
        runner, root = await run_one(tmp_path)
        harness = json.loads((root / "provenance.json").read_text())["harness"]
        assert harness["commit"] and harness["branch"]
        if harness["dirty"]:
            assert harness["dirty_files"], "a dirty tree must name what differs"
            # The status flags occupy the first two columns, so a path that lost
            # its first character means the record is naming files that do not
            # exist -- which reads as clean-ish when it is not.
            assert all(Path(name).name for name in harness["dirty_files"])
            assert not any(name.startswith(("ane_", "ests/", "in/")) for name in harness["dirty_files"])
        if (root / "harness.patch").exists():
            assert (root / "harness.patch").read_text().strip(), "an empty patch discloses nothing"

    async def test_results_survive_an_interrupted_campaign(self, tmp_path):
        """The first cells must be on disk before the last one runs."""
        spec = RunSpec(
            provider="fake",
            probes=[ResponseLatency()],
            configs=[TurnDetection("server_vad", silence_duration_ms=300)],
            voices=["f-us"],
            repeats=3,
            sentinel=False,
            out_root=str(tmp_path),
        )
        runner = Runner(spec, corpus_or_skip(), api_key="unused")

        def stop_after_first(_cell):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            await runner.run(on_cell=stop_after_first)

        rows = [json.loads(line) for line in (runner.root / "cells.jsonl").read_text().splitlines() if line.strip()]
        assert len(rows) == 1 and rows[0]["verdict"]
        summary = json.loads((runner.root / "run.json").read_text())
        assert summary["status"] == "interrupted"
        assert summary["completed_cells"] == 1
        assert len(summary["missing_cells"]) == 2, "an abandoned run must say what it never ran"

    async def test_the_audit_catches_a_cell_that_changed_after_the_run(self, tmp_path):
        from lane_a.audit import audit_run

        runner, root = await run_one(tmp_path)
        assert audit_run(root)["ok"], audit_run(root)["problems"]

        agent = Path(runner.cells[0].artifacts["dir"]) / "agent.wav"
        agent.write_bytes(agent.read_bytes()[: agent.stat().st_size // 2])
        report = audit_run(root)
        assert not report["ok"]
        assert any("agent.wav" in problem for problem in report["problems"])


class TestTheRecordStaysARecord:
    def test_a_probe_carrying_audio_does_not_inline_it(self):
        """Describing a value must not become storing it.

        A probe may hold a whole clip. Writing its bytes into the record inflated
        provenance, the plan and every cell by megabytes each -- while telling a
        reader nothing the checksum does not.
        """
        import json
        from dataclasses import dataclass

        from lane_a.provenance import describe

        @dataclass
        class Holder:
            pcm: bytes
            label: str = "x"

        record = describe(Holder(pcm=b"\x00\x01" * 500_000))
        assert len(json.dumps(record)) < 500
        assert record["pcm"]["bytes"] == 1_000_000
        assert record["pcm"]["sha256"], "the audio must still be identified"


class TestSentinelAndTransforms:
    def test_the_sentinel_leads_every_plan(self, tmp_path):
        """Even a campaign cut short after one cell has the day's noise floor on record."""
        spec = RunSpec(
            provider="fake", probes=[ResponseLatency()], configs=[TurnDetection("manual")],
            voices=["f-us"], repeats=2, out_root=str(tmp_path),
        )
        runner = Runner(spec, corpus_or_skip(), api_key="unused")
        planned = runner.planned
        assert [c.sentinel for c in planned[:3]] == [True, True, True]
        assert all(not c.sentinel for c in planned[3:])
        assert planned[0].cell_id.startswith("sentinel_latency-open.book/server_vad-500ms/f-us/clean/r1")

    def test_a_transform_is_its_own_stratum_with_its_own_directory(self, tmp_path):
        spec = RunSpec(
            provider="fake", probes=[ResponseLatency()], configs=[TurnDetection("manual")],
            voices=["f-us"], repeats=1, transforms=("clean", "telephone"), sentinel=False, out_root=str(tmp_path),
        )
        runner = Runner(spec, corpus_or_skip(), api_key="unused")
        ids = [c.cell_id for c in runner.planned]
        assert ids == [
            "response_latency-open.book/manual/f-us/clean/r1",
            "response_latency-open.book/manual/f-us/telephone/r1",
        ]

    def test_an_unknown_transform_is_refused_before_anything_runs(self, tmp_path):
        spec = RunSpec(
            provider="fake", probes=[ResponseLatency()], configs=[TurnDetection("manual")],
            voices=["f-us"], repeats=1, transforms=("clean", "louder"), out_root=str(tmp_path),
        )
        with pytest.raises(KeyError):
            Runner(spec, corpus_or_skip(), api_key="unused")


class TestDeclaredExclusions:
    async def test_an_unsupported_configuration_is_a_void_not_an_error(self, tmp_path, monkeypatch):
        """The gap is published with its reason; nothing is dialled and nothing is a failure."""
        from lane_a.adapters import fake

        monkeypatch.setattr(fake.FakeAdapter, "supports_manual_commit", False)
        spec = RunSpec(
            provider="fake", probes=[ResponseLatency()], configs=[TurnDetection("manual")],
            voices=["f-us"], repeats=1, sentinel=False, out_root=str(tmp_path),
        )
        runner = Runner(spec, corpus_or_skip(), api_key="unused")
        await runner.run()
        cell = runner.cells[0]
        assert cell.void == "configuration not supported: fake has no manual commit"
        assert cell.error is None
        record = json.loads((Path(cell.artifacts["dir"]) / "cell.json").read_text())
        assert record["error"] is None


class TestASessionTheProviderCloses:
    async def test_a_dead_stream_voids_the_cell_instead_of_hanging(self, tmp_path, monkeypatch):
        """The carrier dies mid-utterance; the probe must not wait on a segment that will never finish."""
        from lane_a.adapters import fake

        sent = {"n": 0}
        original = fake.FakeAdapter.send_audio

        async def dying(self, pcm, t_send=None):
            sent["n"] += 1
            if sent["n"] > 10:
                self.closed.set()
                raise ConnectionError("socket closed by peer")
            return await original(self, pcm, t_send=t_send)

        monkeypatch.setattr(fake.FakeAdapter, "send_audio", dying)
        spec = RunSpec(
            provider="fake", probes=[ResponseLatency()], configs=[TurnDetection("server_vad", silence_duration_ms=300)],
            voices=["f-us"], repeats=1, sentinel=False, out_root=str(tmp_path),
        )
        runner = Runner(spec, corpus_or_skip(), api_key="unused")
        await asyncio.wait_for(runner.run(), timeout=20)
        cell = runner.cells[0]
        assert cell.void and cell.void.startswith("provider closed the session"), cell.void
        assert cell.error and "SessionClosed" in cell.error
        assert (Path(cell.artifacts["dir"]) / "cell.json").exists()
