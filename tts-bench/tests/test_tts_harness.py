"""The TTS runner against the scripted provider: probes behave, artifacts are complete, exclusions are declared."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tts_bench import corpus
from tts_bench.adapters.fake import FakeTTSAdapter
from tts_bench.probes import Cancel, Concurrency, Continuation, OneShot, Repeat, StreamedInput
from tts_bench.report import load_run, render_markdown, summarize_run
from tts_bench.runner import RunSpec, Runner

ITEMS = [corpus.by_id()["prose.greet"], corpus.by_id()["long.prep"], corpus.by_id()["long.summary"]]


async def run(tmp_path: Path, probes, **kwargs) -> Runner:
    spec = RunSpec(provider="fake", probes=probes, items=ITEMS, repeats=1, sentinel=False, store=str(tmp_path), **kwargs)
    runner = Runner(spec, api_key="unused")
    await runner.run()
    return runner


class TestProbes:
    async def test_one_shot_measures_the_scripted_provider_as_scripted(self, tmp_path):
        runner = await run(tmp_path, [OneShot()])
        cell = next(c for c in runner.cells if c.item == "prose.greet")
        v = cell.values
        assert cell.verdict == "pass"
        assert 110 <= v["roundtrip_ms"] <= 160
        assert 50 <= v["leading_silence_ms"] <= 61
        assert v["ttfa_ms"] == pytest.approx(v["roundtrip_ms"] + v["leading_silence_ms"], abs=0.2)
        assert v["underruns"] == 0 and v["ended_by"] == "provider"

    async def test_streamed_input_can_start_before_the_text_ends(self, tmp_path):
        runner = await run(tmp_path, [StreamedInput(words_per_s=20.0)])
        cell = next(c for c in runner.cells if c.item == "long.prep")
        assert cell.values["started_before_input_done"] is True
        assert cell.values["ttfa_from_input_done_ms"] < 0
        assert cell.values["text_frames"] == cell.values["words"]

    async def test_cancel_stops_the_stream_and_records_what_leaked(self, tmp_path):
        runner = await run(tmp_path, [Cancel(after_first_audio_ms=100.0)])
        cell = runner.cells[0]
        assert cell.item == "long.prep" and cell.verdict == "pass"
        assert cell.values["cancel_to_last_chunk_ms"] < 200
        assert cell.values["audio_ms"] < 2000       # the whole item would be ~18 s
        assert cell.values["cancel_ack_ms"] is not None

    async def test_continuation_and_repeat_compare_two_syntheses(self, tmp_path):
        runner = await run(tmp_path, [Continuation(frame_gap_ms=50.0), Repeat()])
        cont = next(c for c in runner.cells if c.probe == "continuation")
        rep = next(c for c in runner.cells if c.probe == "repeat" and c.item == "prose.greet")
        assert cont.values["frames"] >= 3 and cont.values["duration_delta_ms"] == 0.0
        assert rep.values["identical_pcm"] is True and rep.values["duration_delta_ms"] == 0.0
        assert len(list((runner.root / cont.artifacts["slug"]).glob("audio-*.wav"))) == 2

    async def test_concurrency_opens_a_connection_per_stream(self, tmp_path):
        runner = await run(tmp_path, [Concurrency(streams=3)])
        cell = next(c for c in runner.cells if c.item == "prose.greet")
        assert cell.values["streams_with_audio"] == 3 and len(cell.values["per_stream"]) == 3
        assert len(list((runner.root / cell.artifacts["slug"]).glob("audio-stream*.wav"))) == 3


class TestRecord:
    async def test_a_cell_ships_everything_needed_to_recompute_it(self, tmp_path):
        runner = await run(tmp_path, [OneShot()])
        directory = runner.root / runner.cells[0].artifacts["slug"]
        for name in ("audio-main.wav", "events.jsonl", "raw.jsonl", "syntheses.json", "cell.json"):
            assert (directory / name).exists(), name
        record = json.loads((directory / "cell.json").read_text())
        assert record["text"]["spoken_reference"] and record["identity"]["cohort"] == "prose"
        assert record["capabilities"]["streamed_input"] is True
        synth = json.loads((directory / "syntheses.json").read_text())["syntheses"][0]
        assert synth["timeline"]["chunks"] and synth["t_first_text"] is not None
        for name in ("plan.json", "provenance.json", "cells.jsonl", "summary.json"):
            assert (runner.root / name).exists(), name

    async def test_the_sentinel_leads_the_plan(self, tmp_path):
        spec = RunSpec(provider="fake", probes=[OneShot()], items=ITEMS, repeats=1, store=str(tmp_path))
        runner = Runner(spec, api_key="unused")
        assert [c.sentinel for c in runner.planned[:3]] == [True, True, True]
        assert runner.planned[0].cell_id == "sentinel/prose.greet/r1"

    async def test_an_unsupported_feature_is_an_exclusion_not_a_connection(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FakeTTSAdapter, "supports_cancel", False)
        connected = []
        original = FakeTTSAdapter._connect

        async def spy(self):
            connected.append(1)
            await original(self)

        monkeypatch.setattr(FakeTTSAdapter, "_connect", spy)
        runner = await run(tmp_path, [Cancel()])
        cell = runner.cells[0]
        assert cell.void == "configuration not supported: fake has no cancel"
        assert cell.error is None and not connected

    async def test_report_groups_by_probe_and_cohort(self, tmp_path):
        runner = await run(tmp_path, [OneShot(), Cancel(after_first_audio_ms=100.0)])
        report = summarize_run(load_run(runner.root))
        assert "one_shot/prose" in report["by_cohort"] and "cancel-100ms/long" in report["by_cohort"]
        assert report["by_cohort"]["one_shot/prose"]["ttfa_ms"]["n"] == 1
        assert report["counts"]["errors"] == 0
        text = render_markdown(report)
        assert "| one_shot | prose |" in text and "cancel" in text


class TestCorpus:
    def test_every_item_has_a_spoken_reference_in_a_known_cohort(self):
        for item in corpus.ITEMS:
            assert item.cohort in corpus.COHORTS and item.spoken_reference.strip()
            assert item.spoken_reference == item.spoken_reference.lower()
        assert len({i.id for i in corpus.ITEMS}) == len(corpus.ITEMS)

    def test_round_trip_through_the_file_shape(self, tmp_path):
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps(corpus.as_json()))
        version, items = corpus.load(path)
        assert version == corpus.CORPUS_VERSION and items == list(corpus.ITEMS)
