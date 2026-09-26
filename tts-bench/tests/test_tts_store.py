"""The run store: a killed run resumes without repeating finished cells, a run moves and rescores anywhere, and its files are provably intact."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import soundfile as sf

from tts_bench import corpus, store
from tts_bench.adapters.base import AdapterError
from tts_bench.adapters.fake import FakeTTSAdapter
from tts_bench.probes import Cancel, OneShot
from tts_bench.report import load_run, summarize_run
from tts_bench.runner import ResumeRefused, RunSpec, Runner
from tts_bench.score import Instrument, _targets, rescore_run, score_run

ITEMS = [corpus.by_id()["prose.greet"], corpus.by_id()["long.prep"], corpus.by_id()["long.summary"]]


class Killed(BaseException):
    """Stands in for SIGINT or an OOM kill: nothing inside the run may catch it."""


def spec(tmp_path: Path, probes=None) -> RunSpec:
    return RunSpec(provider="fake", probes=probes or [OneShot()], items=ITEMS, repeats=1, sentinel=False,
                   store=str(tmp_path / "store"))


def connect_failing_on(monkeypatch, calls: set[int], exc: BaseException) -> None:
    original = FakeTTSAdapter._connect
    count = {"n": 0}

    async def flaky(self):
        count["n"] += 1
        if count["n"] in calls:
            raise exc
        await original(self)

    monkeypatch.setattr(FakeTTSAdapter, "_connect", flaky)


async def resume(root: Path, **kwargs) -> Runner:
    runner = Runner(RunSpec.from_run(root), "unused", resume=root, **kwargs)
    await runner.run()
    return runner


def progress(root: Path) -> list[dict]:
    return store.read_jsonl(root / "progress.jsonl")


class TestResume:
    async def test_a_killed_run_resumes_and_keeps_the_cells_it_finished(self, tmp_path, monkeypatch):
        connect_failing_on(monkeypatch, {2}, Killed())
        first = Runner(spec(tmp_path), "unused")
        with pytest.raises(Killed):
            await first.run()
        root = first.root
        assert [c["artifacts"]["slug"] for c in store.latest_cells(root)] == ["one_shot/prose.greet/r1"]
        assert (root / store.PARTIAL / "one_shot/long.prep/r1").exists()          # the cell it died in
        assert json.loads((root / "summary.json").read_text())["finished"] is False
        assert progress(root)[-1]["kind"] == "interrupted"

        monkeypatch.undo()
        second = await resume(root)
        assert [c.item for c in second.cells] == ["long.prep", "long.summary"]    # the finished cell is not re-run
        cells = store.latest_cells(root)
        assert len(cells) == 3 and {c["session"] for c in cells} == {1, 2}
        assert not (root / store.PARTIAL).exists()
        assert json.loads((root / "summary.json").read_text())["finished"] is True
        sessions = store.read_jsonl(root / "sessions.jsonl")
        assert [(s["session"], s["kept"], s["to_run"]) for s in sessions] == [(1, 0, 3), (2, 1, 2)]
        assert store.verify(root) == []

    async def test_a_void_is_run_again_and_the_first_attempt_is_kept_aside(self, tmp_path, monkeypatch):
        connect_failing_on(monkeypatch, {1}, AdapterError("connect failed: reset by peer"))
        first = Runner(spec(tmp_path), "unused")
        await first.run()
        assert first.cells[0].void.startswith("provider refused")
        failure = [e for e in progress(first.root) if e["kind"] == "cell_end" and e.get("error_class")]
        assert failure and failure[0]["error_class"] == "AdapterError" and isinstance(failure[0]["raw_tail"], list)

        monkeypatch.undo()
        second = await resume(first.root)
        assert [c.item for c in second.cells] == ["prose.greet"]
        redone = second.cells[0]
        assert redone.attempt == 2 and redone.void is None
        assert (first.root / store.SUPERSEDED / "one_shot/prose.greet/r1/attempt-1/cell.json").exists()
        report = summarize_run(load_run(first.root))
        assert report["counts"]["completed"] == 3 and report["counts"]["voids"] == 0
        rows = store.read_jsonl(first.root / "cells.jsonl")
        assert len(rows) == 4                                                    # the history stays

    async def test_a_declared_exclusion_is_not_retried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FakeTTSAdapter, "supports_cancel", False)
        first = Runner(spec(tmp_path, [Cancel()]), "unused")
        await first.run()
        second = await resume(first.root)
        assert second.cells == []

    async def test_a_different_configuration_is_refused(self, tmp_path):
        first = Runner(spec(tmp_path), "unused")
        await first.run()
        changed = RunSpec.from_run(first.root)
        changed.voice = "another"
        with pytest.raises(ResumeRefused, match="does not match"):
            Runner(changed, "unused", resume=first.root).open_run()

    async def test_different_code_is_refused_unless_allowed_and_then_recorded(self, tmp_path, monkeypatch):
        connect_failing_on(monkeypatch, {3}, Killed())
        first = Runner(spec(tmp_path), "unused")
        with pytest.raises(Killed):
            await first.run()
        monkeypatch.undo()
        import tts_bench.runner as runner_mod

        other = {**first.harness, "commit": "0" * 40}
        monkeypatch.setattr(runner_mod.prov, "harness_state", lambda: other)
        with pytest.raises(ResumeRefused, match="two versions of the code"):
            Runner(RunSpec.from_run(first.root), "unused", resume=first.root).open_run()
        await resume(first.root, allow_harness_change=True)
        last = store.read_jsonl(first.root / "sessions.jsonl")[-1]
        assert last["harness"]["commit"] == "0" * 40 and last["harness_change_allowed"] is True


class _Echo(Instrument):
    """Hears exactly the text that was sent."""

    name = "echo"
    model = "echo-1"

    def __init__(self, texts: dict[str, str]) -> None:
        super().__init__("unused")
        self.texts = texts

    async def transcribe(self, session, wav_path: Path) -> str:
        return self.texts[wav_path.parent.parent.name]


class TestRecords:
    async def test_a_run_moved_elsewhere_still_reports_and_rescores(self, tmp_path):
        first = Runner(spec(tmp_path), "unused")
        await first.run()
        await score_run(first.root, [_Echo({i.id: i.text for i in ITEMS})])
        moved = tmp_path / "elsewhere" / first.root.name
        shutil.copytree(first.root, moved)
        shutil.rmtree(first.root)
        assert summarize_run(load_run(moved))["counts"]["completed"] == 3
        assert {t["wav"].parent for t in _targets(moved)} == {moved / c["artifacts"]["slug"] for c in store.latest_cells(moved)}
        assert rescore_run(moved) == 3
        assert store.verify(moved) == []

    async def test_the_manifest_catches_a_changed_a_missing_and_an_extra_file(self, tmp_path):
        first = Runner(spec(tmp_path), "unused")
        await first.run()
        root = first.root
        assert store.verify(root) == []
        wav = root / "one_shot/prose.greet/r1/audio-main.wav"
        data = bytearray(wav.read_bytes())
        data[-1] ^= 1
        wav.write_bytes(bytes(data))
        (root / "one_shot/long.prep/r1/events.jsonl").unlink()
        (root / "stray.txt").write_text("x")
        assert sorted(store.verify(root)) == ["changed: one_shot/prose.greet/r1/audio-main.wav",
                                              "missing: one_shot/long.prep/r1/events.jsonl",
                                              "not in manifest: stray.txt"]

    async def test_archive_is_sample_exact_and_scoring_reads_the_flac(self, tmp_path):
        first = Runner(spec(tmp_path), "unused")
        await first.run()
        root = first.root
        wavs = sorted(root.rglob("audio-*.wav"))
        before = {w.relative_to(root).with_suffix(".flac").as_posix(): sf.read(w, dtype="int16")[0].tobytes() for w in wavs}
        result = store.archive(root)
        assert result["converted"] == len(wavs) and not list(root.rglob("audio-*.wav"))
        record = json.loads((root / "archive.json").read_text())["files"]
        for rel, pcm in before.items():
            assert sf.read(root / rel, dtype="int16")[0].tobytes() == pcm
            assert record[rel]["pcm_sha256"] == hashlib.sha256(pcm).hexdigest()
        assert store.verify(root) == []
        assert {t["wav"].suffix for t in _targets(root)} == {".flac"}

    async def test_a_line_cut_short_by_a_kill_is_skipped(self, tmp_path):
        first = Runner(spec(tmp_path), "unused")
        await first.run()
        with open(first.root / "cells.jsonl", "a") as handle:
            handle.write('{"artifacts": {"slug": "one_shot/x')
        assert len(store.latest_cells(first.root)) == 3

    async def test_the_store_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "shared"))
        runner = Runner(RunSpec(provider="fake", probes=[OneShot()], items=ITEMS[:1], repeats=1, sentinel=False), "unused")
        await runner.run()
        assert runner.root.parent == tmp_path / "shared"
        assert [s["completed"] for s in store.list_runs(tmp_path / "shared")] == [1]

    async def test_the_logs_cover_every_cell(self, tmp_path):
        first = Runner(spec(tmp_path), "unused")
        await first.run()
        kinds = [e["kind"] for e in progress(first.root)]
        assert kinds == ["run_start"] + ["cell_start", "cell_end"] * 3 + ["run_end"]
        end = progress(first.root)[-1]
        assert end["done"] == 3 and end["errors"] == 0
        assert store.verify(first.root) == []                                    # the manifest covers the final log lines
        assert (first.root / "run.log").read_text().count("cell_end") == 3


def finish_failing_with(monkeypatch, message: str) -> None:
    async def refuse(self, context_id):
        self._on_error(context_id, message)

    monkeypatch.setattr(FakeTTSAdapter, "_finish", refuse)


class TestAccountRefusals:
    async def test_a_balance_or_rate_limit_error_is_a_void_that_a_resume_retries(self, tmp_path, monkeypatch):
        finish_failing_with(monkeypatch, "organization_balance_exhausted: Organization balance exhausted")
        first = Runner(spec(tmp_path), "unused")
        await first.run()
        assert all(c.void.startswith("provider refused: organization_balance_exhausted") and c.verdict is None
                   for c in first.cells)
        monkeypatch.undo()
        second = await resume(first.root)
        assert len(second.cells) == 3 and all(c.verdict == "pass" for c in second.cells)

    async def test_a_service_error_on_an_allowed_request_stays_a_failure(self, tmp_path, monkeypatch):
        finish_failing_with(monkeypatch, "synthesis failed: internal error")
        runner = Runner(spec(tmp_path), "unused")
        await runner.run()
        assert all(c.verdict == "fail" and c.void is None for c in runner.cells)
        assert runner.pending() == []

    def test_the_messages_that_mean_the_account_not_the_voice(self):
        from tts_bench.adapters.base import is_refusal

        for message in ("HTTP 429: Too Many Requests", "HTTP 402: payment required", "HTTP 401: {}", "quota exceeded",
                        "Invalid API key", "rate_limit_exceeded", "insufficient credits"):
            assert is_refusal(message), message
        for message in ("HTTP 500: internal", "invalid_argument: text too long", "receive loop: ConnectionClosed()", None):
            assert not is_refusal(message), message


class TestPlanAndSite:
    def test_cancel_runs_every_long_item_five_times_and_continuation_every_long_item(self, tmp_path):
        from tts_bench.probes import Continuation

        runner = Runner(RunSpec(provider="fake", probes=[Cancel(), Continuation()], repeats=3, sentinel=False,
                                store=str(tmp_path)), "unused")
        counts = {}
        for cell in runner.planned:
            counts[cell.probe.name] = counts.get(cell.probe.name, 0) + 1
            assert cell.item.cohort == "long"
        assert counts == {"cancel": 20, "continuation": 12}

    async def test_each_session_records_where_it_ran_and_a_resume_rebuilds_the_cancel_repeats(self, tmp_path, monkeypatch):
        runner = Runner(spec(tmp_path, probes=[Cancel(repeats=2)]), "unused", site="lab-a")
        await runner.run()
        provenance = json.loads((runner.root / "provenance.json").read_text())
        assert provenance["client"]["site"] == "lab-a"
        again = await resume(runner.root, site="lab-b")
        assert len(again.planned) == len(runner.planned) == 4          # the run's two long items, twice each
        assert [s["client"]["site"] for s in store.read_jsonl(runner.root / "sessions.jsonl")] == ["lab-a", "lab-b"]


class TestScoringJournal:
    async def test_a_scoring_pass_that_dies_keeps_its_transcripts_and_the_next_asks_only_for_the_rest(self, tmp_path):
        runner = Runner(spec(tmp_path), "unused")
        await runner.run()
        texts = {i.id: i.text for i in ITEMS}
        asked: list[str] = []

        class Dies(_Echo):
            async def transcribe(self, session, wav_path):
                asked.append(wav_path.parent.parent.name)
                if len(asked) == 3:
                    raise Killed()
                return await super().transcribe(session, wav_path)

        with pytest.raises(Killed):
            await score_run(runner.root, [Dies(texts)], concurrency=1)
        assert len(store.read_jsonl(runner.root / "transcripts.jsonl")) == 2      # paid for, kept

        asked.clear()
        assert await score_run(runner.root, [_Echo(texts)], concurrency=1) == 3
        rows = store.read_jsonl(runner.root / "scores.jsonl")
        assert len(rows) == 3 and all(r["instruments"]["echo"]["best"]["wer"] == 0 for r in rows)
        assert len(store.read_jsonl(runner.root / "transcripts.jsonl")) == 3      # only the missing one was asked for
