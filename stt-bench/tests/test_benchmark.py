import asyncio
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest
import soundfile as sf

from stt_bench.data import FRAME_BYTES, load_manifest, prepare, prepare_audio, read_fleurs, sha256, write_json
from stt_bench.deepgram import exchange, query_url, reduce_events
from stt_bench.score import aggregate_wer, entity_errors, score, word_errors
from stt_bench.streaming import EventLog, pacing_metrics, read_events, stream_audio

CONFIG = json.loads(Path("config/deepgram.json").read_text())


def result(text, at, *, final=True, ack=False, start=0, duration=1, metadata=True):
    msg = {"type": "Results", "channel_index": [0, 1], "start": start, "duration": duration,
           "is_final": final, "from_finalize": ack, "channel": {"alternatives": [{"transcript": text}]}}
    if metadata:
        msg["metadata"] = {"model_info": {"version": CONFIG["version"]}, "model_uuid": CONFIG["expected_model_uuid"]}
    return {"kind": "provider_message", "time_seconds": at, "message": msg}


def markers():
    return [{"kind": "speech_end", "time_seconds": 2},
            {"kind": "finalize_requested", "time_seconds": 2.001},
            {"kind": "audio_complete", "time_seconds": 3}]


def test_corpus_wer_is_weighted_by_reference_words():
    short = word_errors("cat", "dog")
    sentence = "the quick brown fox jumps over the lazy sleeping dog"
    long = word_errors(sentence, sentence)
    total = aggregate_wer([short, long])
    assert total["wer"] == 1 / 11
    assert total["wer"] != .5


def test_empty_hypothesis_counts_deletions():
    assert word_errors("the red car", "")["deletions"] == 3


def test_final_segments_and_blank_ack_are_preserved():
    events = markers() + [result("hello", 1, start=0), result("world", 2.1, start=1),
                          result("", 2.2, ack=True, start=2)]
    reduced = reduce_events(events, CONFIG)
    assert reduced["transcript"] == "hello world"
    assert reduced["finalize_latency_ms"] == pytest.approx(200)
    assert reduced["first_partial_after_t0"] is None
    assert reduced["model_verified"]
    assert not reduced["transcript_complete"]  # Ack confirms a flush, not eventual stream completion.


def test_regular_segment_final_does_not_complete_the_clip():
    reduced = reduce_events(markers() + [result("hello", 2.1)], CONFIG)
    assert reduced["finalize_latency_ms"] is None
    assert not reduced["transcript_complete"]


def test_partial_must_be_nonempty_and_after_our_t0():
    reduced = reduce_events(markers() + [result("early", 1.8, final=False), result("", 2.1, final=False),
                                          result("hello", 2.2, final=False), result("hello", 2.3, ack=True)], CONFIG)
    assert reduced["first_partial_after_t0"]["latency_ms"] == pytest.approx(200)
    assert reduced["transcript"] == "hello"


def test_close_flush_is_never_finalize_latency():
    events = markers() + [{"kind": "close_stream_requested", "time_seconds": 7},
                          result("hello", 7.1, ack=True),
                          {"kind": "provider_message", "time_seconds": 7.2, "message": {"type": "Metadata"}}]
    reduced = reduce_events(events, CONFIG)
    assert reduced["transcript_complete"]
    assert reduced["transcript"] == "hello"
    assert reduced["finalize_latency_ms"] is None


def test_duplicate_event_does_not_duplicate_transcript_but_repeated_speech_does():
    first = result("yes", 1, start=0)
    events = markers() + [first, first, result("yes", 2.1, start=1, ack=True)]
    assert reduce_events(events, CONFIG)["transcript"] == "yes yes"


def test_wrong_model_is_not_valid_benchmark_evidence():
    bad = result("hello", 2.1, ack=True)
    bad["message"]["metadata"]["model_info"]["version"] = "other"
    assert not reduce_events(markers() + [bad], CONFIG)["model_verified"]


def test_error_after_final_ack_still_excludes_failed_clip():
    events = markers() + [result("hello", 2.1, ack=True), {"kind": "error", "time_seconds": 4}]
    assert not reduce_events(events, CONFIG)["transcript_complete"]


def test_model_and_endpoint_are_pinned():
    url = query_url(CONFIG)
    assert "endpointing=false" in url and "version=2025-04-17.21547" in url
    bad = copy.deepcopy(CONFIG)
    bad["version"] = "latest"
    with pytest.raises(ValueError):
        query_url(bad)


def test_entity_scoring_uses_raw_reference_and_requires_annotation():
    ref = "Pay $20 to Jane."
    entity = {"type": "amount", "start": 4, "end": 7, "text": "$20"}
    assert entity_errors(ref, "Pay $20 to Jane.", [entity])["errors"] == 0
    assert entity_errors(ref, "Pay $30 to Jane.", [entity])["errors"] == 1
    assert entity_errors(ref, "Pay twenty dollars to Jane.", [entity])["errors"] == 1
    assert entity_errors(ref, ref, None)["status"] == "not_annotated"
    assert entity_errors(ref, ref, [entity])["reference_entities"] == 1


def test_entity_duplicate_mentions_count_independently():
    ref = "20 then 20"
    entities = [{"type": "number", "start": 0, "end": 2, "text": "20"},
                {"type": "number", "start": 8, "end": 10, "text": "20"}]
    assert entity_errors(ref, "20 then 30", entities)["errors"] == 1


def test_tsv_literal_quotes_are_not_csv_escaping(tmp_path):
    source = tmp_path / "test.tsv"
    sf.write(tmp_path / "a.wav", np.zeros(320), 16000)
    source.write_text('7\ta.wav\t"hello there"\thello there\th e l l o\t320\tMALE\n')
    rows = read_fleurs(source, tmp_path)
    assert rows[0]["reference"] == '"hello there"'


def test_all_silent_audio_rejected(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="No speech"):
        prepare_audio(path)


def test_frozen_sample_has_exact_silence_and_hashes():
    manifest_path = Path("datasets/fleurs-en-us-smoke-v1/manifest.json")
    manifest = load_manifest(manifest_path)
    assert len(manifest["clips"]) == 3
    assert len({c["source_id"] for c in manifest["clips"]}) == 3
    for clip in manifest["clips"]:
        pcm, _ = sf.read(manifest_path.parent / clip["audio"], dtype="int16")
        assert not np.any(pcm[-16000:])
        assert clip["total_frames"] - clip["speech_frames"] == 50


def test_selection_and_audio_are_reproducible(tmp_path):
    regenerated = prepare(Path("test.tsv"), Path("test"), tmp_path / "frozen", 3, 42)
    assert regenerated.read_bytes() == Path("datasets/fleurs-en-us-smoke-v1/manifest.json").read_bytes()


def test_tampered_audio_rejected(tmp_path):
    import shutil
    root = Path("datasets/fleurs-en-us-smoke-v1")
    target = tmp_path / "sample"
    shutil.copytree(root, target)
    audio = next((target / "audio").glob("*.wav"))
    with audio.open("ab") as h:
        h.write(b"tamper")
    with pytest.raises(ValueError, match="Frozen audio changed"):
        load_manifest(target / "manifest.json")


def test_real_clock_pacing_does_not_burst_after_stall(tmp_path):
    async def scenario():
        log = EventLog(tmp_path / "pace.jsonl")
        sends, finalized = [], []
        async def send(frame):
            sends.append(time.perf_counter())
            assert len(frame) == FRAME_BYTES
            if len(sends) == 2:
                await asyncio.sleep(.080)
        async def finalize(t0):
            finalized.append((len(sends), t0))
        await stream_audio(b"\0" * FRAME_BYTES * 53, 3, send, finalize, log)
        log.close()
        assert finalized[0][0] == 3
        assert min(np.diff(sends)) >= .018
        evidence = read_events(tmp_path / "pace.jsonl")
        assert not pacing_metrics(evidence)["valid"]  # Deliberate 80 ms stall is rejected.
        assert sum(e.get("phase") == "silence" for e in evidence) == 50
    asyncio.run(scenario())


def test_duplex_streaming_with_early_ack_still_sends_full_silence(tmp_path):
    class FakeSocket:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.audio_count = 0
            self.finalize_at_frame = None
        def __aiter__(self):
            return self
        async def __anext__(self):
            item = await self.queue.get()
            if item is None:
                raise StopAsyncIteration
            return json.dumps(item)
        async def send(self, message):
            if isinstance(message, bytes):
                self.audio_count += 1
            elif json.loads(message)["type"] == "Finalize":
                self.finalize_at_frame = self.audio_count
                await self.queue.put(result("synthetic fixture", 0, ack=True)["message"])
            else:
                await self.queue.put({"type": "Metadata"})
                await self.queue.put(None)
    async def scenario():
        log = EventLog(tmp_path / "duplex.jsonl")
        ws = FakeSocket()
        await exchange(ws, b"\0" * FRAME_BYTES * 53, 3, CONFIG, log)
        log.close()
        assert ws.finalize_at_frame == 3
        assert ws.audio_count == 53
        reduced = reduce_events(read_events(tmp_path / "duplex.jsonl"), CONFIG)
        assert reduced["transcript"] == "synthetic fixture"
        assert reduced["transcript_complete"]
        assert reduced["finalize_latency_ms"] is not None
    asyncio.run(scenario())


@pytest.mark.parametrize("mode,failed,empty,expected_scored", [
    ("live", False, False, 1), ("dry_run", False, False, 0),
    ("live", True, False, 0), ("live", False, True, 1),
])
def test_offline_report_excludes_invalid_runs_and_scores_empty_completed_text(tmp_path, mode, failed, empty, expected_scored):
    root = tmp_path / "run"
    (root / "raw").mkdir(parents=True)
    write_json(root / "manifest.json", {"clips": [{"clip_id": "fixture", "condition": "public_anchor",
               "reference": "hello", "submitted_seconds": 1.02, "entities": None}]})
    write_json(root / "run.json", {"mode": mode, "config": CONFIG,
               "manifest_sha256": sha256(root / "manifest.json")})
    events = [{"kind": "audio_sent", "time_seconds": (i + 1) * .020, "ideal_seconds": (i + 1) * .020,
               "index": i, "bytes": FRAME_BYTES, "phase": "speech" if i == 0 else "silence"} for i in range(51)]
    events += [{"kind": "speech_end", "time_seconds": .020},
               {"kind": "finalize_requested", "time_seconds": .0201},
               result("" if empty else "hello", .120, ack=True),
               {"kind": "audio_complete", "time_seconds": 1.02},
               {"kind": "close_stream_requested", "time_seconds": 1.03},
               {"kind": "provider_message", "time_seconds": 1.1, "message": {"type": "Metadata"}}]
    if failed:
        # The transport can complete while pacing quality fails. Exclude the result.
        events[10]["time_seconds"] += .100
    (root / "raw" / "fixture.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    report = score(root, tmp_path / "report")
    group = report["results"][0]
    assert group["scored_clips"] == expected_scored
    assert group["finalize_latency"]["headline_eligible"] is False
    if expected_scored:
        assert group["wer"] == (1 if empty else 0)
        assert group["finalize_latency"]["p50_ms"] == pytest.approx(100)
    else:
        assert group["wer"] is None
        assert group["finalize_latency"]["p50_ms"] is None
    assert group["entity_error_rate"] is None
    assert all(g["private_minus_public_wer"] is None for g in report["public_private_gap"])
