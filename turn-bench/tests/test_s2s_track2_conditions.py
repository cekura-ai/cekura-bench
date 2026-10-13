"""Track 2 caller conditions: same service transforms as Track 1, deterministic, causal."""
import asyncio
import json
import wave

import numpy as np
import pytest

from turn_bench.author import service_bridge
from turn_bench.s2s import default_config
from turn_bench.s2s import track2
from turn_bench.s2s.track2_conditions import Condition

service_bridge._ensure_path()
from service import transforms as T  # noqa: E402
from service.caller import Clip  # noqa: E402

from tests.test_s2s_track2 import SpeechFake, _clip, needs_assets, vad  # noqa: E402,F401

RATE = 24000


def _stream(seconds=3.0, seed=3):
    rng = np.random.default_rng(seed)
    x = np.zeros(int(RATE * seconds))
    seg = x[RATE // 2:int(0.8 * RATE)]
    seg[:] = rng.normal(0, 4000, seg.size)  # a "line" with silence around it
    if seconds >= 2.6:
        x[2 * RATE:int(2.6 * RATE)] = rng.normal(0, 2500, int(0.6 * RATE))
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


def _chunked(cond, pcm, chunk=480):
    """Feed the stream through the stream stage 20 ms at a time, as the carrier does."""
    n = len(pcm) // 2
    return b"".join(cond.process(pcm[p * 2:min(n, p + chunk) * 2], p) for p in range(0, n, chunk))


@pytest.mark.parametrize("name", ["noise-10db", "noise-20db"])
def test_stream_noise_is_the_service_transform_applied_causally(name):
    pcm = _stream()
    n = len(pcm) // 2
    cond = Condition(name)
    level = T._rms(np.frombuffer(pcm, "<i2").astype(np.float64))
    cond.set_noise(level, RATE, period=n)  # whole-stream level and length, as the service sees it
    assert _chunked(cond, pcm) == T.get(name).apply(pcm, RATE)  # byte-identical
    # causal: a chunk's output depends only on its own samples and position
    half = len(pcm) // 2  # bytes: the first n/2 samples
    tail_changed = pcm[:half] + bytes(len(pcm) - half)  # later audio replaced
    a, b = _chunked(cond, pcm), _chunked(cond, tail_changed)
    assert a[:half] == b[:half] and a[half:] != b[half:]


def test_dropouts_are_the_service_transform_at_stream_positions():
    pcm = _stream()
    cond = Condition("dropouts")
    cond.rate = RATE
    assert _chunked(cond, pcm, chunk=480) == T.get("dropouts").apply(pcm, RATE)
    assert _chunked(cond, pcm, chunk=317) == T.get("dropouts").apply(pcm, RATE)  # any chunking


@pytest.mark.parametrize("name", ["telephone", "reverb"])
def test_clip_stage_is_the_service_transform_on_each_whole_clip(name):
    clips = {"a": Clip("a", _stream(1.2, 1), RATE), "b": Clip("b", _stream(0.9, 2), 16000)}
    one = Condition(name).prepare_clips(clips, RATE)
    two = Condition(name).prepare_clips(clips, RATE)
    for k in clips:
        assert one[k].rate == RATE
        assert one[k].pcm == two[k].pcm  # same seed -> byte-identical degraded clip
        assert one[k].pcm == T.get(name).apply(clips[k].at_rate(RATE).pcm, RATE)
        assert one[k].pcm != clips[k].at_rate(RATE).pcm
    assert Condition(name).process(b"\x01\x00" * 480, 0) == b"\x01\x00" * 480  # carrier left as is


def test_noise_level_fixed_per_conversation_and_present_between_lines():
    clips = {"a": Clip("a", _stream(1.2, 1), RATE), "b": Clip("b", _stream(0.9, 2), RATE)}
    c1, c2 = Condition("noise-10db"), Condition("noise-10db")
    c1.prepare_clips(clips, RATE); c2.prepare_clips(clips, RATE)
    assert c1.provenance() == c2.provenance()
    joined = np.concatenate([np.frombuffer(c.pcm, "<i2") for c in clips.values()]).astype(np.float64)
    assert c1.level == pytest.approx(T._rms(joined))
    silence = bytes(2 * 480)
    gap = np.frombuffer(c1.process(silence, 12345), "<i2").astype(np.float64)
    assert gap.std() > 0 and c1.process(silence, 12345) == c2.process(silence, 12345)
    p = c1.provenance()
    assert (p["transform"], p["seed"], p["stage"]) == ("noise-10db", 101, "stream")
    assert p["noise_rms_dbfs"] == pytest.approx(p["level_dbfs"] - 10)
    long = c1.process(bytes(2 * RATE * 20), 0)  # a bed of 20 s is close to the stated level
    got = 20 * np.log10(T._rms(np.frombuffer(long, "<i2").astype(np.float64)) / 32768)
    assert abs(got - p["noise_rms_dbfs"]) < 1.0


def test_clean_is_identity_and_unknown_condition_fails():
    clips = {"a": Clip("a", _stream(1.0), RATE)}
    assert Condition().prepare_clips(clips, RATE)["a"].pcm == clips["a"].pcm
    with pytest.raises(KeyError):
        Condition("clipping-hard")


def _wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


@needs_assets
@pytest.mark.parametrize("name", ["telephone", "noise-10db", "noise-20db", "reverb", "dropouts"])
def test_live_loop_under_condition_records_and_scores(tmp_path, name, vad):
    items = [{"id": "c1", "speaker": "caller", "kind": "turn", "after": None, "gap_s": 0.1},
             {"id": "a1", "speaker": "agent", "kind": "turn", "trigger_caller_id": "c1"},
             {"id": "c2", "speaker": "caller", "kind": "turn", "after": None, "gap_s": 0.3},
             {"id": "a2", "speaker": "agent", "kind": "turn", "trigger_caller_id": "c2"},
             {"id": "x1", "speaker": "caller", "kind": "barge_in", "into": "a2", "offset_s": 2.2}]
    conv = {"conversation_id": "1", "spec_id": "T", "caller_voice": "f-us", "timing": {"start_s": 0.0},
            "items": items, "void": None}
    conv["schedule"] = track2.schedule(conv)
    clips = {i["id"]: _clip(i["id"]) for i in items if i["speaker"] == "caller"}
    # the fake's peak endpointer must not hear the noise bed as speech
    events, ctls = asyncio.run(track2.run_conversation(
        conv, tmp_path, default_config("fake"), tmp_path / "r", vad=vad, adapter_factory=SpeechFake,
        adapter_kwargs={"endpoint_ms": 300, "yields_to_barge_in": True, "silence_threshold": 6000},
        clips=clips, onset_timeout_s=5, quiet_timeout_s=15, timeout_s=60, condition=name))
    (event,) = events
    assert event["stage"] == "injected" and event["condition"] == name
    assert event["void"] is None or event["void"].startswith("speech_"), event["void"]
    assert all(c["condition"] == name for c in ctls) and any(not c["void"] for c in ctls)
    result = json.loads((tmp_path / "r" / "result.json").read_text())
    assert result["condition"]["transform"] == name and result["events"][0]["condition"] == name
    sent = _wav(tmp_path / "r" / "caller.wav")
    first, last = event["caller_first_sample"], event["caller_last_sample"]
    import hashlib
    assert hashlib.sha256(sent[first:last].tobytes()).hexdigest() == event["sent_stream_sha256"]
    head = sent[: int(0.05 * RATE)].astype(np.float64)  # carrier before the first line
    if name.startswith("noise-"):
        assert head.std() > 20  # noise between lines
    else:
        assert not head.any()  # digital carrier silence stays silence
    if name in ("telephone", "reverb"):  # clip stage: the transmitted clip is the service transform's output
        degraded = T.get(name).apply(clips["x1"].pcm, RATE)
        assert event["sent_pcm_sha256"] == hashlib.sha256(degraded).hexdigest() == event["sent_stream_sha256"]
    summary = track2.summarize(events, ctls)
    assert summary["condition"] == name and summary["caller_voices"] == ["f-us"]


def test_combine_refuses_mixed_conditions(tmp_path):
    for name, cond in (("a", None), ("b", {"condition": "noise-10db"})):
        d = tmp_path / name; d.mkdir()
        (d / "t2-events.json").write_text(json.dumps({"events": [], "controls": []}))
        info = {"provider": "fake", "model": "fake"}
        if cond:
            info["condition"] = cond
        (d / "run.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="one caller condition"):
        track2.combine([tmp_path / "a", tmp_path / "b"], tmp_path / "out")
