"""Track 2 v0.7: speech-cessation scoring, causal schedule, live loop with controls."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from turn_taking_bench.author import service_bridge
from turn_taking_bench.s2s import SYSTEM_PROMPT, SYSTEM_PROMPT_TRACK2, default_config, prompt_sha256
from turn_taking_bench.s2s import authored_speech as S
from turn_taking_bench.s2s import track2

service_bridge._ensure_path()
from service.adapters.fake import FakeAdapter  # noqa: E402
from service.caller import Clip  # noqa: E402

VAD = track2.VAD_MODEL
SPEECH = Path(__file__).resolve().parent.parent / ".cache" / "renders" / "elevenlabs" / "5c4f121dc39011be47febc88.wav"
needs_assets = pytest.mark.skipif(not (VAD.is_file() and SPEECH.is_file()), reason="cached VAD/speech assets absent")


@pytest.fixture(scope="module")
def vad():
    from turn_taking_bench.s2s.authored_vad import SpeechVeto
    return SpeechVeto(VAD)


def _speech():
    x, rate = sf.read(SPEECH, dtype="int16")
    return x, rate


def _score(x, rate, at, vad, cut=None):
    tl = service_bridge.audio().AudioTimeline(rate)
    tl.record(len(x), 0.0)
    if cut is not None:
        tl.cut(cut)
    rec = {"conversation_id": "c", "event_id": "e", "kind": "barge_in", "caller_start_s": at,
           "observation_end_s": at + 1.2, "void": None}
    return S.score_event(rec, SimpleNamespace(agent_pcm=x.astype("<i2").tobytes(), agent_timeline=tl), vad)


def _speaking_point(x, rate, vad):
    for at in np.arange(1.5, len(x) / rate - 1.3, 0.05):
        r = _score(x, rate, float(at), vad)
        y = x.copy(); y[int((at + .77) * rate):] = 0
        c = _score(y, rate, float(at), vad)
        if not r["void"] and not r["stopped"] and not c["void"]:
            return float(at)
    raise AssertionError("no continuous-speech point in fixture")



def test_higgs_runs_with_the_private_sets_turn_detection_in_both_tracks():
    from turn_taking_bench.s2s import PINNED_TURN_DETECTION
    from turn_taking_bench.s2s.model import session_config

    config = default_config("higgs-realtime")
    detection = session_config(config, type("Entry", (), {"default_voice": "default"})()).turn_detection
    assert (detection.silence_duration_ms, detection.threshold, detection.prefix_padding_ms) == (1000, 0.35, 300)
    assert detection.create_response is True and detection.interrupt_response is True
    assert PINNED_TURN_DETECTION.keys() == {"higgs-realtime"}
    other = session_config(default_config("openai-realtime"), type("Entry", (), {"default_voice": "marin"})()).turn_detection
    assert (other.silence_duration_ms, other.threshold, other.prefix_padding_ms) == (None, None, None)

def test_track2_prompt_is_separate_and_neutral():
    assert SYSTEM_PROMPT_TRACK2 != SYSTEM_PROMPT and "Reply briefly" in SYSTEM_PROMPT
    low = SYSTEM_PROMPT_TRACK2.lower()
    for word in ("brief", "interrupt", "backchannel", "turn", "short"):
        assert word not in low
    assert track2.prompt_info()["track1_prompt_sha256"] == prompt_sha256(SYSTEM_PROMPT)


def test_schedule_is_model_independent_and_readmits_end_relative():
    conv = {"items": [
        {"id": "a1", "speaker": "agent", "kind": "turn"},
        {"id": "x1", "speaker": "caller", "kind": "barge_in", "into": "a1", "offset_s": 4.5},
        {"id": "x2", "speaker": "caller", "kind": "barge_in", "into": "a2", "offset_s": 0.0, "offset_from_end_s": 1.2},
        {"id": "x3", "speaker": "caller", "kind": "barge_in", "into": "a3", "offset_s": 0.7},
        *[{"id": f"b{k}", "speaker": "caller", "kind": "backchannel", "into": "a4", "offset_s": 3.0 + 5 * k} for k in range(5)]]}
    sched = track2.schedule(conv)
    assert [sched[k]["nominal_offset_s"] for k in ("x1", "x2", "x3")] == [3.0, 3.0, 1.0]
    assert [sched[f"b{k}"]["nominal_offset_s"] for k in range(5)] == [2.5, 5.0, 7.5, 10.0, None]


@needs_assets
def test_cessation_under_residual_noise_is_a_stop_and_continuation_is_not(vad):
    x, rate = _speech()
    at = _speaking_point(x, rate, vad)
    for db in (None, -55, -35, -25):
        y = x.copy().astype(float)
        cut = int((at + .77) * rate)
        y[cut:] = 0 if db is None else np.random.default_rng(3).normal(0, 32768 * 10 ** (db / 20), len(y) - cut)
        r = _score(np.clip(np.rint(y), -32768, 32767).astype(np.int16), rate, at, vad)
        if db is None:
            assert r["void"] is None and r["stopped"] is True, (db, r["void"])
        # heavy residual noise: a stop or an explicit exclusion, never "kept speaking"
        assert r["void"] is not None or r["stopped"] is True, (db, r)
        if db is None:
            assert abs(r["stop_ms"] - 770) <= 20 and r["latency_retained"]
        elif not r["void"]:
            assert r["latency_retained"] is False  # masked by residual noise
    r = _score(x, rate, at, vad)
    assert r["void"] is None and r["stopped"] is False


@needs_assets
def test_click_after_stop_does_not_delay_speech_cessation(vad):
    x, rate = _speech()
    at = _speaking_point(x, rate, vad)
    y = x.copy(); y[int((at + .237) * rate):] = 0
    c = int((at + .4) * rate); y[c:c + int(.002 * rate)] = 16000
    r = _score(y, rate, at, vad)
    assert r["void"] is None and r["stopped"] and abs(r["stop_ms"] - 237) <= 20


class SpeechFake(FakeAdapter):
    """Fake provider that replies with real recorded speech (so the VAD sees speech)."""

    async def _speak(self):
        x, _ = _speech()
        pcm = x.astype("<i2").tobytes()
        per = int(self.output_rate * self.chunk_ms / 1000) * 2
        for start in range(0, len(pcm), per):
            self._on_agent_audio(pcm[start:start + per])
            await asyncio.sleep(self.chunk_ms / 1000)
        self._on_agent_audio_done()


def _clip(name, s=0.35):
    t = np.arange(int(24000 * s)) / 24000
    noise = np.random.default_rng(len(name)).normal(0, 6000, len(t)) * np.sin(2 * np.pi * 3 * t) ** 2
    pcm = np.concatenate([np.zeros(2400), noise, np.zeros(1200)]).astype("<i2").tobytes()
    return Clip(name, pcm, 24000, name)


@needs_assets
@pytest.mark.parametrize("kind,yields", [("barge_in", True), ("backchannel", False)])
def test_live_loop_scores_events_and_matched_controls(tmp_path, kind, yields, vad):
    items = [{"id": "c1", "speaker": "caller", "kind": "turn", "after": None, "gap_s": 0.1},
             {"id": "a1", "speaker": "agent", "kind": "turn", "trigger_caller_id": "c1"},
             {"id": "c2", "speaker": "caller", "kind": "turn", "after": None, "gap_s": 0.3},
             {"id": "a2", "speaker": "agent", "kind": "turn", "trigger_caller_id": "c2"},
             {"id": "x1", "speaker": "caller", "kind": kind, "into": "a2", "offset_s": 2.2}]
    conv = {"conversation_id": "1", "spec_id": "T", "caller_voice": "f-us", "timing": {"start_s": 0.0},
            "items": items, "void": None}
    conv["schedule"] = track2.schedule(conv)
    clips = {i["id"]: _clip(i["id"]) for i in items if i["speaker"] == "caller"}
    events, ctls = asyncio.run(track2.run_conversation(
        conv, tmp_path, default_config("fake"), tmp_path / "r", vad=vad, adapter_factory=SpeechFake,
        adapter_kwargs={"endpoint_ms": 300, "yields_to_barge_in": yields, "silence_threshold": 200},
        clips=clips, onset_timeout_s=5, quiet_timeout_s=15, timeout_s=60))
    (event,) = events
    assert event["stage"] == "injected"
    if yields:
        assert event["void"] is None, event["void"]
    if event["void"] is None:
        assert event["stopped"] is yields and event["verdict"] == "pass"
    else:  # natural pauses in the recorded reply can make an event unscorable, never wrong
        assert event["void"].startswith("speech_")
    assert abs(event["actual_offset_s"] - 2.0) <= track2.SPEAK_GRACE_S + 0.05
    assert abs(event["injection_lateness_ms"]) <= 25
    scored = [c for c in ctls if not c["void"]]
    # reply a1 (no stimulus) hosts the controls; a2's controls end before x1 or are void
    assert any(c["into"] == "a1" for c in scored)
    for c in ctls:
        if c["into"] == "a2" and not c["void"]:
            assert c["caller_start_s"] + 1.2 <= event["caller_start_s"] + 1e-6
    result = json.loads((tmp_path / "r" / "result.json").read_text())
    assert result["prompt"]["sha256"] == prompt_sha256(SYSTEM_PROMPT_TRACK2)
    summary = track2.summarize(events, ctls)
    assert summary["interruption" if kind == "barge_in" else "backchannel"]["injected"] == 1


# ---- v0.7b: sounds not frame runs, floor outside the VAD region, residual checks ----

class _LagVad:
    """Deterministic stand-in for Silero: a chunk is speech once it and the two
    previous chunks are loud (onset lag), so sounds shorter than ~64 ms never
    enter the stateful region -- the failure mode of short truncated onsets."""

    def __init__(self, on_db=-35.0):
        import threading
        self._lock, self._vad, self.on_db = threading.Lock(), self, on_db
        self.provenance = {"model": "lag-fake"}
        self.hist = []

    def reset(self):
        self.hist = []

    def prob(self, chunk, _t):
        rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float64))))) + 1e-12
        self.hist.append(20 * np.log10(rms) > self.on_db)
        return 1.0 if len(self.hist) >= 3 and all(self.hist[-3:]) else 0.0


RATE = 24000


def _voiced(seconds, db, f0=140.0, seed=0):
    """Harmonic 'voiced' signal (low spectral flatness) at about ``db`` dBFS."""
    t = np.arange(int(round(seconds * RATE))) / RATE
    x = sum(np.sin(2 * np.pi * f0 * k * t + k) / k for k in range(1, 12))
    x = x / np.sqrt(np.mean(x ** 2))
    return x * 32768 * 10 ** (db / 20)


def _pcm(*parts):
    return np.clip(np.rint(np.concatenate(parts)), -32768, 32767).astype(np.int16)


def _z(seconds):
    return np.zeros(int(round(seconds * RATE)))


def _rec_score(x, at, vad, observed=None):
    tl = service_bridge.audio().AudioTimeline(RATE)
    tl.record(len(x), 0.0)
    rec = {"conversation_id": "c", "event_id": "e", "kind": "barge_in", "caller_start_s": at,
           "observation_end_s": observed or at + 1.6, "void": None}
    return S.score_event(rec, SimpleNamespace(agent_pcm=x.astype("<i2").tobytes(), agent_timeline=tl), vad)


def test_groups_join_closure_dips_and_chain_end_follows_the_sound():
    m = np.zeros(60, bool)
    m[0:10] = m[14:20] = m[21:25] = m[40:45] = True  # 4- and 1-frame dips, then a 15-frame gap
    assert S._groups(m, 10) == [(0, 25), (40, 45)]
    assert S._chain_end(m, 5, 10) == 25           # inside the sound: to its end, across dips
    assert S._chain_end(m, 12, 10) == 25          # in a short dip just after it: same sound
    assert S._chain_end(m, 35, 10) == 35          # 10+ frames after the sound: nothing sounding
    assert S._chain_end(m, 60, 10) == 60


def test_floor_ignores_the_vad_onset_pad():
    noise = np.random.default_rng(1).normal(0, 32768 * 10 ** (-55 / 20), int(0.12 * RATE))
    ramp = np.random.default_rng(2).normal(0, 32768 * 10 ** (-45 / 20), int(0.24 * RATE))  # leading ramp, VAD-negative
    x = _pcm(noise, ramp, _voiced(1.5, -20))
    an = S.Analysis(x, RATE, 0.0, _LagVad(), level_until=1.2)
    assert an.floor is not None and abs(an.floor - (-55)) < 2.0  # not pulled up to the ramp
    assert an.threshold <= -48.5


def test_truncated_onset_after_release_extends_the_speech():
    # word ends, 20 ms closure, 60 ms fricative, 5 ms dip, loud 32 ms onset cut at +237 ms
    at = 1.0
    fric = np.random.default_rng(3).normal(0, 32768 * 10 ** (-45 / 20), int(0.06 * RATE))
    x = _pcm(_voiced(at + 0.12, -20), _z(0.02), fric, _z(0.005), _voiced(0.032, -15), _z(1.6))
    r = _rec_score(x, at, _LagVad())
    assert r["void"] is None and r["stopped"] is True
    assert abs(r["stop_ms"] - 237) <= 20, r["stop_ms"]


def test_fragment_after_inserted_pause_is_judged_with_what_follows():
    # 190 ms pause, then a 20 ms fragment, 25 ms dip, speech continues: not a stop
    at = 1.0
    x = _pcm(_voiced(at + 0.237, -20), _z(0.19), _voiced(0.02, -20), _z(0.025), _voiced(1.6, -20))
    r = _rec_score(x, at, _LagVad())
    assert r["void"] is None and r["stopped"] is False, (r["void"], r["stop_ms"])


def test_soft_sound_under_raised_threshold_is_never_a_stop():
    # a modulated -46 dBFS sound the VAD calls non-speech, -55 dBFS noise throughout:
    # the raised threshold hides it, so the 'pause' must not be scored as a stop
    at = 1.0
    t = np.arange(int(0.45 * RATE)) / RATE
    soft = _voiced(0.45, -46) * 10 ** (4 * np.sin(2 * np.pi * 4 * t) / 20)
    clean = np.concatenate([_voiced(at + 0.3, -20), soft, _voiced(1.4, -20)])
    x = _pcm(clean + np.random.default_rng(4).normal(0, 32768 * 10 ** (-55 / 20), len(clean)))
    r = _rec_score(x, at, _LagVad())
    assert r["void"] is not None or r["stopped"] is False, (r["void"], r["stop_ms"])


def test_scoring_voids_cover_every_reason_score_event_returns():
    import inspect
    import re
    reasons = set(re.findall(r'_void\(record, "([a-z0-9_]+)"', inspect.getsource(S.score_event)))
    assert reasons and reasons <= S.SCORING_VOIDS


@needs_assets
def test_rescore_reproduces_live_scores_keeps_live_voids_and_never_writes_the_source(tmp_path, vad):
    items = [{"id": "c1", "speaker": "caller", "kind": "turn", "after": None, "gap_s": 0.1},
             {"id": "a1", "speaker": "agent", "kind": "turn", "trigger_caller_id": "c1"},
             {"id": "c2", "speaker": "caller", "kind": "turn", "after": None, "gap_s": 0.3},
             {"id": "a2", "speaker": "agent", "kind": "turn", "trigger_caller_id": "c2"},
             {"id": "x1", "speaker": "caller", "kind": "barge_in", "into": "a2", "offset_s": 2.2}]
    conv = {"conversation_id": "1", "spec_id": "T", "caller_voice": "f-us", "timing": {"start_s": 0.0},
            "items": items, "void": None}
    conv["schedule"] = track2.schedule(conv)
    clips = {i["id"]: _clip(i["id"]) for i in items if i["speaker"] == "caller"}
    run = tmp_path / "run"
    events, ctls = asyncio.run(track2.run_conversation(
        conv, tmp_path, default_config("fake"), run / "1-r1", vad=vad, adapter_factory=SpeechFake,
        adapter_kwargs={"endpoint_ms": 300, "yields_to_barge_in": True, "silence_threshold": 200},
        clips=clips, onset_timeout_s=5, quiet_timeout_s=15, timeout_s=60))
    info = {"schema": track2.SCHEMA, "status": "running", "provider": "fake", "model": "fake",
            "speech_policy": S.policy(), "conversations": [{"conversation_id": "1", "repeat": 1}]}
    (run / "run.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="not complete"):
        track2.rescore_run(run, tmp_path / "x")
    info["status"] = "complete"
    (run / "run.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="inside the source"):
        track2.rescore_run(run, run / "rescored")
    # a scoring void is re-evaluated; a live void is kept
    path = run / "1-r1" / "result.json"
    saved = json.loads(path.read_text())
    saved["events"][0] = {**saved["events"][0], "void": "agent_not_speaking_at_injection", "stopped": None}
    saved["controls"].append({**saved["controls"][0], "event_id": "control:live", "void": "no_live_agent_reply_anchor"})
    path.write_text(json.dumps(saved))
    before = {p.name: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    result = track2.rescore_run(run, tmp_path / "rescored")
    assert {p.name: p.read_bytes() for p in run.rglob("*") if p.is_file()} == before
    data = json.loads((tmp_path / "rescored" / "t2-events.json").read_text())
    (ev,) = data["events"]
    assert (ev["void"], ev["stopped"], ev["stop_ms"]) == (events[0]["void"], events[0]["stopped"], events[0]["stop_ms"])
    live = [c for c in data["controls"] if c["event_id"] == "control:live"]
    assert live[0]["void"] == "no_live_agent_reply_anchor"
    for old, new in zip(ctls, data["controls"]):
        assert (old["void"], old["stopped"]) == (new["void"], new["stopped"])
    assert result["summary"]["rescore"]["provider_contacted"] is False
    combined = track2.combine([tmp_path / "rescored"], tmp_path / "pooled")
    assert combined["interruption"]["injected"] == 1


def test_pacing_stall_voids_only_overlapping_rows():
    from service.events import Event

    events = [Event(10.10, "caller.pacing.slip", {"chunk": 1, "slip_ms": 100.0}),
              Event(10.12, "caller.pacing.slip", {"chunk": 2, "slip_ms": 80.0}),
              Event(30.0, "caller.pacing.slip", {"chunk": 3, "slip_ms": 20.0})]  # under the void threshold
    stalls = track2.slip_windows(events)
    assert stalls == [(10.0, 10.12)]
    assert track2._stalled({"caller_start_s": 10.5}, stalls)          # stall in the 1 s guard before the stimulus
    assert track2._stalled({"caller_start_s": 9.0}, stalls)           # stall inside the observation window
    assert not track2._stalled({"caller_start_s": 12.0}, stalls)      # far from the stall: kept
    assert not track2._stalled({"caller_start_s": 30.0}, stalls)
    assert not track2._stalled({"caller_start_s": None}, stalls)
    assert track2.slip_windows([{"t": 5.0, "kind": "caller.pacing.slip", "slip_ms": 40.0}]) == [(4.96, 5.0)]
