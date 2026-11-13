"""reson8_turns against a scripted fake Reson8 websocket (no network, no key)."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest

from turn_taking_bench.adapters import model_names, reson8_turns
from turn_taking_bench.adapters.reson8_turns import Reson8Turns
from turn_taking_bench.runner import stream_speaker


class FakeReson8:
    """No ready message; sends scripted messages after given counts of audio frames."""

    def __init__(self, script: dict[int, list[dict]], close_after: int | None = None) -> None:
        self.script = script
        self.close_after = close_after
        self.urls: list[str] = []
        self.headers: list[dict] = []
        self.frame_bytes: list[int] = []
        self.texts: list[str] = []

    async def connect(self, url, headers, timeout_s):
        self.urls.append(url)
        self.headers.append(headers)
        return FakeSocket(self)


class FakeSocket:
    def __init__(self, server: FakeReson8) -> None:
        self.server = server
        self.queue: asyncio.Queue = asyncio.Queue()
        self.frames = 0
        self.close_code = None

    async def send_bytes(self, data: bytes) -> None:
        self.server.frame_bytes.append(len(data))
        self.frames += 1
        for message in self.server.script.get(self.frames, []):
            await self.queue.put(json.dumps(message))
        if self.frames == self.server.close_after:
            self.close_code = 1011
            await self.queue.put(None)

    async def send_str(self, data: str) -> None:
        self.server.texts.append(data)

    async def close(self) -> None:
        self.close_code = 1000
        await self.queue.put(None)

    async def __aiter__(self):
        while (item := await self.queue.get()) is not None:
            yield item


CONFIG = {"flush_grace_s": 0.05}


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("RESON_API_KEY", "test-key-not-real")


def test_turn_messages_map_to_causal_eot_and_interruption():
    candidate = {"type": "turn_end_candidate", "text": "hello there", "start_ms": 200, "duration_ms": 500}
    fake = FakeReson8({10: [{"type": "turn_start"}], 30: [candidate], 40: [candidate, {"type": "turn_end"}]})
    model = Reson8Turns(CONFIG, connector=fake.connect)
    sample_rate = 48000
    samples = (0.1 * np.sin(np.arange(sample_rate) * 0.05)).astype(np.float32)

    run = stream_speaker(model, samples, sample_rate, realtime=True)  # causality checked by the runner

    assert [e.kind for e in run.events] == ["interruption", "eot"]
    start, end = (e.commit_time_s for e in run.events)
    assert 0.2 - 1e-9 <= start < end <= run.audio_time_s
    assert end >= 0.8 - 1e-9
    assert set(fake.frame_bytes) == {640} and len(fake.frame_bytes) == 50  # 48k -> 16k pcm_s16le
    url = fake.urls[0]
    assert url.startswith("wss://api.global.reson8.dev/v1/speech-to-text/turns?")
    for part in ("model=rumble-2", "encoding=pcm_s16le", "sample_rate=16000", "language=en",
                 "eager_turn_probability=0.5", "final_turn_probability=0.92", "include_timestamps=true"):
        assert part in url
    assert fake.headers[0] == {"Authorization": "ApiKey test-key-not-real"}
    assert fake.texts == []  # no flush_request: end of audio must not force a turn_end
    diag = run.diagnostics
    assert [m["message"]["type"] for m in diag["messages"]] == [
        "turn_start", "turn_end_candidate", "turn_end_candidate", "turn_end"]
    assert diag["reson8"]["sent_sample_rate"] == 16000
    assert "test-key-not-real" not in json.dumps(diag)


def test_thresholds_are_configurable():
    fake = FakeReson8({})
    model = Reson8Turns({**CONFIG, "eager_turn_probability": 0.3, "final_turn_probability": 0.8},
                        connector=fake.connect)
    stream_speaker(model, np.zeros(1600, dtype=np.float32), 16000, realtime=True)
    assert "eager_turn_probability=0.3" in fake.urls[0] and "final_turn_probability=0.8" in fake.urls[0]


def test_server_close_mid_stream_fails_the_channel():
    fake = FakeReson8({}, close_after=3)
    model = Reson8Turns(CONFIG, connector=fake.connect)
    with pytest.raises(RuntimeError, match="close code 1011"):
        stream_speaker(model, np.zeros(16000, dtype=np.float32), 16000, realtime=True)


def test_refuses_without_realtime_pacing():
    fake = FakeReson8({})
    model = Reson8Turns(CONFIG, connector=fake.connect)
    with pytest.raises(RuntimeError, match="reson8_turns requires real-time pacing"):
        stream_speaker(model, np.zeros(3 * 16000, dtype=np.float32), 16000, realtime=False)
    assert len(fake.frame_bytes) < 60


def test_registered_with_provenance():
    assert "reson8_turns" in model_names()
    info = reson8_turns.provenance(reson8_turns.DEFAULT_CONFIG)
    assert info["requires_realtime"] is True
    assert info["request"]["final_turn_probability"] == 0.92


def test_host_pacing_slip_is_recorded_and_voids_past_threshold():
    fake = FakeReson8({})
    model = Reson8Turns(CONFIG, connector=fake.connect)
    samples = np.zeros(16000, dtype=np.float32)
    run = stream_speaker(model, samples, 16000, realtime=True)
    pacing = run.diagnostics["pacing"]
    assert pacing["void_threshold_ms"] == 25.0 and pacing["pacing_void"] is False

    class LateModel(Reson8Turns):
        def push(self, frame, t_end_s):
            if abs(t_end_s - 0.5) < 1e-9:
                time.sleep(0.06)  # our host stalls 60 ms mid-stream
            return super().push(frame, t_end_s)

    late = LateModel(CONFIG, connector=FakeReson8({}).connect)
    run = stream_speaker(late, samples, 16000, realtime=True)
    assert run.diagnostics["pacing"]["max_slip_ms"] > 25.0
    assert run.diagnostics["pacing"]["pacing_void"] is True
