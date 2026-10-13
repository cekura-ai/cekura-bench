"""cascade_flux against a scripted fake Flux websocket (no network, no key)."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from turn_taking_bench.adapters import cascade_flux, model_names
from turn_taking_bench.adapters.cascade_flux import CascadeFlux, Decimator
from turn_taking_bench.runner import stream_speaker


class FakeFlux:
    """Sends Connected, then TurnInfo events after given counts of audio frames."""

    def __init__(self, script: dict[int, list[dict]]) -> None:
        self.script = script  # frames received -> messages to send
        self.urls: list[str] = []
        self.headers: list[dict] = []
        self.frame_bytes: list[int] = []
        self.texts: list[str] = []

    async def connect(self, url, headers, timeout_s):
        self.urls.append(url)
        self.headers.append(headers)
        return FakeSocket(self)


class FakeSocket:
    def __init__(self, server: FakeFlux) -> None:
        self.server = server
        self.queue: asyncio.Queue = asyncio.Queue()
        self.queue.put_nowait(json.dumps({"type": "Connected", "request_id": "fake"}))
        self.frames = 0

    async def send_bytes(self, data: bytes) -> None:
        self.server.frame_bytes.append(len(data))
        self.frames += 1
        for message in self.server.script.get(self.frames, []):
            await self.queue.put(json.dumps(message))

    async def send_str(self, data: str) -> None:
        self.server.texts.append(data)
        if json.loads(data).get("type") == "CloseStream":
            await self.queue.put(None)

    async def close(self) -> None:
        await self.queue.put(None)

    async def __aiter__(self):
        while (item := await self.queue.get()) is not None:
            yield item


def turn(event: str, index: int = 0) -> dict:
    return {"type": "TurnInfo", "event": event, "turn_index": index, "transcript": "hi"}


CONFIG = {"flush_grace_s": 0.05}


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-not-real")


def test_flux_events_map_to_causal_eot_and_interruption():
    fake = FakeFlux({10: [turn("StartOfTurn")], 20: [turn("Update")], 40: [turn("EndOfTurn")]})
    model = CascadeFlux(CONFIG, connector=fake.connect)
    sample_rate = 48000
    samples = (0.1 * np.sin(np.arange(int(1.0 * sample_rate)) * 0.05)).astype(np.float32)

    run = stream_speaker(model, samples, sample_rate, realtime=True)  # causality checked by the runner

    kinds = [e.kind for e in run.events]
    assert kinds == ["interruption", "eot"]
    start, end = (e.commit_time_s for e in run.events)
    # Committed no earlier than the audio that triggered them and never past the audio heard.
    assert 0.2 - 1e-9 <= start < end <= run.audio_time_s
    assert end >= 0.8 - 1e-9
    # 48 kHz is decimated to 16 kHz: 320 int16 samples per 20 ms frame.
    assert set(fake.frame_bytes) == {640} and len(fake.frame_bytes) == 50
    url = fake.urls[0]
    assert "model=flux-general-en" in url and "sample_rate=16000" in url
    assert "eot_threshold=0.7" in url and "eot_timeout_ms=5000" in url and "eager" not in url
    assert fake.headers[0] == {"Authorization": "Token test-key-not-real"}
    assert json.loads(fake.texts[-1]) == {"type": "CloseStream"}
    diag = run.diagnostics
    assert [m["message"].get("event") for m in diag["messages"][1:]] == ["StartOfTurn", "Update", "EndOfTurn"]
    assert diag["flux"]["sent_sample_rate"] == 16000
    assert "test-key-not-real" not in json.dumps(diag)


def test_refuses_without_realtime_pacing():
    fake = FakeFlux({})
    model = CascadeFlux(CONFIG, connector=fake.connect)
    samples = np.zeros(3 * 16000, dtype=np.float32)
    with pytest.raises(RuntimeError, match="--realtime"):
        stream_speaker(model, samples, 16000, realtime=False)
    assert len(fake.frame_bytes) < 60  # refused within the first ~second of audio


def test_provider_error_fails_the_channel():
    fake = FakeFlux({3: [{"type": "Error", "code": "BAD", "description": "nope"}]})
    model = CascadeFlux(CONFIG, connector=fake.connect)
    with pytest.raises(RuntimeError, match="Flux error: BAD"):
        stream_speaker(model, np.zeros(16000, dtype=np.float32), 16000, realtime=True)


def test_decimator_keeps_passband_and_rejects_alias_band():
    rate, factor = 48000, 3
    t = np.arange(rate) / rate
    decimate = Decimator(factor)
    for tone, expected in ((1000, 1.0), (12000, 0.0)):
        decimate = Decimator(factor)
        signal = np.sin(2 * np.pi * tone * t)
        out = np.concatenate([decimate(signal[i : i + 960]) for i in range(0, rate, 960)])
        assert len(out) == rate // factor
        amplitude = np.sqrt(2) * np.sqrt(np.mean(out[1000:] ** 2))
        assert amplitude == pytest.approx(expected, abs=0.02)


def test_registered_with_provenance():
    assert "cascade_flux" in model_names()
    info = cascade_flux.provenance(cascade_flux.DEFAULT_CONFIG)
    assert info["requires_realtime"] is True
    if info["stt_bench_config"] is not None:
        assert all(info["stt_bench_config"]["matches"].values())
