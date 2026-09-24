"""The TTS numbers are computed from timestamps and PCM alone; check them against constructed inputs."""

from __future__ import annotations

import math

import numpy as np

from tts_bench.common.audio import AudioTimeline, to_pcm

from tts_bench.adapters.base import Synthesis
from tts_bench.metrics import cancel, first_audible_offset_ms, playout, summarize, trailing_silence_ms

RATE = 24000


def tone(ms: float, amplitude: float = 0.3) -> bytes:
    t = np.arange(int(RATE * ms / 1000)) / RATE
    return to_pcm(amplitude * 32767 * np.sin(2 * math.pi * 220 * t))


def silence(ms: float) -> bytes:
    return b"\x00\x00" * int(RATE * ms / 1000)


class TestOnset:
    def test_leading_silence_is_found_to_the_millisecond(self):
        pcm = silence(80) + tone(200)
        offset = first_audible_offset_ms(pcm, RATE)
        assert 70 <= offset <= 81   # the 10 ms window straddles the edge

    def test_dc_offset_is_not_audio(self):
        dc = (np.ones(RATE // 10) * 3000).astype("<i2").tobytes()
        assert first_audible_offset_ms(dc, RATE) is None

    def test_silence_only_is_none_and_trailing_is_symmetric(self):
        assert first_audible_offset_ms(silence(300), RATE) is None
        pcm = tone(100) + silence(150)
        assert 140 <= trailing_silence_ms(pcm, RATE) <= 151


def _synthesis(arrivals: list[tuple[float, bytes]], t0: float = 1.0, cancel_at: float | None = None) -> Synthesis:
    s = Synthesis("c", RATE, AudioTimeline(RATE))
    s.t_first_text = t0
    s.t_input_done = t0
    for t, pcm in arrivals:
        s.timeline.record_pcm(pcm, t)
        s.pcm.extend(pcm)
        s.t_first_chunk = s.t_first_chunk if s.t_first_chunk is not None else t
        s.t_last_chunk = t
    s.chars_sent = 40
    s.t_cancel = cancel_at
    return s


class TestPlayout:
    def test_clock_starts_at_first_audible_sample_not_at_request(self):
        # 500 ms of round trip, 100 ms of leading silence, then 100 ms chunks
        # arriving exactly on time. Anchored at the request this would be an
        # underrun; anchored at the onset it is a clean playout.
        arrivals = [(1.5, silence(100) + tone(100))]
        for i in range(1, 5):
            arrivals.append((1.5 + 0.2 * i, tone(100)))
        s = _synthesis(arrivals)
        values = summarize(s)
        assert values["roundtrip_ms"] == 500.0
        assert 90 <= values["leading_silence_ms"] <= 101
        assert 590 <= values["ttfa_ms"] <= 601
        # Each 100 ms chunk is due 100 ms after the previous; they arrive 200 ms
        # apart. The first is exactly on time, the other three are late.
        assert values["underruns"] == 3
        assert values["min_margin_ms"] < 0
        assert values["stall_ms"] > 0

    def test_fast_delivery_has_positive_margin_and_no_stall(self):
        arrivals = [(1.5, tone(100))] + [(1.5 + 0.02 * i, tone(100)) for i in range(1, 5)]
        values = playout(_synthesis(arrivals).timeline, 0.0)
        assert values["underruns"] == 0 and values["stall_ms"] == 0.0 and values["min_margin_ms"] > 0


class TestCancel:
    def test_audio_after_cancel_is_counted_from_arrival(self):
        arrivals = [(1.5, tone(100)), (1.6, tone(100)), (2.05, tone(100)), (2.3, tone(50))]
        s = _synthesis(arrivals, cancel_at=2.0)
        values = cancel(s)
        assert values["chunks_after_cancel"] == 2
        assert values["audio_after_cancel_ms"] == 150.0
        assert values["cancel_to_last_chunk_ms"] == 300.0
        assert values["cancel_at_ms"] == 1000.0
