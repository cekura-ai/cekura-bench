"""The streaming resampler reproduces the service bench's published filter."""

import numpy as np
import pytest

from turn_taking_bench.author import service_bridge
from turn_taking_bench.s2s.resample import StreamingResampler, design


@pytest.mark.parametrize("src,dst", [(48000, 24000), (48000, 16000), (16000, 24000)])
def test_streaming_matches_published_resampler_sample_for_sample(src, dst):
    audio = service_bridge.audio()
    rng = np.random.default_rng(3)
    seconds = 0.5
    signal = (0.3 * rng.standard_normal(int(src * seconds))).astype(np.float64)
    # The published function, on int16 input, whole clip at once.
    pcm = audio.to_pcm(signal * 32767.0)
    expected = np.frombuffer(audio.resample(pcm, src, dst), dtype="<i2").astype(np.float64)

    resampler = StreamingResampler(src, dst)
    frame = int(0.02 * src)
    pieces = []
    source = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    for start in range(0, len(source), frame):
        pieces.append(resampler.process(source[start : start + frame]))
    pieces.append(resampler.flush())
    streamed = np.concatenate(pieces)

    assert streamed.size == expected.size
    rounded = np.clip(np.rint(streamed), -32768, 32767)
    # Same arithmetic in a different summation order: at most one LSB apart after rounding.
    assert np.abs(rounded - expected).max() <= 1
    assert np.mean(rounded == expected) > 0.99


def test_filter_design_is_the_service_bench_filter():
    h, up, down = design(48000, 16000)
    assert (up, down) == (1, 3)
    assert h.size == 2 * 32 * 3 + 1
    assert abs(h.sum() - up) < 1e-9
    assert StreamingResampler(48000, 16000).delay_s == pytest.approx(96 / 48000)


def test_identity_rates_pass_through():
    resampler = StreamingResampler(24000, 24000)
    frame = np.arange(480, dtype=np.float64)
    assert np.array_equal(resampler.process(frame), frame)
    assert resampler.flush().size == 0
