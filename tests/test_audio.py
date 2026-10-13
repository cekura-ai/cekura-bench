"""The sample-to-wall-clock mapping, the resampler, and G.711.

The timeline tests are the important ones. A benchmark that anchors a boundary to
"the arrival time of the chunk it fell in" credits a provider that ships 500 ms in
one frame with speaking 500 ms earlier than it did, and that error is larger than
most of the gaps this benchmark exists to resolve.
"""

from __future__ import annotations

import numpy as np
import pytest

from service.audio import (
    AudioTimeline,
    SAMPLE_WIDTH,
    iter_chunks,
    mix,
    pcm_to_ulaw,
    pink_noise,
    resample,
    silence,
    to_array,
    to_pcm,
    ulaw_to_pcm,
)

RATE = 24000


class TestTimeline:
    def test_a_sample_inside_a_chunk_is_interpolated(self):
        timeline = AudioTimeline(RATE)
        timeline.record(480, 10.0)   # 20 ms
        timeline.record(480, 10.02)
        assert timeline.time_of_sample(0) == pytest.approx(10.0)
        assert timeline.time_of_sample(240) == pytest.approx(10.01)
        assert timeline.time_of_sample(479) == pytest.approx(10.0 + 479 / RATE)
        assert timeline.time_of_sample(480) == pytest.approx(10.02)

    def test_a_batched_first_frame_is_not_credited_to_its_arrival(self):
        """The listener hears sample k of a big frame k/rate after it lands."""
        batched = AudioTimeline(RATE)
        batched.record(12000, 11.0)          # 500 ms in one frame
        assert batched.time_of_sample(6000) == pytest.approx(11.25)

        streamed = AudioTimeline(RATE)
        for index in range(25):
            streamed.record(480, 11.0 + index * 0.02)
        # Same audio, same start: the midpoint is audible at the same instant.
        assert streamed.time_of_sample(6000) == pytest.approx(batched.time_of_sample(6000), abs=0.001)

    def test_samples_past_the_end_have_no_time(self):
        timeline = AudioTimeline(RATE)
        timeline.record(480, 1.0)
        assert timeline.time_of_sample(480) is None
        assert timeline.time_of_sample(-1) is None

    def test_duration_tracks_recorded_samples(self):
        timeline = AudioTimeline(RATE)
        timeline.record_pcm(silence(RATE, 250.0), 0.0)
        assert timeline.duration_s == pytest.approx(0.25)


class TestChunking:
    def test_chunks_are_whole_frames_and_cover_the_input(self):
        pcm = silence(RATE, 100.0)
        chunks = list(iter_chunks(pcm, RATE, 20.0))
        assert len(chunks) == 5
        assert all(len(c) == int(RATE * 0.02) * SAMPLE_WIDTH for c in chunks)
        assert b"".join(chunks) == pcm

    def test_a_short_tail_is_still_emitted(self):
        chunks = list(iter_chunks(silence(RATE, 30.0), RATE, 20.0))
        assert len(chunks) == 2 and len(chunks[1]) < len(chunks[0])


class TestResample:
    @pytest.mark.parametrize("target", [16000, 8000, 48000])
    def test_a_tone_keeps_its_frequency_and_level(self, target):
        t = np.arange(RATE) / RATE
        source = to_pcm(8000 * np.sin(2 * np.pi * 1000.0 * t))
        out = to_array(resample(source, RATE, target)).astype(float)
        assert abs(out.size - target) <= 2

        window = out[target // 8 : -target // 8]
        spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
        peak_hz = np.fft.rfftfreq(window.size, 1.0 / target)[spectrum.argmax()]
        assert abs(peak_hz - 1000.0) < 15.0

        reference = to_array(source).astype(float)[RATE // 8 : -RATE // 8]
        assert np.sqrt((window**2).mean()) / np.sqrt((reference**2).mean()) == pytest.approx(1.0, abs=0.05)

    def test_matching_rates_are_returned_untouched(self):
        pcm = silence(RATE, 10.0)
        assert resample(pcm, RATE, RATE) is pcm


class TestUlaw:
    def test_round_trip_stays_within_g711_error(self):
        t = np.arange(8000) / 8000
        source = to_pcm(9000 * np.sin(2 * np.pi * 440.0 * t))
        back = to_array(ulaw_to_pcm(pcm_to_ulaw(source))).astype(float)
        original = to_array(source).astype(float)
        assert np.sqrt(((back - original) ** 2).mean()) / np.sqrt((original**2).mean()) < 0.05

    def test_one_byte_per_sample(self):
        assert len(pcm_to_ulaw(silence(8000, 100.0))) == 800


class TestNoise:
    def test_level_is_what_was_asked_for(self):
        samples = to_array(pink_noise(RATE, 1000.0, -30.0, seed=3)).astype(float)
        level = 20 * np.log10(np.sqrt((samples**2).mean()) / 32768)
        assert level == pytest.approx(-30.0, abs=0.5)

    def test_the_same_seed_gives_the_same_bed(self):
        assert pink_noise(RATE, 200.0, -30.0, seed=5) == pink_noise(RATE, 200.0, -30.0, seed=5)
        assert pink_noise(RATE, 200.0, -30.0, seed=5) != pink_noise(RATE, 200.0, -30.0, seed=6)

    def test_mixing_a_bed_raises_the_floor_without_clipping(self):
        speech_pcm = to_pcm(6000 * np.sin(2 * np.pi * 200 * np.arange(RATE) / RATE))
        mixed = to_array(mix(speech_pcm, pink_noise(RATE, 100.0, -40.0, seed=1))).astype(float)
        assert np.abs(mixed).max() < 32768
        assert np.sqrt((mixed**2).mean()) > np.sqrt((to_array(speech_pcm).astype(float) ** 2).mean()) * 0.99
