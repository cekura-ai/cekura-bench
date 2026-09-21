"""The detector's own error, measured against boundaries that are exact by construction.

Every published latency is a difference between two decisions, so the detector's
error is a floor on what this benchmark can resolve. These tests pin the two
properties that matter and the one failure mode that would silently manufacture a
result: under noise the detector runs *late*, which if left unchecked would turn
an instrument artefact into "providers are slower in noisy conditions".
"""

from __future__ import annotations

import numpy as np
import pytest

from service.audio import to_pcm
from service.detector import detect_offset, detect_onset, speech_bounds

RATE = 24000


def speech(ms: float, f0: float = 120.0, rate: int = RATE, seed: int = 0) -> np.ndarray:
    """A speech surrogate: harmonic stack, 4 Hz syllabic envelope, sharp attack."""
    n = int(rate * ms / 1000.0)
    t = np.arange(n) / rate
    stack = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t - np.pi / 2)
    attack = np.minimum(t / 0.015, 1.0)
    return 8000 * stack / 3.0 * envelope * attack


def clip(lead_ms: float, speech_ms: float, tail_ms: float, noise_db: float | None = None, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(RATE * lead_ms / 1000.0)), speech(speech_ms), np.zeros(int(RATE * tail_ms / 1000.0))]
    signal = np.concatenate(parts)
    if noise_db is not None:
        signal = signal + rng.standard_normal(signal.size) * (32768 * 10 ** (noise_db / 20.0))
    return to_pcm(signal)


class TestOnset:
    @pytest.mark.parametrize("lead_ms", [0.0, 120.0, 500.0, 1500.0])
    def test_clean_onset_lands_within_ten_milliseconds(self, lead_ms):
        decision = detect_onset(clip(lead_ms, 800, 300), RATE)
        assert decision.found
        assert abs(decision.time_ms - lead_ms) <= 10.0, decision

    def test_a_single_click_does_not_rescale_the_threshold(self):
        """A peak-relative threshold would be set by the click and miss the speech."""
        samples = np.frombuffer(clip(300, 800, 300), dtype="<i2").astype(np.float64)
        samples[10] = 32767  # one sample of digital full scale in the lead silence
        decision = detect_onset(to_pcm(samples), RATE)
        assert decision.found and abs(decision.time_ms - 300.0) <= 20.0, decision

    def test_nothing_is_found_in_silence(self):
        assert not detect_onset(to_pcm(np.zeros(RATE)), RATE).found


class TestOffset:
    def test_offset_is_reported_where_speech_stopped_not_where_we_were_sure(self):
        decision = detect_offset(clip(200, 900, 600), RATE)
        assert decision.found
        # 200 ms of hysteresis must not be added to the reported boundary.
        assert abs(decision.time_ms - 1100.0) <= 20.0, decision

    def test_leading_silence_is_never_mistaken_for_the_end(self):
        decision = detect_offset(clip(1500, 600, 500), RATE)
        assert decision.found and decision.time_ms > 1500.0


class TestBounds:
    def test_bounds_work_on_a_clip_trimmed_at_both_ends(self):
        """A TTS render has no trailing silence, so detect_offset returns nothing."""
        trimmed = to_pcm(speech(900))
        assert not detect_offset(trimmed, RATE).found
        bounds = speech_bounds(trimmed, RATE)
        assert bounds.found
        assert abs(bounds.start_ms) <= 20.0 and abs(bounds.end_ms - 900.0) <= 20.0, bounds

    def test_padding_keeps_the_floor_out_of_the_speech(self):
        """Without the silence pad the noise-floor percentile lands inside speech."""
        bounds = speech_bounds(to_pcm(speech(2000)), RATE)
        assert bounds.noise_floor_db < -60.0, bounds

    def test_bounds_track_a_known_window(self):
        bounds = speech_bounds(clip(250, 700, 400), RATE)
        assert abs(bounds.start_ms - 250.0) <= 15.0
        assert abs(bounds.end_ms - 950.0) <= 25.0


class TestNoiseIsTheKnownFailureMode:
    def test_the_detector_runs_late_under_noise_rather_than_early(self):
        """Documented limit: this detector may not be pointed at a noisy channel.

        The service bench is safe because noise sits on the caller channel we author while
        the provider's returned audio comes back clean. The agent bench's phone leg is not,
        and needs harmonicity-based detection instead of broadband energy.
        """
        errors = []
        for seed in range(12):
            decision = detect_onset(clip(400, 900, 300, noise_db=-34.0, seed=seed), RATE)
            if decision.found:
                errors.append(decision.time_ms - 400.0)
        assert errors, "detector lost the onset entirely"
        assert np.mean(errors) > -5.0, "noise must not make the detector fire early"
