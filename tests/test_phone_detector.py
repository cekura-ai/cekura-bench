"""The phone-leg detector, whose failures would be published as provider latency.

These run offline against constructed signals. A detector that is wrong here is
wrong in a way no live call would reveal: a boundary invented from line noise
does not look like an error, it looks like a fast reply.
"""

from __future__ import annotations

import numpy as np

from service.audio import pcm_to_ulaw, ulaw_to_pcm
from agent.calibrate import line_noise, through_the_phone, voiced
from agent.detector import detect_offset, detect_onset, harmonicity

RATE = 8000


def pcm(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()


class TestPeriodicitySeparatesSpeechFromNoise:
    def test_voiced_speech_and_line_noise_at_the_same_level_do_not_score_alike(self):
        """Level cannot tell them apart, which is the whole reason for this detector."""
        rng = np.random.default_rng(1)
        speech = voiced(1.0, RATE)
        speech = speech / np.abs(speech).max() * 0.3
        noise = line_noise(speech.size, RATE, 0.3, rng)

        assert harmonicity(pcm(speech), RATE).mean() > 0.6
        assert harmonicity(pcm(noise), RATE).mean() < 0.3

    def test_the_missing_fundamental_still_scores_as_voiced(self):
        """The telephone band starts near 300 Hz, so most voices lose their f0.

        Autocorrelation recovers the period from the harmonics, which is why it
        is the estimator here; anything looking for energy at f0 would score the
        audio this lane exists to measure as unvoiced.
        """
        t = np.arange(RATE) / RATE
        f0 = 120.0
        without_f0 = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(3, 12))
        without_f0 = without_f0 / np.abs(without_f0).max() * 0.3

        assert harmonicity(pcm(without_f0), RATE).mean() > 0.6


class TestASilentLineIsNotATurn:
    def test_transients_on_a_quiet_line_do_not_start_a_turn(self):
        """A click is loud and brief. Treating it as speech starts the clock early.

        This is not a rare case: switching clicks, crosstalk and a dropout
        returning all produce one. The measured cost of thresholding energy
        alone here is a false onset in roughly four calls out of five.
        """
        rng = np.random.default_rng(7)
        buffer = line_noise(int(3.0 * RATE), RATE, 0.02, rng)
        for at in (int(0.5 * RATE), int(1.4 * RATE), int(2.2 * RATE)):
            buffer[at : at + 240] += rng.normal(0, 0.25, 240)

        assert not detect_onset(ulaw_to_pcm(pcm_to_ulaw(pcm(buffer))), RATE).found


class TestOffsetIsAnchoredAfterOnset:
    def test_leading_silence_is_not_reported_as_the_end_of_the_turn(self):
        """On a phone leg the quiet before the caller speaks is most of the buffer.

        Searched from the start, that silence is itself the first long unvoiced
        run, and every turn ends before it began -- which surfaces as a missing
        or negative boundary rather than as an obviously wrong one.
        """
        rng = np.random.default_rng(3)
        speech = voiced(1.2, 16000)
        clip = np.concatenate([np.zeros(int(0.6 * 16000)), speech, np.zeros(int(0.6 * 16000))])
        heard = through_the_phone(clip, 25.0, rng)

        onset, offset = detect_onset(heard, RATE), detect_offset(heard, RATE)

        assert onset.found and offset.found
        assert offset.time_ms > onset.time_ms
        assert abs(offset.time_ms - 1800) < 120        # 600 ms lead + 1200 ms of speech
