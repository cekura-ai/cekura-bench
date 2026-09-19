"""The injected-tone instrument that gates every Lane B latency.

If this is wrong, a transport measurement is wrong, and every phone-leg latency
corrected against it is wrong by the same amount in the same direction -- which
is the kind of error that survives review, because nothing about the published
numbers looks odd.
"""

from __future__ import annotations

import numpy as np

from lane_a.audio import pcm_to_ulaw, to_array, to_pcm, ulaw_to_pcm
from lane_b.calibrate import line_noise, voiced
from lane_b.tone import RATE, chirp, find, round_trip


def through_a_line(signal: np.ndarray, noise_rms: float, rng: np.random.Generator) -> bytes:
    noisy = signal + line_noise(signal.size, RATE, noise_rms, rng)
    return ulaw_to_pcm(pcm_to_ulaw(to_pcm(np.clip(noisy, -1, 1) * 32767)))


def delayed(reference: bytes, delay_ms: float, tail_ms: float = 1000.0) -> np.ndarray:
    return np.concatenate([
        np.zeros(int(RATE * delay_ms / 1000.0)),
        to_array(reference).astype(np.float64) / 32768.0,
        np.zeros(int(RATE * tail_ms / 1000.0)),
    ])


class TestRecoveringTheChirp:
    def test_the_delay_is_recovered_to_well_under_a_millisecond(self):
        """The whole point is to resolve the transport, which is tens of ms.

        An instrument with millisecond error would be adequate; this one is
        better than that, and the test pins it so a change to the correlation
        cannot quietly give some of it back.
        """
        rng = np.random.default_rng(5)
        reference = chirp()
        for true_delay in (120.0, 337.5, 800.0):
            heard = through_a_line(delayed(reference, true_delay), 0.02, rng)
            found = find(heard, reference)

            assert found.found
            assert abs(found.offset_ms - true_delay) < 0.5

    def test_it_still_works_with_the_chirp_buried_in_the_noise(self):
        """A quiet calibration signal is the polite one: it is played into a live
        call. It has to survive being quieter than the line it travels on."""
        rng = np.random.default_rng(11)
        reference = chirp()
        heard = through_a_line(delayed(reference, 250.0), 0.5, rng)   # 0 dB SNR

        found = find(heard, reference)
        assert found.found
        assert abs(found.offset_ms - 250.0) < 1.0


class TestRefusingToGuess:
    def test_speech_is_not_mistaken_for_the_chirp(self):
        """A calibration that finds itself in ordinary audio is worse than none:
        it produces a transport correction out of the agent's own voice."""
        rng = np.random.default_rng(13)
        speech = voiced(1.5, RATE)
        speech = speech / np.abs(speech).max() * 0.4
        heard = through_a_line(np.concatenate([np.zeros(int(0.3 * RATE)), speech]), 0.03, rng)

        assert not find(heard, chirp()).found

    def test_a_signal_lost_in_the_noise_is_reported_missing_not_estimated(self):
        rng = np.random.default_rng(17)
        heard = through_a_line(delayed(chirp(), 300.0), 1.6, rng)     # about -10 dB

        assert not find(heard, chirp()).found


class TestRoundTrip:
    def test_the_interval_is_measured_from_when_the_first_sample_left(self):
        rng = np.random.default_rng(19)
        reference = chirp()
        heard = through_a_line(delayed(reference, 500.0), 0.02, rng)

        result = round_trip(sent_at_ms=140.0, recording=heard, reference=reference)

        assert result is not None
        assert abs(result.round_trip_ms - 360.0) < 1.0
        assert abs(result.one_way_ms - 180.0) < 0.5

    def test_audio_that_came_back_before_it_was_sent_is_rejected(self):
        """A negative interval means the clocks or the recording are wrong.

        Returning it as a small positive number, or as zero, would put a
        nonsensical measurement into the correction every latency is adjusted by.
        """
        rng = np.random.default_rng(23)
        reference = chirp()
        heard = through_a_line(delayed(reference, 100.0), 0.02, rng)

        assert round_trip(sent_at_ms=400.0, recording=heard, reference=reference) is None
