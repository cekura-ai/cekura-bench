"""Degradations must be reproducible and must do what their name says.

A transform that drifted between runs would turn the delta-from-clean into
noise, and one whose level or bandwidth was not what its name claims would
publish a robustness result about a degradation nobody can regenerate.
"""

from __future__ import annotations

import numpy as np
import pytest

from lane_a.audio import to_array, to_pcm
from lane_a.detector import frame_energy_db
from lane_a.transforms import TRANSFORMS, get, noise, telephone

RATE = 24000


def speech_like(seconds: float = 1.0) -> bytes:
    t = np.arange(int(RATE * seconds)) / RATE
    tone = 6000 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    bright = 2000 * np.sin(2 * np.pi * 6000 * t)     # energy above the telephone band
    return to_pcm(tone + bright)


class TestEveryTransform:
    @pytest.mark.parametrize("name", sorted(TRANSFORMS))
    def test_same_input_same_output(self, name):
        pcm = speech_like()
        once = TRANSFORMS[name].apply(pcm, RATE)
        twice = TRANSFORMS[name].apply(pcm, RATE)
        assert once == twice, f"{name} is not deterministic"

    @pytest.mark.parametrize("name", sorted(TRANSFORMS))
    def test_length_is_preserved(self, name):
        """The caller-side boundary is a sample index; a transform may not move it."""
        pcm = speech_like()
        assert len(TRANSFORMS[name].apply(pcm, RATE)) == len(pcm)

    def test_names_resolve_and_unknown_names_do_not(self):
        assert get("clean").name == "clean"
        with pytest.raises(KeyError):
            get("louder")


class TestNoise:
    def test_the_snr_is_what_the_name_says(self):
        pcm = speech_like()
        clean = to_array(pcm).astype(float)
        degraded = to_array(noise(20.0)(pcm, RATE)).astype(float)
        added = degraded - clean
        snr = 20 * np.log10(np.sqrt(np.mean(clean**2)) / np.sqrt(np.mean(added**2)))
        assert snr == pytest.approx(20.0, abs=0.5)


class TestTelephone:
    def test_the_band_above_four_kilohertz_is_gone(self):
        pcm = speech_like()
        before = np.abs(np.fft.rfft(to_array(pcm).astype(float))) ** 2
        after = np.abs(np.fft.rfft(to_array(telephone(pcm, RATE)).astype(float))) ** 2
        freqs = np.fft.rfftfreq(len(to_array(pcm)), 1.0 / RATE)
        high = freqs > 4500
        assert before[high].sum() > 0.1 * before.sum(), "the fixture must carry energy above the band"
        assert after[high].sum() < 1e-3 * before[high].sum()

    def test_the_speech_band_survives(self):
        pcm = speech_like()
        energy_before = frame_energy_db(pcm, RATE).mean()
        energy_after = frame_energy_db(telephone(pcm, RATE), RATE).mean()
        assert abs(energy_before - energy_after) < 6.0


class TestDropouts:
    def test_holes_are_silent_and_periodic(self):
        pcm = speech_like(2.0)
        out = to_array(TRANSFORMS["dropouts"].apply(pcm, RATE))
        hole = slice(int(RATE * 0.25), int(RATE * 0.25) + int(RATE * 0.06))
        assert np.all(out[hole] == 0)
        assert not np.all(out[int(RATE * 0.35): int(RATE * 0.40)] == 0)
