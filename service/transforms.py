"""Degradation transforms: published, deterministic, applied to our own audio.

Robustness is reported as the **delta from clean**, never as an absolute. The
same authored clip goes through the same transform for every provider, so the
only thing that differs between the clean cell and the degraded one is the
degradation -- and anyone can regenerate the degraded audio from the clean
master and the seed, which is what makes the delta auditable.

Everything here is synthesized from the clip and a seed rather than mixed from
recordings. A recording would add realism at the cost of reproducibility, and a
benchmark that cannot regenerate its own inputs cannot be checked. Recorded
beds can join later as checksummed files; they do not replace these.

Each transform is named once, in ``TRANSFORMS``, and that name is the identity
that travels into the cell record. Changing a transform's parameters is a new
name, not an edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from service.audio import pcm_to_ulaw, resample, to_array, to_pcm, ulaw_to_pcm

Transform = Callable[[bytes, int], bytes]


def clean(pcm: bytes, rate: int) -> bytes:
    return pcm


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


def _pink(n: int, rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    shaped = spectrum / np.sqrt(np.maximum(freqs, freqs[1] if freqs.size > 1 else 1.0))
    noise = np.fft.irfft(shaped, n=n)
    return noise / (_rms(noise) or 1.0)


def noise(snr_db: float, seed: int = 101) -> Transform:
    """Pink noise under the whole clip at a stated SNR against the clip's own RMS.

    SNR is relative to the speech level rather than to full scale, so a quiet
    render and a loud one are degraded by the same amount.
    """

    def apply(pcm: bytes, rate: int) -> bytes:
        samples = to_array(pcm).astype(np.float64)
        if samples.size == 0:
            return pcm
        level = _rms(samples) or 1.0
        bed = _pink(samples.size, rate, seed) * level / (10 ** (snr_db / 20.0))
        return to_pcm(samples + bed)

    return apply


def telephone(pcm: bytes, rate: int) -> bytes:
    """A narrowband phone leg: 8 kHz round trip through G.711 mu-law.

    The band limit comes from the resampler's own anti-alias filter at 8 kHz,
    so nothing above 4 kHz survives, and the mu-law step adds the codec's
    quantisation. Restored to the original rate so the clip's timing is
    unchanged and the caller-side boundary stays exact.
    """
    narrow = resample(pcm, rate, 8000)
    coded = ulaw_to_pcm(pcm_to_ulaw(narrow))
    restored = resample(coded, 8000, rate)
    # The resampler's group delay is symmetric and the filter is applied twice,
    # so the round trip preserves length up to rounding; pad or trim to match.
    want = len(pcm)
    if len(restored) < want:
        restored += b"\x00" * (want - len(restored))
    return restored[:want]


def reverb(rt60_s: float = 0.4, direct_ratio: float = 0.6, seed: int = 202) -> Transform:
    """A far-field room: a seeded exponentially decaying noise tail behind the direct path.

    Not a measured room; a shape every room shares, at a stated decay time.
    ``direct_ratio`` is the direct path's share of energy, which is what places
    the talker nearer or further from the microphone.
    """

    def apply(pcm: bytes, rate: int) -> bytes:
        samples = to_array(pcm).astype(np.float64)
        if samples.size == 0:
            return pcm
        rng = np.random.default_rng(seed)
        length = int(rate * rt60_s)
        t = np.arange(length) / rate
        tail = rng.standard_normal(length) * np.exp(-6.9 * t / rt60_s)  # -60 dB at rt60
        tail /= np.sqrt(np.sum(tail**2)) or 1.0
        impulse = np.zeros(length)
        impulse[0] = 1.0
        impulse = np.sqrt(direct_ratio) * impulse + np.sqrt(1.0 - direct_ratio) * tail
        wet = np.convolve(samples, impulse)[: samples.size]
        wet *= _rms(samples) / (_rms(wet) or 1.0)  # keep the level; only the room changes
        return to_pcm(wet)

    return apply


def clipping(gain_db: float = 12.0) -> Transform:
    """Overdriven input: gain, hard clip at full scale, level restored."""

    def apply(pcm: bytes, rate: int) -> bytes:
        samples = to_array(pcm).astype(np.float64)
        if samples.size == 0:
            return pcm
        driven = np.clip(samples * 10 ** (gain_db / 20.0), -32768, 32767)
        driven *= _rms(samples) / (_rms(driven) or 1.0)
        return to_pcm(driven)

    return apply


def dropouts(hole_ms: float = 60.0, every_ms: float = 500.0, offset_ms: float = 250.0) -> Transform:
    """Packet loss as periodic silent holes, fixed in position so every run loses the same audio."""

    def apply(pcm: bytes, rate: int) -> bytes:
        samples = to_array(pcm).copy()
        hole = int(rate * hole_ms / 1000.0)
        step = int(rate * every_ms / 1000.0)
        start = int(rate * offset_ms / 1000.0)
        for begin in range(start, samples.size, step):
            samples[begin : begin + hole] = 0
        return samples.tobytes()

    return apply


@dataclass(frozen=True)
class NamedTransform:
    name: str
    apply: Transform
    description: str


TRANSFORMS: dict[str, NamedTransform] = {
    t.name: t
    for t in (
        NamedTransform("clean", clean, "the master as rendered"),
        NamedTransform("noise-20db", noise(20.0), "pink noise at 20 dB SNR against the speech"),
        NamedTransform("noise-10db", noise(10.0), "pink noise at 10 dB SNR against the speech"),
        NamedTransform("telephone", telephone, "8 kHz G.711 mu-law round trip"),
        NamedTransform("reverb", reverb(), "far-field room, 0.4 s decay, 60% direct energy"),
        NamedTransform("clipping", clipping(), "12 dB overdrive hard-clipped, level restored"),
        NamedTransform("dropouts", dropouts(), "60 ms silent holes every 500 ms"),
    )
}


def get(name: str) -> NamedTransform:
    try:
        return TRANSFORMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown transform {name!r}; known: {sorted(TRANSFORMS)}") from exc
