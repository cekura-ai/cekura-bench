"""Adaptive speech-onset detector, published beside the fixed leading-silence rule.

The fixed rule in ``tts_bench.metrics`` is what TTFA is built from, because a fixed
threshold compares across providers. This detector is the second opinion: its
threshold sits a margin above an estimated noise floor, it requires several frames
of evidence, and it reports the start of the first frame of the run so that the
evidence does not add latency to the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FRAME_MS = 10.0
SILENCE_DB = -90.0          # floor for an all-zero frame
ONSET_FRAMES = 3            # 30 ms of evidence to declare speech
MARGIN_DB = 8.0             # how far above the noise floor counts as speech


@dataclass(frozen=True)
class Decision:
    """A boundary, with enough context to audit it after the fact."""

    time_ms: float | None
    confidence_db: float      # how far past threshold the deciding frames sat
    noise_floor_db: float
    threshold_db: float
    sharpness_db: float       # energy rise across the decision; a slow fade is ambiguous

    @property
    def found(self) -> bool:
        return self.time_ms is not None


def frame_energy_db(pcm: bytes | np.ndarray, rate: int, frame_ms: float = FRAME_MS) -> np.ndarray:
    """Per-frame RMS in dBFS. Frames are non-overlapping; a partial tail is dropped."""
    samples = (
        np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray)) else pcm
    ).astype(np.float64)
    if samples.size == 0:
        return np.zeros(0)
    n = max(1, int(rate * frame_ms / 1000.0))
    usable = samples.size - (samples.size % n)
    if usable == 0:
        return np.zeros(0)
    frames = samples[:usable].reshape(-1, n)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return np.maximum(20.0 * np.log10(np.maximum(rms, 1e-9) / 32768.0), SILENCE_DB)


def noise_floor_db(energy: np.ndarray, percentile: float = 10.0) -> float:
    """Estimate the quiet level from the low percentile of frame energies.

    A percentile rather than the minimum: the minimum is one frame and a single
    digital-silence gap inside speech would drag the floor to -90 dB and make
    every subsequent frame look like speech.
    """
    return float(np.percentile(energy, percentile)) if energy.size else SILENCE_DB


def _run_start(mask: np.ndarray, length: int) -> int | None:
    """First index beginning ``length`` consecutive True values."""
    if mask.size < length:
        return None
    if length == 1:
        hits = np.flatnonzero(mask)
        return int(hits[0]) if hits.size else None
    windows = np.lib.stride_tricks.sliding_window_view(mask, length)
    hits = np.flatnonzero(windows.all(axis=1))
    return int(hits[0]) if hits.size else None


def detect_onset(
    pcm: bytes | np.ndarray,
    rate: int,
    frame_ms: float = FRAME_MS,
    frames: int = ONSET_FRAMES,
    margin_db: float = MARGIN_DB,
) -> Decision:
    """First audible sample: the start of the first sustained run above the floor."""
    energy = frame_energy_db(pcm, rate, frame_ms)
    floor = noise_floor_db(energy)
    threshold = floor + margin_db
    idx = _run_start(energy > threshold, frames)
    if idx is None:
        return Decision(None, 0.0, floor, threshold, 0.0)
    deciding = energy[idx : idx + frames]
    before = float(energy[max(0, idx - frames) : idx].mean()) if idx else floor
    return Decision(
        time_ms=idx * frame_ms,
        confidence_db=float(deciding.min() - threshold),
        noise_floor_db=floor,
        threshold_db=threshold,
        sharpness_db=float(deciding.mean() - before),
    )
