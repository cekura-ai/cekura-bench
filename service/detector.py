"""Speech onset/offset detection — the clock every service-bench number is measured against.

Every published latency is a difference between two of these decisions, so the
detector's own error is a floor on what the benchmark can resolve. It therefore
reports a confidence with every decision, and ``calibrate()`` measures the error
against signals whose true boundaries are known by construction.

Design notes, since the obvious implementation is wrong in ways that do not show
up until they have already moved a ranking:

* The threshold is relative to an estimated **noise floor**, not to the clip's
  peak. A peak-relative threshold is set by the loudest sample anywhere in the
  file, so one click or one loud syllable silently rescales the decision for the
  whole clip.
* Onset requires several consecutive frames above threshold, and offset several
  below. Without hysteresis a codec click starts "speech" and an inter-word stop
  ends it.
* The reported time is the start of the **first** frame of the run, not the
  frame that completed it. Requiring N frames of evidence must not add N frames
  of latency to the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_MS = 10.0
SILENCE_DB = -90.0          # floor for an all-zero frame
ONSET_FRAMES = 3            # 30 ms of evidence to declare speech
OFFSET_FRAMES = 20          # 200 ms of quiet to declare the turn ended
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


def detect_offset(
    pcm: bytes | np.ndarray,
    rate: int,
    frame_ms: float = FRAME_MS,
    frames: int = OFFSET_FRAMES,
    margin_db: float = MARGIN_DB,
) -> Decision:
    """End of speech: the start of the first sustained quiet run after onset.

    Anchored after the onset so leading silence is never mistaken for the end,
    and reported at the first quiet frame rather than where the run completed —
    the speech stopped when it went quiet, not 200 ms later when we were sure.
    """
    energy = frame_energy_db(pcm, rate, frame_ms)
    floor = noise_floor_db(energy)
    threshold = floor + margin_db
    onset = detect_onset(pcm, rate, frame_ms, margin_db=margin_db)
    if not onset.found:
        return Decision(None, 0.0, floor, threshold, 0.0)

    start = int(onset.time_ms / frame_ms)
    quiet = energy[start:] <= threshold
    idx = _run_start(quiet, frames)
    if idx is None:
        # Speech ran to the end of the buffer; the boundary is outside the clip.
        return Decision(None, 0.0, floor, threshold, 0.0)
    absolute = start + idx
    deciding = energy[absolute : absolute + frames]
    before = float(energy[max(start, absolute - frames) : absolute].mean())
    return Decision(
        time_ms=absolute * frame_ms,
        confidence_db=float(threshold - deciding.max()),
        noise_floor_db=floor,
        threshold_db=threshold,
        sharpness_db=float(before - deciding.mean()),
    )


# ── extent of speech inside an authored clip ─────────────────────────────────
#
# Distinct from ``detect_offset``, and the distinction is load-bearing.
# ``detect_offset`` answers "has this turn ended?", so it demands a sustained
# quiet run and deliberately reports nothing when speech runs to the end of the
# buffer. ``speech_bounds`` answers "where inside this file is the speech?" for a
# clip we authored and can inspect offline. A TTS render is usually trimmed hard
# at both ends, so asking the first question of a corpus file returns None and
# would silently cost us the one boundary this benchmark gets for free.
#
# The padding is not cosmetic: the noise floor is a low percentile of frame
# energy, and in a clip that is nearly all speech that percentile lands *inside*
# speech and drags the threshold up. Padding with digital silence gives the
# estimator quiet material that is known-quiet by construction.

BOUNDS_PAD_MS = 200.0


@dataclass(frozen=True)
class Bounds:
    """Speech extent within a clip, in milliseconds from the clip's first sample."""

    start_ms: float | None
    end_ms: float | None
    noise_floor_db: float
    threshold_db: float

    @property
    def found(self) -> bool:
        return self.start_ms is not None and self.end_ms is not None

    @property
    def duration_ms(self) -> float:
        return 0.0 if not self.found else self.end_ms - self.start_ms


def speech_bounds(
    pcm: bytes | np.ndarray,
    rate: int,
    frame_ms: float = FRAME_MS,
    frames: int = ONSET_FRAMES,
    margin_db: float = MARGIN_DB,
) -> Bounds:
    """First and last sustained speech frame in an authored clip.

    The end is reported at the *end* of the last speech frame, not its start:
    this is where the audio we wrote actually stops, and it is the caller-side
    boundary every service-bench latency is measured from.
    """
    samples = (
        np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray)) else pcm
    ).astype(np.float64)
    pad = np.zeros(int(rate * BOUNDS_PAD_MS / 1000.0))
    padded = np.concatenate([pad, samples, pad])

    energy = frame_energy_db(padded, rate, frame_ms)
    floor = noise_floor_db(energy)
    threshold = floor + margin_db
    loud = energy > threshold

    start = _run_start(loud, frames)
    end = _run_start(loud[::-1], frames)
    if start is None or end is None:
        return Bounds(None, None, floor, threshold)

    offset_frames = BOUNDS_PAD_MS / frame_ms
    last_frame = loud.size - 1 - end  # index of the final speech frame
    return Bounds(
        start_ms=(start - offset_frames) * frame_ms,
        end_ms=(last_frame + 1 - offset_frames) * frame_ms,
        noise_floor_db=floor,
        threshold_db=threshold,
    )
