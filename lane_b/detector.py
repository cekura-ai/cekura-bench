"""Speech detection on a phone leg, where energy alone is not enough.

Lane A's detector thresholds frame energy against an estimated noise floor. That
works because both sides of a direct websocket are clean: the only thing on the
channel is the audio we sent and the audio the provider returned. A phone leg is
not that channel. It carries comfort noise, line hum and codec artifacts
continuously, and the caller's own audio arrives after μ-law at 8 kHz. The
energy detector's measured behaviour under noise -- systematically late at 10 dB
SNR, failing entirely at 0 dB -- is a statement about this channel, which is why
Lane A's rule forbids pointing it at one.

What separates speech from line noise is not level, it is **periodicity**.
Voiced speech repeats at the speaker's pitch, 70-400 Hz for adult voices; hum is
periodic far below that band, and comfort noise is not periodic at all. So the
decision here is energy *and* periodicity, and the periodicity is measured by
normalized autocorrelation over the pitch band.

Autocorrelation is the right estimator for this channel specifically. The
telephone band starts near 300 Hz, so the fundamental of most voices is simply
not present in the signal -- but the harmonics are, and they still repeat at the
period of the missing fundamental. A pitch estimator that looked for energy at
f0 would find nothing on exactly the audio this lane is built to measure.

**The cost, stated because it is a bias and not a bug.** Unvoiced sounds are not
periodic, so a turn opening on a fricative is detected at its first voiced frame
rather than at the fricative. That is late by a measurable amount rather than
wrong, and `lane_b/calibrate.py` measures it. The alternative -- accepting any
high-energy frame -- reintroduces exactly the false onsets on noise this detector
exists to avoid, and a false onset is not a bias, it is a wrong answer.

Nothing here may be used to publish a latency until `calibrate.py` has been run
against the transport actually in use.
"""

from __future__ import annotations

import numpy as np

from lane_a.detector import Decision, _run_start, frame_energy_db, noise_floor_db

FRAME_MS = 30.0          # >= 2 periods at the lowest pitch we accept
HOP_MS = 10.0            # decision resolution, matched to Lane A's
PITCH_MIN_HZ = 70.0
PITCH_MAX_HZ = 400.0
HARMONIC_THRESHOLD = 0.45
ONSET_FRAMES = 3         # 30 ms of evidence
OFFSET_FRAMES = 20       # 200 ms of quiet
MARGIN_DB = 6.0          # lower than Lane A's: periodicity carries the decision


def _frames(samples: np.ndarray, rate: int, frame_ms: float, hop_ms: float) -> np.ndarray:
    """Overlapping frames as a 2-D view. Empty when the signal is shorter than one frame."""
    n = max(1, int(rate * frame_ms / 1000.0))
    hop = max(1, int(rate * hop_ms / 1000.0))
    if samples.size < n:
        return np.zeros((0, n))
    count = 1 + (samples.size - n) // hop
    return np.lib.stride_tricks.as_strided(
        samples, shape=(count, n), strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )


def harmonicity(
    pcm: bytes | np.ndarray,
    rate: int,
    frame_ms: float = FRAME_MS,
    hop_ms: float = HOP_MS,
) -> np.ndarray:
    """Per-frame peak normalized autocorrelation over the pitch band, in [0, 1].

    Normalized by the energy of both halves of each lagged product rather than by
    the frame energy alone: an unnormalized peak grows with loudness, which would
    make the score a second energy detector wearing a different name.
    """
    samples = (
        np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray)) else pcm
    ).astype(np.float64)
    frames = _frames(samples, rate, frame_ms, hop_ms)
    if frames.shape[0] == 0:
        return np.zeros(0)

    # Mean removal per frame: a DC offset or a strong sub-band hum correlates with
    # itself at every lag and would raise the score on silence.
    frames = frames - frames.mean(axis=1, keepdims=True)

    lag_min = max(1, int(rate / PITCH_MAX_HZ))
    lag_max = min(frames.shape[1] - 1, int(rate / PITCH_MIN_HZ))
    if lag_max <= lag_min:
        return np.zeros(frames.shape[0])

    best = np.zeros(frames.shape[0])
    for lag in range(lag_min, lag_max + 1):
        a, b = frames[:, :-lag], frames[:, lag:]
        num = np.einsum("ij,ij->i", a, b)
        den = np.sqrt(np.einsum("ij,ij->i", a, a) * np.einsum("ij,ij->i", b, b))
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(den > 0, num / den, 0.0)
        best = np.maximum(best, score)
    return np.clip(best, 0.0, 1.0)


def _speech_mask(
    pcm: bytes | np.ndarray, rate: int, frame_ms: float, hop_ms: float,
    margin_db: float, threshold: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Frames that are both loud enough and periodic enough to be voiced speech."""
    tone = harmonicity(pcm, rate, frame_ms, hop_ms)
    energy = frame_energy_db(pcm, rate, hop_ms)[: tone.size]
    if energy.size < tone.size:                      # ragged tail, align to the shorter
        tone = tone[: energy.size]
    floor = noise_floor_db(energy)
    level = floor + margin_db
    return (energy > level) & (tone > threshold), energy, floor, level


def detect_onset(
    pcm: bytes | np.ndarray,
    rate: int,
    frame_ms: float = FRAME_MS,
    hop_ms: float = HOP_MS,
    frames: int = ONSET_FRAMES,
    margin_db: float = MARGIN_DB,
    threshold: float = HARMONIC_THRESHOLD,
) -> Decision:
    """First voiced sample, as the start of the first sustained periodic run."""
    mask, energy, floor, level = _speech_mask(pcm, rate, frame_ms, hop_ms, margin_db, threshold)
    idx = _run_start(mask, frames)
    if idx is None:
        return Decision(None, 0.0, floor, level, 0.0)
    deciding = energy[idx : idx + frames]
    before = float(energy[max(0, idx - frames) : idx].mean()) if idx else floor
    return Decision(
        time_ms=idx * hop_ms,
        confidence_db=float(deciding.min() - level),
        noise_floor_db=floor,
        threshold_db=level,
        sharpness_db=float(deciding.mean() - before),
    )


def detect_offset(
    pcm: bytes | np.ndarray,
    rate: int,
    frame_ms: float = FRAME_MS,
    hop_ms: float = HOP_MS,
    frames: int = OFFSET_FRAMES,
    margin_db: float = MARGIN_DB,
    threshold: float = HARMONIC_THRESHOLD,
) -> Decision:
    """End of speech: the first sustained unvoiced run *after* onset.

    Anchored after the onset, or the leading silence on the line -- which on a
    phone leg is most of the buffer before the caller speaks -- is itself the
    first long unvoiced run and every turn ends before it began. Reported at the
    first quiet frame rather than where the run completed: the speech stopped
    when it went quiet, not 200 ms later when we were sure.
    """
    mask, energy, floor, level = _speech_mask(pcm, rate, frame_ms, hop_ms, margin_db, threshold)
    onset = detect_onset(pcm, rate, frame_ms, hop_ms, margin_db=margin_db, threshold=threshold)
    if not onset.found:
        return Decision(None, 0.0, floor, level, 0.0)

    start = int(onset.time_ms / hop_ms)
    idx = _run_start(~mask[start:], frames)
    if idx is None:
        # Speech ran to the end of the buffer; the boundary is outside the clip.
        return Decision(None, 0.0, floor, level, 0.0)
    absolute = start + idx
    deciding = energy[absolute : absolute + frames]
    before = float(energy[max(start, absolute - frames) : absolute].mean())
    return Decision(
        time_ms=absolute * hop_ms,
        confidence_db=float(level - deciding.max()) if deciding.size else 0.0,
        noise_floor_db=floor,
        threshold_db=level,
        sharpness_db=float(before - deciding.mean()) if deciding.size else 0.0,
    )
