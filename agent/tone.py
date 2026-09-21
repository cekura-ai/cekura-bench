"""Measuring what the carrier itself adds, with a signal whose start time we know.

The agent bench's detector calibration bounds the *detector*. It says nothing about the
transport: a phone call adds delay in the carrier, in the codec, and in whatever
jitter buffer sits between, and it need not add the same amount in each
direction. Until that is measured, an agent-bench latency is a number of unknown
origin, so no the agent bench latency may be published before this has been run against
the carrier and transport actually in use.

The method is the only one that does not assume what it is trying to establish:
play a signal whose first sample leaves at a known instant, recover it from the
audio that comes back, and take the difference. The far end is a loopback we
control -- an endpoint that returns each inbound frame immediately -- so the
measured interval contains the carrier and nothing that reasons.

**A linear chirp, not a tone.** A steady tone cross-correlates broadly: every
cycle looks like every other, so the peak is a plateau a few milliseconds wide
and the answer is only as good as the plateau is narrow. A sweep correlates
sharply against exactly one alignment, and a sweep across the telephone band
survives the band-pass and the codec that a click would not.

**One-way is halved round trip, and that is an assumption.** The two directions
are separate paths and need not be symmetric. The halving is stated with the
result rather than folded into it, and the asymmetry is exactly what a second
loopback in the other direction would measure if a result ever turned on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from service.audio import to_array, to_pcm

RATE = 8000
CHIRP_MS = 120.0
F_START = 400.0
F_END = 3000.0


def chirp(rate: int = RATE, duration_ms: float = CHIRP_MS,
          f_start: float = F_START, f_end: float = F_END, level: float = 0.5) -> bytes:
    """A linear frequency sweep across the telephone band, as int16 PCM.

    Bounded inside 300-3400 Hz so the band-pass a phone network applies does not
    remove the ends of the sweep and blunt the correlation peak.
    """
    n = int(rate * duration_ms / 1000.0)
    t = np.arange(n) / rate
    seconds = duration_ms / 1000.0
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * seconds))
    # Tapered ends: an abrupt start is a click, which spreads energy outside the
    # band and comes back as ringing that widens the peak.
    taper = np.minimum(1.0, np.minimum(t, seconds - t) / 0.01)
    return to_pcm(np.clip(np.sin(phase) * taper * level, -1, 1) * 32767)


@dataclass(frozen=True)
class Alignment:
    """Where a reference signal sits inside a recording, and how sure that is."""

    offset_ms: float | None
    peak: float                 # normalized correlation at the winning alignment
    runner_up: float            # best peak at least one chirp-length away

    @property
    def found(self) -> bool:
        return self.offset_ms is not None

    @property
    def margin(self) -> float:
        """How far the winner beat the best unrelated alignment.

        A high peak is not on its own evidence: noise that happens to resemble a
        sweep scores high everywhere. A high peak with nothing near it is.
        """
        return self.peak - self.runner_up


def find(recording: bytes | np.ndarray, reference: bytes | np.ndarray,
         rate: int = RATE, min_peak: float = 0.25, min_margin: float = 0.08) -> Alignment:
    """Locate ``reference`` inside ``recording`` by normalized cross-correlation.

    Normalized per alignment rather than globally: a phone leg's level drifts,
    and an unnormalized correlation would then prefer the loudest stretch of the
    recording over the one that actually matches.
    """
    haystack = to_array(recording).astype(np.float64) if isinstance(recording, (bytes, bytearray)) else np.asarray(recording, dtype=np.float64)
    needle = to_array(reference).astype(np.float64) if isinstance(reference, (bytes, bytearray)) else np.asarray(reference, dtype=np.float64)
    if haystack.size < needle.size or needle.size == 0:
        return Alignment(None, 0.0, 0.0)

    needle = needle - needle.mean()
    needle_norm = np.sqrt(np.sum(needle**2))
    if needle_norm == 0:
        return Alignment(None, 0.0, 0.0)

    correlation = np.correlate(haystack, needle, mode="valid")
    # Per-alignment energy via a sliding sum, so each lag is normalized by the
    # energy of the window it actually overlaps.
    squares = np.concatenate([[0.0], np.cumsum(haystack**2)])
    window = squares[needle.size :] - squares[: -needle.size]
    means = (np.concatenate([[0.0], np.cumsum(haystack)])[needle.size :] -
             np.concatenate([[0.0], np.cumsum(haystack)])[: -needle.size]) / needle.size
    energy = np.sqrt(np.maximum(window - needle.size * means**2, 1e-12))
    scores = correlation / (energy * needle_norm)

    best = int(np.argmax(scores))
    guard = needle.size
    masked = scores.copy()
    masked[max(0, best - guard) : best + guard] = -np.inf
    runner_up = float(masked.max()) if np.isfinite(masked).any() else 0.0
    peak = float(scores[best])
    if peak < min_peak or peak - runner_up < min_margin:
        return Alignment(None, peak, runner_up)
    return Alignment(offset_ms=best * 1000.0 / rate, peak=peak, runner_up=runner_up)


@dataclass(frozen=True)
class RoundTrip:
    """One loopback measurement, with everything needed to reject it."""

    round_trip_ms: float
    one_way_ms: float           # half the round trip; symmetry is an assumption
    peak: float
    margin: float


def round_trip(sent_at_ms: float, recording: bytes, reference: bytes,
               rate: int = RATE, **kw) -> RoundTrip | None:
    """Round trip from when the chirp's first sample left to where it came back.

    ``sent_at_ms`` and the recording share one clock -- both are ours -- which is
    what makes the difference meaningful without any assumption about the
    carrier's own timestamps.
    """
    alignment = find(recording, reference, rate, **kw)
    if not alignment.found:
        return None
    delta = alignment.offset_ms - sent_at_ms
    if delta < 0:
        return None
    return RoundTrip(
        round_trip_ms=delta,
        one_way_ms=delta / 2.0,
        peak=alignment.peak,
        margin=alignment.margin,
    )
