"""PCM handling and the sample-to-wall-clock mapping every service-bench latency rests on.

A realtime API is a stream of chunks, and a chunk is not an instant. If a boundary
is reported as "the arrival time of the chunk it fell in", a provider that ships
500 ms of audio in one frame is credited with speaking 500 ms earlier than it did,
and a provider that ships 20 ms frames is not. That difference is larger than most
of the gaps this benchmark is trying to resolve, so the timeline below carries the
in-chunk sample offset and every published boundary goes through it.

The convention, stated once and published with the results:

* **Outbound** — a chunk handed to the socket at ``t`` carries its first sample at
  ``t``; sample ``k`` within it is at ``t + k/rate``. We pace sends in realtime, so
  this is also when the provider could first have heard it.
* **Inbound** — a chunk that arrives at ``t`` is playable from ``t``; sample ``k``
  within it reaches a listener's ear at ``t + k/rate``. Arrival, not generation, is
  the anchor: a provider that batches its first response into one large frame *is*
  later for the listener, and is scored later.
"""

from __future__ import annotations

import bisect
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SAMPLE_WIDTH = 2  # int16 mono everywhere; the providers all speak this


# ── the timeline ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    """One transfer, and where it sits in the stream."""

    seq: int
    first_sample: int
    n_samples: int
    t_wall: float  # monotonic seconds: send time outbound, arrival time inbound

    @property
    def last_sample(self) -> int:
        return self.first_sample + self.n_samples


class AudioTimeline:
    """Sample index -> wall clock, for one direction of one stream."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.chunks: list[Chunk] = []
        self._starts: list[int] = []  # parallel to chunks, for bisect
        self._playout: list[float] = []      # parallel to chunks: when a realtime player starts each one
        self._playout_end: list[float] = []  # ...and when it stops playing it (early, if cut)
        self.cuts: list[float] = []          # instants the player discarded what it held
        self._n_samples = 0

    def record(self, n_samples: int, t_wall: float) -> Chunk:
        chunk = Chunk(len(self.chunks), self._n_samples, n_samples, t_wall)
        # A player that is still busy with the previous chunk cannot start this
        # one at its arrival; it starts when the previous one drains.
        previous_end = self._playout_end[-1] if self.chunks else 0.0
        start = max(t_wall, previous_end)
        self._playout.append(start)
        self._playout_end.append(start + n_samples / self.rate)
        self.chunks.append(chunk)
        self._starts.append(chunk.first_sample)
        self._n_samples += n_samples
        return chunk

    def cut(self, at: float) -> None:
        """The player discarded everything it was holding at ``at``.

        This is what a client does on the provider's interrupt signal, and it
        is the only way a provider that ships a whole reply in one burst can be
        heard to stop. Chunks playing at the cut end there; chunks queued behind
        it are never heard, and the next arrival starts fresh.
        """
        self.cuts.append(at)
        for index, (start, end) in enumerate(zip(self._playout, self._playout_end)):
            if end <= at:
                continue                       # finished before the cut
            if start >= at:                    # queued behind it: never heard
                self._playout[index] = at
            self._playout_end[index] = at      # playing at the cut: ends there

    def record_pcm(self, pcm: bytes, t_wall: float) -> Chunk:
        return self.record(len(pcm) // SAMPLE_WIDTH, t_wall)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def duration_s(self) -> float:
        return self._n_samples / self.rate

    def chunk_of_sample(self, index: int) -> Chunk | None:
        if not self.chunks or index < 0 or index >= self._n_samples:
            return None
        return self.chunks[bisect.bisect_right(self._starts, index) - 1]

    def time_of_sample(self, index: int) -> float | None:
        """When a listener reaches ``index``. None if the stream never got there."""
        chunk = self.chunk_of_sample(index)
        if chunk is None:
            return None
        return chunk.t_wall + (index - chunk.first_sample) / self.rate

    # -- the realtime-player view -----------------------------------------
    #
    # ``time_of_sample`` is when a sample could first be heard: the onset anchor.
    # Providers ship audio faster than realtime, so the end of speech is taken
    # from a player that starts each chunk at its arrival or when the previous
    # one drains, whichever is later.

    def playout_start(self, chunk_index: int) -> float:
        return self._playout[chunk_index]

    def playout_time_of_sample(self, index: int) -> float | None:
        """When a realtime player, buffering nothing away, reaches ``index``."""
        chunk = self.chunk_of_sample(index)
        if chunk is None:
            return None
        return self._playout[chunk.seq] + (index - chunk.first_sample) / self.rate

    def playout_span(self, chunk_index: int) -> tuple[float, float]:
        """When the player starts and stops a chunk; equal if it was never heard."""
        return self._playout[chunk_index], self._playout_end[chunk_index]

    def playout_end(self) -> float | None:
        """When such a player goes quiet. None if nothing was ever received."""
        return self._playout_end[-1] if self.chunks else None

    def discarded_s(self, first_chunk: int = 0, last_chunk: int | None = None) -> float:
        """Audio delivered in a range of chunks that a cut kept the listener from hearing."""
        stop = len(self.chunks) if last_chunk is None else last_chunk + 1
        total = 0.0
        for index in range(first_chunk, stop):
            nominal = self.chunks[index].n_samples / self.rate
            total += nominal - (self._playout_end[index] - self._playout[index])
        return max(total, 0.0)

    # -- persistence ------------------------------------------------------
    #
    # Per-chunk timestamps are what make a published latency reproducible from
    # the artifacts: one row per chunk, four numbers, in the order below.

    FIELDS = ("seq", "first_sample", "n_samples", "t_wall")

    def as_json(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "fields": list(self.FIELDS),
            "chunks": [[c.seq, c.first_sample, c.n_samples, round(c.t_wall, 6)] for c in self.chunks],
            "cuts": [round(t, 6) for t in self.cuts],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "AudioTimeline":
        """Rebuilt in arrival order with the cuts replayed where they fell."""
        timeline = cls(int(payload["rate"]))
        cuts = sorted(float(t) for t in payload.get("cuts", []))
        for _seq, _first, n_samples, t_wall in payload["chunks"]:
            while cuts and cuts[0] <= float(t_wall):
                timeline.cut(cuts.pop(0))
            timeline.record(int(n_samples), float(t_wall))
        for at in cuts:
            timeline.cut(at)
        return timeline


# ── pcm plumbing ─────────────────────────────────────────────────────────────

def iter_chunks(pcm: bytes, rate: int, chunk_ms: float = 20.0) -> Iterator[bytes]:
    """Split for realtime sending. 20 ms keeps pacing error inside detector error."""
    step = max(SAMPLE_WIDTH, int(rate * chunk_ms / 1000.0) * SAMPLE_WIDTH)
    for start in range(0, len(pcm), step):
        yield pcm[start : start + step]


def silence(rate: int, duration_ms: float) -> bytes:
    return b"\x00\x00" * int(round(rate * duration_ms / 1000.0))


def to_array(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2")


def to_pcm(samples: np.ndarray) -> bytes:
    return np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()


def read_wav(path: str | Path) -> tuple[bytes, int]:
    """16-bit mono PCM out of a wav file, with its rate."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != SAMPLE_WIDTH or handle.getnchannels() != 1:
            raise ValueError(f"{path}: benchmark audio is 16-bit mono only")
        return handle.readframes(handle.getnframes()), handle.getframerate()


def write_wav(path: str | Path, pcm: bytes, rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(rate)
        handle.writeframes(pcm)


# ── resampling ───────────────────────────────────────────────────────────────
#
# One canonical master per clip, resampled to each provider's rate by the
# function below. Published with the corpus, because a benchmark that ships
# differently-filtered audio to different providers is measuring its own
# resampler. Kaiser-windowed polyphase FIR: deterministic, no scipy.

def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Rational resample by the published filter. A no-op when the rates match."""
    if src_rate == dst_rate:
        return pcm
    samples = to_array(pcm).astype(np.float64)
    if samples.size == 0:
        return b""
    divisor = np.gcd(src_rate, dst_rate)
    up, down = dst_rate // divisor, src_rate // divisor
    taps = 2 * 32 * max(up, down) + 1
    n = np.arange(taps) - (taps - 1) / 2.0
    cutoff = 1.0 / max(up, down)
    h = np.sinc(cutoff * n) * np.kaiser(taps, 8.6)
    h *= up / h.sum()
    stuffed = np.zeros(samples.size * up, dtype=np.float64)
    stuffed[::up] = samples
    filtered = np.convolve(stuffed, h, mode="same")
    return to_pcm(filtered[::down])


# ── mu-law, for the telephony leg ────────────────────────────────────────────
#
# G.711 by table rather than stdlib ``audioop``: that module is deprecated in
# 3.12 and gone in 3.13, and a benchmark should not carry a removal date.

_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635


def pcm_to_ulaw(pcm: bytes) -> bytes:
    samples = to_array(pcm).astype(np.int32)
    sign = (samples < 0).astype(np.uint8) * 0x80
    magnitude = np.minimum(np.abs(samples), _ULAW_CLIP) + _ULAW_BIAS
    exponent = np.maximum(np.floor(np.log2(np.maximum(magnitude, 1))).astype(np.int32) - 7, 0)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4).astype(np.uint8) | mantissa.astype(np.uint8))).astype(np.uint8).tobytes()


def ulaw_to_pcm(ulaw: bytes) -> bytes:
    encoded = ~np.frombuffer(ulaw, dtype=np.uint8).astype(np.int32)
    sign, exponent, mantissa = encoded & 0x80, (encoded >> 4) & 0x07, encoded & 0x0F
    magnitude = ((mantissa << 3) + _ULAW_BIAS) << exponent
    return to_pcm(np.where(sign, _ULAW_BIAS - magnitude, magnitude - _ULAW_BIAS))


# ── noise beds ───────────────────────────────────────────────────────────────
#
# Synthesized rather than sampled, and that is the honest choice here: a noise
# bed carries no content, so there is nothing a recording would supply that a
# seeded generator does not, and a generator is reproducible by anyone who reads
# the code.

def pink_noise(rate: int, duration_ms: float, level_dbfs: float = -30.0, seed: int = 0) -> bytes:
    """1/f noise at a stated RMS level. Deterministic for a given seed."""
    n = int(round(rate * duration_ms / 1000.0))
    if n <= 0:
        return b""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    shaped = spectrum / np.sqrt(np.maximum(freqs, freqs[1] if freqs.size > 1 else 1.0))
    noise = np.fft.irfft(shaped, n=n)
    rms = np.sqrt(np.mean(noise**2)) or 1.0
    return to_pcm(noise / rms * (32768.0 * 10 ** (level_dbfs / 20.0)))


def mix(base: bytes, overlay: bytes, gain: float = 1.0) -> bytes:
    """Sum two PCM buffers, looping the overlay to cover the base."""
    a = to_array(base).astype(np.float64)
    b = to_array(overlay).astype(np.float64)
    if b.size == 0 or a.size == 0:
        return base
    tiled = np.resize(b, a.size) * gain
    return to_pcm(a + tiled)
