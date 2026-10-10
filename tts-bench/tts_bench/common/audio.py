"""PCM handling and the sample-to-wall-clock mapping every latency here rests on.

A streaming TTS API is a sequence of chunks, and a chunk is not an instant. If a
boundary is reported as "the arrival time of the chunk it fell in", a provider that
ships 500 ms of audio in one frame is credited with speaking 500 ms earlier than it
did, and a provider that ships 20 ms frames is not. The timeline below carries the
in-chunk sample offset, and every published boundary goes through it.

Inbound convention, published with the results: a chunk that arrives at ``t`` is
playable from ``t``; sample ``k`` within it reaches a listener's ear at
``t + k/rate``. Arrival, not generation, is the anchor.
"""

from __future__ import annotations

import bisect
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_WIDTH = 2  # int16 mono everywhere; the providers all speak this


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

    def time_of_ms(self, offset_ms: float) -> float | None:
        return self.time_of_sample(int(round(offset_ms * self.rate / 1000.0)))

    # -- the realtime-player view -----------------------------------------
    #
    # ``time_of_sample`` says when a sample *could* first be heard: it is the
    # anchor for onset, where nothing is queued ahead of it. It is the wrong
    # clock for when speech *ends*. Providers ship audio faster than realtime,
    # so the last chunk of a reply can arrive seconds before a listener reaches
    # it, and a reply's end measured from arrival would be credited too early --
    # which would score a provider that dumps its whole reply in one burst as
    # having yielded the floor before the listener heard it. The playout view
    # follows a player that starts each chunk at its arrival or when the previous
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
    # Without this the published artifacts cannot reproduce a published number.
    # The wav says what the audio was and the event log says what happened, but
    # only the per-chunk timestamps say *when each sample became audible*, and
    # every latency in this benchmark is a difference between two of those.
    # Stored as plain arrays rather than objects: one row per chunk, four numbers,
    # in the order below.

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


def to_array(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2")


def to_pcm(samples: np.ndarray) -> bytes:
    return np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()


def write_wav(path: str | Path, pcm: bytes, rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(rate)
        handle.writeframes(pcm)
