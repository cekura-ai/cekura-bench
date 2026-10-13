"""Where the speech is inside a rendered clip, pause by pause.

The service bench's detector (``service/detector.py``) answers "where does the
speech in this clip start and end?" with ``speech_bounds``: frame energy in dB
against a noise floor estimated from the quiet frames, a margin above it, and
a few frames of evidence. Authored lines also contain *internal* pauses — the
``<break/>`` markers — so this module reuses the detector's frame energy, noise
floor and constants (10 ms frames, floor + 8 dB, 3 frames of onset evidence,
200 ms of known-silent padding) and adds the split into segments, two ways:

* ``speech_segments`` — threshold only: runs of loud frames separated by less
  than ``min_pause_s`` of quiet are one segment, longer quiet splits them.
* ``scripted_segments`` — count-driven, what the build uses: the script says
  how many pauses a line has, so the ``n - 1`` longest quiet gaps are the
  pauses and every other gap is inside a segment. Each chosen gap must be at
  least ``MIN_PAUSE_S`` (a break that rendered shorter is not a pause a model
  could hear, and the line is rejected), and the longest unchosen gap must be
  clearly shorter than the shortest chosen one (``AMBIGUITY_RATIO``), so a
  natural sentence pause can never be mistaken for a break. A fixed threshold
  alone cannot do this: real TTS puts 0.6-0.8 s between sentences, which is as
  long as a short break rendered a little short.

Boundaries are reported on the 10 ms frame grid: a segment starts at the first
frame whose energy clears the threshold and ends at the end of the last one
that does, so on clean audio they are exact to within one frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from turn_taking_bench.author import service_bridge

MIN_PAUSE_S = 0.6
AMBIGUITY_RATIO = 0.75  # an unscripted gap longer than this fraction of a scripted pause is ambiguous


class BoundsError(ValueError):
    """The render does not hold the pauses the script asked for."""


@dataclass(frozen=True)
class Segment:
    """One run of speech inside a clip, seconds from the clip's first sample."""

    start_s: float
    end_s: float


@dataclass(frozen=True)
class Bounds:
    segments: list[Segment]
    noise_floor_db: float
    threshold_db: float
    frame_s: float
    # Quiet gaps of at least the pause minimum that lie *inside* a segment
    # (``scripted_segments`` only): natural pauses the script did not ask for.
    # They stay unlabelled, as in TurnBench's annotation (a turn is one turn), so
    # a model firing there is neither rewarded nor penalised.
    unscripted_pauses: list[Segment] = field(default_factory=list)


def speech_segments(
    samples: np.ndarray,
    sample_rate: int,
    *,
    min_pause_s: float = MIN_PAUSE_S,
    margin_db: float | None = None,
) -> Bounds:
    """Speech segments of a float32 clip, split at pauses of ``min_pause_s`` or more."""
    det = service_bridge.detector()
    frame_ms = det.FRAME_MS
    margin = det.MARGIN_DB if margin_db is None else margin_db
    pad = np.zeros(int(sample_rate * det.BOUNDS_PAD_MS / 1000.0))
    scaled = np.asarray(samples, dtype=np.float64) * 32768.0  # the detector is int16-scaled
    padded = np.concatenate([pad, scaled, pad])
    energy = det.frame_energy_db(padded, sample_rate, frame_ms)
    floor = det.noise_floor_db(energy)
    threshold = floor + margin
    loud = energy > threshold

    runs: list[list[int]] = []  # [first_frame, last_frame_exclusive]
    index = 0
    while index < loud.size:
        if not loud[index]:
            index += 1
            continue
        start = index
        while index < loud.size and loud[index]:
            index += 1
        if index - start >= det.ONSET_FRAMES:
            runs.append([start, index])
    min_gap_frames = int(round(min_pause_s * 1000.0 / frame_ms))
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < min_gap_frames:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    pad_frames = det.BOUNDS_PAD_MS / frame_ms
    frame_s = frame_ms / 1000.0
    segments = [
        Segment(start_s=(start - pad_frames) * frame_s, end_s=(end - pad_frames) * frame_s)
        for start, end in merged
    ]
    return Bounds(segments=segments, noise_floor_db=floor, threshold_db=threshold, frame_s=frame_s)


def scripted_segments(
    samples: np.ndarray,
    sample_rate: int,
    n_segments: int,
    *,
    min_pause_s: float = MIN_PAUSE_S,
    ambiguity_ratio: float = AMBIGUITY_RATIO,
    margin_db: float | None = None,
) -> Bounds:
    """Exactly ``n_segments`` segments: split at the ``n_segments - 1`` longest
    quiet gaps between runs of speech (see module docstring for the rules).

    Raises ``BoundsError`` when a chosen gap is shorter than ``min_pause_s`` or
    an unchosen gap is not clearly shorter than the chosen ones.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    raw = speech_segments(samples, sample_rate, min_pause_s=0.0, margin_db=margin_db)
    runs = raw.segments
    if not runs:
        raise BoundsError("no speech detected in the render")
    gaps = [(runs[i + 1].start_s - runs[i].end_s, i) for i in range(len(runs) - 1)]
    wanted = n_segments - 1
    if wanted > len(gaps):
        raise BoundsError(f"{n_segments} scripted segment(s) but only {len(runs)} run(s) of speech detected")
    by_length = sorted(gaps, key=lambda g: (-g[0], g[1]))
    chosen, unchosen = by_length[:wanted], by_length[wanted:]
    if chosen:
        shortest = min(length for length, _ in chosen)
        if shortest < min_pause_s:
            short = [f"{length:.2f}s" for length, _ in sorted(chosen, key=lambda g: g[1]) if length < min_pause_s]
            raise BoundsError(
                f"{wanted} scripted pause(s) but only {sum(1 for l, _ in chosen if l >= min_pause_s)} gap(s) of "
                f">= {min_pause_s:.2f}s quiet detected; the shortest chosen gap(s) {short} are below the minimum"
            )
        if unchosen and unchosen[0][0] > ambiguity_ratio * shortest:
            raise BoundsError(
                f"ambiguous pauses: an unscripted gap of {unchosen[0][0]:.2f}s is not clearly shorter than the "
                f"shortest scripted pause of {shortest:.2f}s (ratio > {ambiguity_ratio})"
            )
    split_after = {i for _, i in chosen}
    segments: list[Segment] = []
    start = runs[0].start_s
    for index, run in enumerate(runs):
        if index in split_after:
            segments.append(Segment(start_s=start, end_s=run.end_s))
            start = runs[index + 1].start_s
    segments.append(Segment(start_s=start, end_s=runs[-1].end_s))
    unscripted = [
        Segment(start_s=runs[i].end_s, end_s=runs[i + 1].start_s)
        for length, i in sorted(unchosen, key=lambda g: g[1])
        if length >= min_pause_s
    ]
    return Bounds(
        segments=segments,
        noise_floor_db=raw.noise_floor_db,
        threshold_db=raw.threshold_db,
        frame_s=raw.frame_s,
        unscripted_pauses=unscripted,
    )
