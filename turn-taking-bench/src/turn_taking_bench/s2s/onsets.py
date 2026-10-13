"""Reading an S2S model's turn-taking decisions off the audio it produced.

Pure functions over recorded artifacts — the model's output PCM, the service
bench's arrival-anchored ``AudioTimeline`` for it, the normalized event log,
and the caller PCM that was sent — so the readout can be recomputed from a
run's files by anyone, and tested without a provider.

The rules, each the service bench's or TurnBench's rather than new:

* **Responses.** The adapter marks the first chunk of every reply
  (``agent.audio.start``, with its sample index); a response spans from that
  sample to the next response's first sample.
* **Speech segments inside a response** use the service detector's frame
  energy against its noise-floor estimate (``service.detector``: 10 ms frames,
  3 frames of evidence for an onset, 200 ms of quiet for an offset, 8 dB
  margin, 200 ms of zero padding so the floor estimate has known-quiet
  material). Segments closer than ``merge_gap_s`` are one segment (TurnBench's
  ``MERGE_GAP_S`` in ``baselines/gemini_vad``).
* **When an onset is audible.** The first segment of a response is anchored on
  *arrival*: sample ``k`` of the chunk that arrived at ``t`` is audible at
  ``t + k/rate`` (nothing was queued ahead of a reply's first sample). A later
  segment in the same response is anchored on the *playout* clock, because the
  provider ships audio faster than real time and a listener hears those samples
  only when the player reaches them. A segment the player never reached (cut by
  the provider's interrupt signal) is not an onset.
* **Caller activity.** An onset while the caller's own channel is active is the
  model talking over the caller, not a decision that the turn had ended, and is
  dropped (TurnBench's ``gemini_vad`` rule). Activity comes from the same energy
  detector on the caller PCM that was sent, without the offset hysteresis (the
  caller is active while a frame is loud, not for 200 ms after), merged over
  gaps shorter than ``caller_merge_gap_s`` (0 by default; TurnBench's 0.5 s would
  hide a cut-in during a short scripted pause).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from turn_taking_bench.author import service_bridge

MERGE_GAP_S = 0.5


@dataclass(frozen=True)
class Segment:
    """A stretch of model speech inside one response, in frames of the response's audio."""

    start_frame: int
    end_frame: int  # exclusive


@dataclass(frozen=True)
class Onset:
    """One onset of model speech, on the run clock (seconds since the session's origin)."""

    t: float
    sample: int  # index into the model's output PCM
    response: int  # which response (0-based) the onset belongs to
    segment: int  # which speech segment of that response
    anchor: str  # "arrival" | "playout"
    lead_silence_ms: float  # silence the provider shipped before this speech
    first_chunk_ms: float  # size of the chunk the response began with (batching)
    chunk_arrival_t: float

    def as_json(self) -> dict[str, Any]:
        record = asdict(self)
        for key in ("t", "chunk_arrival_t"):
            record[key] = round(record[key], 6)
        return record


# ---- speech segments ---------------------------------------------------------------


def speech_segments(
    pcm: np.ndarray | bytes,
    rate: int,
    *,
    merge_gap_s: float = MERGE_GAP_S,
    frame_ms: float | None = None,
    onset_frames: int | None = None,
    offset_frames: int | None = None,
    margin_db: float | None = None,
) -> list[Segment]:
    """Sustained speech runs in ``pcm`` (int16), as frame indices of the unpadded
    audio, with the service detector's constants unless overridden."""
    detector = service_bridge.detector()
    frame_ms = detector.FRAME_MS if frame_ms is None else frame_ms
    onset_frames = detector.ONSET_FRAMES if onset_frames is None else onset_frames
    offset_frames = detector.OFFSET_FRAMES if offset_frames is None else offset_frames
    margin_db = detector.MARGIN_DB if margin_db is None else margin_db

    samples = (np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray)) else np.asarray(pcm)).astype(np.float64)
    pad_frames = int(round(detector.BOUNDS_PAD_MS / frame_ms))
    pad = np.zeros(pad_frames * int(rate * frame_ms / 1000.0))
    energy = detector.frame_energy_db(np.concatenate([pad, samples, pad]), rate, frame_ms)
    floor = detector.noise_floor_db(energy)
    loud = energy > floor + margin_db
    n_frames = loud.size - 2 * pad_frames

    segments: list[Segment] = []
    speaking = False
    start = 0
    index = 0
    while index < loud.size:
        if not speaking:
            if loud[index : index + onset_frames].all() and index + onset_frames <= loud.size:
                speaking, start = True, index
                index += onset_frames
            else:
                index += 1
        else:
            if not loud[index : index + offset_frames].any() and index + offset_frames <= loud.size:
                segments.append(Segment(start, index))
                speaking = False
                index += offset_frames
            else:
                index += 1
    if speaking:
        segments.append(Segment(start, loud.size))

    # Back to the unpadded frame grid; drop anything that lies in the padding.
    shifted = []
    for segment in segments:
        begin = max(0, segment.start_frame - pad_frames)
        end = min(n_frames, segment.end_frame - pad_frames)
        if end > begin:
            shifted.append(Segment(begin, end))
    return merge_segments(shifted, int(round(merge_gap_s * 1000.0 / frame_ms)))


def merge_segments(segments: list[Segment], gap_frames: int) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if merged and segment.start_frame - merged[-1].end_frame < gap_frames:
            merged[-1] = Segment(merged[-1].start_frame, max(merged[-1].end_frame, segment.end_frame))
        else:
            merged.append(segment)
    return merged


# ---- responses and onsets ------------------------------------------------------------


def response_starts(events: Iterable[Any]) -> list[int]:
    """First-sample index of every response, from the adapter's ``agent.audio.start`` events."""
    ev = service_bridge.events()
    return [int(e.data["sample"]) for e in events if e.kind == ev.AGENT_AUDIO_START]


def output_onsets(
    agent_pcm: bytes,
    timeline: Any,
    events: Iterable[Any],
    *,
    merge_gap_s: float = MERGE_GAP_S,
    frame_ms: float | None = None,
) -> list[Onset]:
    """Every audible onset of model speech in the recorded output, on the run clock."""
    detector = service_bridge.detector()
    frame_ms = detector.FRAME_MS if frame_ms is None else frame_ms
    rate = timeline.rate
    frame_samples = int(rate * frame_ms / 1000.0)
    starts = response_starts(events)
    total = len(agent_pcm) // 2
    onsets: list[Onset] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else total
        if end <= start:
            continue
        pcm = agent_pcm[start * 2 : end * 2]
        first_chunk = timeline.chunk_of_sample(start)
        for seg_index, segment in enumerate(speech_segments(pcm, rate, merge_gap_s=merge_gap_s, frame_ms=frame_ms)):
            sample = start + segment.start_frame * frame_samples
            chunk = timeline.chunk_of_sample(sample)
            if chunk is None:
                continue
            if seg_index == 0:
                at = timeline.time_of_sample(sample)
                anchor = "arrival"
            else:
                played_start, played_end = timeline.playout_span(chunk.seq)
                if played_end <= played_start:
                    continue  # delivered, then discarded before anyone heard it
                at = timeline.playout_time_of_sample(sample)
                anchor = "playout"
            if at is None:
                continue
            onsets.append(
                Onset(
                    t=float(at),
                    sample=int(sample),
                    response=index,
                    segment=seg_index,
                    anchor=anchor,
                    lead_silence_ms=round(segment.start_frame * frame_ms, 1) if seg_index == 0 else 0.0,
                    first_chunk_ms=round(1000.0 * first_chunk.n_samples / rate, 1) if first_chunk else 0.0,
                    chunk_arrival_t=float(chunk.t_wall),
                )
            )
    return sorted(onsets, key=lambda o: o.t)


# ---- caller activity -----------------------------------------------------------------


def active_mask(
    pcm: np.ndarray | bytes,
    rate: int,
    *,
    merge_gap_s: float = 0.0,
    frame_ms: float | None = None,
    onset_frames: int | None = None,
    margin_db: float | None = None,
) -> np.ndarray:
    """Per-frame "the caller is speaking" over the caller PCM that was sent: loud
    frames (energy above the detector's floor + margin) that belong to a run of
    at least ``onset_frames``, gaps shorter than ``merge_gap_s`` filled."""
    detector = service_bridge.detector()
    frame_ms = detector.FRAME_MS if frame_ms is None else frame_ms
    onset_frames = detector.ONSET_FRAMES if onset_frames is None else onset_frames
    margin_db = detector.MARGIN_DB if margin_db is None else margin_db
    samples = (np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray)) else np.asarray(pcm)).astype(np.float64)
    energy = detector.frame_energy_db(samples, rate, frame_ms)
    if energy.size == 0:
        return np.zeros(0, dtype=bool)
    loud = energy > detector.noise_floor_db(energy) + margin_db
    mask = np.zeros_like(loud)
    index = 0
    while index < loud.size:
        if loud[index]:
            end = index
            while end < loud.size and loud[end]:
                end += 1
            if end - index >= onset_frames:
                mask[index:end] = True
            index = end
        else:
            index += 1
    gap_frames = int(round(merge_gap_s * 1000.0 / frame_ms))
    if gap_frames > 0:
        index = 0
        while index < mask.size:
            if not mask[index]:
                end = index
                while end < mask.size and not mask[end]:
                    end += 1
                if 0 < index and end < mask.size and end - index < gap_frames:
                    mask[index:end] = True
                index = end
            else:
                index += 1
    return mask


def active_at(mask: np.ndarray, t_audio_s: float, frame_ms: float | None = None) -> bool:
    detector = service_bridge.detector()
    frame_ms = detector.FRAME_MS if frame_ms is None else frame_ms
    index = int(t_audio_s * 1000.0 / frame_ms)
    return bool(0 <= index < mask.size and mask[index])
