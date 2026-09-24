"""Every published TTS number, computed from one ``Synthesis`` record.

Nothing here needs the provider: the inputs are the per-chunk arrival times and
the PCM, both of which every cell writes to disk, so anyone holding the
artifacts can recompute the row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tts_bench.common.audio import AudioTimeline
from tts_bench.common.detector import detect_onset

from tts_bench.adapters.base import Synthesis

# Leading-silence detector. Fixed, published, deliberately simple: the first
# 10 ms window whose DC-removed RMS exceeds 1% of full scale, scanned at 1 ms
# hops. A fixed threshold is comparable across providers in a way an adaptive
# one is not, and it sits above every noise floor seen in provider output while
# well below any voiced onset.
AUDIBLE_RMS = 0.01
FRAME_S = 0.010
HOP_S = 0.001
PCM16_FULL_SCALE = 32768.0


def first_audible_offset_ms(pcm: bytes, rate: int) -> float | None:
    """Milliseconds of stream before the first audible sample; None if nothing is audible."""
    if not pcm or rate <= 0:
        return None
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / PCM16_FULL_SCALE
    frame = max(1, round(FRAME_S * rate))
    hop = max(1, round(HOP_S * rate))
    if samples.size < frame:
        return None
    pad = frame // 2
    padded = np.pad(samples, pad, mode="edge")
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    rms = np.sqrt(np.mean((frames - frames.mean(axis=1, keepdims=True)) ** 2, axis=1))
    audible = np.flatnonzero(rms > AUDIBLE_RMS)
    if audible.size == 0:
        return None
    return float(audible[0]) * hop / rate * 1000.0


def trailing_silence_ms(pcm: bytes, rate: int) -> float | None:
    """Same rule from the end: how much silence the provider pads onto the tail."""
    if not pcm:
        return None
    reversed_pcm = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2")[::-1].tobytes()
    return first_audible_offset_ms(reversed_pcm, rate)


def _adaptive_onset_ms(pcm: bytes, rate: int) -> float | None:
    if len(pcm) < rate // 10:
        return None
    decision = detect_onset(pcm, rate)
    return round(decision.time_ms, 1) if decision.found else None


def _ms(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round((a - b) * 1000.0, 1)


def playout(timeline: AudioTimeline, onset_ms: float | None) -> dict[str, Any]:
    """How a realtime player fared, with its clock started at the first audible sample.

    Starting the clock at the request would count connection and generation
    latency as underrun, which TTFA already reports. Starting it at the first
    audible sample asks the only question left: once the listener hears speech,
    does the stream keep up? ``min_margin_ms`` is the closest any chunk came to
    its own deadline (negative = late); ``underruns`` counts chunks that missed;
    ``stall_ms`` is the total silence a listener would have heard as a result.
    """
    chunks = timeline.chunks
    if not chunks or onset_ms is None:
        return {"underruns": None, "min_margin_ms": None, "stall_ms": None}
    onset_sample = int(round(onset_ms * timeline.rate / 1000.0))
    first = timeline.chunk_of_sample(onset_sample)
    if first is None:
        return {"underruns": None, "min_margin_ms": None, "stall_ms": None}
    # The listener hears the onset sample when it arrives; every later sample is
    # due exactly its offset later. A chunk is late if it arrives after the
    # instant its first sample is due.
    clock_zero = first.t_wall + (onset_sample - first.first_sample) / timeline.rate
    margins: list[float] = []
    for chunk in chunks[first.seq + 1 :]:
        due = clock_zero + (chunk.first_sample - onset_sample) / timeline.rate
        margins.append((due - chunk.t_wall) * 1000.0)
    stall = 0.0
    for index in range(first.seq + 1, len(chunks)):
        start, _ = timeline.playout_span(index)
        _, previous_end = timeline.playout_span(index - 1)
        stall += max(0.0, start - previous_end)
    return {
        "underruns": sum(1 for m in margins if m < 0),
        "min_margin_ms": round(min(margins), 1) if margins else None,
        "stall_ms": round(stall * 1000.0, 1),
    }


def summarize(synthesis: Synthesis) -> dict[str, Any]:
    """The full row for one synthesis. Missing pieces are None, never zero."""
    pcm = bytes(synthesis.pcm)
    rate = synthesis.rate
    onset = first_audible_offset_ms(pcm, rate)
    roundtrip = _ms(synthesis.t_first_chunk, synthesis.t0)
    audio_s = synthesis.timeline.duration_s
    generation_s = None
    if synthesis.t_first_chunk is not None and synthesis.t_last_chunk is not None:
        generation_s = synthesis.t_last_chunk - synthesis.t_first_chunk
    out: dict[str, Any] = {
        "chars": synthesis.chars_sent,
        "text_frames": len(synthesis.text_frames),
        "chunks": len(synthesis.timeline.chunks),
        "audio_ms": round(audio_s * 1000.0, 1),
        "roundtrip_ms": roundtrip,
        "leading_silence_ms": None if onset is None else round(onset, 1),
        # A second opinion from the adaptive detector (noise floor + margin,
        # 30 ms of evidence). Published beside the fixed rule so a disagreement
        # between the two is visible in the row rather than buried in a choice.
        "leading_silence_adaptive_ms": _adaptive_onset_ms(pcm, rate),
        "trailing_silence_ms": trailing_silence_ms(pcm, rate),
        # Perceived first-audio latency: the wait to the first chunk plus the
        # silence inside the stream before anything can be heard.
        "ttfa_ms": None if roundtrip is None or onset is None else round(roundtrip + onset, 1),
        "ttfa_from_input_done_ms": None if onset is None else (
            None if (v := _ms(synthesis.t_first_chunk, synthesis.t_input_done)) is None else round(v + onset, 1)
        ),
        "input_duration_ms": _ms(synthesis.t_input_done, synthesis.t0),
        "completion_ms": _ms(synthesis.t_done, synthesis.t0),
        "last_chunk_ms": _ms(synthesis.t_last_chunk, synthesis.t0),
        # Delivery speed relative to realtime: 0.25 = the whole utterance was
        # delivered in a quarter of its own duration.
        "generation_over_realtime": None if generation_s is None or audio_s <= 0 else round(generation_s / audio_s, 3),
        "audio_per_char_ms": None if synthesis.chars_sent == 0 else round(audio_s * 1000.0 / synthesis.chars_sent, 2),
        "ended_by": synthesis.meta.get("ended_by"),
        "finish_reason": synthesis.meta.get("finish_reason"),
        "error": synthesis.error,
        **playout(synthesis.timeline, onset),
    }
    if synthesis.t_cancel is not None:
        out.update(cancel(synthesis))
    return out


def cancel(synthesis: Synthesis) -> dict[str, Any]:
    """What happened after we asked the provider to stop."""
    after = [c for c in synthesis.timeline.chunks if c.t_wall > synthesis.t_cancel]
    audio_after_ms = sum(c.n_samples for c in after) / synthesis.rate * 1000.0
    last_after = max((c.t_wall for c in after), default=None)
    return {
        "cancel_at_ms": _ms(synthesis.t_cancel, synthesis.t0),
        "cancel_to_last_chunk_ms": 0.0 if last_after is None else round((last_after - synthesis.t_cancel) * 1000.0, 1),
        "audio_after_cancel_ms": round(audio_after_ms, 1),
        "chunks_after_cancel": len(after),
        "cancel_ack_ms": _ms(synthesis.t_cancel_ack, synthesis.t_cancel),
    }
