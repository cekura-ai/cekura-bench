"""Metrics: artifacts in, published numbers out.

Everything here is a pure function of the recorded audio and the event log, so a
sceptic can recompute a cell from the released artifacts, or disagree with our
detector and run their own over the same audio. Nothing reads provider telemetry.

The asymmetry worth restating: the caller-side boundary comes from
``Utterance.speech_end_sample`` -- audio written here, so it is exact -- and only
the agent side goes through the detector. Published latency therefore carries a
single detector error, about +-10 ms on clean audio, rather than one at each end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from lane_a import events as ev
from lane_a.adapters.base import RealtimeAdapter
from lane_a.audio import SAMPLE_WIDTH, AudioTimeline
from lane_a.caller import Utterance


@dataclass(frozen=True)
class AgentOnset:
    """When the agent's audio actually became audible to a listener."""

    t: float
    sample: int
    chunk_seq: int
    arrival_t: float           # when the containing chunk landed
    lead_silence_ms: float     # padding the provider shipped before its own speech
    first_chunk_ms: float      # batching: a big first frame is late for the listener

    @property
    def batching_penalty_ms(self) -> float:
        """How much of the onset is the provider's own framing rather than thinking."""
        return (self.t - self.arrival_t) * 1000.0


def agent_onset_after(adapter: RealtimeAdapter, after_t: float) -> AgentOnset | None:
    """Resolve the first agent speech arriving after ``after_t`` to the sample.

    Slices from the first chunk that landed after the boundary rather than
    scanning the whole stream, so a previous turn's audio cannot be mistaken for
    this turn's reply.
    """
    timeline: AudioTimeline = adapter.agent_timeline
    chunk = next((c for c in timeline.chunks if c.t_wall > after_t), None)
    if chunk is None:
        return None
    tail = bytes(adapter.agent_pcm[chunk.first_sample * SAMPLE_WIDTH :])
    bounds = adapter.speech_bounds(tail, timeline.rate)
    if not bounds.found:
        return None
    sample = chunk.first_sample + int(round(max(bounds.start_ms, 0.0) * timeline.rate / 1000.0))
    at = timeline.time_of_sample(sample)
    if at is None:
        return None
    return AgentOnset(
        t=at,
        sample=sample,
        chunk_seq=chunk.seq,
        arrival_t=chunk.t_wall,
        lead_silence_ms=max(bounds.start_ms, 0.0),
        first_chunk_ms=1000.0 * chunk.n_samples / timeline.rate,
    )


def speech_runs(adapter: RealtimeAdapter, gap_ms: float = 250.0) -> list[tuple[float, float]]:
    """Agent audio grouped into stretches a listener would hear, as (start, end).

    A realtime provider does not mark its own turns reliably -- a response
    cancelled in flight may never send a done event -- so stretches are derived
    from the audio itself, on the **playout** clock: each chunk plays at its
    arrival or when the previous one drains, whichever is later. On arrival
    times alone a provider that ships a reply in one burst would be credited
    with finishing seconds before anyone heard the end of it. A gap wider than
    ``gap_ms`` between one chunk draining and the next starting begins a new
    stretch.
    """
    runs: list[tuple[float, float]] = []
    timeline = adapter.agent_timeline
    for chunk in timeline.chunks:
        start, end = timeline.playout_span(chunk.seq)
        if end <= start:
            continue  # delivered, then discarded at a cut before anyone heard it
        if runs and (start - runs[-1][1]) * 1000.0 <= gap_ms:
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((start, end))
    return runs


def agent_audio_ends_at(adapter: RealtimeAdapter, after_t: float, gap_ms: float = 250.0) -> float | None:
    """When the stretch of speech that was under way at ``after_t`` finishes.

    Not "the last chunk after ``after_t``": once a caller barges in, the provider
    usually endpoints them and begins a *fresh* reply a few hundred milliseconds
    later. Counting that second reply as the tail of the first reports a provider
    that yielded the floor instantly as one that talked over the caller for the
    better part of a second -- which is exactly the direction a benchmark must
    not get wrong, since it turns a good behaviour into a bad score.
    """
    runs = [run for run in speech_runs(adapter, gap_ms) if run[0] <= after_t]
    return runs[-1][1] if runs else None


@dataclass(frozen=True)
class ResponseLatency:
    """The headline number, with enough context to argue about it."""

    latency_ms: float | None
    caller_end_t: float
    onset: AgentOnset | None
    turn_detection: str
    note: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "turn_detection": self.turn_detection,
            "lead_silence_ms": None if not self.onset else round(self.onset.lead_silence_ms, 1),
            "first_chunk_ms": None if not self.onset else round(self.onset.first_chunk_ms, 1),
            "batching_penalty_ms": None if not self.onset else round(self.onset.batching_penalty_ms, 1),
            "note": self.note,
        }


def response_latency(adapter: RealtimeAdapter, utterance: Utterance) -> ResponseLatency:
    """Authored speech-end to first audible agent sample."""
    caller_end_t = adapter.caller_timeline.time_of_sample(max(utterance.speech_end_sample - 1, 0))
    if caller_end_t is None:
        return ResponseLatency(None, 0.0, None, adapter.config.turn_detection.label, "caller boundary not on timeline")
    onset = agent_onset_after(adapter, caller_end_t)
    if onset is None:
        return ResponseLatency(None, caller_end_t, None, adapter.config.turn_detection.label, "no agent audio")
    return ResponseLatency(
        latency_ms=(onset.t - caller_end_t) * 1000.0,
        caller_end_t=caller_end_t,
        onset=onset,
        turn_detection=adapter.config.turn_detection.label,
    )


@dataclass(frozen=True)
class BargeIn:
    """Did the agent yield the floor, and how fast.

    ``stop_ms`` is what a listener experiences: caller onset to the agent going
    quiet, on the reference client -- one that plays audio in realtime and
    clears its buffer when the provider reports that the caller started
    speaking or that it interrupted its reply, which is what these protocols
    document a client should do. A provider whose endpointer reacts promptly
    stops the listener hearing it promptly, however much reply it had already
    pushed down the wire; one that never signals is heard to the end of
    whatever it sent, because a client has no other way to know.
    ``discarded_ms`` is the audio the signal threw away: delivered, paid for,
    never heard. ``cancelled_by_provider`` says whether the provider also
    stopped generating, which is its decision rather than the client's.
    """

    stopped: bool
    stop_ms: float | None            # caller onset -> the listener stops hearing the agent
    discarded_ms: float | None       # reply delivered but never heard, thanks to the interrupt
    cancelled_by_provider: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "stopped": self.stopped,
            "stop_ms": None if self.stop_ms is None else round(self.stop_ms, 1),
            "discarded_ms": None if self.discarded_ms is None else round(self.discarded_ms, 1),
            "provider_cancelled": self.cancelled_by_provider,
        }


def barge_in(adapter: RealtimeAdapter, utterance: Utterance, settle_ms: float = 2000.0) -> BargeIn:
    """Measured from the caller's first authored speech sample, not the clip start.

    Stop time is when the listener last heard the interrupted speech: the end of
    the latest stretch of agent audio that overlapped the caller's utterance,
    including one that resumed after a pause. Stretches that begin after the
    caller finished speaking are the agent's *reply* to the interruption, not the
    speech that was interrupted, and are excluded.
    """
    caller_start_t = adapter.caller_timeline.time_of_sample(utterance.speech_start_sample)
    caller_end_t = adapter.caller_timeline.time_of_sample(utterance.speech_end_sample) or caller_start_t
    if caller_start_t is None:
        return BargeIn(False, None, None, False)
    cancelled = any(e.kind == ev.AGENT_INTERRUPTED and e.t >= caller_start_t for e in adapter.log.events)
    overlapping = [run for run in speech_runs(adapter) if run[1] > caller_start_t and run[0] < caller_end_t]
    if not overlapping:
        # Nothing was audible from the caller's first word to their last: the
        # agent was already quiet, or stopped inside detector resolution.
        return BargeIn(True, 0.0, 0.0, cancelled)
    stop_ms = max((overlapping[-1][1] - caller_start_t) * 1000.0, 0.0)
    timeline = adapter.agent_timeline
    in_runs = [
        c.seq for c in timeline.chunks
        if any(run[0] <= timeline.playout_span(c.seq)[0] <= run[1] for run in overlapping)
    ]
    discarded_ms = 1000.0 * timeline.discarded_s(min(in_runs), max(in_runs)) if in_runs else 0.0
    return BargeIn(
        stopped=stop_ms < settle_ms,
        stop_ms=stop_ms,
        discarded_ms=discarded_ms,
        cancelled_by_provider=cancelled,
    )


def spoke_between(adapter: RealtimeAdapter, start_t: float, end_t: float) -> bool:
    """Any agent audio arriving inside a window. The endpointing ladder's verdict."""
    return any(start_t < c.t_wall <= end_t for c in adapter.agent_timeline.chunks)


def false_triggers(adapter: RealtimeAdapter, start_t: float, end_t: float) -> int:
    """Responses started while the caller was silent."""
    return sum(1 for e in adapter.log.events if e.kind == ev.AGENT_AUDIO_START and start_t < e.t <= end_t)


def _flatten_usage(payload: dict[str, Any], prefix: str = "") -> dict[str, int]:
    """Every count the provider reports, including the nested breakdowns.

    Audio and text tokens are priced differently, and the provider reports the
    split one level down. Summing only the top level gives a total that cannot be
    turned into a cost -- and cost per hour is a column this benchmark publishes.
    """
    flat: dict[str, int] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            flat[name] = value
        elif isinstance(value, dict):
            flat.update(_flatten_usage(value, f"{name}."))
    return flat


def usage(adapter: RealtimeAdapter) -> dict[str, Any]:
    """Token counts as the provider reports them. Cost is derived at publication."""
    totals: dict[str, int] = {}
    for event in adapter.log.of_kind(ev.RESPONSE_DONE):
        for key, value in _flatten_usage(event.data.get("usage") or {}).items():
            totals[key] = totals.get(key, 0) + value
    return totals


def percentiles(values: Sequence[float], points: Sequence[int] = (50, 90)) -> dict[str, float]:
    """P50/P90 only until a declared minimum n; tails need samples we do not have yet."""
    import numpy as np

    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    return {f"p{p}": round(float(np.percentile(clean, p)), 1) for p in points}
