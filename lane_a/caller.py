"""The branching caller: clips selected by what the agent actually did.

This is a *caller technology*, not a lane. The same state machine drives a
provider websocket directly (Lane A) or a phone call (Lane B); determinism of
the caller and realism of the transport are independent axes, and conflating
them is what made the earlier design look twice as expensive as it is.

Why not a fixed tape with absolute offsets. For isolated stimulus-response
probes a tape is fine and we still use one. For anything interactive it is
invalid: when a real caller would interrupt depends on when the agent started
talking and how long it talks, so a fixed tape punishes a verbose model with
artificial overlap and rewards a silent one with an unrealistic pause. Every
turn here is anchored to an observed event instead.

Two clocks, kept apart on purpose:

* **Control** -- deciding when to speak next -- runs off provider events and
  audio arrival. Coarse, live, good enough to steer a conversation.
* **Measurement** -- what gets published -- runs offline over the recorded
  audio with ``lane_a.detector``. A provider that pads its reply with trailing
  silence must not be credited with speaking for that long.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from lane_a import events as ev
from lane_a.adapters.base import RealtimeAdapter
from lane_a.audio import SAMPLE_WIDTH, iter_chunks, resample, silence
from lane_a.detector import speech_bounds

CHUNK_MS = 20.0          # pacing granularity, kept well inside detector error
POLL_S = 0.005
SLIP_WARN_MS = 15.0      # a send this late means the host, not the provider, is the story


@dataclass(frozen=True)
class Clip:
    """One piece of caller audio, with the ground truth we authored it from."""

    name: str
    pcm: bytes
    rate: int
    text: str = ""

    @property
    def duration_ms(self) -> float:
        return 1000.0 * len(self.pcm) / (2 * self.rate)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.pcm).hexdigest()

    def at_rate(self, rate: int) -> "Clip":
        if rate == self.rate:
            return self
        return Clip(self.name, resample(self.pcm, self.rate, rate), rate, self.text)


# ── anchors ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Utterance:
    """Exactly what was sent, and where the authored speech inside it ended.

    ``speech_end_sample`` is the whole point of Lane A. This audio was written
    here, so the caller-side boundary is a known sample index rather than a
    detected one, and it contributes no error to the latency measured from it.
    Detecting it instead would put the instrument at both ends of every
    measurement.
    """

    clip: Clip
    first_sample: int
    last_sample: int
    speech_start_sample: int
    speech_end_sample: int
    t_start: float
    t_end: float


@dataclass(frozen=True)
class Anchor:
    """When a turn starts. ``kind`` names the event it hangs off."""

    kind: str = "immediate"          # immediate | agent_quiet | agent_onset | tool_call
    delay_ms: float = 0.0            # extra wait after the anchor fires
    gap_ms: float = 300.0            # agent_quiet: how much silence counts as "stopped"
    tool: str | None = None
    timeout_s: float = 30.0

    @staticmethod
    def now(delay_ms: float = 0.0) -> "Anchor":
        return Anchor("immediate", delay_ms=delay_ms)

    @staticmethod
    def after_agent(gap_ms: float = 300.0, delay_ms: float = 0.0, timeout_s: float = 30.0) -> "Anchor":
        """Wait for the agent to finish, the way a polite caller would."""
        return Anchor("agent_quiet", delay_ms=delay_ms, gap_ms=gap_ms, timeout_s=timeout_s)

    @staticmethod
    def over_agent(delay_ms: float = 700.0, timeout_s: float = 30.0) -> "Anchor":
        """Barge in ``delay_ms`` after the agent starts speaking."""
        return Anchor("agent_onset", delay_ms=delay_ms, timeout_s=timeout_s)

    @staticmethod
    def after_tool(name: str, delay_ms: float = 0.0, timeout_s: float = 30.0) -> "Anchor":
        return Anchor("tool_call", delay_ms=delay_ms, tool=name, timeout_s=timeout_s)


@dataclass(frozen=True)
class Turn:
    """One caller action: wait for the anchor, then play the clip."""

    clip: Clip | None
    anchor: Anchor = field(default_factory=Anchor)
    commit: bool = False             # manual mode: declare the boundary at the last sample
    label: str = ""
    # Branch on what the agent has done so far; returns extra turns to splice in.
    branch: Callable[["CallerState"], Sequence["Turn"]] | None = None


@dataclass
class CallerState:
    """What the caller can see. Passed to branch functions."""

    agent_text: list[str]
    caller_text: list[str]
    tool_calls: list[dict[str, Any]]
    turns_taken: int

    def said(self, *needles: str) -> bool:
        joined = " ".join(self.agent_text).lower()
        return any(needle.lower() in joined for needle in needles)

    def called(self, name: str) -> bool:
        return any(call["name"] == name for call in self.tool_calls)


# ── the caller ───────────────────────────────────────────────────────────────

@dataclass
class _Segment:
    """A clip queued onto the carrier, plus where it landed once it went out."""

    clip: Clip
    pcm: bytes
    speech_start_offset: int      # samples into the clip
    speech_end_offset: int
    trim_tail: bool
    done: asyncio.Event
    first_sample: int = -1
    last_sample: int = -1
    t_start: float = 0.0
    t_end: float = 0.0
    consumed: int = 0             # bytes


class BranchingCaller:
    """Plays turns into an adapter, anchored on observed agent behaviour.

    The caller holds one **continuous** realtime stream open for the whole call
    and drops clips into it, rather than sending a clip and going quiet. That is
    not a stylistic choice: server-side endpointers decide a turn has ended by
    observing silence *in the stream*, so a caller that simply stops sending is
    never heard to stop talking and the provider waits forever. It is also what a
    phone line does, which keeps Lane A and the telephony back-end identical in
    shape.
    """

    def __init__(self, adapter: RealtimeAdapter, log: ev.EventLog) -> None:
        self.adapter = adapter
        self.log = log
        self.max_slip_ms = 0.0
        self.utterances: list[Utterance] = []
        self.bed: bytes | None = None          # room tone / noise, looped under everything
        self._queue: list[_Segment] = []
        self._current: _Segment | None = None
        self._bed_offset = 0
        self._stream: asyncio.Task | None = None
        self._stop = asyncio.Event()

    # -- the carrier ------------------------------------------------------

    async def __aenter__(self) -> "BranchingCaller":
        self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.stop()

    def start(self) -> None:
        if self._stream is None:
            self._stream = asyncio.create_task(self._carrier(), name="lane-a-carrier")

    async def stop(self) -> None:
        self._stop.set()
        if self._stream:
            await asyncio.gather(self._stream, return_exceptions=True)
            self._stream = None

    @property
    def _chunk_bytes(self) -> int:
        return int(self.adapter.input_rate * CHUNK_MS / 1000.0) * SAMPLE_WIDTH

    def _bed_chunk(self) -> bytes:
        """Background under the caller: digital silence unless a bed is loaded."""
        size = self._chunk_bytes
        if not self.bed:
            return b"\x00\x00" * (size // SAMPLE_WIDTH)
        out = bytearray()
        while len(out) < size:
            take = self.bed[self._bed_offset : self._bed_offset + size - len(out)]
            if not take:
                self._bed_offset = 0
                continue
            out.extend(take)
            self._bed_offset += len(take)
        return bytes(out)

    def _next_chunk(self) -> bytes:
        """One chunk of outbound audio, and the bookkeeping that goes with it.

        Segments always begin on a chunk boundary. The alignment error that buys
        is under 20 ms and it keeps sample accounting exact, which matters more:
        the caller-side boundary is the one number in this benchmark that carries
        no detector error, and it must not acquire one here.
        """
        size = self._chunk_bytes
        position = self.adapter.caller_timeline.n_samples

        if self._current is None and self._queue:
            self._current = self._queue.pop(0)
            self._current.first_sample = position
            self._current.t_start = self.log.clock.now()
            self.log.emit(
                ev.CALLER_AUDIO_START,
                clip=self._current.clip.name,
                sha256=self._current.clip.sha256[:16],
                ms=round(self._current.clip.duration_ms, 1),
                sample=position,
            )

        segment = self._current
        if segment is None:
            return self._bed_chunk()

        take = segment.pcm[segment.consumed : segment.consumed + size]
        segment.consumed += len(take)
        if segment.consumed >= len(segment.pcm):
            segment.last_sample = segment.first_sample + len(segment.pcm) // SAMPLE_WIDTH
            segment.t_end = self.log.clock.now()
            self._finish(segment)
            self._current = None
        if len(take) < size:
            take = take + self._bed_chunk()[len(take) :]
        return take

    def _finish(self, segment: _Segment) -> None:
        speech_end = segment.speech_end_offset if not segment.trim_tail else len(segment.pcm) // SAMPLE_WIDTH
        utterance = Utterance(
            clip=segment.clip,
            first_sample=segment.first_sample,
            last_sample=segment.last_sample,
            speech_start_sample=segment.first_sample + segment.speech_start_offset,
            speech_end_sample=segment.first_sample + min(speech_end, len(segment.pcm) // SAMPLE_WIDTH),
            t_start=segment.t_start,
            t_end=segment.t_end,
        )
        self.utterances.append(utterance)
        self.log.emit(
            ev.CALLER_AUDIO_END,
            clip=segment.clip.name,
            sample=utterance.last_sample,
            speech_end_sample=utterance.speech_end_sample,
        )
        segment.done.set()

    async def _carrier(self) -> None:
        """One paced stream for the whole call.

        Scheduled against a fixed origin rather than by sleeping a chunk at a
        time, so scheduler jitter does not accumulate into drift. Slip is
        measured and published: if this host could not keep up, that is our
        defect, and the run says so rather than blaming the provider for a late
        reply.
        """
        origin = self.log.clock.now()
        index = 0
        while not self._stop.is_set():
            target = origin + index * CHUNK_MS / 1000.0
            now = self.log.clock.now()
            if now < target:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=target - now)
                    return
                except asyncio.TimeoutError:
                    pass
            actual = self.log.clock.now()
            slip_ms = (actual - target) * 1000.0
            self.max_slip_ms = max(self.max_slip_ms, slip_ms)
            if slip_ms > SLIP_WARN_MS:
                self.log.emit(ev.CALLER_SLIP, chunk=index, slip_ms=round(slip_ms, 2))
            try:
                await self.adapter.send_audio(self._next_chunk(), t_send=max(target, actual))
            except Exception as exc:  # noqa: BLE001 -- a dead socket ends the run, not the process
                self.log.emit(ev.SESSION_ERROR, error=repr(exc), where="carrier")
                return
            index += 1

    # -- speaking ---------------------------------------------------------

    async def play(self, clip: Clip, trim_tail: bool = False) -> Utterance:
        """Queue a clip onto the carrier and wait until its last sample is out."""
        clip = clip.at_rate(self.adapter.input_rate)
        bounds = speech_bounds(clip.pcm, clip.rate)
        total = len(clip.pcm) // SAMPLE_WIDTH
        start_offset = int(round((bounds.start_ms or 0.0) * clip.rate / 1000.0))
        end_offset = min(total, int(round((bounds.end_ms if bounds.found else clip.duration_ms) * clip.rate / 1000.0)))
        pcm = clip.pcm[: end_offset * SAMPLE_WIDTH] if trim_tail else clip.pcm

        segment = _Segment(
            clip=clip,
            pcm=pcm,
            speech_start_offset=start_offset,
            speech_end_offset=end_offset,
            trim_tail=trim_tail,
            done=asyncio.Event(),
        )
        self._queue.append(segment)
        self.start()
        await segment.done.wait()
        return self.utterances[-1]

    async def wait(self, duration_ms: float) -> None:
        """Hold the line. The carrier keeps the stream alive underneath."""
        await asyncio.sleep(duration_ms / 1000.0)

    # -- observation ------------------------------------------------------

    def _last_agent_audio_at(self) -> float | None:
        chunks = self.adapter.agent_timeline.chunks
        return chunks[-1].t_wall if chunks else None

    async def wait_agent_onset(self, timeout_s: float, since_sample: int | None = None) -> bool:
        """Until agent audio newer than ``since_sample`` arrives."""
        mark = self.adapter.agent_timeline.n_samples if since_sample is None else since_sample
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.adapter.agent_timeline.n_samples > mark:
                return True
            await asyncio.sleep(POLL_S)
        return False

    async def wait_agent_quiet(self, gap_ms: float, timeout_s: float) -> bool:
        """Until the agent has produced no audio for ``gap_ms``.

        Gap-based rather than event-based because a response cancelled in flight
        may never send its done event, and a caller waiting for one would hang
        instead of taking its turn.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            last = self._last_agent_audio_at()
            if last is not None and (self.log.clock.now() - last) * 1000.0 >= gap_ms:
                return True
            await asyncio.sleep(POLL_S)
        return False

    async def wait_tool_call(self, name: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if any(call["name"] == name for call in self.adapter.tool_calls):
                return True
            await asyncio.sleep(POLL_S)
        return False

    async def _await_anchor(self, anchor: Anchor) -> bool:
        fired = True
        if anchor.kind == "agent_quiet":
            fired = await self.wait_agent_quiet(anchor.gap_ms, anchor.timeout_s)
        elif anchor.kind == "agent_onset":
            fired = await self.wait_agent_onset(anchor.timeout_s)
        elif anchor.kind == "tool_call" and anchor.tool:
            fired = await self.wait_tool_call(anchor.tool, anchor.timeout_s)
        if anchor.delay_ms:
            await self.wait(anchor.delay_ms)
        return fired

    # -- running a scenario ----------------------------------------------

    def state(self) -> CallerState:
        return CallerState(
            agent_text=list(self.adapter.agent_text),
            caller_text=list(self.adapter.caller_text),
            tool_calls=list(self.adapter.tool_calls),
            turns_taken=len(self.utterances),
        )

    async def run(self, turns: Sequence[Turn]) -> None:
        self.start()
        pending = list(turns)
        index = 0
        while pending:
            turn = pending.pop(0)
            label = turn.label or (turn.clip.name if turn.clip else "wait")
            self.log.emit(ev.STEP_START, step=index, label=label, anchor=turn.anchor.kind)
            fired = await self._await_anchor(turn.anchor)
            if not fired:
                self.log.emit(ev.STEP_END, step=index, label=label, anchor_timeout=True)
                index += 1
                continue
            if turn.clip is not None:
                await self.play(turn.clip, trim_tail=turn.commit)
            if turn.commit:
                await self.adapter.commit()
            if turn.branch is not None:
                extra = list(turn.branch(self.state()))
                if extra:
                    pending = extra + pending
            self.log.emit(ev.STEP_END, step=index, label=label)
            index += 1
