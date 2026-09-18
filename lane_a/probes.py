"""The Lane A probe set.

Each probe is one question asked of one provider configuration, producing one
row. Probes never average over anything -- repeats, voices and configurations
stay separate populations all the way to publication, because a mean over a
condition is how a benchmark loses the finding it was built to make.

Open-loop probes use a fixed stimulus and are valid precisely because nothing the
agent does should change what the caller says next. Closed-loop probes anchor
every turn on an observed event, because an absolute-offset tape punishes a
verbose model with artificial overlap and rewards a silent one with an
unrealistic pause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import asyncio

from lane_a import events as ev
from lane_a.adapters.base import RealtimeAdapter
from lane_a.audio import pink_noise
from lane_a.caller import BranchingCaller, Clip
from lane_a.clips import Corpus
from lane_a.metrics import barge_in, response_latency, speech_runs, spoke_between, usage
from lane_a.scenarios import SCENARIOS, ScenarioSpec, by_id
from lane_a.transcript import compare_transcripts
from lane_a.transforms import TRANSFORMS, NamedTransform
from mock_tools.server import MockToolServer
from mock_tools.verifier import verify_trace


@dataclass
class ProbeContext:
    adapter: RealtimeAdapter
    caller: BranchingCaller
    corpus: Corpus
    voice: str
    log: ev.EventLog
    tools: MockToolServer | None = None
    transform: NamedTransform = TRANSFORMS["clean"]

    def clip(self, clip_id: str) -> Clip:
        """The authored clip, degraded by the cell's transform where one applies.

        The transform runs on the master before any resampling, so a provider
        with a different input rate receives the same degradation through the
        same published filter as every other.
        """
        clip = self.corpus.load(clip_id, self.voice)
        if self.transform.name == "clean":
            return clip
        return Clip(f"{clip.name}+{self.transform.name}", self.transform.apply(clip.pcm, clip.rate), clip.rate, clip.text)

    @property
    def is_text(self) -> bool:
        return self.adapter.config.modality == "text"

    async def say(self, clip_id: str, trim_tail: bool = False):
        """Deliver one authored line, as speech or as text.

        This one method is the whole text control arm. The scenario, the tools
        and the model are identical in both modes; the only difference is whether
        the words arrive as audio. That is what turns "the model failed the
        booking" into "the model does this fine in text and the speech pathway
        broke it" -- the claim an S2S benchmark exists to make, and one that a
        separate text model could not support.
        """
        if self.is_text:
            await self.adapter.send_text(self.corpus.specs[clip_id].text)
            return None
        return await self.caller.play(self.clip(clip_id), trim_tail=trim_tail)

    def _activity(self) -> int:
        """Anything the agent is still doing: replies finished plus tools called.

        Waiting for a single ``response.done`` is not enough. Answering a tool
        call creates a response of its own, so a turn can be "replied to" by a
        response the caller never prompted -- which walks the script out of step
        with the conversation and loses the last turn. Settling on *quiet* rather
        than on a count keeps the two aligned.
        """
        return sum(1 for _ in self.log.of_kind(ev.RESPONSE_DONE)) + len(self.adapter.tool_calls)

    async def wait_reply(self, settle_ms: float = 800.0, timeout_s: float = 45.0) -> bool:
        """Until the agent has finished replying and gone quiet, in either modality."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s

        if not self.is_text:
            if not await self.caller.wait_agent_onset(timeout_s=timeout_s):
                return False
            if not await self.caller.wait_agent_quiet(settle_ms, timeout_s=timeout_s):
                return False
            # Audio can go quiet while a tool call is still in flight.
            return await self._settle(deadline, settle_ms)

        # A reply has landed when a transcript arrives after our turn *and* after
        # the last tool call: the tool result provokes a further response, and a
        # caller that took the pre-tool text as the reply spoke into the model's
        # next sentence and was heard as an interruption.
        sent_at = self.log.clock.now()
        while loop.time() < deadline:
            replies = [e.t for e in self.log.of_kind(ev.AGENT_TRANSCRIPT) if e.t > sent_at]
            calls = [e.t for e in self.log.of_kind(ev.TOOL_CALL) if e.t > sent_at]
            if replies and (not calls or max(replies) > max(calls)):
                break
            await asyncio.sleep(0.01)
        return await self._settle(deadline, settle_ms)

    async def _settle(self, deadline: float, settle_ms: float) -> bool:
        loop = asyncio.get_running_loop()
        while loop.time() < deadline:
            mark = self._activity()
            quiet_until = loop.time() + settle_ms / 1000.0
            while loop.time() < quiet_until:
                await asyncio.sleep(0.01)
                if self._activity() != mark:
                    break
            else:
                return True
        return False


async def pump_tools(ctx: ProbeContext, stop: asyncio.Event) -> None:
    """Answer tool calls from the published contract as they arrive.

    Runs as a background task rather than inside the turn loop: a model may call a
    tool at any point in its reply, including before it has said anything, and a
    caller that only checked between turns would deadlock against it.
    """
    served = 0
    while not stop.is_set():
        while ctx.tools is not None and served < len(ctx.adapter.tool_calls):
            call = ctx.adapter.tool_calls[served]
            served += 1
            output = ctx.tools.call(call["name"], call.get("arguments", {}))
            await ctx.adapter.send_tool_result(call.get("call_id", ""), output)
        await asyncio.sleep(0.01)


@dataclass
class ProbeResult:
    """One cell. ``void`` means the run did not happen; it is never a score."""

    probe: str
    values: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None          # "pass" / "fail" for pass-fail probes
    void: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {"probe": self.probe, "verdict": self.verdict, "void": self.void, **self.values}


class Probe(Protocol):
    name: str

    @property
    def slug(self) -> str:
        """Filesystem identity. Two rungs of one ladder are two probes, not one.

        ``name`` groups rows for reporting; ``slug`` keeps their artifacts apart.
        Without the distinction every rung of a parameter sweep writes to the same
        directory and only the last one's audio survives -- which would quietly
        gut the claim that any published cell can be recomputed from its files.
        """
        return self.name

    async def run(self, ctx: ProbeContext) -> ProbeResult: ...


# ── open loop ────────────────────────────────────────────────────────────────

@dataclass
class ResponseLatency:
    """Authored speech-end to first audible agent sample.

    Run under both turn-detection configurations. The manual-commit number is a
    generation-latency floor with zero endpointing error, which is only available
    to a benchmark that authored the audio; the difference between it and the
    native-VAD number is that provider's endpointing cost, measured rather than
    quoted from its documentation.
    """

    clip_id: str = "open.book"
    settle_ms: float = 400.0
    name: str = "response_latency"

    @property
    def slug(self) -> str:
        return f"{self.name}-{self.clip_id}"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        manual = ctx.adapter.config.turn_detection.is_manual
        utterance = await ctx.caller.play(ctx.clip(self.clip_id), trim_tail=manual)
        if manual:
            await ctx.adapter.commit()
        if not await ctx.caller.wait_agent_onset(timeout_s=20):
            return ProbeResult(self.name, verdict="fail", values={"reason": "agent never spoke"})
        await ctx.caller.wait_agent_quiet(self.settle_ms, timeout_s=25)

        latency = response_latency(ctx.adapter, utterance)
        return ProbeResult(
            self.name,
            verdict="pass" if latency.latency_ms is not None else "fail",
            values={
                **latency.as_json(),
                "clip": self.clip_id,
                "agent_text": " ".join(ctx.adapter.agent_text)[:400],
                "caller_asr": " ".join(ctx.adapter.caller_text)[:400],
                "usage": usage(ctx.adapter),
            },
        )


@dataclass
class EndpointingLadder:
    """Does the provider interrupt a mid-utterance pause of ``gap_ms``?

    One rung per run, never a mean over the ladder: the published artifact is the
    curve, which gives each provider's actual patience threshold instead of a
    single number that hides where it breaks. The two halves are bit-identical
    across every rung, so the gap is the only thing that varies.
    """

    gap_ms: float
    first: str = "phone.part1"
    second: str = "phone.part2"
    name: str = "endpointing_ladder"

    @property
    def slug(self) -> str:
        return f"{self.name}-gap{self.gap_ms:.0f}ms"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        if ctx.adapter.config.turn_detection.is_manual:
            return ProbeResult(self.name, void="manual commit has no endpointer to probe")
        await ctx.caller.play(ctx.clip(self.first))
        gap_start = ctx.log.clock.now()
        await ctx.caller.wait(self.gap_ms)
        gap_end = ctx.log.clock.now()
        interrupted = spoke_between(ctx.adapter, gap_start, gap_end)
        await ctx.caller.play(ctx.clip(self.second))
        await ctx.caller.wait_agent_quiet(400.0, timeout_s=20)
        return ProbeResult(
            self.name,
            verdict="fail" if interrupted else "pass",   # pass = waited for the caller to finish
            values={
                "gap_ms": self.gap_ms,
                "interrupted_in_gap": interrupted,
                "caller_asr": " ".join(ctx.adapter.caller_text)[:400],
            },
        )


@dataclass
class FalseTrigger:
    """Noise on the line, caller silent. How often does the provider speak anyway?

    Reported as a rate per minute rather than a count, so windows of different
    lengths stay comparable. Note the noise sits on the *caller* channel; the
    provider's returned audio is clean, which is what keeps the energy detector
    valid here.
    """

    duration_ms: float = 20000.0
    level_dbfs: float = -30.0
    seed: int = 7
    name: str = "false_trigger"

    @property
    def slug(self) -> str:
        return f"{self.name}-{abs(self.level_dbfs):.0f}dbfs"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        if ctx.adapter.config.turn_detection.is_manual:
            return ProbeResult(self.name, void="manual commit cannot false-trigger")
        ctx.caller.bed = pink_noise(ctx.adapter.input_rate, 2000.0, self.level_dbfs, self.seed)
        start = ctx.log.clock.now()
        await ctx.caller.wait(self.duration_ms)
        end = ctx.log.clock.now()
        ctx.caller.bed = None
        triggers = sum(1 for e in ctx.log.events if e.kind == ev.AGENT_AUDIO_START and start < e.t <= end)
        minutes = (end - start) / 60.0
        return ProbeResult(
            self.name,
            verdict="pass" if triggers == 0 else "fail",
            values={
                "triggers": triggers,
                "per_minute": round(triggers / minutes, 2) if minutes else None,
                "window_ms": round((end - start) * 1000.0, 1),
                "noise_level_dbfs": self.level_dbfs,
            },
        )


# ── closed loop ──────────────────────────────────────────────────────────────

@dataclass
class BargeIn:
    """Caller speaks over the agent. Does it yield the floor, and how fast?

    Anchored on the agent's own onset rather than on an absolute offset, so a
    model that starts talking late is interrupted at the same point *in its
    reply* as one that starts early.
    """

    after_onset_ms: float = 900.0
    clip_id: str = "bargein.stop"
    opener: str = "open.book"
    name: str = "barge_in"

    @property
    def slug(self) -> str:
        return f"{self.name}-at{self.after_onset_ms:.0f}ms"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        manual = ctx.adapter.config.turn_detection.is_manual
        await ctx.caller.play(ctx.clip(self.opener), trim_tail=manual)
        if manual:
            await ctx.adapter.commit()
        if not await ctx.caller.wait_agent_onset(timeout_s=20):
            return ProbeResult(self.name, void="agent never spoke; nothing to interrupt")
        await ctx.caller.wait(self.after_onset_ms)
        if not ctx.adapter.agent_speaking:
            return ProbeResult(self.name, void="agent finished before the barge-in point")

        utterance = await ctx.caller.play(ctx.clip(self.clip_id), trim_tail=manual)
        if manual:
            await ctx.adapter.commit()
        await ctx.caller.wait(2000.0)
        result = barge_in(ctx.adapter, utterance)
        return ProbeResult(
            self.name,
            verdict="pass" if result.stopped else "fail",
            values={**result.as_json(), "after_onset_ms": self.after_onset_ms},
        )


@dataclass
class BackchannelTolerance:
    """Caller says "mm hmm" mid-reply. Correct behaviour is to keep going.

    The failure this catches is the one every voice agent has: a listener noise
    read as a turn, so the agent stops and answers it. Scored from the *shape of
    the agent's speech*, not from provider cancellation events -- a cancel that
    arrives after a reply has already finished is a no-op, and an earlier version
    of this probe scored one as a failure, which is the sort of thing that turns
    an instrument artefact into a published ranking.

    Three outcomes, and the void matters as much as the other two: if the agent's
    reply ended before the provider ever registered the backchannel, nothing was
    tested and the cell must say so rather than quietly scoring a pass.
    """

    after_onset_ms: float = 400.0
    settle_ms: float = 2500.0
    clip_id: str = "token.backchannel"
    opener: str = "open.book"
    name: str = "backchannel_tolerance"

    @property
    def slug(self) -> str:
        return f"{self.name}-at{self.after_onset_ms:.0f}ms"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        if ctx.adapter.config.turn_detection.is_manual:
            return ProbeResult(self.name, void="manual commit never yields on a backchannel")
        await ctx.caller.play(ctx.clip(self.opener))
        if not await ctx.caller.wait_agent_onset(timeout_s=20):
            return ProbeResult(self.name, void="agent never spoke")
        await ctx.caller.wait(self.after_onset_ms)
        if not ctx.adapter.agent_speaking:
            return ProbeResult(self.name, void="agent finished before the backchannel point")

        started = ctx.log.clock.now()
        utterance = await ctx.caller.play(ctx.clip(self.clip_id))
        heard_at = ctx.caller.adapter.caller_timeline.time_of_sample(utterance.speech_start_sample) or started
        await ctx.caller.wait(self.settle_ms)

        runs = speech_runs(ctx.adapter)
        in_flight = next((run for run in runs if run[0] <= heard_at <= run[1]), None)
        if in_flight is None:
            return ProbeResult(
                self.name,
                void="the agent's reply ended before the backchannel was heard; nothing was tested",
                values={"heard_at": round(heard_at, 3)},
            )

        # The test is valid only if the whole backchannel was spoken while the
        # reply was still audible. That is judged from our own audio, which every
        # provider gets alike. Whether the provider's endpointer *registered* it
        # is reported separately where the provider says so -- an endpointer
        # that never fires on "mm hmm" is one way of tolerating it, not a reason
        # to throw the cell away.
        spoken_until = ctx.caller.adapter.caller_timeline.time_of_sample(
            max(utterance.speech_end_sample - 1, 0)
        ) or heard_at
        if spoken_until > in_flight[1]:
            return ProbeResult(
                self.name,
                void="reply finished before the backchannel had been fully spoken; nothing was tested",
                values={"spoken_after_reply_ms": round((spoken_until - in_flight[1]) * 1000.0, 1)},
            )
        registered = None
        if ctx.adapter.emits_vad_events:
            registered = next(
                (e.t for e in ctx.log.events if e.kind == ev.VAD_SPEECH_START and e.t >= heard_at), None
            )

        # Two ways to fail: stop mid-reply, or answer the backchannel as a turn.
        kept_talking = in_flight[1] > spoken_until
        new_response = next((run for run in runs if run[0] > in_flight[1]), None)
        answered_it = new_response is not None

        return ProbeResult(
            self.name,
            verdict="pass" if kept_talking and not answered_it else "fail",
            values={
                "kept_talking": kept_talking,
                "answered_the_backchannel": answered_it,
                "held_floor_after_ms": round((in_flight[1] - spoken_until) * 1000.0, 1),
                "new_reply_after_ms": None if not answered_it else round((new_response[0] - in_flight[1]) * 1000.0, 1),
                "provider_registered_it": None if not ctx.adapter.emits_vad_events else registered is not None,
                "registration_lag_ms": None if registered is None else round((registered - heard_at) * 1000.0, 1),
            },
        )


# ── caller transcription ─────────────────────────────────────────────────────

@dataclass
class CallerTranscription:
    """How accurately the provider heard what we said, against an exact reference.

    The authored text is the ground truth, so no second transcriber is needed
    and no judge is involved. Digits are scored separately from words because a
    phone number heard as nine digits fails the task however good the word
    error rate looks, and the two are reported apart so a provider's weakness
    is visible rather than averaged away.
    """

    clip_id: str = "task.identify"
    timeout_s: float = 30.0
    name: str = "caller_transcription"

    @property
    def slug(self) -> str:
        return f"{self.name}-{self.clip_id}"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        manual = ctx.adapter.config.turn_detection.is_manual
        clip = ctx.clip(self.clip_id)
        await ctx.caller.play(clip, trim_tail=manual)
        if manual:
            await ctx.adapter.commit()
        heard = await self._wait_transcript(ctx)
        await ctx.caller.wait_agent_quiet(400.0, timeout_s=15)
        if heard is None:
            return ProbeResult(self.name, void="provider returned no transcript of the caller")
        scored = compare_transcripts(clip.text, heard)
        verdict = scored["digits_match"] if scored["digits_expected"] else scored["wer"] <= 0.25
        return ProbeResult(
            self.name,
            verdict="pass" if verdict else "fail",
            values={**scored, "clip": self.clip_id, "reference": clip.text, "heard": heard},
        )

    async def _wait_transcript(self, ctx: ProbeContext) -> str | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_s
        while loop.time() < deadline:
            if ctx.adapter.caller_text:
                return ctx.adapter.caller_text[-1]
            await asyncio.sleep(0.02)
        return None


@dataclass
class FilledPause:
    """A mid-utterance pause the caller fills with "um, let me think".

    The unfilled ladder asks whether silence of a given length ends the turn;
    this asks whether a filler buys the caller more time, as it does with a
    human listener. Same two halves as the ladder, so the only difference from
    a rung of the same gap is the filler.
    """

    gap_ms: float = 1500.0
    first: str = "phone.part1"
    filler: str = "token.hesitation"
    second: str = "phone.part2"
    name: str = "filled_pause"

    @property
    def slug(self) -> str:
        return f"{self.name}-gap{self.gap_ms:.0f}ms"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        if ctx.adapter.config.turn_detection.is_manual:
            return ProbeResult(self.name, void="manual commit has no endpointer to probe")
        await ctx.caller.play(ctx.clip(self.first))
        gap_start = ctx.log.clock.now()
        await ctx.caller.wait(self.gap_ms / 3.0)
        await ctx.caller.play(ctx.clip(self.filler))
        await ctx.caller.wait(self.gap_ms / 3.0)
        gap_end = ctx.log.clock.now()
        interrupted = spoke_between(ctx.adapter, gap_start, gap_end)
        await ctx.caller.play(ctx.clip(self.second))
        await ctx.caller.wait_agent_quiet(400.0, timeout_s=20)
        return ProbeResult(
            self.name,
            verdict="fail" if interrupted else "pass",
            values={
                "gap_ms": self.gap_ms,
                "interrupted_in_gap": interrupted,
                "caller_asr": " ".join(ctx.adapter.caller_text)[:400],
            },
        )


# ── tools and state ──────────────────────────────────────────────────────────

@dataclass
class TaskScenario:
    """A multi-turn task against a published mock-tool contract, driven by a scenario.

    Scored on the **tool-call trace**, not on a judge's opinion of the
    conversation: a model scoring whether an agent "handled it well" imports the
    judge's taste into the ranking, while comparing calls against a declared
    expectation does not. What the agent said is recorded and published, but it
    does not decide the cell.

    The caller **branches**, because this is a closed-loop scenario: agents ask
    for the phone number, the reason and the date in whatever order their prompt
    implies, and a fixed script silently runs out of turns against one that asks
    them in a different order -- scoring the ordering of our script rather than
    the agent. Routing is the scenario's published table of literal patterns.

    Runs in Lane A, and in the text arm with the identical script. The pair is
    the point -- it separates "this model cannot do the task" from "this model
    cannot do the task *by voice*".
    """

    spec: ScenarioSpec = field(default_factory=lambda: by_id("book.morning"))
    settle_ms: float = 800.0
    name: str = "task"

    @property
    def slug(self) -> str:
        return f"{self.name}-{self.spec.id}"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        if ctx.tools is None:
            return ProbeResult(self.name, void="no tool server configured")
        if ctx.tools.suite != self.spec.contract:
            return ProbeResult(self.name, void=f"scenario needs the {self.spec.contract} contract, got {ctx.tools.suite}")

        stop = asyncio.Event()
        pump = asyncio.create_task(pump_tools(ctx, stop), name="tool-pump")
        manual = ctx.adapter.config.turn_detection.is_manual
        script: list[str] = []
        ended_by = "max_turns"
        try:
            clip_id = self.spec.opener
            for _ in range(self.spec.max_turns):
                script.append(clip_id)
                await ctx.say(clip_id, trim_tail=manual)
                if manual and not ctx.is_text:
                    await ctx.adapter.commit()
                if not await ctx.wait_reply(self.settle_ms):
                    ended_by = f"no reply after {clip_id}"
                    break
                last = ctx.adapter.agent_text[-1] if ctx.adapter.agent_text else ""
                if self.spec.finished(last, ctx.adapter.tool_calls):
                    ended_by = "scenario complete"
                    break
                clip_id = self.spec.route(last)
            await asyncio.sleep(1.0)  # let a trailing tool call land
        finally:
            stop.set()
            await asyncio.gather(pump, return_exceptions=True)

        verdict = verify_trace(ctx.adapter.tool_calls, expected=self.spec.expected, forbidden=self.spec.forbidden)
        return ProbeResult(
            self.name,
            verdict="pass" if verdict.passed else "fail",
            values={
                **verdict.as_json(),
                "scenario": self.spec.id,
                "contract": self.spec.contract,
                "modality": ctx.adapter.config.modality,
                "turns": len(script),
                "script": script,
                "ended_by": ended_by,
                "tool_calls": [
                    {"name": c["name"], "arguments": c.get("arguments", {})} for c in ctx.adapter.tool_calls
                ],
                "unmatched_tool_inputs": sum(1 for c in ctx.tools.calls if not c.matched),
                "agent_text": " ".join(ctx.adapter.agent_text)[:1200],
                "usage": usage(ctx.adapter),
            },
        )


def BookingTask(**kwargs: Any) -> TaskScenario:
    """The baseline booking scenario, kept under its original name."""
    return TaskScenario(spec=by_id("book.morning"), **kwargs)


def route_booking_reply(text: str) -> str:
    """The baseline scenario's routing, exposed for the routing tests."""
    return by_id("book.morning").route(text)


def task_probes(contract: str | None = None, scenarios: Sequence[ScenarioSpec] = SCENARIOS) -> list[TaskScenario]:
    """One probe per scenario, optionally for a single contract."""
    return [TaskScenario(spec=s) for s in scenarios if contract is None or s.contract == contract]
