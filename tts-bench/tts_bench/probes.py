"""What a cell does to a provider, and what it writes down.

A probe is a small script against the adapter's context API. It never computes
a statistic; it returns the measurements for one cell, and the report layer
does the aggregation. A probe that cannot run on a provider is declared, not
discovered: ``needs`` names the adapter features it requires, and the runner
publishes the gap as an exclusion without connecting.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from tts_bench.common.events import Clock, EventLog

from tts_bench.adapters.base import Synthesis, TTSAdapter
from tts_bench.corpus import Item
from tts_bench.metrics import summarize

WAIT_S = 30.0          # a synthesis that takes longer than this is a failure, not a slow success
FIRST_AUDIO_WAIT_S = 15.0


@dataclass
class ProbeResult:
    name: str
    verdict: str | None = None          # "pass" | "fail" | None for pure measurements
    void: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    syntheses: list[Synthesis] = field(default_factory=list)


@dataclass
class ProbeContext:
    adapter: TTSAdapter
    item: Item
    clock: Clock
    log: EventLog
    # For probes that need more than one connection.
    new_adapter: Callable[[], Awaitable[TTSAdapter]] | None = None


class Probe:
    name = "probe"
    needs: tuple[str, ...] = ()
    # Which corpus item this probe runs on when it is not the item under test.
    fixed_item: str | None = None

    @property
    def slug(self) -> str:
        return self.name

    def params(self) -> dict[str, Any]:
        return {}

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        raise NotImplementedError


async def one_shot(adapter: TTSAdapter, context_id: str, text: str, wait_s: float = WAIT_S) -> Synthesis:
    await adapter.open_context(context_id)
    await adapter.send(context_id, text)
    await adapter.finish(context_id)
    return await adapter.wait(context_id, wait_s)


def _verdict(synthesis: Synthesis) -> str:
    if synthesis.error or not synthesis.pcm:
        return "fail"
    if synthesis.meta.get("ended_by") == "timeout":
        return "fail"
    return "pass"


class OneShot(Probe):
    """The whole text in one frame: the number every other benchmark publishes, plus playout."""

    name = "one_shot"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        synthesis = await one_shot(ctx.adapter, "main", ctx.item.text)
        return ProbeResult(self.name, _verdict(synthesis), values=summarize(synthesis), syntheses=[synthesis])


class StreamedInput(Probe):
    """Text arrives the way an LLM produces it, and the question is when speech starts.

    Words are sent one frame at a time at a fixed cadence. Two latencies come
    out: from the first word (what the listener waits after the model starts
    answering) and from the last word (how much of the wait the provider hid
    behind the input). A negative second number means the provider was already
    speaking before the sentence was finished.
    """

    name = "streamed_input"
    needs = ("streamed_input",)

    def __init__(self, words_per_s: float = 30.0) -> None:
        self.words_per_s = words_per_s

    @property
    def slug(self) -> str:
        return f"{self.name}-{self.words_per_s:g}wps"

    def params(self) -> dict[str, Any]:
        return {"words_per_s": self.words_per_s}

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        adapter = ctx.adapter
        await adapter.open_context("main")
        words = ctx.item.text.split()
        gap = 1.0 / self.words_per_s
        for index, word in enumerate(words):
            await adapter.send("main", word + (" " if index < len(words) - 1 else ""))
            if index < len(words) - 1:
                await asyncio.sleep(gap)
        await adapter.finish("main")
        synthesis = await adapter.wait("main", WAIT_S)
        values = summarize(synthesis)
        values["words"] = len(words)
        values["started_before_input_done"] = (
            None if values["ttfa_from_input_done_ms"] is None else values["ttfa_from_input_done_ms"] < 0
        )
        return ProbeResult(self.name, _verdict(synthesis), values=values, syntheses=[synthesis])


class Cancel(Probe):
    """How long after we say stop does the audio actually stop.

    Runs on the long item so there is something left to cancel. The cancel is
    sent a fixed interval after the first chunk arrives; what is measured is
    how much audio the provider still delivered after that instant and when its
    last byte landed. This is what barge-in responsiveness is made of.
    """

    name = "cancel"
    needs = ("cancel",)
    fixed_item = "long.prep"

    def __init__(self, after_first_audio_ms: float = 300.0) -> None:
        self.after_first_audio_ms = after_first_audio_ms

    @property
    def slug(self) -> str:
        return f"{self.name}-{self.after_first_audio_ms:g}ms"

    def params(self) -> dict[str, Any]:
        return {"after_first_audio_ms": self.after_first_audio_ms}

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        adapter = ctx.adapter
        synthesis = await adapter.open_context("main")
        await adapter.send("main", ctx.item.text)
        await adapter.finish("main")
        deadline = ctx.clock.now() + FIRST_AUDIO_WAIT_S
        while synthesis.t_first_chunk is None and not synthesis.error and ctx.clock.now() < deadline:
            await asyncio.sleep(0.005)
        if synthesis.t_first_chunk is None:
            await adapter.wait("main", 1.0)
            return ProbeResult(self.name, "fail", values=summarize(synthesis), syntheses=[synthesis])
        await asyncio.sleep(self.after_first_audio_ms / 1000.0)
        await adapter.cancel("main")
        await adapter.wait("main", 10.0, quiet_s=1.5)
        values = summarize(synthesis)
        # Delivered means "would have been heard": the provider stopped when its
        # last chunk landed, not when it said so.
        stopped = values["cancel_to_last_chunk_ms"] is not None and values["cancel_to_last_chunk_ms"] < 2000.0
        return ProbeResult(self.name, "pass" if stopped else "fail", values=values, syntheses=[synthesis])


class Continuation(Probe):
    """One utterance sent as several frames: does the prosody survive the seams.

    The same text is synthesised once whole and once as sentence-sized frames
    with a pause between them, the way an LLM's sentences reach a TTS service.
    Duration and onset are compared here; the two recordings are kept so the
    round-trip transcripts can be compared offline.
    """

    name = "continuation"
    needs = ("continuation",)
    fixed_item = "long.summary"

    def __init__(self, frame_gap_ms: float = 300.0) -> None:
        self.frame_gap_ms = frame_gap_ms

    def params(self) -> dict[str, Any]:
        return {"frame_gap_ms": self.frame_gap_ms}

    @staticmethod
    def frames(text: str) -> list[str]:
        parts = [p for p in re.split(r"(?<=[.?!:])\s+", text.strip()) if p]
        return [p + " " for p in parts[:-1]] + [parts[-1]]

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        adapter = ctx.adapter
        whole = await one_shot(adapter, "whole", ctx.item.text)
        await adapter.open_context("framed")
        frames = self.frames(ctx.item.text)
        for index, frame in enumerate(frames):
            await adapter.send("framed", frame)
            if index < len(frames) - 1:
                await asyncio.sleep(self.frame_gap_ms / 1000.0)
        await adapter.finish("framed")
        framed = await adapter.wait("framed", WAIT_S)
        w, f = summarize(whole), summarize(framed)
        values = {
            "frames": len(frames),
            "whole": w,
            "framed": f,
            "duration_delta_ms": None if None in (w["audio_ms"], f["audio_ms"]) else round(f["audio_ms"] - w["audio_ms"], 1),
            "ttfa_delta_ms": None if None in (w["ttfa_ms"], f["ttfa_ms"]) else round(f["ttfa_ms"] - w["ttfa_ms"], 1),
            "framed_underruns": f["underruns"],
            "framed_stall_ms": f["stall_ms"],
        }
        verdict = "pass" if _verdict(whole) == "pass" and _verdict(framed) == "pass" else "fail"
        return ProbeResult(self.name, verdict, values=values, syntheses=[whole, framed])


class Repeat(Probe):
    """The same text twice on one connection: is the service deterministic, and how much does it vary."""

    name = "repeat"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        first = await one_shot(ctx.adapter, "first", ctx.item.text)
        second = await one_shot(ctx.adapter, "second", ctx.item.text)
        a, b = summarize(first), summarize(second)
        values = {
            "first": a,
            "second": b,
            "identical_pcm": bytes(first.pcm) == bytes(second.pcm) and bool(first.pcm),
            "duration_delta_ms": None if None in (a["audio_ms"], b["audio_ms"]) else round(b["audio_ms"] - a["audio_ms"], 1),
            "ttfa_delta_ms": None if None in (a["ttfa_ms"], b["ttfa_ms"]) else round(b["ttfa_ms"] - a["ttfa_ms"], 1),
            # The second synthesis on a warm connection is the number a busy
            # agent sees; the first is what the sentinel and one_shot see.
            "ttfa_ms": b["ttfa_ms"],
        }
        verdict = "pass" if _verdict(first) == "pass" and _verdict(second) == "pass" else "fail"
        return ProbeResult(self.name, verdict, values=values, syntheses=[first, second])


class Concurrency(Probe):
    """N one-shots at once on N connections under one key: what the tail looks like under load."""

    name = "concurrency"

    def __init__(self, streams: int = 8) -> None:
        self.streams = streams

    @property
    def slug(self) -> str:
        return f"{self.name}-{self.streams}"

    def params(self) -> dict[str, Any]:
        return {"streams": self.streams}

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        if ctx.new_adapter is None:
            return ProbeResult(self.name, void="harness error: concurrency needs an adapter factory")
        adapters = [ctx.adapter] + [await ctx.new_adapter() for _ in range(self.streams - 1)]
        try:
            for adapter in adapters[1:]:
                await adapter.connect()
            results = await asyncio.gather(
                *(one_shot(adapter, f"stream{index}", ctx.item.text) for index, adapter in enumerate(adapters))
            )
        finally:
            for adapter in adapters[1:]:
                await adapter.close()
        rows = [summarize(s) for s in results]
        ttfas = sorted(r["ttfa_ms"] for r in rows if r["ttfa_ms"] is not None)
        values = {
            "streams": self.streams,
            "streams_with_audio": len(ttfas),
            "per_stream": rows,
            "ttfa_ms": ttfas[len(ttfas) // 2] if ttfas else None,      # median across the burst
            "ttfa_max_ms": ttfas[-1] if ttfas else None,
            "ttfa_min_ms": ttfas[0] if ttfas else None,
            "underruns_total": sum(r["underruns"] or 0 for r in rows),
        }
        verdict = "pass" if len(ttfas) == self.streams else "fail"
        return ProbeResult(self.name, verdict, values=values, syntheses=list(results))
