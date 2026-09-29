"""The caller's state machine, against a scripted agent with known reply timing.

Ground truth is available here and nowhere else: the fake agent's endpointing
delay and reply length are set by the test, so a metric can be checked against
the number it should have produced rather than against one that merely looks
plausible.
"""

from __future__ import annotations


import numpy as np
import pytest

from service import events as ev
from service.adapters.base import SessionConfig, TurnDetection
from service.adapters.fake import FakeAdapter
from service.audio import to_pcm
from service.caller import BranchingCaller, Clip
from service.metrics import barge_in, response_latency, spoke_between

pytestmark = pytest.mark.asyncio


def speech_clip(name: str = "utterance", ms: float = 800.0, rate: int = 24000) -> Clip:
    """A clip with unambiguous energy and 100 ms of trailing render silence."""
    n = int(rate * ms / 1000.0)
    t = np.arange(n) / rate
    body = 9000 * np.sin(2 * np.pi * 180.0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t))
    tail = np.zeros(int(rate * 0.1))
    return Clip(name, to_pcm(np.concatenate([body, tail])), rate, "test utterance")


async def harness(**adapter_kwargs):
    log = ev.EventLog(ev.Clock())
    config = adapter_kwargs.pop("config", SessionConfig(turn_detection=TurnDetection("server_vad", silence_duration_ms=300)))
    adapter = FakeAdapter(log=log, config=config, **adapter_kwargs)
    await adapter.connect()
    return adapter, BranchingCaller(adapter, log), log


class TestCarrier:
    async def test_stream_keeps_running_between_clips(self):
        adapter, caller, _ = await harness(reply_ms=200)
        async with caller:
            await caller.wait(200)
            sent = adapter.caller_timeline.n_samples
            assert sent > 0, "the carrier must hold the line open while nobody speaks"
            await caller.wait(200)
            assert adapter.caller_timeline.n_samples > sent
        await adapter.close()

    async def test_clip_samples_are_accounted_exactly(self):
        adapter, caller, _ = await harness(reply_ms=100)
        clip = speech_clip(ms=500)
        async with caller:
            utterance = await caller.play(clip)
        expected = len(clip.pcm) // 2
        assert utterance.last_sample - utterance.first_sample == expected
        # The authored boundary sits inside the clip, before the render tail.
        assert utterance.first_sample < utterance.speech_end_sample <= utterance.last_sample
        assert utterance.last_sample - utterance.speech_end_sample > 0
        await adapter.close()

    async def test_trim_tail_stops_at_the_authored_boundary(self):
        adapter, caller, _ = await harness(reply_ms=100)
        async with caller:
            utterance = await caller.play(speech_clip(ms=500), trim_tail=True)
        assert utterance.speech_end_sample == utterance.last_sample
        await adapter.close()


class TestAnchors:
    async def test_agent_quiet_waits_for_the_reply_to_finish(self):
        adapter, caller, log = await harness(reply_ms=600, config=SessionConfig(
            turn_detection=TurnDetection("server_vad", silence_duration_ms=200)))
        async with caller:
            await caller.play(speech_clip(ms=400))
            assert await caller.wait_agent_onset(timeout_s=5)
            started = log.clock.now()
            assert await caller.wait_agent_quiet(gap_ms=150, timeout_s=5)
            waited_ms = (log.clock.now() - started) * 1000.0
        # The agent speaks for 600 ms; we must not take our turn before then.
        assert waited_ms > 300, waited_ms
        await adapter.close()

    async def test_agent_onset_times_out_when_nothing_replies(self):
        adapter, caller, _ = await harness(config=SessionConfig(turn_detection=TurnDetection("manual")))
        async with caller:
            assert not await caller.wait_agent_onset(timeout_s=0.3)
        await adapter.close()


class TestMeasurement:
    async def test_manual_commit_latency_excludes_endpointing(self):
        """The floor: we declare the boundary, so only generation is measured."""
        adapter, caller, _ = await harness(reply_ms=300, config=SessionConfig(turn_detection=TurnDetection("manual")))
        async with caller:
            utterance = await caller.play(speech_clip(ms=500), trim_tail=True)
            await adapter.commit()
            assert await caller.wait_agent_onset(timeout_s=5)
            await caller.wait_agent_quiet(200, timeout_s=5)
        latency = response_latency(adapter, utterance)
        assert latency.latency_ms is not None
        # Nothing but scheduling sits between commit and the first tone sample.
        assert 0 <= latency.latency_ms < 150, latency.as_json()
        await adapter.close()

    async def test_native_vad_latency_contains_the_configured_silence(self):
        silence_ms = 400
        adapter, caller, _ = await harness(reply_ms=300, config=SessionConfig(
            turn_detection=TurnDetection("server_vad", silence_duration_ms=silence_ms)))
        async with caller:
            utterance = await caller.play(speech_clip(ms=500))
            assert await caller.wait_agent_onset(timeout_s=5)
            await caller.wait_agent_quiet(200, timeout_s=5)
        latency = response_latency(adapter, utterance)
        # The endpointer starts counting from the last loud chunk, and the clip
        # carries 100 ms of render tail, so the wait lands just under the setting.
        assert latency.latency_ms > silence_ms - 150, latency.as_json()
        await adapter.close()

    async def test_barge_in_is_detected_and_timed(self):
        adapter, caller, log = await harness(reply_ms=3000, config=SessionConfig(
            turn_detection=TurnDetection("server_vad", silence_duration_ms=200)))
        async with caller:
            await caller.play(speech_clip(ms=400))
            assert await caller.wait_agent_onset(timeout_s=5)
            await caller.wait(500)
            assert adapter.agent_speaking
            interrupt = await caller.play(speech_clip("interrupt", ms=300))
            await caller.wait(500)
        result = barge_in(adapter, interrupt)
        assert result.stopped and result.cancelled_by_provider
        assert result.stop_ms is not None and result.stop_ms < 500, result
        await adapter.close()

    async def test_an_agent_that_holds_the_floor_fails_barge_in(self):
        adapter, caller, _ = await harness(reply_ms=3000, yields_to_barge_in=False, config=SessionConfig(
            turn_detection=TurnDetection("server_vad", silence_duration_ms=200)))
        async with caller:
            await caller.play(speech_clip(ms=400))
            assert await caller.wait_agent_onset(timeout_s=5)
            await caller.wait(500)
            interrupt = await caller.play(speech_clip("interrupt", ms=300))
            await caller.wait(400)
            result = barge_in(adapter, interrupt, settle_ms=300)
        assert not result.stopped and not result.cancelled_by_provider
        await adapter.close()

    async def test_spoke_between_sees_an_interrupting_agent(self):
        adapter, caller, log = await harness(reply_ms=500, config=SessionConfig(
            turn_detection=TurnDetection("server_vad", silence_duration_ms=150)))
        async with caller:
            await caller.play(speech_clip(ms=300))
            gap_start = log.clock.now()
            await caller.wait(900)
            gap_end = log.clock.now()
        assert spoke_between(adapter, gap_start, gap_end), "a 150 ms endpointer must fire inside a 900 ms gap"
        await adapter.close()
