"""What a call consumed, and what that costs.

Cost is one of the few columns a benchmark is read for and the only one that
cannot be recovered afterwards: consumption is reported during the call and then
it is gone. It does reach the trace store, but that empties after thirty days
and a board is questioned months later. So it is recorded on the run, and priced
here, later, from a table that can be corrected without re-running anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent / "reference-agents" / "pipecat-s2s"
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot  # noqa: E402
from pricing.cost import load_table, price_call  # noqa: E402


def metrics(*entries):
    from pipecat.frames.frames import MetricsFrame

    return MetricsFrame(data=list(entries))


def pushed(frame):
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    return FramePushed(
        source=None, destination=None, frame=frame,
        direction=FrameDirection.DOWNSTREAM, timestamp=0,
    )


def tokens(**fields):
    from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData

    fields.setdefault("prompt_tokens", 0)
    fields.setdefault("completion_tokens", 0)
    fields.setdefault("total_tokens", 0)
    return LLMUsageMetricsData(processor="test", value=LLMTokenUsage(**fields))


class TestCountingWhatACallUsed:
    @pytest.mark.asyncio
    async def test_the_lanes_stay_apart(self):
        """Audio costs a multiple of text and cached input a fraction of fresh.

        A single total cannot be priced at all, so the split is the measurement.
        """
        meter = bot.UsageMeter()
        await meter.on_push_frame(pushed(metrics(
            tokens(input_audio_tokens=1000, output_audio_tokens=400, cache_read_input_tokens=90)
        )))
        usage = meter.as_metadata()["usage"]
        assert usage["input_audio_tokens"] == 1000
        assert usage["output_audio_tokens"] == 400
        assert usage["cache_read_input_tokens"] == 90

    @pytest.mark.asyncio
    async def test_reports_accumulate_across_a_call(self):
        meter = bot.UsageMeter()
        for _ in range(3):
            await meter.on_push_frame(pushed(metrics(tokens(input_audio_tokens=100))))
        usage = meter.as_metadata()["usage"]
        assert usage["input_audio_tokens"] == 300
        assert usage["usage_reports"] == 3

    @pytest.mark.asyncio
    async def test_the_same_frame_is_not_billed_twice(self):
        # Metrics frames are broadcast, so one arrives more than once.
        meter = bot.UsageMeter()
        frame = metrics(tokens(input_audio_tokens=100))
        await meter.on_push_frame(pushed(frame))
        await meter.on_push_frame(pushed(frame))
        assert meter.as_metadata()["usage"]["input_audio_tokens"] == 100

    @pytest.mark.asyncio
    async def test_a_cascade_is_measured_the_way_it_is_billed(self):
        from pipecat.metrics.metrics import STTUsage, STTUsageMetricsData, TTSUsageMetricsData

        meter = bot.UsageMeter()
        await meter.on_push_frame(pushed(metrics(
            STTUsageMetricsData(processor="stt", value=STTUsage(audio_seconds=12.5)),
            TTSUsageMetricsData(processor="tts", value=340),
        )))
        usage = meter.as_metadata()["usage"]
        assert usage["stt_audio_seconds"] == 12.5
        assert usage["tts_characters"] == 340

    @pytest.mark.asyncio
    async def test_a_silent_provider_is_visible_as_silent(self):
        """Zero tokens and no reports are different findings from zero tokens."""
        usage = bot.UsageMeter().as_metadata()["usage"]
        assert usage["usage_reports"] == 0
        assert "input_audio_tokens" not in usage

    def test_the_length_of_the_call_is_recorded(self):
        # Some rows are billed by the minute, so duration is part of usage.
        assert bot.UsageMeter().as_metadata()["usage"]["call_seconds"] >= 0


class TestPricingACall:
    def test_a_token_row_is_priced_by_lane(self):
        price = price_call("openai-realtime", {"input_audio_tokens": 1_000_000})
        assert price.usd == pytest.approx(32.0)
        assert price.basis == "per_million_tokens"

    def test_a_per_minute_row_is_priced_by_the_clock(self):
        price = price_call("gpt-live", {"call_seconds": 120})
        assert price.usd == pytest.approx(0.10)

    def test_a_row_with_no_rate_yields_a_reason_not_a_zero(self):
        price = price_call("qwen-realtime", {"call_seconds": 60})
        assert price.usd is None and price.reason

    def test_a_provider_that_reported_nothing_is_not_free(self):
        """Free and "we could not measure it" are opposite findings."""
        price = price_call("nova-sonic", {"call_seconds": 60, "usage_reports": 0})
        assert price.usd is None
        assert "no usage" in price.reason

    def test_an_unknown_row_is_named_in_the_reason(self):
        assert "not-a-provider" in price_call("not-a-provider", {}).reason


class TestTheTableItself:
    def test_nothing_is_publishable_until_a_rate_is_checked(self):
        # An unverified price may be computed and looked at; publishing one
        # would be asking to be believed rather than checked.
        price = price_call("openai-realtime", {"input_audio_tokens": 1_000_000})
        assert price.usd is not None
        assert price.publishable is False, "a rate must be verified before a row goes out"

    def test_every_row_the_agent_can_run_has_an_entry(self):
        table = load_table()
        priced = set(table["rows"]) | set(table["cascade_text_models"])
        missing = (set(bot.PROVIDERS) | set(bot.TEXT_MODELS)) - priced
        assert not missing, f"a row with no price entry would publish a blank cost: {sorted(missing)}"

    def test_a_recorded_rate_carries_the_date_it_was_read(self):
        # An undated price is not reproducible.
        table = load_table()
        for name, entry in {**table["rows"], **table["cascade_text_models"]}.items():
            if entry.get("rates"):
                assert entry.get("read_on"), f"{name} has rates but no date"

    def test_the_table_is_valid_json_on_disk(self):
        json.loads((Path(__file__).resolve().parent.parent / "pricing" / "prices.json").read_text())


class TestTheMeterIsActuallyConnected:
    """A meter that works and is not attached measures nothing, silently."""

    @pytest.mark.asyncio
    async def test_a_metrics_frame_in_a_running_pipeline_reaches_it(self):
        import asyncio

        from pipecat.frames.frames import Frame, StartFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        class Reporter(FrameProcessor):
            def __init__(self):
                super().__init__()
                self.done = asyncio.Event()

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, StartFrame):
                    self.create_task(self._report())
                await self.push_frame(frame, direction)

            async def _report(self):
                await asyncio.sleep(0.05)
                await self.push_frame(metrics(tokens(input_audio_tokens=777)))
                await asyncio.sleep(0.3)
                self.done.set()

        meter = bot.UsageMeter()
        reporter = Reporter()
        task = PipelineTask(Pipeline([reporter]), params=PipelineParams(), observers=[meter])
        running = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
        await asyncio.wait_for(reporter.done.wait(), timeout=30)
        await task.stop_when_done()
        await asyncio.wait_for(running, timeout=30)

        assert meter.as_metadata()["usage"]["input_audio_tokens"] == 777
