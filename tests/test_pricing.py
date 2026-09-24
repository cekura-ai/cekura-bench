"""What a call consumed, and what that costs.

Consumption is reported during the call and then it is gone, so it is recorded
on the run and priced here, later, from a table that can be corrected without
re-running anything.
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
from pricing.cost import lanes, load_table, price_call  # noqa: E402


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


# One realtime call as the record keeps it: the audio and cached counts sit
# inside the prompt and output totals.
OPENAI_CALL = {
    "usage_reports": 15, "call_seconds": 112.4,
    "prompt_tokens": 51_733, "cache_read_input_tokens": 42_560,
    "input_audio_tokens": 1_933, "cache_read_input_audio_tokens": 1_024,
    "completion_tokens": 2_058, "output_audio_tokens": 1_316,
}


class TestPricingACall:

    def test_the_totals_split_into_the_lanes_vendors_price(self):
        assert lanes(OPENAI_CALL) == {
            "text_input": 8_264, "text_input_cached": 41_536,
            "audio_input": 909, "audio_input_cached": 1_024,
            "text_output": 742, "audio_output": 1_316,
        }

    def test_every_lane_is_priced_not_only_the_audio(self):
        # Text is most of the tokens on a realtime row: the prompt and the tools
        # are read again on every response. Pricing audio alone undercounts.
        price = price_call("openai-realtime", OPENAI_CALL)
        expected = (8_264 * 4 + 41_536 * 0.4 + 909 * 32 + 1_024 * 0.4 + 742 * 24 + 1_316 * 64) / 1e6
        assert price.usd == pytest.approx(expected)
        assert price.unmetered, "what the figure leaves out travels with it"

    def test_a_smaller_tier_is_priced_at_its_own_rates(self):
        # Same service, same usage fields: only the row tells the two apart, so
        # the row has to carry the smaller model's rates, not the larger one's.
        mini = price_call("openai-realtime-mini", OPENAI_CALL)
        expected = (8_264 * 0.6 + 41_536 * 0.06 + 909 * 10 + 1_024 * 0.3 + 742 * 2.4 + 1_316 * 20) / 1e6
        assert mini.usd == pytest.approx(expected)
        assert mini.usd < price_call("openai-realtime", OPENAI_CALL).usd
        # Each row names the model its rates were read for.
        table = load_table()
        for row in ("openai-realtime-mini", "gemini-flash-live"):
            assert table["rows"][row]["components"][0]["name"] in bot.PROVIDERS[row].default_model

    def test_reasoning_reported_beside_the_output_is_billed_as_output(self):
        usage = {"usage_reports": 1, "prompt_tokens": 100, "input_audio_tokens": 100,
                 "completion_tokens": 50, "output_audio_tokens": 50, "reasoning_tokens": 30}
        assert lanes(usage, reasoning_in_output=False)["text_output"] == 30
        assert lanes(usage, reasoning_in_output=True)["text_output"] == 0

    def test_a_live_row_is_priced_on_its_audio_seconds_plus_its_backend(self):
        usage = {"usage_reports": 8, "call_seconds": 106.5, "live_audio_seconds": 89.0,
                 "prompt_tokens": 48_620, "cache_read_input_tokens": 44_656, "completion_tokens": 234}
        price = price_call("gpt-live", usage)
        assert price.lines["gpt-live-1"] == pytest.approx(0.05 * 89 / 60)
        assert price.lines["backend gpt-6-sol"] == pytest.approx(
            (3_964 * 2 + 44_656 * 0.2 + 234 * 10) / 1e6)
        assert price.usd == pytest.approx(sum(price.lines.values()))

    def test_a_live_row_without_its_seconds_is_not_priced_on_the_clock(self):
        # The vendor bills the seconds it reports, not the pipeline's lifetime.
        price = price_call("gpt-live", {"usage_reports": 1, "call_seconds": 120})
        assert price.usd is None and "live_audio_seconds" in price.reason

    def test_a_per_minute_row_is_priced_by_the_clock(self):
        price = price_call("grok-realtime", {"call_seconds": 120, "usage_reports": 3})
        assert price.usd == pytest.approx(0.16)

    def test_a_record_without_the_speech_split_is_not_priced_as_text(self):
        # Speech costs about ten times text there; the total alone would look cheap.
        price = price_call("nova-sonic", {"usage_reports": 140, "prompt_tokens": 3_571, "completion_tokens": 1_991})
        assert price.usd is None and "input_audio_tokens" in price.reason

    def test_a_billed_lane_with_no_rate_is_a_refusal_not_free(self):
        usage = {"usage_reports": 1, "prompt_tokens": 10, "input_audio_tokens": 10,
                 "cache_read_input_tokens": 10, "cache_read_input_audio_tokens": 10,
                 "completion_tokens": 5, "output_audio_tokens": 5}
        price = price_call("gemini-live", usage)
        assert price.usd is None and "audio_input_cached" in price.reason

    def test_usage_that_does_not_add_up_is_refused(self):
        price = price_call("openai-realtime", {"usage_reports": 1, "prompt_tokens": 10,
                                               "input_audio_tokens": 50, "output_audio_tokens": 1})
        assert price.usd is None and "does not add up" in price.reason

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
        usage = {"usage_reports": 1, "prompt_tokens": 100, "input_audio_tokens": 80,
                 "completion_tokens": 60, "output_audio_tokens": 50}
        price = price_call("nova-sonic", usage)
        assert price.usd is not None
        assert price.publishable is False, "a rate must be verified before a row goes out"

    def test_every_row_the_agent_can_run_has_an_entry(self):
        table = load_table()
        priced = set(table["rows"]) | set(table["cascade_text_models"])
        missing = (set(bot.PROVIDERS) | set(bot.TEXT_MODELS)) - priced
        assert not missing, f"a row with no price entry would publish a blank cost: {sorted(missing)}"

    def test_a_recorded_rate_carries_the_date_and_page_it_was_read_from(self):
        # An undated price is not reproducible.
        table = load_table()
        for name, entry in {**table["rows"], **table["cascade_text_models"]}.items():
            if any(c.get("rates") for c in entry.get("components") or ()):
                assert entry.get("read_on") and entry.get("source"), f"{name} has rates but no date or source"

    def test_every_rate_names_a_lane_the_arithmetic_knows(self):
        table = load_table()
        known = set(lanes({}))
        for name, entry in table["rows"].items():
            for component in entry.get("components") or ():
                if component["basis"] == "per_million_tokens":
                    assert set(component["rates"]) <= known, f"{name}: {set(component['rates']) - known}"

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
