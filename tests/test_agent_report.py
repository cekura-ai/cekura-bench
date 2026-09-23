"""The agent bench's own board columns, built from the records the agent stamped on its runs."""

from __future__ import annotations

import json

import pytest

from agent.report import ReportError, load_runs, summarize, turns


def record(**overrides):
    base = {
        "agent_commit": "a" * 40, "config": "grok-realtime", "s2s_provider": "grok-realtime",
        "s2s_model": "m", "s2s_voice": "v", "stack": "native", "turn_source": "provider",
        "pipeline_sample_rate": 24000, "agent_definition": "appointments",
        "system_prompt_sha256": "p", "first_message_sha256": "f", "tools": "t",
        "pipecat_version": "1.11.0", "cekura_version": "1.6.7", "divergences": {},
        "grok_reasoning": "high",
        "usage": {"call_seconds": 60.0, "usage_reports": 3},
        "integrity": {"checks": ["ok"], "hangup_tail_ms": 20},
        "tool_calls": [
            {"name": "lookup", "arguments": {"id": "1"}, "resolution": "exact"},
            {"name": "end_call", "arguments": {}, "resolution": "exact"},
        ],
    }
    base.update(overrides)
    return base


def runs(*records):
    return [{"run_id": i, "scenario_id": i % 2, "custom_metadata": r} for i, r in enumerate(records)]


class TestOneRowIsOneBuild:
    def test_runs_from_two_builds_are_refused(self):
        with pytest.raises(ReportError, match="agent_commit"):
            summarize(runs(record(), record(agent_commit="b" * 40)))

    def test_runs_from_two_configurations_are_refused(self):
        with pytest.raises(ReportError, match="s2s_voice"):
            summarize(runs(record(), record(s2s_voice="other")))

    def test_what_a_row_discloses_is_carried_by_name(self):
        row = summarize(runs(record(), record()))["rows"][0]
        assert row["configuration"]["disclosures"] == {"grok_reasoning": "high"}

    def test_a_run_without_an_agent_record_is_refused(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        path.write_text(json.dumps({"run_id": 1}) + "\n")
        with pytest.raises(ReportError, match="no agent record"):
            load_runs([path])


class TestCost:
    def test_per_minute_is_a_ratio_of_totals(self):
        # 60 s and 180 s at $0.08/min: $0.08 + $0.24 over 4 minutes.
        row = summarize(runs(record(), record(usage={"call_seconds": 180.0, "usage_reports": 1})))["rows"][0]
        assert row["cost"]["per_minute_usd"] == pytest.approx(0.08)
        assert row["cost"]["per_call_usd"] == pytest.approx(0.16)
        assert row["cost"]["publishable"] is True

    def test_an_unpriced_run_is_counted_and_blocks_publication(self):
        nova = {"config": "nova-sonic", "s2s_provider": "nova-sonic"}
        row = summarize(runs(
            record(**nova, usage={"call_seconds": 60, "usage_reports": 5, "prompt_tokens": 90, "completion_tokens": 40}),
        ))["rows"][0]
        assert row["cost"]["priced_runs"] == 0
        assert row["cost"]["publishable"] is False
        assert any("input_audio_tokens" in reason for reason in row["cost"]["unpriced_runs"])


class TestIntegrityAndTools:
    def test_a_flagged_run_is_counted_not_dropped(self):
        flagged = record(integrity={"checks": ["hangup_held"], "hangup_tail_ms": 4200})
        row = summarize(runs(record(), flagged))["rows"][0]
        assert row["runs"] == 2
        assert row["integrity"]["ok_runs"] == 1
        assert row["integrity"]["flags"] == {"hangup_held": 1}
        assert row["integrity"]["hangup_tail_ms"]["max"] == 4200

    def test_a_repeated_call_and_a_missing_hang_up_are_counted(self):
        repeated = record(tool_calls=[
            {"name": "lookup", "arguments": {"id": "1"}, "resolution": "exact"},
            {"name": "lookup", "arguments": {"id": "1"}, "resolution": "exact"},
        ])
        withdrawn = record(tool_calls=[
            {"name": "lookup", "arguments": {"id": "1"}, "resolution": "exact", "cancelled": True},
            {"name": "lookup", "arguments": {"id": "1"}, "resolution": "exact"},
            {"name": "end_call", "arguments": {}},
        ])
        tools = summarize(runs(repeated, withdrawn))["rows"][0]["tools"]
        assert tools["runs_with_a_repeated_call"] == 1, "a withdrawn call is not a repeat"
        assert tools["runs_not_closed_by_the_agent"] == 1
        assert tools["cancelled"] == 1


class TestEveryTurnIsWrittenOut:
    def test_each_timed_turn_is_one_row_with_its_place_in_the_call(self):
        timed = record(timing={
            "reply": {"p50_ms": 900, "turns": [{"caller_stopped_at_s": 4.2, "ms": 850},
                                               {"caller_stopped_at_s": 11.0, "ms": 950}]},
            "endpointing": {"p50_ms": 40, "turns": [{"caller_stopped_at_s": 4.2, "ms": 40}]},
        })
        rows = turns(runs(timed, record()))
        assert [(r["measure"], r["caller_stopped_at_s"], r["ms"]) for r in rows] == [
            ("agent_reply_ms", 4.2, 850), ("agent_reply_ms", 11.0, 950), ("agent_endpointing_ms", 4.2, 40),
        ]
        assert {r["run_id"] for r in rows} == {0}

    def test_who_closed_each_call_is_tallied(self):
        by_agent = record(integrity={"checks": ["ok"], "hangup_tail_ms": 20, "closed_by": "agent"})
        by_backstop = record(integrity={"checks": ["ok"], "hangup_tail_ms": 20, "closed_by": "harness"})
        left_open = record(integrity={"checks": ["ok"]})
        closers = summarize(runs(by_agent, by_backstop, left_open))["rows"][0]["integrity"]["closed_by"]
        assert closers == {"agent": 1, "harness": 1, "caller_or_timeout": 1}
