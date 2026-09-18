"""The aggregation rules, checked on cells whose right answer is known.

These are the rules that turn a directory of cells into a published row, so a
mistake here is a mistake in every row. They are tested on synthetic cells
rather than on a run, because the run would only show that the code ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lane_a.report import (
    MIN_TAIL_N,
    TIE_MS,
    clustered_success,
    compare_medians,
    latency_summary,
    load_run,
    rate_summary,
    render_markdown,
    summarize_run,
)


class TestLatency:
    def test_tails_are_withheld_below_the_declared_minimum(self):
        few = latency_summary([900.0 + i for i in range(MIN_TAIL_N - 1)])
        assert "p95" not in few and "tails_withheld" in few
        enough = latency_summary([900.0 + i for i in range(MIN_TAIL_N)])
        assert "p95" in enough and "p99" in enough

    def test_the_interval_is_reproducible_and_brackets_the_median(self):
        values = [1000.0, 1010.0, 990.0, 1005.0, 1020.0]
        a, b = latency_summary(values), latency_summary(values)
        assert a["p50_ci95"] == b["p50_ci95"]
        lo, hi = a["p50_ci95"]
        assert lo <= a["p50"] <= hi

    def test_nones_are_not_zeros(self):
        assert latency_summary([None, None])["n"] == 0
        assert latency_summary([None, 800.0])["p50"] == 800.0


class TestTies:
    def test_a_gap_inside_detector_resolution_is_a_tie(self):
        a = latency_summary([1000.0] * 5)
        b = latency_summary([1000.0 + TIE_MS * 0.9] * 5)
        assert compare_medians(a, b)["verdict"] == "tie"

    def test_a_gap_outside_it_is_a_ranking(self):
        a = latency_summary([1000.0] * 5)
        b = latency_summary([1000.0 + TIE_MS * 3] * 5)
        assert compare_medians(a, b)["verdict"] == "a"


class TestSuccess:
    def test_all_repeats_is_what_was_observed_not_a_power(self):
        """Four of five is 0.8 per run and 0.0 all-repeats; never 0.8 ** 5."""
        result = clustered_success({"s1": [True, True, True, True, False]})
        assert result["per_run_rate"] == 0.8
        assert result["all_repeats_rate"] == 0.0
        assert result["all_repeats_passed"] == []

    def test_clusters_are_scenarios(self):
        result = clustered_success({"a": [True] * 5, "b": [False] * 5, "c": [True, True, True, True, False]})
        assert result["scenarios"] == 3
        assert result["per_run_rate"] == pytest.approx(9 / 15, abs=1e-3)
        assert result["all_repeats_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert result["all_repeats_passed"] == ["a"]
        lo, hi = result["per_run_ci95"]
        assert lo <= result["per_run_rate"] <= hi

    def test_rates_ignore_nothing_silently(self):
        assert rate_summary([])["n"] == 0
        assert rate_summary([True, False])["rate"] == 0.5


def _cell(probe, variant, config, voice, transform, repeat, *, verdict=None, void=None, values=None, slug=None):
    return {
        "provider": "fake", "model": "scripted", "probe": probe, "variant": variant, "config": config,
        "voice": voice, "transform": transform, "repeat": repeat, "verdict": verdict, "void": void,
        "values": values or {}, "artifacts": {"slug": slug or f"{variant}/{config}/{voice}/{transform}/r{repeat}"},
        "usage": {}, "error": None,
    }


def write_run(root: Path, cells, plan):
    root.mkdir(parents=True)
    (root / "cells.jsonl").write_text("\n".join(json.dumps(c) for c in cells) + "\n")
    (root / "plan.json").write_text(json.dumps({"run_id": root.name, "cells": plan}))
    (root / "provenance.json").write_text(json.dumps({
        "run_id": root.name, "provider": "fake", "model": "scripted", "methodology_version": "t",
        "harness": {"commit": "abc", "dirty": False}, "corpus_version": "0.2.0",
    }))


class TestRunReport:
    def test_strata_stay_apart_and_deltas_are_from_clean(self, tmp_path):
        cells = []
        plan = []
        for repeat in (1, 2, 3):
            for transform, ms in (("clean", 1000.0), ("telephone", 1150.0)):
                cell = _cell("response_latency", "response_latency-open.book", "server_vad-500ms", "f-us", transform, repeat,
                             verdict="pass", values={"latency_ms": ms + repeat})
                cells.append(cell)
                plan.append({"cell_id": cell["artifacts"]["slug"], "sentinel": False})
        sentinel = _cell("sentinel_latency", "sentinel_latency-open.book", "server_vad-500ms", "f-us", "clean", 1,
                         verdict="pass", values={"latency_ms": 990.0})
        cells.append(sentinel)
        plan.append({"cell_id": sentinel["artifacts"]["slug"], "sentinel": True})
        voided = _cell("response_latency", "response_latency-open.book", "semantic_vad", "f-us", "clean", 1,
                       void="configuration not supported: fake has no semantic turn detection")
        cells.append(voided)
        plan.append({"cell_id": voided["artifacts"]["slug"], "sentinel": False})

        write_run(tmp_path / "run", cells, plan)
        report = summarize_run(load_run(tmp_path / "run"))

        groups = report["groups"]
        assert "response_latency/response_latency-open.book/server_vad-500ms/f-us/clean" in groups
        assert "response_latency/response_latency-open.book/server_vad-500ms/f-us/telephone" in groups
        delta = report["degradation_deltas"]["response_latency/response_latency-open.book/server_vad-500ms/f-us/telephone"]
        assert delta["p50_delta_ms"] == 150.0
        assert report["sentinel"]["latency"]["p50"] == 990.0
        # the sentinel is not a stratum of the result
        assert not any(key.startswith("sentinel") for key in groups)
        excluded = groups["response_latency/response_latency-open.book/semantic_vad/f-us/clean"]
        assert excluded["scored"] == 0 and list(excluded["voids"]) == ["excluded: fake has no semantic turn detection"]
        text = render_markdown(report)
        assert "Sentinel" in text and "telephone" in text and "Voids" in text
