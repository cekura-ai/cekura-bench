"""Comparisons: the score difference and its interval, and which skill changes count as clear."""

import json

from test_site_export import export, make_inputs

from s2s_human_bench.bench import Entry
from s2s_human_bench.calls import Call
from s2s_human_bench.comparisons import bh, paired, skill_changes, t_quantile


def call(code, score, tested=(), misses=()):
    return Call(
        code=code,
        call_id=code,
        url="",
        seconds=60,
        verdicts={},
        explanations={},
        families={},
        outcome=None,
        outcome_lines=[],
        score=score,
        latency_ms=None,
        turn_words=None,
        tested=list(tested),
        misses=list(misses),
    )


def entry(cfg, index, runs):
    return Entry(model=cfg.models[index], runs=runs)


def test_t_quantile_matches_tables():
    assert abs(t_quantile(0.975, 69) - 1.995) < 0.002 and abs(t_quantile(0.975, 20) - 2.086) < 0.003


def test_bh_adjusts_in_input_order():
    assert bh([0.01, 0.04, 0.03, 0.5]) == [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5]


def test_identical_models_are_not_separable(cfg):
    runs = [{f"S{i}": call(f"S{i}", 70 + i % 5) for i in range(20)} for _ in range(3)]
    d = paired(entry(cfg, 0, runs), entry(cfg, 1, runs), lambda c: c.score, 0.95)
    assert d == {"diff": 0.0, "lo": 0.0, "hi": 0.0, "scenarios": 20, "separable": False}


def test_a_steady_gain_is_separable_and_signed_b_minus_a(cfg):
    a = [{f"S{i}": call(f"S{i}", 70 + i % 5) for i in range(20)} for _ in range(3)]
    b = [{f"S{i}": call(f"S{i}", 75 + i % 7) for i in range(20)} for _ in range(3)]
    d = paired(entry(cfg, 0, a), entry(cfg, 1, b), lambda c: c.score, 0.95)
    assert d["diff"] > 0 and d["lo"] > 0 and d["separable"]


def runs_with(skill, passes_per_run, judged=10):
    """Three runs where ``skill`` is judged on ``judged`` calls and passes on the given counts."""
    return [{f"S{i}": call(f"S{i}", 80, tested=[skill], misses=[] if i < k else [skill]) for i in range(judged)} for k in passes_per_run]


def test_clear_change_needs_every_run_and_significance(cfg):
    settings = {"skillQ": 0.1, "minSkillCallsPerRun": 5}
    clear, within = skill_changes(cfg, entry(cfg, 0, runs_with("1.1", [2, 3, 2])), entry(cfg, 1, runs_with("1.1", [9, 10, 9])), settings)
    assert [s["skill"] for s in clear] == ["1.1"] and clear[0]["a"] == 23 and clear[0]["b"] == 93 and within == 0
    # every run moves the same way, but by too little for 30 calls: counted, not listed
    clear, within = skill_changes(cfg, entry(cfg, 0, runs_with("1.1", [6, 7, 6])), entry(cfg, 1, runs_with("1.1", [8, 8, 8])), settings)
    assert clear == [] and within == 1
    # a big pooled gap that one run contradicts is not clear
    clear, _ = skill_changes(cfg, entry(cfg, 0, runs_with("1.1", [2, 3, 9])), entry(cfg, 1, runs_with("1.1", [9, 10, 8])), settings)
    assert clear == []


def test_thin_skills_are_left_out(cfg):
    clear, within = skill_changes(
        cfg,
        entry(cfg, 0, runs_with("1.1", [0, 0, 0], judged=4)),
        entry(cfg, 1, runs_with("1.1", [4, 4, 4], judged=4)),
        {"skillQ": 0.1, "minSkillCallsPerRun": 5},
    )
    assert clear == [] and within == 0


def test_export_writes_every_pair(cfg, tmp_path):
    make_inputs(tmp_path)
    doc = export(tmp_path, cfg, tmp_path / "out")["comparisons.json"]
    pairs = {(p["a"], p["b"]): p for p in doc["pairs"]}
    assert set(pairs) == {("alpha-site", "beta-site"), ("alpha-site", "ref-site"), ("beta-site", "ref-site")}
    ab = pairs["alpha-site", "beta-site"]
    assert ab["score"]["diff"] < 0 and ab["score"]["scenarios"] == 4  # beta is worse; b minus a
    assert set(ab["families"]) == {"Hearing", "Task"} and doc["skills"]["1.1"] == "Skill 1.1"
    json.dumps(doc)  # serialisable as written
