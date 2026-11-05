"""Call score, ratings, tie margin, void-run guard, pass³ and the suite / language splits."""

import pytest

from conftest import BAD, write_call, write_run
from s2s_human_bench import rating
from s2s_human_bench.bench import family_means, load_bench, pass_all, scenario_scores
from s2s_human_bench.calls import Call, call_score, family_scores, load_call, model_runs, usable_run, verdict
from s2s_human_bench.config import ConfigError, site_order
from s2s_human_bench.languages import group_scenarios
from s2s_human_bench.s2s import suite_runs

# ---- call score ------------------------------------------------------------------


def test_verdict_kinds(cfg):
    boolean, count = cfg.skill("1.1"), cfg.skill("2.2")
    assert verdict(boolean, {"score": True}) is True
    assert verdict(boolean, {"score": None}) is None
    assert verdict(count, {"score": 0}) is True  # no problems
    assert verdict(count, {"score": 2}) is False


def test_call_score_blends_outcome_into_task(cfg):
    verdicts = {"1.1": True, "1.2": False, "2.1": None, "2.2": True}
    fam = family_scores(cfg, verdicts, outcome=50.0)
    assert fam == {"hearing": 50.0, "task": 0.5 * 100 + 0.5 * 50}
    assert call_score(cfg.weights, fam) == pytest.approx((50 + 75) / 2)


def test_unreached_family_drops_out_of_the_score(cfg):
    fam = family_scores(cfg, {"1.1": True, "1.2": True, "2.1": None, "2.2": None}, outcome=None)
    assert fam["task"] is None
    assert call_score(cfg.weights, fam) == 100.0
    assert call_score(cfg.weights, {"hearing": None, "task": None}) is None


def test_model_family_is_the_mean_of_its_skill_rates(cfg):
    def call(tested, misses, outcome):
        return Call(
            code="B01",
            call_id=None,
            url="",
            seconds=60,
            verdicts={},
            explanations={},
            families={},
            outcome=outcome,
            outcome_lines=[],
            score=None,
            latency_ms=None,
            turn_words=None,
            tested=tested,
            misses=misses,
        )

    # Skill 1.1 passes on three calls, skill 1.2 fails on its one call: each skill counts once, not each call.
    calls = [call(["1.1"], [], 100.0)] * 3 + [call(["1.2", "2.1"], ["1.2"], 0.0)]
    fam = family_means(cfg, calls)
    assert fam["Hearing"] == 50.0
    assert fam["Task"] == 0.5 * 100 + 0.5 * 75


def test_load_call(cfg, tmp_path):
    write_call(tmp_path / "r", "B01", {101: True, 102: None, 201: None, 202: 1}, ["✅ a", "❌ b", "🟡 c"])
    call = load_call(cfg, next((tmp_path / "r").glob("calls/*/*/call.json")))
    assert call.verdicts == {"1.1": True, "1.2": None, "2.1": None, "2.2": False}
    assert call.outcome == 50.0 and not call.task_done
    assert call.score == pytest.approx((100 + 25) / 2)
    assert call.tested == ["1.1", "2.2"] and call.misses == ["2.2"]
    assert call.seconds == 90 and call.latency_ms == 1200 and call.spoke


# ---- ratings ---------------------------------------------------------------------


def test_tie_margin():
    assert rating.outcome(80.0, 85.0, 5.0) == 0.5
    assert rating.outcome(80.0, 85.1, 5.0) == 0.0
    assert rating.outcome(90.0, 80.0, 5.0) == 1.0
    scores = {"S1": {"a": 80.0, "b": 84.0}, "S2": {"a": 90.0}}  # S2: b has no call, no game
    assert rating.matches_for(scores, ["a", "b"], 5.0) == [("a", "b", 0.5)]


def test_bradley_terry_orders_and_centres():
    scores = {f"S{i}": {"a": 90.0, "b": 70.0 if i % 3 else 95.0, "c": 50.0} for i in range(12)}
    r = rating.bradley_terry(rating.matches_for(scores, ["a", "b", "c"], 5.0), ["a", "b", "c"])
    assert r["a"] > r["b"] > r["c"]
    assert sum(r.values()) / 3 == pytest.approx(1000.0)
    even = rating.bradley_terry([("a", "b", 0.5)] * 5, ["a", "b"])
    assert even["a"] == pytest.approx(even["b"]) == pytest.approx(1000.0)


def test_reference_is_anchored_and_moves_no_one():
    scores = {f"S{i}": {"a": 90.0, "b": 70.0 if i % 3 else 95.0, "ref": 99.0} for i in range(9)}
    alone = rating.bradley_terry(rating.matches_for(scores, ["a", "b"], 5.0), ["a", "b"])
    both = rating.ratings(scores, ["a", "b"], ["ref"], 5.0)
    assert both["a"] == alone["a"] and both["b"] == alone["b"]
    assert both["ref"] > both["a"]  # it beat everyone
    assert rating.anchored_rating("ref", {"S1": {"a": 1.0}}, alone, 5.0) is None  # never played


def test_bootstrap_is_seeded():
    scores = {f"S{i}": {"a": 90.0 - i, "b": 70.0 + 3 * i} for i in range(8)}
    one = rating.bootstrap_ci(scores, ["a", "b"], 5.0, draws=30, seed=3, iters=20)
    assert one == rating.bootstrap_ci(scores, ["a", "b"], 5.0, draws=30, seed=3, iters=20)
    assert one["a"][0] <= one["a"][1]


# ---- runs ------------------------------------------------------------------------


def test_void_run_guard(cfg, tmp_path):
    write_run(tmp_path / "run_a1_r1", {}, silent=1)  # agent silent on 1 of 4: kept
    write_run(tmp_path / "run_a1_r2", {}, silent=2)  # 2 of 4: a broken run, dropped
    write_run(tmp_path / "run_a1_r3", {})
    (tmp_path / "run_a1_r4" / "calls").mkdir(parents=True)  # incomplete export: dropped
    assert usable_run(cfg, tmp_path / "run_a1_r1") and not usable_run(cfg, tmp_path / "run_a1_r2")
    model = cfg.models[0]
    assert [p.name for p in model_runs(cfg, model, tmp_path)] == ["run_a1_r1", "run_a1_r3"]


def test_models_without_enough_runs_are_left_out(cfg, tmp_path):
    for name in ("run_a1_r1", "run_a1_r2", "run_a2_r1"):
        write_run(tmp_path / name, {})
    assert [e.model.key for e in load_bench(cfg, tmp_path)] == ["alpha"]


def test_pass_all_needs_every_run(cfg, tmp_path):
    write_run(tmp_path / "run_a1_r1", {})
    write_run(tmp_path / "run_a1_r2", {"B02": BAD})
    (entry,) = load_bench(cfg, tmp_path)
    assert pass_all(cfg, entry.runs, "B01")
    assert not pass_all(cfg, entry.runs, "B02")
    assert not pass_all(cfg, [entry.runs[0], {}], "B01")  # missing from a run


# ---- splits ----------------------------------------------------------------------


def test_suite_split(cfg, tmp_path):
    write_run(tmp_path / "run_a1_r1", {})
    write_run(tmp_path / "run_a1_r2", {})
    (entry,) = load_bench(cfg, tmp_path)
    assert [sorted(r) for r in suite_runs(entry.runs, "C")] == [["C01", "C02"]] * 2
    assert sorted(scenario_scores([entry], lambda c: c.startswith("B"))) == ["B01", "B02"]


def test_language_groups_follow_the_switch_skill(cfg, tmp_path):
    write_run(tmp_path / "run_a1_r1", {})
    write_run(tmp_path / "run_a1_r2", {})
    entries = load_bench(cfg, tmp_path)
    groups = {g["id"]: codes for g, codes in group_scenarios(cfg, entries)}
    assert groups == {"all": {"B01", "B02", "C01", "C02"}, "en": {"B01", "B02", "C01"}, "xx": {"C02"}}
    cfg.raw["languages"]["groups"][2]["scenarios"] = ["C01"]  # config disagrees with where the skill applies
    with pytest.raises(ConfigError):
        group_scenarios(cfg, entries)


def test_site_order():
    assert sorted(["1.10", "3.5b", "1.8", "3.5a", "3.6"], key=site_order) == ["1.8", "1.10", "3.5a", "3.5b", "3.6"]
