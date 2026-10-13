"""Conversational Dynamics scorecard and conversation-level bootstrap: the
mapping from Sesame's rates, n/a handling, seeded determinism, pooled-count
resampling on a hand-computed case, and the two-run comparison."""

import copy
import math

import numpy as np
import pytest
from turnbench.score import ConversationScores, TaskScore

from turn_bench.rescore import compare_reports, render_compare
from turn_bench.scorecard import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    SCORECARD,
    Counts,
    bootstrap_block,
    draw_matrix,
    intervals_disjoint,
    resolve_weights,
    task_rates,
)


def conv(eot=(0, 0, 0, 0), int_=(0, 0, 0, 0), eot_latencies=(), int_latencies=()) -> ConversationScores:
    return ConversationScores(
        TaskScore(*eot, latencies_ms=list(eot_latencies)),
        TaskScore(*int_, latencies_ms=list(int_latencies)),
    )


def test_scorecard_maps_pooled_rates_onto_the_four_names():
    # Pooled over both conversations: EOT tp=3 fn=1 (recall .75), fp=1 tn=3 (fp_rate .25);
    # INT tp=1 fn=1 (recall .5), fp=2 tn=2 (fp_rate .5).
    scores = [
        conv(eot=(2, 0, 1, 1), int_=(1, 0, 1, 1), eot_latencies=(100.0, 300.0), int_latencies=(50.0,)),
        conv(eot=(1, 1, 0, 2), int_=(0, 1, 1, 1), eot_latencies=(200.0,)),
    ]
    card = bootstrap_block(scores)["scorecard"]
    assert card["pause_handling"]["value"] == 0.75 and card["pause_handling"]["n"] == 4
    assert card["turn_taking"]["value"] == 0.75 and card["turn_taking"]["n"] == 4
    assert card["interruption_handling"]["value"] == 0.5 and card["interruption_handling"]["n"] == 2
    assert card["backchannel_handling"]["value"] == 0.5 and card["backchannel_handling"]["n"] == 4
    assert card["overall"]["value"] == pytest.approx((0.75 + 0.75 + 0.5 + 0.5) / 4)
    assert card["overall"]["defined"] == list(SCORECARD) and card["overall"]["omitted"] == {}
    latency = card["turn_taking"]["latency_ms"]
    assert latency["n"] == 3 and latency["p50"]["value"] == 200.0 and latency["p90"]["value"] == pytest.approx(280.0)
    for name in SCORECARD:
        ci = card[name]["ci95"]
        assert ci["lo"] <= card[name]["value"] <= ci["hi"] and ci["draws_defined"] == BOOTSTRAP_DRAWS


def test_undefined_values_are_na_never_zero_or_one_and_the_overall_says_so():
    # An agent-like block: EOT events only, no INT positives and no INT negatives.
    scores = [conv(eot=(2, 0, 0, 0), eot_latencies=(10.0, 20.0)), conv(eot=(1, 1, 0, 0), eot_latencies=(30.0,))]
    out = bootstrap_block(scores)
    card = out["scorecard"]
    assert card["pause_handling"] == {**card["pause_handling"], "value": None, "n": 0}
    assert card["pause_handling"]["ci95"] == {"lo": None, "hi": None, "draws_defined": 0}
    assert card["interruption_handling"]["value"] is None and card["backchannel_handling"]["value"] is None
    assert card["turn_taking"]["value"] == 0.75
    assert card["overall"]["value"] == 0.75
    assert card["overall"]["defined"] == ["turn_taking"]
    assert set(card["overall"]["omitted"]) == {"pause_handling", "interruption_handling", "backchannel_handling"}
    assert card["overall"]["weights"] == {"turn_taking": 1.0}
    assert out["ci95"]["int"]["recall"]["lo"] is None and out["ci95"]["int"]["latency_ms"]["p50"] is None
    # Fewer than two conversations: values, but no interval at all.
    single = bootstrap_block(scores[:1])["scorecard"]
    assert single["turn_taking"]["value"] == 1.0 and single["turn_taking"]["ci95"] is None
    assert single["overall"]["ci95"] is None


def test_bootstrap_is_deterministic_for_the_recorded_seed():
    rng = np.random.default_rng(0)
    scores = [
        conv(eot=tuple(int(v) for v in rng.integers(0, 6, 4)), int_=tuple(int(v) for v in rng.integers(0, 4, 4)),
             eot_latencies=rng.uniform(0, 500, 3).tolist())
        for _ in range(10)
    ]
    first, second = bootstrap_block(scores), bootstrap_block(scores)
    assert first == second
    assert np.array_equal(draw_matrix(10), draw_matrix(10, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED))
    assert not np.array_equal(draw_matrix(10, seed=BOOTSTRAP_SEED), draw_matrix(10, seed=BOOTSTRAP_SEED + 1))
    assert first["scorecard"]["turn_taking"]["ci95"] != bootstrap_block(scores, seed=BOOTSTRAP_SEED + 1)["scorecard"]["turn_taking"]["ci95"]


def test_draws_pool_counts_rather_than_averaging_per_conversation_rates():
    # A: 1 EOT, detected (recall 1). B: 3 EOTs, all missed (recall 0).
    # Pairs with replacement: {A,A} -> 1/1, {A,B} -> 1/4, {B,B} -> 0/3.
    # Averaging per-conversation rates would give 0.5 for {A,B}; pooling never does.
    a, b = conv(eot=(1, 0, 0, 1), eot_latencies=(10.0,)), conv(eot=(0, 3, 1, 0))
    counts = Counts([a.task_eot, b.task_eot])
    assert counts.pooled() == {"tp": 1, "fn": 3, "fp": 1, "tn": 1}
    draws = draw_matrix(2)
    recall = task_rates(counts.pooled(draws))["recall"]
    assert set(np.round(recall, 6).tolist()) == {0.0, 0.25, 1.0}
    # Each draw's recall is the pooled count ratio of exactly that draw.
    for row, value in zip(draws[:50], recall[:50]):
        tp, fn = counts.tp[row].sum(), counts.fn[row].sum()
        assert value == tp / (tp + fn)
    out = bootstrap_block([a, b])
    assert out["scorecard"]["turn_taking"]["value"] == 0.25 and out["scorecard"]["turn_taking"]["n"] == 4
    assert out["scorecard"]["turn_taking"]["ci95"] == {"lo": 0.0, "hi": 1.0, "draws_defined": BOOTSTRAP_DRAWS}
    # Latency pools the resampled conversations' TPs: A's single TP is the only
    # latency, so p50 is 10 ms in every draw that includes A, undefined in {B,B}.
    p50 = out["scorecard"]["turn_taking"]["latency_ms"]["p50"]
    assert p50["value"] == 10.0 and p50["ci95"]["lo"] == p50["ci95"]["hi"] == 10.0
    assert p50["ci95"]["draws_defined"] == int(np.count_nonzero((draws == 0).any(axis=1)))
    assert math.isnan(counts.latency_percentile(50, np.array([1, 1])))


def test_weights_are_a_presentation_choice_and_are_validated():
    scores = [conv(eot=(1, 0, 0, 1), int_=(0, 1, 1, 0)), conv(eot=(1, 1, 1, 1), int_=(1, 0, 0, 1))]
    equal = bootstrap_block(scores)["scorecard"]
    weighted = bootstrap_block(scores, {"turn_taking": 3})["scorecard"]
    values = {name: equal[name]["value"] for name in SCORECARD}
    assert equal["overall"]["value"] == pytest.approx(sum(values.values()) / 4)
    assert weighted["overall"]["value"] == pytest.approx((3 * values["turn_taking"] + sum(values.values()) - values["turn_taking"]) / 6)
    assert weighted["overall"]["weights"]["turn_taking"] == 3.0
    assert {name: weighted[name]["value"] for name in SCORECARD} == values  # the four rows never move
    assert resolve_weights(None) == {name: 1.0 for name in SCORECARD}
    with pytest.raises(KeyError):
        resolve_weights({"latency": 1})
    with pytest.raises(ValueError):
        resolve_weights({"turn_taking": 0})


def _report(values: dict[str, tuple[float, float, float]], conversations: int = 10) -> dict:
    card = {
        name: {"value": v, "ci95": {"lo": lo, "hi": hi, "draws_defined": 2000}, "n": 5}
        for name, (v, lo, hi) in values.items()
    }
    card["overall"] = {"value": 0.5, "ci95": {"lo": 0.4, "hi": 0.6, "draws_defined": 2000}, "defined": list(values), "omitted": {}, "weights": {}}
    return {
        "conversations_scored": conversations,
        "bootstrap": {"draws": 2000, "seed": 7},
        "blocks": {block: {"scorecard": copy.deepcopy(card)} for block in ("caller", "agent", "all")},
        "breakdowns": {},
    }


def test_compare_flags_only_rows_whose_intervals_do_not_overlap():
    a = _report({"pause_handling": (0.9, 0.8, 0.95), "turn_taking": (0.5, 0.4, 0.6), "interruption_handling": (0.2, 0.1, 0.3), "backchannel_handling": (0.7, 0.6, 0.8)})
    b = _report({"pause_handling": (0.5, 0.3, 0.7), "turn_taking": (0.55, 0.45, 0.65), "interruption_handling": (0.5, 0.35, 0.6), "backchannel_handling": (0.7, 0.6, 0.8)})
    b["blocks"]["agent"]["scorecard"]["interruption_handling"] = {"value": None, "ci95": {"lo": None, "hi": None, "draws_defined": 0}, "n": 0}
    comparison = compare_reports(a, b)
    by_row = {(r["label"], r["row"]): r for r in comparison["rows"]}
    assert by_row[("caller", "pause_handling")]["ci_disjoint"] is True and by_row[("caller", "pause_handling")]["higher"] == "A"
    assert by_row[("caller", "interruption_handling")]["ci_disjoint"] is True and by_row[("caller", "interruption_handling")]["higher"] == "B"
    assert by_row[("caller", "turn_taking")]["ci_disjoint"] is False  # overlapping: not flagged, not a tie
    assert by_row[("caller", "backchannel_handling")]["ci_disjoint"] is False  # identical
    assert by_row[("agent", "interruption_handling")]["ci_disjoint"] is None  # n/a on one side
    assert len(comparison["flagged"]) == 2 * 3 - 1  # two rows per block, one of them n/a in the agent block
    assert intervals_disjoint({"lo": 0.1, "hi": 0.2}, {"lo": 0.2, "hi": 0.3}) is False  # touching = overlapping
    text = render_compare("A", "B", a, b, comparison)
    assert "Pause handling" in text and "<- A higher (CIs disjoint)" in text and "<- B higher (CIs disjoint)" in text
    assert "flagged 5 of 15 rows" in text and "n/a" in text
