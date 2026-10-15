"""site-export: published values are verified, models added, overall rank and breakdowns derived."""

import json
from argparse import Namespace

import numpy as np
import pytest

from turn_taking_bench.site_export import ExportError, eot_rows, export, interruption_rows, run_site_export

CONDITIONS = ["clean", "telephone", "noise-20db", "noise-10db", "reverb", "dropouts"]
# Three calls per condition: (script, family, call kind, caller voice).
CALLS = [("S1", "pauses", "filler", "f-us"), ("S2", "pauses", "lookup", "m-us"), ("S3", "turn_ends", "one_word", "f-us")]
# Caller-block (tp, fn, fp, tn) per model and call; model B misses more turn ends in heavy noise.
COUNTS = {
    "model-a": {"S1": (4, 0, 1, 3), "S2": (3, 1, 2, 2), "S3": (5, 0, 0, 0)},
    "model-b": {"S1": (4, 0, 2, 2), "S2": (2, 2, 1, 3), "S3": (4, 1, 0, 0)},
}


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def rates(c):
    tp, fn, fp, tn = c
    pause = 1 - fp / (fp + tn) if fp + tn else None
    turn = tp / (tp + fn)
    return pause, turn, turn if pause is None else (pause + turn) / 2


def condition_counts(slug, condition):
    counts = {s: list(v) for s, v in COUNTS[slug].items()}
    if slug == "model-b" and condition == "noise-10db":
        counts["S3"] = [2, 3, 0, 0]
    return counts


def build_root(root, voided=("model-a", "reverb", "S2")):
    for k, cond in enumerate(CONDITIONS):
        suffix = "" if cond == "clean" else str(k)
        cid = {s: f"{s}{suffix}" for s, *_ in CALLS}
        truth = {cid[s]: {"spec_id": s, "family": fam, "subtype": kind, "pairing": f"{voice}/x", "voices": {"caller": voice}}
                 for s, fam, kind, voice in CALLS}
        write(root / "datasets" / f"t1-{cond}" / "ground_truth.json", truth)
        rows, track2 = [], {}
        for slug in COUNTS:
            counts = condition_counts(slug, cond)
            excluded = [cid[voided[2]]] if (slug, cond) == voided[:2] else []
            write(root / "t1" / cond / slug / "report.json", {"blocks": {"caller": {"per_conversation": {
                cid[s]: {"eot": dict(zip(("tp", "fn", "fp", "tn"), v))} for s, v in counts.items()}}}})
            pooled = np.sum([v for s, v in counts.items() if cid[s] not in excluded], axis=0)
            pause, turn, score = rates(pooled)
            rows.append({
                "slug": slug, "total_conversations": 3, "scored_conversations": 3 - len(excluded),
                "pacing_voided_excluded": [{"conversation_id": c} for c in excluded], "failures": [],
                "caller": {"overall": {"value": score, "ci95": {"lo": score - 0.1, "hi": score + 0.1}},
                           "pause_handling": {"value": pause}, "turn_taking": {"value": turn, "latency_ms": {"p50": {"value": 900.0}, "p90": {"value": 1400.0}}}},
            })
            stops, mmhmm = (1.0, 0.1) if slug == "model-a" else (0.8, 0.6)
            track2[slug] = {"yield_rate": stops, "yield_n": 18, "barge_planned": 20, "backchannel_n": 14 if slug == "model-b" else 19,
                            "backchannel_planned": 20, "excess_yield": 0.7, "excess_ci95": [0.6, 0.8], "backchannel_continue": mmhmm,
                            "stop_ms": {"p50": 300.0}}
        for rank, row in enumerate(sorted(rows, key=lambda r: -r["caller"]["overall"]["value"]), 1):
            row["rank"] = rank
        write(root / "t1" / cond / "SCORECARD.json", {"rows": rows})
        write(root / "t2" / cond / "TRACK2.json", {"rows": track2})


def published_site(root, public):
    """The site as published before model B: model A's rows and block, from the same run outputs."""
    names = {"model-a": "Model A"}
    eot = [r for c in CONDITIONS for r in eot_rows(json.loads((root / "t1" / c / "SCORECARD.json").read_text()), c, names, set(), True)]
    cut = [r for c in CONDITIONS for r in interruption_rows(json.loads((root / "t2" / c / "TRACK2.json").read_text()), c, names, True)]
    for r in eot:
        r.pop("_p90")
    pub_eot = eot_rows(json.loads((public / "t1" / "clean" / "SCORECARD.json").read_text()), "clean", names, set(), False)
    pub_cut = interruption_rows(json.loads((public / "t2" / "clean" / "TRACK2.json").read_text()), "clean", names, False)
    block = {
        "vendor": "A", "status": "ranked", "insights": ["kept as written"], "setup": {},
        "endOfTurn": {r["condition"]: {"score": r["overall"], "rank": None} for r in eot},
        "interruptions": {r["condition"]: {"stopsOnWait": r["yieldRate"], "ignoresMmhmm": r["backchannelContinue"], "lowCoverage": r["lowCoverage"]} for r in cut},
    }
    return {
        "benchmark": {"models": 1, "conditions": [{"id": c, "label": c} for c in CONDITIONS], "sets": {"public": {"conditionsRun": ["clean"]}}},
        "models": {"Model A": block}, "modelNotes": {},
        "results": {"private": {"endOfTurn": eot, "interruptions": cut}, "public": {"endOfTurn": pub_eot, "interruptions": pub_cut}},
    }


@pytest.fixture
def roots(tmp_path):
    runs, public = tmp_path / "runs", tmp_path / "public"
    build_root(runs)
    build_root(public)
    return runs, public


NEW = {"model-b": {"name": "Model B", "vendor": "B", "insights": ["new"], "setup": {"modelId": "b"}}}


def test_export_adds_model_overall_rank_and_breakdowns(roots):
    runs, public = roots
    site, breakdowns = export(published_site(runs, public), runs, public, NEW)

    a, b = site["models"]["Model A"], site["models"]["Model B"]
    assert site["benchmark"]["models"] == 2 and a["insights"] == ["kept as written"]
    assert [r["model"] for r in site["results"]["private"]["leaderboard"]] == ["Model A", "Model B"]
    # Overall = mean of the six per-condition scores, which exclude the voided call.
    assert a["overall"]["endOfTurn"]["score"] == pytest.approx(np.mean([a["endOfTurn"][c]["score"] for c in CONDITIONS]))
    assert a["endOfTurn"]["reverb"]["score"] == pytest.approx(rates(np.add(COUNTS["model-a"]["S1"], COUNTS["model-a"]["S3"]))[2])
    lo, hi = a["overall"]["endOfTurn"]["ci"]
    assert lo <= a["overall"]["endOfTurn"]["score"] <= hi
    # Cut-in handling is reported beside the rank; B is indicative (70% of "mm-hmm" moments scorable).
    assert a["overall"]["cutInHandling"] == {"score": pytest.approx(0.55), "indicative": False, "indicativeConditions": []}
    assert b["overall"]["cutInHandling"]["indicative"] and b["overall"]["cutInHandling"]["indicativeConditions"] == CONDITIONS
    assert b["endOfTurn"]["noise-10db"]["rank"] == 2 and b["stability"]["worstCondition"] == "noise-10db"

    cell = breakdowns["models"]["Model A"]["reverb"]
    assert cell["callKind"]["lookup"] == {"score": None, "pauseHandling": None, "turnTaking": None, "scored": 0, "total": 1}
    assert cell["callKind"]["one_word"]["pauseHandling"] is None and cell["callKind"]["one_word"]["score"] == 1.0
    assert cell["family"]["pauses"]["scored"] == 1 and cell["callerVoice"]["f-us"]["scored"] == 2
    assert breakdowns["groups"]["callKind"] == ["filler", "lookup", "one_word"]


def test_unreproduced_published_value_writes_nothing(roots, tmp_path):
    runs, public = roots
    site = published_site(runs, public)
    site["results"]["private"]["endOfTurn"][0]["pauseHandling"] += 0.01
    (tmp_path / "site.json").write_text(json.dumps(site))
    args = Namespace(site=tmp_path / "site.json", runs=runs, public_runs=public, add_models=None, alias=[],
                     out=tmp_path / "out.json", breakdowns_out=tmp_path / "breakdowns.json")
    with pytest.raises(ExportError, match="pauseHandling"):
        run_site_export(args)
    assert not (tmp_path / "out.json").exists() and not (tmp_path / "breakdowns.json").exists()


def test_site_model_without_result_rows_needs_an_alias(roots):
    runs, public = roots
    site = published_site(runs, public)
    site["models"]["Model A (renamed)"] = site["models"].pop("Model A")
    with pytest.raises(ExportError, match="--alias"):
        export(site, runs, public)
    assert export(site, runs, public, aliases={"Model A (renamed)": "model-a"})[0]["models"]["Model A (renamed)"]["overall"]
