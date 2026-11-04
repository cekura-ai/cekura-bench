"""site-export end to end on synthetic runs: every file is written, shapes hold, output is deterministic."""

import json
from argparse import Namespace

import pytest

from conftest import BAD, write_run
from s2s_human_bench.site_export import run_site_export

PUBLIC_S2S = {
    "schemaVersion": 3,
    "build": {"agentCommit": "abc"},
    "models": [{"id": "alpha-site", "name": "Alpha", "logo": "alpha.svg"}],
    "results": [{"model": "alpha-site", "suite": "all", "costPublished": True, "costPerMinuteUsd": 0.1, "costNote": "list price"}],
}


def make_inputs(tmp_path):
    runs = tmp_path / "runs"
    for agent, quality in ((1, {}), (2, {"B01": BAD, "C02": BAD}), (3, {"B02": BAD, "C01": BAD, "C02": BAD})):
        for r in (1, 2):
            write_run(runs / f"run_a{agent}_r{r}", quality)
    (tmp_path / "public.json").write_text(json.dumps(PUBLIC_S2S))
    (tmp_path / "titles.json").write_text(json.dumps({"calls": [{"id": "B01", "title": "Synthetic title"}]}))
    (runs / "scenario_clips.json").write_text(json.dumps({"B01": ["x", "y"], "C01": ["z"]}))
    return runs


def export(tmp_path, cfg, out, reviewed=None, clips=None):
    args = Namespace(
        config=tmp_path / "config.json",
        runs=tmp_path / "runs",
        public_s2s=tmp_path / "public.json",
        titles=tmp_path / "titles.json",
        out=out,
        clips=None,
        name=["languages=lang.json"],
        date="2026-01-01",
        exclude_unhealthy=False,
        clip_picks=None,
        reviewed_examples=reviewed,
        reviewed_clips=clips,
    )
    assert run_site_export(args) == 0
    return {p.name: json.loads(p.read_text()) for p in out.iterdir()}


def test_site_export(cfg, tmp_path):
    make_inputs(tmp_path)
    docs = export(tmp_path, cfg, tmp_path / "out1")
    assert sorted(docs) == ["benchmark.json", "examples.json", "insights.json", "lang.json", "reports.json", "skills.json"]
    assert export(tmp_path, cfg, tmp_path / "out2") == docs  # deterministic

    bench = docs["benchmark.json"]
    assert [s["id"] for s in bench["suites"]] == ["bank", "clinic"] and bench["repeats"] == 2 and bench["calls"] == 24
    alpha = next(m for m in bench["models"] if m["id"] == "alpha-site")
    assert alpha["logo"] == "alpha.svg" and alpha["voice"] == "-"  # public metadata kept, defaults filled
    overall = {r["model"]: r for r in bench["results"] if r["suite"] == "all"}
    assert overall["alpha-site"]["costPerMinuteUsd"] == 0.1 and overall["beta-site"]["costPublished"] is False
    assert overall["alpha-site"]["passAll"]["overall"] == 4 and overall["beta-site"]["passAll"]["overall"] == 2
    assert overall["alpha-site"]["elo"] > overall["beta-site"]["elo"]
    assert "elo" in overall["ref-site"]  # the reference is rated, on the same scale

    skills = docs["skills.json"]
    elo = {r["model"]: r for r in skills["elo"]}
    assert elo["ref-site"]["lo"] is None and elo["alpha-site"]["lo"] is not None
    assert skills["callSummary"]["humanClips"] == 3 and skills["limits"] == ["4 scenarios."]

    lang = docs["lang.json"]["groups"]
    assert [g["id"] for g in lang] == ["all", "en", "xx"] and [g["calls"] for g in lang] == [4, 3, 1]
    assert lang[0]["models"]["alpha-site"]["elo"] == elo["alpha-site"]["elo"]
    assert lang[2]["models"]["beta-site"]["passAll"] == 0.0

    assert docs["reports.json"]["reports"]["alpha-site"] == {k: "https://example.test/run_a1_r1" for k in ("all", "bank", "clinic")}
    ins = docs["insights.json"]
    assert [c["key"] for c in ins["categories"]] == ["hear", "switch"] and ins["models"]["beta-site"]["byCategory"]["switch"]["scenarios"] == 1
    ex = docs["examples.json"]
    assert ex["examples"] == {}  # nothing is published until a person has reviewed it


def test_reviewed_examples(cfg, tmp_path):
    make_inputs(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(json.dumps({"alpha-site": ["1"]}))
    ex = export(tmp_path, cfg, tmp_path / "out", reviewed)["examples.json"]["examples"]
    assert list(ex) == ["alpha-site"] and len(ex["alpha-site"]) == 1
    assert ex["alpha-site"][0]["url"].startswith("https://example.test/run_a1_")


def test_reviewed_clips(cfg, tmp_path):
    make_inputs(tmp_path)
    sid = cfg.skills[0].id
    side = dict(model="alpha-site", scenario="B01", finding="f", tag="t", audio="/a.mp3", start_s=1.0, end_s=9.0, turns=[])
    clips = tmp_path / "clips.json"
    clips.write_text(json.dumps({sid: {"setup": "s", "should": "w", "pass": side, "fail": side}}))
    moments = export(tmp_path, cfg, tmp_path / "out", clips=clips)["examples.json"]["moments"]
    assert list(moments) == [sid] and moments[sid]["pass"]["audio"] == "/a.mp3"
    clips.write_text(json.dumps({sid: {"pass": dict(side, model="nobody"), "fail": side}}))
    with pytest.raises(ValueError, match="unknown model"):
        export(tmp_path, cfg, tmp_path / "out2", clips=clips)
