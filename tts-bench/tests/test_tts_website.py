"""The website export: one small summary per run, joined into the site file, with exclusions as null."""

from __future__ import annotations

import json

from tts_bench import corpus, website
from tts_bench.probes import Cancel, Concurrency, OneShot, Repeat
from tts_bench.runner import RunSpec, Runner


async def test_a_run_reduces_to_published_numbers_and_joins_the_site_file(tmp_path):
    items = [corpus.by_id()["prose.greet"], corpus.by_id()["long.prep"], corpus.by_id()["paragraph.benefits"]]
    runner = Runner(RunSpec(provider="fake", probes=[OneShot(), Repeat(), Cancel(repeats=1), Concurrency(streams=2)],
                            items=items, repeats=1, sentinel=False, store=str(tmp_path)), "unused", site="lab")
    await runner.run()
    summary = website.summarize(runner.root)
    assert summary["cells"]["planned"] == summary["cells"]["completed"]
    assert summary["latency"]["ttfa"]["n"] == 3 and summary["site"] == "lab"
    assert summary["underLoad"]["perStreamTtfa"]["n"] == 6
    assert summary["throughput"]["charsPerSecondParagraph"]["n"] == 1
    assert summary["reliability"]["runaways"] == 0 and summary["accuracy"] == {}
    assert len(json.dumps(summary)) < 20000                   # numbers only: no audio, no transcripts

    models = {"campaign": "t", "models": {
        f"fake/{summary['model']}": {"id": "fake", "name": "Fake", "vendor": "Test", "modelId": summary["model"], "voice": "v", "price": None},
        "other/x": {"id": "other", "name": "Other", "vendor": "Test", "modelId": "x", "voice": "v", "price": None}}}
    site = website.combine([summary], models)
    assert [m["measured"] for m in site["models"]] == [True, False]
    assert site["results"][0]["id"] == "fake" and site["build"]["sites"] == ["lab"]
