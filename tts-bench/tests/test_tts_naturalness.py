"""The naturalness journal: which recordings it covers, how an amendment joins it, and how the export reads it."""

from __future__ import annotations

import json

from tts_bench import corpus, naturalness, website
from tts_bench.probes import Cancel, OneShot
from tts_bench.runner import RunSpec, Runner


async def test_one_shot_recordings_are_scored_once_and_the_export_reads_the_journal(tmp_path):
    by_id = corpus.by_id()
    run = Runner(RunSpec(provider="fake", probes=[OneShot(), Cancel(repeats=1)], items=[by_id["prose.greet"], by_id["long.prep"]],
                         repeats=2, sentinel=True, store=str(tmp_path)), "unused", site="lab")
    await run.run()

    todo = naturalness.recordings(run.root)
    assert sorted(r["item"] for r in todo) == ["long.prep", "long.prep", "prose.greet", "prose.greet"]   # one-shots only: no sentinel, no cancel
    assert all(r["audio"].exists() for r in todo)
    assert website.summarize(run.root)["naturalness"] is None                                          # unscored is null, never zero

    # A journal as the scorer would leave it, without needing the model here.
    with open(run.root / naturalness.JOURNAL, "w") as handle:
        for rec, mos in zip(todo, (4.2, 4.4, 3.1, 3.3)):
            handle.write(json.dumps({"cell_id": rec["cell_id"], "item": rec["item"], "cohort": rec["cohort"], "repeat": rec["repeat"],
                                     "audio_s": 1.0, "model": "utmos22-strong", "hub": "pinned", "mos": mos}) + "\n")
    assert [r for r in naturalness.recordings(run.root) if r["cell_id"] not in naturalness.scored(run.root)] == []
    summary = website.summarize(run.root)["naturalness"]
    assert summary["model"] == "utmos22-strong" and summary["mos"]["n"] == 4 and summary["mos"]["mean"] == 3.75
    assert set(summary["byCohort"]) == {"prose", "long"}

    # A later run on the same model re-scores one recording; the newer score wins at the join.
    amend = Runner(RunSpec(provider="fake", probes=[OneShot()], items=[by_id["prose.greet"]], repeats=1, sentinel=False,
                           store=str(tmp_path)), "unused", site="lab")
    await amend.run()
    rec = naturalness.recordings(amend.root)[0]
    (amend.root / naturalness.JOURNAL).write_text(json.dumps({**{k: rec[k] for k in ("cell_id", "item", "cohort", "repeat")},
                                                               "audio_s": 1.0, "model": "utmos22-strong", "hub": "pinned", "mos": 5.0}) + "\n")
    joined = website.summarize(run.root, [amend.root])["naturalness"]
    assert joined["mos"]["n"] == 4 and joined["mos"]["mean"] > 3.75
