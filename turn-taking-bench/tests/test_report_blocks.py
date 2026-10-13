"""Caller / agent / all scoring blocks: each block is Sesame's scorer on one
speaker's gold, the oracle is perfect in every block, and caller + agent
partition the all block (tp/fn/fp/tn add up, latencies concatenate)."""

from pathlib import Path

import pytest
from turnbench.data import conversation, conversation_ids, resolve_dataset
from turnbench.submission import SCHEMA_VERSION, ConversationPrediction, SpeakerEvents, Submission

from turn_taking_bench.author.build import build_dataset
from turn_taking_bench.author.check import oracle_submission
from turn_taking_bench.author.listen import CUE_COLUMNS, write_listening_pack
from turn_taking_bench.author.render import MockRenderer, RenderCache
from turn_taking_bench.report import BLOCKS, authored_groups, render_markdown, score_report

COUNTS = ("tp", "fn", "fp", "tn")


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory, seed_specs_dir):
    out = tmp_path_factory.mktemp("authored") / "mock"
    build_dataset(seed_specs_dir, out, MockRenderer(), cache=RenderCache(tmp_path_factory.mktemp("renders")))
    return out


@pytest.fixture(scope="module")
def dataset(dataset_dir):
    return resolve_dataset(source=str(dataset_dir))


def assert_blocks_partition(report):
    for task in ("eot", "int"):
        for key in COUNTS:
            caller = report["blocks"]["caller"]["aggregate"][task][key]
            agent = report["blocks"]["agent"]["aggregate"][task][key]
            assert caller + agent == report["blocks"]["all"]["aggregate"][task][key] == report["aggregate"][task][key]
        for cid in report["per_conversation"]:
            for key in COUNTS:
                assert (
                    report["blocks"]["caller"]["per_conversation"][cid][task][key]
                    + report["blocks"]["agent"]["per_conversation"][cid][task][key]
                    == report["blocks"]["all"]["per_conversation"][cid][task][key]
                )


def test_oracle_is_perfect_in_every_block_and_blocks_partition_all(dataset):
    report = score_report(oracle_submission(dataset), dataset)
    assert report["conversations_scored"] == 3
    for block in BLOCKS:
        for task in ("eot", "int"):
            scores = report["blocks"][block]["aggregate"][task]
            if scores["tp"] + scores["fn"]:
                assert scores["recall"] == 1.0
            if scores["fp"] + scores["tn"]:
                assert scores["fp_rate"] == 0.0
    # The seed specs plant events on both sides, so both role blocks are non-trivial.
    assert report["blocks"]["caller"]["aggregate"]["eot"]["tp"] > 0
    assert report["blocks"]["agent"]["aggregate"]["eot"]["tp"] > 0
    assert report["blocks"]["caller"]["aggregate"]["int"]["tp"] == 1  # TB-09's barge-in is the caller's
    assert report["blocks"]["agent"]["aggregate"]["int"] == {**report["blocks"]["agent"]["aggregate"]["int"], "tp": 0, "fn": 0, "fp": 0, "tn": 0}
    assert report["blocks"]["caller"]["speakers"] == [1] and report["blocks"]["all"]["speakers"] == [1, 2]
    assert_blocks_partition(report)


def test_a_caller_only_model_scores_on_the_caller_block_and_misses_the_agent_block(dataset):
    oracle = oracle_submission(dataset).by_conversation()
    predictions = []
    for cid in conversation_ids(dataset):
        duration = conversation(dataset, cid).duration_s
        # Caller: oracle plus a spurious fire inside every planted pause; agent: nothing at all.
        caller = oracle[cid].speaker_1
        predictions.append(
            ConversationPrediction(
                conversation_id=cid,
                speaker_1=SpeakerEvents(eot=sorted(set(caller.eot + [min(t + 0.3, duration) for t in caller.eot])), interruption=caller.interruption),
                speaker_2=SpeakerEvents(eot=[], interruption=[]),
            )
        )
    report = score_report(Submission(schema_version=SCHEMA_VERSION, predictions=predictions), dataset)
    caller, agent, both = (report["blocks"][b]["aggregate"]["eot"] for b in BLOCKS)
    assert caller["recall"] == 1.0 and caller["fn"] == 0
    assert agent["recall"] == 0.0 and agent["tp"] == 0 and agent["fn"] > 0
    assert both["tp"] == caller["tp"] and both["fn"] == agent["fn"]
    assert_blocks_partition(report)
    # Latencies partition too: the all block's list is the two role lists together.
    assert report["blocks"]["all"]["aggregate"]["eot"]["latency_ms"]["p50"] == caller["latency_ms"]["p50"]
    markdown = render_markdown(
        report,
        {"model": {"name": "m", "config": {}}, "dataset": {"split": "x", "source": "s"}, "turnbench": {"commit": "c", "version": "v"}, "harness": {"version": "0", "git_commit": "g", "git_dirty": False}},
    )
    assert "## Caller (speaker_1)" in markdown and "## Agent (speaker_2)" in markdown and "## All speakers" in markdown
    assert markdown.index("## Caller") < markdown.index("## Agent") < markdown.index("## All speakers")


def test_listening_pack_writes_stereo_audio_and_a_cue_per_labelled_event(dataset_dir, dataset):
    import csv

    import soundfile as sf

    written = write_listening_pack(dataset_dir)
    assert len(written) == 6 and all(p.parent == dataset_dir / "listening" for p in written)
    wav = dataset_dir / "listening" / "9091-TB-09.wav"
    samples, rate = sf.read(wav, dtype="float32", always_2d=True)
    conv = conversation(dataset, "9091")
    assert rate == 48000 and samples.shape == (len(conv.audio(1)[0]), 2)
    assert abs(samples[:, 0] - conv.audio(1)[0]).max() < 1e-4 and abs(samples[:, 1] - conv.audio(2)[0]).max() < 1e-4
    with (dataset_dir / "listening" / "9091-TB-09.cues.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and tuple(rows[0]) == CUE_COLUMNS
    events = [r["event"] for r in rows]
    assert events.count("segment") == 8 and events.count("int_positive") == 1 and events.count("eot_negative") == 1
    barge = next(r for r in rows if r["event"] == "int_positive")
    assert barge["speaker"] == "caller" and barge["channel"] == "L"
    assert [float(r["time_s"]) for r in rows] == sorted(float(r["time_s"]) for r in rows)


def test_breakdowns_by_family_partition_the_blocks(dataset_dir, dataset):
    groups = authored_groups(dataset_dir)
    assert groups["family"] == {"pauses": ["9041"], "backchannels": ["8291"], "barge_in": ["9091"]}
    assert set(groups["difficulty"]) <= {"easy", "medium", "hard"}
    report = score_report(oracle_submission(dataset), dataset, groups)
    families = report["breakdowns"]["family"]
    assert set(families) == {"pauses", "backchannels", "barge_in"} and all(g["conversations"] == 1 for g in families.values())
    for block in BLOCKS:
        for task in ("eot", "int"):
            for key in COUNTS:
                assert sum(g["blocks"][block][task][key] for g in families.values()) == report["blocks"][block]["aggregate"][task][key]
    assert families["barge_in"]["blocks"]["caller"]["int"]["tp"] == 1 and families["backchannels"]["blocks"]["caller"]["int"]["tn"] == 6
    # A group naming ids outside the scored set is ignored; an empty group is dropped.
    extra = score_report(oracle_submission(dataset), dataset, {"family": {"pauses": ["9041", "nope"], "ghost": ["nope"]}})
    assert set(extra["breakdowns"]["family"]) == {"pauses"}
    markdown = render_markdown(
        report,
        {"model": {"name": "m", "config": {}}, "dataset": {"split": "x", "source": "s"}, "turnbench": {"commit": "c", "version": "v"}, "harness": {"version": "0", "git_commit": "g", "git_dirty": False}},
    )
    assert "## Caller (speaker_1) by family" in markdown and "| barge_in | 1 | INT | 1.000 |" in markdown
    assert authored_groups(dataset_dir / "missing") == {}


def test_oracle_scorecard_is_one_everywhere_with_degenerate_intervals(dataset_dir, dataset):
    from turn_taking_bench.scorecard import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, SCORECARD

    report = score_report(oracle_submission(dataset), dataset, authored_groups(dataset_dir))
    assert report["bootstrap"]["draws"] == BOOTSTRAP_DRAWS and report["bootstrap"]["seed"] == BOOTSTRAP_SEED
    cards = [report["blocks"][b]["scorecard"] for b in BLOCKS]
    cards += [g["scorecard"][b] for names in report["breakdowns"].values() for g in names.values() for b in BLOCKS if g["conversations"] >= 2]
    defined = 0
    for card in cards:
        for name in tuple(SCORECARD) + ("overall",):
            row = card[name]
            if row["value"] is None:
                assert row["n"] == 0 and name in card["overall"]["omitted"]
                continue
            defined += 1
            assert row["value"] == 1.0
            assert (row["ci95"]["lo"], row["ci95"]["hi"]) == (1.0, 1.0), (name, row)
        assert card["overall"]["defined"] and card["overall"]["value"] == 1.0
    assert defined > 0
    # The agent block has no INT events at all: its two INT rows are n/a and the overall says so.
    agent = report["blocks"]["agent"]["scorecard"]
    assert agent["interruption_handling"]["value"] is None and agent["backchannel_handling"]["value"] is None
    assert "interruption_handling" in agent["overall"]["omitted"] and "turn_taking" in agent["overall"]["defined"]
    markdown = render_markdown(report, RUN_INFO)
    assert "## Conversational Dynamics scorecard" in markdown and "Caller (speaker_1) — headline" in markdown
    assert "| recall 95% CI |" in markdown and "n/a (n=0)" in markdown


RUN_INFO = {"model": {"name": "m", "config": {}}, "dataset": {"split": "x", "source": "s"}, "turnbench": {"commit": "c", "version": "v"}, "harness": {"version": "0", "git_commit": "g", "git_dirty": False}}


def _write_run(run_dir: Path, dataset_dir: Path, submission: Submission) -> Path:
    import json

    run_dir.mkdir(parents=True)
    (run_dir / "predictions-mock.json").write_text(submission.model_dump_json(indent=2), encoding="utf-8")
    run_info = {
        **RUN_INFO,
        "run_id": run_dir.name,
        "dataset": {"kind": "local", "split": dataset_dir.name, "source": str(dataset_dir), "revision": None, "manifest": None},
        "files": {"predictions": "predictions-mock.json"},
        "scores": None,
    }
    (run_dir / "run.json").write_text(json.dumps(run_info), encoding="utf-8")
    return run_dir


def test_report_command_rescores_a_run_without_the_model_and_compare_reads_it(dataset_dir, dataset, tmp_path, capsys):
    import json

    from turn_taking_bench.rescore import rescore_run, run_compare

    oracle = oracle_submission(dataset)
    # A second run: the oracle with the caller's EOT fires removed on one conversation.
    worse = Submission(
        schema_version=SCHEMA_VERSION,
        predictions=[
            ConversationPrediction(conversation_id=p.conversation_id, speaker_1=SpeakerEvents(eot=[] if p.conversation_id == "9041" else p.speaker_1.eot, interruption=p.speaker_1.interruption), speaker_2=p.speaker_2)
            for p in oracle.predictions
        ],
    )
    run_a = _write_run(tmp_path / "oracle", dataset_dir, oracle)
    run_b = _write_run(tmp_path / "worse", dataset_dir, worse)
    report_a = rescore_run(run_a)  # dataset from run.json
    report_b = rescore_run(run_b, dataset_dir)  # dataset given explicitly
    expected = score_report(oracle, dataset, authored_groups(dataset_dir))
    written = json.loads((run_a / "report.json").read_text(encoding="utf-8"))
    for report in (report_a, written):
        assert report["aggregate"] == expected["aggregate"] and report["blocks"]["caller"]["aggregate"] == expected["blocks"]["caller"]["aggregate"]
        assert report["blocks"]["caller"]["scorecard"] == expected["blocks"]["caller"]["scorecard"]
        assert report["breakdowns"] == expected["breakdowns"] and report["partial"] is False
    assert written["rescored"]["predictions"] == "predictions-mock.json"
    assert "## Conversational Dynamics scorecard" in (run_a / "report.md").read_text(encoding="utf-8")
    assert report_b["blocks"]["caller"]["aggregate"]["eot"]["fn"] > 0 and report_b["blocks"]["caller"]["scorecard"]["turn_taking"]["value"] < 1.0
    assert json.loads((run_b / "run.json").read_text(encoding="utf-8"))["scores"] is None  # run.json untouched

    assert run_compare(run_a, run_b) == 0
    out = capsys.readouterr().out
    assert "compare: A = oracle  vs  B = worse" in out and "Turn taking" in out and "flagged" in out
    assert run_compare(run_a, tmp_path / "nowhere") == 2
