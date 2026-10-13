"""End to end: build three seed specs (first voice pairing) with the mock
renderer into a dataset directory, load it with Sesame's own loader, round-trip
the gold through Sesame's scorer, and stream it through the harness."""

import json
import shutil
from pathlib import Path

import pytest
import soundfile as sf
from turnbench.data import conversation, conversation_ids, resolve_dataset

from turn_taking_bench.author.build import build_dataset
from turn_taking_bench.author.check import gold_check, oracle_ok, oracle_scores
from turn_taking_bench.author.render import MockRenderer, RenderCache
from turn_taking_bench.author.spec import SpecError, normalise_breaks, spec_from_json
from turn_taking_bench.data import load_local_dataset, local_split, locate_rows
from turn_taking_bench.manifest import ManifestError, load_manifest, verify_snapshot
from turn_taking_bench.runner import run_benchmark

SEED_IDS = ["8291", "9041", "9091"]  # NB-29, TB-04, TB-09, first pairing (f-us caller / m-us agent)


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory, seed_specs_dir):
    out = tmp_path_factory.mktemp("authored") / "mock"
    build_dataset(seed_specs_dir, out, MockRenderer(), transform="clean", cache=RenderCache(tmp_path_factory.mktemp("renders")))
    return out


def test_sesame_loads_the_directory_and_its_gold_is_the_planted_gold(dataset_dir):
    dataset = resolve_dataset(source=str(dataset_dir))
    assert conversation_ids(dataset) == SEED_IDS
    conv = conversation(dataset, "9091")
    samples, rate = conv.audio(1)
    assert rate == 48000
    assert len(samples) == len(conv.audio(2)[0]) == round(conv.duration_s * rate)
    assert conv.annotations[(1, "a")] == conv.annotations[(1, "b")] == conv.annotations[(1, "c")]
    durations = json.loads((dataset_dir / "durations.json").read_text())["durations"]
    assert durations["9091"] == pytest.approx(conv.duration_s)  # header and sidecar agree
    assert load_local_dataset(dataset_dir).durations == {k: pytest.approx(v) for k, v in durations.items()}

    check = gold_check(dataset_dir, dataset)
    assert check.ok, check.table()
    assert [(r["planted"], r["derived"]) for r in check.rows] == [
        ({"eot_pos": 7, "eot_neg": 0, "int_pos": 0, "int_neg": 6},) * 2,
        ({"eot_pos": 7, "eot_neg": 3, "int_pos": 0, "int_neg": 0},) * 2,
        ({"eot_pos": 7, "eot_neg": 1, "int_pos": 1, "int_neg": 0},) * 2,
    ]
    scores = oracle_scores(dataset)
    assert oracle_ok(scores)
    assert (scores.task_eot.tp, scores.task_eot.tn, scores.task_int.tp, scores.task_int.tn) == (21, 4, 1, 6)
    truth = json.loads((dataset_dir / "ground_truth.json").read_text())
    assert truth["9091"]["family"] == "barge_in" and truth["9091"]["pairing"] == "f-us/m-us" and truth["9091"]["difficulty"]


def test_manifest_pins_the_files_and_catches_edits(dataset_dir, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(dataset_dir, copy)
    manifest = load_manifest(copy / "manifest.json")
    assert manifest["kind"] == "local" and manifest["conversations"]["ids"] == SEED_IDS
    assert {f["path"] for f in manifest["files"]} == {"durations.json", "ground_truth.json", "mock-00000-of-00001.parquet"}
    assert manifest["authored"]["renderer"] == {"name": "mock", "model": "mock-v1"}
    verify_snapshot(manifest, copy, cache_path=tmp_path / "verify.json")
    (copy / "durations.json").write_text("{}")
    with pytest.raises(ManifestError, match="durations.json"):
        verify_snapshot(manifest, copy, cache_path=tmp_path / "verify.json")


def test_harness_streams_a_local_dataset_and_scores_it(dataset_dir, tmp_path):
    split = local_split(dataset_dir)
    assert set(locate_rows(split)) == set(SEED_IDS)
    info = run_benchmark(model_name="rms_vad", dataset_dir=dataset_dir, workers=1, out=tmp_path / "run")
    assert info["failures"] == []
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert set(report["breakdowns"]["family"]) == {"pauses", "backchannels", "barge_in"}
    assert "## Caller (speaker_1) by family" in (tmp_path / "run" / "report.md").read_text()
    assert info["dataset"]["kind"] == "local" and info["dataset"]["manifest"]["verified"] is True
    assert info["scores"]["eot"]["recall"] == 1.0  # a VAD finds every planted turn end...
    assert info["scores"]["eot"]["fp_rate"] == 1.0  # ...and fires on every planted pause
    assert info["scores"]["int"]["recall"] == 1.0 and info["scores"]["int"]["fp_rate"] == 1.0
    assert (tmp_path / "run" / "report.md").exists()


def test_transform_changes_audio_but_not_labels(tmp_path):
    # A short script: the transforms work on whole channels, and a minute of mock
    # audio per channel is more memory than this check needs.
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "tiny.json").write_text(json.dumps({
        "schema_version": 2, "id": "TINY", "conversation_id": "9011", "title": "tiny",
        "voices": {"caller": "f-us", "agent": "m-us"},
        "items": [
            {"id": "g", "speaker": "agent", "text": "Hello."},
            {"id": "c1", "speaker": "caller", "text": "Hi. <1s> One change."},
            {"id": "a1", "speaker": "agent", "text": "Sure."},
        ],
    }))
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    cache = RenderCache(tmp_path / "renders")
    build_dataset(specs, clean, MockRenderer(), transform="clean", cache=cache)
    build_dataset(specs, noisy, MockRenderer(), transform="noise-10db", cache=cache)
    assert json.loads((clean / "ground_truth.json").read_text())["9011"]["planted"] == json.loads((noisy / "ground_truth.json").read_text())["9011"]["planted"]
    a, b = resolve_dataset(source=str(clean)), resolve_dataset(source=str(noisy))
    clean_audio, noisy_audio = conversation(a, "9011").audio(1)[0], conversation(b, "9011").audio(1)[0]
    assert len(clean_audio) == len(noisy_audio)
    assert abs(clean_audio[:1000]).max() == 0.0 and abs(noisy_audio[:1000]).max() > 0.0  # noise under the leading silence
    assert load_manifest(noisy / "manifest.json")["authored"]["transform"] == "noise-10db"


def test_spec_validation():
    assert normalise_breaks("A <1.5s> B <300ms> C") == 'A <break time="1.5s"/> B <break time="300ms"/> C'
    base = {"schema_version": 2, "id": "X", "conversation_id": "5", "title": "t", "voices": {"caller": "f-us", "agent": "m-us"}}
    with pytest.raises(SpecError, match="digits"):
        spec_from_json({**base, "conversation_id": "TB-01", "items": [{"id": "c1", "speaker": "caller", "text": "Hi."}]})
    with pytest.raises(SpecError, match="earlier item"):
        spec_from_json({**base, "items": [{"id": "b", "speaker": "caller", "kind": "backchannel", "text": "Mm.", "into": "a1"}]})
    with pytest.raises(SpecError, match="without speech on both sides"):
        spec_from_json({**base, "items": [{"id": "c1", "speaker": "caller", "text": "Hi. <1s>"}]})
