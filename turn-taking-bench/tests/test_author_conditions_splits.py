"""Condition variants and template splits of an authored dataset: labels copied
unchanged, ids suffixed, every pairing of a template in one split, and the
result still round-trips through Sesame's gold and scorer."""

import json
from pathlib import Path

import pytest

from turn_taking_bench.author.build import build_dataset
from turn_taking_bench.author.check import gold_check, oracle_ok, oracle_scores
from turn_taking_bench.author.degrade import CONDITIONS, degrade_dataset, variant_id
from turn_taking_bench.author.render import MockRenderer, RenderCache
from turn_taking_bench.author.splits import assign_templates, load_assignment, split_dataset, write_assignment
from turn_taking_bench.data import load_audio_conversation, load_local_dataset, local_split, locate_rows
from turn_taking_bench.report import authored_groups

from tests.conftest import subset_specs


@pytest.fixture(scope="module")
def clean_dir(tmp_path_factory):
    specs = subset_specs(tmp_path_factory.mktemp("specs"), pairings=2)  # 3 templates x 2 pairings
    out = tmp_path_factory.mktemp("authored") / "clean"
    build_dataset(specs, out, MockRenderer(), cache=RenderCache(tmp_path_factory.mktemp("renders")))
    return out


@pytest.fixture(scope="module")
def combined_dir(clean_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("authored") / "combined"
    degrade_dataset(clean_dir, out, ("telephone", "dropouts"), workers=2)  # the worker-pool path, rows still in id order
    return out


def test_variants_keep_the_labels_and_suffix_the_ids(clean_dir, combined_dir):
    clean = json.loads((clean_dir / "ground_truth.json").read_text())
    combined = json.loads((combined_dir / "ground_truth.json").read_text())
    assert sorted(combined, key=int) == sorted([variant_id(cid, c) for cid in clean for c in ("clean", "telephone", "dropouts")], key=int)
    for cid, record in clean.items():
        for condition in ("clean", "telephone", "dropouts"):
            variant = combined[variant_id(cid, condition)]
            assert variant["condition"] == condition and variant["source_conversation_id"] == cid
            assert variant["planted"] == record["planted"] and variant["annotations"] == record["annotations"]
    assert variant_id("9011", "telephone") == "90111" and CONDITIONS["clean"]["suffix"] == ""
    # The audio differs under a condition and is byte-identical for the clean copy.
    locations = locate_rows(local_split(combined_dir))
    durations = load_local_dataset(combined_dir).durations
    clean_audio = load_audio_conversation("9041", locations["9041"], durations["9041"]).audio_bytes
    source = load_audio_conversation("9041", locate_rows(local_split(clean_dir))["9041"], durations["9041"]).audio_bytes
    assert clean_audio == source
    degraded = load_audio_conversation("90415", locations["90415"], durations["90415"])
    assert degraded.audio_bytes != clean_audio
    assert len(degraded.audio(1)[0]) == len(load_audio_conversation("9041", locations["9041"], durations["9041"]).audio(1)[0])

    check = gold_check(combined_dir)
    assert check.ok, check.table()
    assert oracle_ok(oracle_scores(load_local_dataset(combined_dir)))
    groups = authored_groups(combined_dir)
    assert set(groups["condition"]) == {"clean", "telephone", "dropouts"}
    assert set(groups["pairing"]) == {"f-us/m-us", "m-us/f-gb"} and set(groups["caller_voice"]) == {"f-us", "m-us"}


def test_split_assigns_whole_templates_stratified_by_family(combined_dir, tmp_path):
    families = {f"P-{i:02d}": "pauses" for i in range(10)} | {f"B-{i:02d}": "barge_in" for i in range(5)} | {"L-01": "long", "L-02": "long"}
    assignment = assign_templates(families, seed=7)
    assert assignment == assign_templates(families, seed=7)  # deterministic
    assert sorted(sum(assignment.values(), [])) == sorted(families)
    assert sum(t.startswith("P-") for t in assignment["holdout"]) == 2 and sum(t.startswith("P-") for t in assignment["dev"]) == 3
    assert sum(t.startswith("B-") for t in assignment["holdout"]) == 1 and sum(t.startswith("B-") for t in assignment["dev"]) == 2
    assert sorted(assignment["dev"] + assignment["test"]).count("L-01") + sorted(assignment["dev"] + assignment["test"]).count("L-02") == 2
    # Linked templates share a split (the minimal pair), whatever the draw gave the second one.
    pair = assign_templates({**families, "NB-31": "backchannels", "NB-32": "barge_in"}, seed=7)
    assert len({name for name, ids in pair.items() if "NB-31" in ids or "NB-32" in ids}) == 1

    real = assign_templates({"TB-04": "pauses", "NB-29": "backchannels", "TB-09": "barge_in"}, seed=1, holdout=0.0, dev=0.5)
    public, hidden = tmp_path / "splits.json", tmp_path / "hidden" / "holdout.json"
    real["holdout"], real["test"] = ["TB-09"], [t for t in real["test"] if t != "TB-09"]
    real["dev"] = [t for t in real["dev"] if t != "TB-09"]
    write_assignment(real, public, hidden, seed=1, holdout=0.0, dev=0.5)
    assert "holdout" not in json.loads(public.read_text()) and json.loads(public.read_text())["holdout_count"] == 1
    assert load_assignment(public) == real

    outs = {"dev": tmp_path / "dev", "test": tmp_path / "test", "holdout": tmp_path / "hidden" / "holdout"}
    manifests = split_dataset(combined_dir, outs, real)
    truth = json.loads((combined_dir / "ground_truth.json").read_text())
    for name, path in outs.items():
        if name not in manifests:
            assert not path.exists()
            continue
        split_truth = json.loads((path / "ground_truth.json").read_text())
        assert {r["spec_id"] for r in split_truth.values()} == set(real[name])
        assert all(truth[cid] == record for cid, record in split_truth.items())
        assert gold_check(path).ok and oracle_ok(oracle_scores(load_local_dataset(path)))
    assert {r["spec_id"] for r in json.loads((outs["holdout"] / "ground_truth.json").read_text()).values()} == {"TB-09"}
    assert sum(m["conversations"]["count"] for m in manifests.values()) == len(truth)
    with pytest.raises(ValueError, match="no split"):
        split_dataset(combined_dir, outs, {"dev": ["TB-04"], "test": [], "holdout": []})
