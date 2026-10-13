"""Cache-only planning: exact selection, no guessing and no rendering."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from turn_taking_bench.author.render import MockRenderer, RenderCache
from turn_taking_bench.author.spec import load_specs
from turn_taking_bench.s2s.authored_plan import CacheOnlyRenderer, PreflightError, VerifiedCache, build_plan, load_clips


def fixture_cache(tmp_path, *, end_relative=False, splice=False):
    specs = tmp_path / "authored" / "specs"
    specs.mkdir(parents=True)
    items = [
        {"id": "g", "speaker": "agent", "text": "How can I help?"},
        {"id": "c1", "speaker": "caller", "text": "Tell me everything. <0.8s> Take your time.", "splice_breaks": splice},
        {"id": "a1", "speaker": "agent", "text": "A long answer."},
        {"id": "b1", "speaker": "caller", "kind": "backchannel", "text": "Mm-hmm.", "into": "a1", "offset_s": 1.5},
        {"id": "x1", "speaker": "caller", "kind": "barge_in", "text": "Wait, stop.", "into": "a1", "offset_s": 12.0},
        {"id": "a2", "speaker": "agent", "text": "What would you like?"},
        {"id": "cz", "speaker": "caller", "text": "Goodbye."},
    ]
    if end_relative:
        items[4].pop("offset_s")
        items[4]["offset_from_end_s"] = 1.2
    raw = {"schema_version": 2, "id": "TB-09", "conversation_id": "909", "title": "Test",
           "family": "barge_in", "pairings": [{"caller": "f-us", "agent": "m-us"},
           {"caller": "f-us", "agent": "f-gb"}, {"caller": "m-us", "agent": "f-gb"}],
           "timing": {"gap_range_s": [0.5, 1.0], "gap_seed": 2026}, "items": items}
    (specs / "tb-09.json").write_text(json.dumps(raw))
    (specs.parent / "splits.json").write_text(json.dumps({"dev": [], "test": ["TB-09"]}))
    cache = RenderCache(tmp_path / "cache")
    # Deterministic offline audio placed under the actual ElevenLabs key/meta
    # contract, so the real planner exercises every validation path without TTS.
    renderer = replace(MockRenderer(), name="elevenlabs", model="eleven_turbo_v2_5")
    for item in load_specs(specs)[0].items:
        if item.speaker == "caller":
            for text in item.segments if item.splice_breaks else [item.text]:
                cache.get(renderer, text, "f-us", item.speed)
    return specs, cache.directory


@pytest.mark.parametrize("splice", [False, True])
def test_cached_plan_is_exact_read_only_and_loads_only_caller_audio(tmp_path, splice):
    specs, cache = fixture_cache(tmp_path, splice=splice)
    before = {p: p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    plan = build_plan(specs, cache, ids=["TB-09"])
    conversation = plan["conversations"][0]
    assert conversation["conversation_id"] == "9091" and conversation["split"] == "test"
    assert "2 matching pairings" in conversation["selection_rule"]
    assert conversation["source_sha256"] == hashlib.sha256((specs / "tb-09.json").read_bytes()).hexdigest()
    assert conversation["void"] is None and plan["counts"]["barge_in"] == 1
    by_id = {i["id"]: i for i in conversation["items"]}
    assert by_id["a1"]["trigger_caller_id"] == "c1" and by_id["a2"]["trigger_caller_id"] == "x1"
    assert by_id["x1"]["offset_s"] == 12.0  # long offsets are never shortened
    clips = load_clips(conversation, cache)
    assert set(clips) == {"c1", "b1", "x1", "cz"}
    assert clips["c1"].sha256 == conversation["clips"]["c1"]["pcm_sha256"]
    assert {p: p.read_bytes() for p in cache.rglob("*") if p.is_file()} == before
    assert build_plan(specs, cache, ids=["9092"])["conversations"][0]["conversation_id"] == "9092"
    with pytest.raises(PreflightError, match="differs"):
        build_plan(specs, cache, ids=["9093"])
    with pytest.raises(PreflightError, match="ambiguous"):
        build_plan(specs, cache, ids=["TB-09", "9092"])
    with pytest.raises(PreflightError, match="unknown"):
        build_plan(specs, cache, ids=["missing"])


@pytest.mark.parametrize("corrupt", ["missing", "hash", "metadata", "sample_count"])
def test_missing_or_corrupt_cache_fails_without_rendering(tmp_path, corrupt):
    specs, cache = fixture_cache(tmp_path)
    wav = next(cache.rglob("*.wav"))
    meta_path = wav.with_suffix(".json")
    if corrupt == "missing":
        wav.unlink()
    elif corrupt == "hash":
        wav.write_bytes(wav.read_bytes() + b"invalid")
    else:
        meta = json.loads(meta_path.read_text())
        meta["text" if corrupt == "metadata" else "samples"] = "wrong"
        meta_path.write_text(json.dumps(meta))
    with pytest.raises(PreflightError, match="cached"):
        build_plan(specs, cache)
    with pytest.raises(PreflightError, match="forbidden"):
        CacheOnlyRenderer().render("Hello", "f-us")


def test_future_end_offsets_exclude_whole_conversation_and_count_all_events(tmp_path):
    specs, cache = fixture_cache(tmp_path, end_relative=True)
    plan = build_plan(specs, cache)
    assert "noncausal" in plan["conversations"][0]["void"]
    assert plan["exclusions"][0]["event_ids"] == ["x1"]
    assert plan["counts"]["excluded_conversations"] == 1
    assert plan["counts"]["eligible_conversations"] == 0
    assert plan["counts"]["excluded_events"] == 2
    assert plan["counts"]["barge_in"] == plan["counts"]["backchannel"] == 0


def test_cache_change_after_preflight_fails_before_clip_loading(tmp_path):
    specs, cache = fixture_cache(tmp_path)
    conversation = build_plan(specs, cache)["conversations"][0]
    meta_path = next(cache.rglob("*.json"))
    meta = json.loads(meta_path.read_text())
    meta["rendered_at"] = "changed"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(PreflightError, match="changed since preflight"):
        load_clips(conversation, cache)


def test_default_split_filters_templates_and_rejects_hidden_or_other_split(tmp_path):
    specs, cache = fixture_cache(tmp_path)
    raw = json.loads((specs / "tb-09.json").read_text())
    for sid, cid in [("TB-10", "910"), ("TB-11", "911")]:
        extra = dict(raw, id=sid, conversation_id=cid)
        (specs / f"{sid.lower()}.json").write_text(json.dumps(extra))
    (specs.parent / "splits.json").write_text(json.dumps({"dev": ["TB-10"], "test": ["TB-09"]}))
    assert [c["spec_id"] for c in build_plan(specs, cache)["conversations"]] == ["TB-09"]
    assert [c["spec_id"] for c in build_plan(specs, cache, split="dev")["conversations"]] == ["TB-10"]
    for forbidden in ["TB-10", "TB-11", "9111"]:
        with pytest.raises(PreflightError, match="not explicitly assigned"):
            build_plan(specs, cache, ids=[forbidden])
    with pytest.raises(PreflightError, match="only dev or test"):
        build_plan(specs, cache, split="holdout")


def test_no_split_metadata_requires_explicit_local_fixture_ids(tmp_path):
    specs, cache = fixture_cache(tmp_path)
    (specs.parent / "splits.json").unlink()
    with pytest.raises(PreflightError, match="explicit IDs are required"):
        build_plan(specs, cache)
    conversation = build_plan(specs, cache, ids=["TB-09"])["conversations"][0]
    assert conversation["split"] is None


def test_legacy_missing_speed_is_allowed_only_for_canonical_natural_rate_keys(tmp_path):
    specs, cache_dir = fixture_cache(tmp_path)
    for sidecar in cache_dir.rglob("*.json"):
        meta = json.loads(sidecar.read_text())
        meta.pop("speed")
        sidecar.write_text(json.dumps(meta))
    before = {p: p.read_bytes() for p in cache_dir.rglob("*") if p.is_file()}
    conversation = build_plan(specs, cache_dir)["conversations"][0]
    assert all(record["legacy_natural_speed"] for record in conversation["clips"].values())
    assert set(load_clips(conversation, cache_dir)) == {"c1", "b1", "x1", "cz"}
    assert {p: p.read_bytes() for p in cache_dir.rglob("*") if p.is_file()} == before
    offline_renderer = replace(MockRenderer(), name="elevenlabs", model="eleven_turbo_v2_5")
    writable = RenderCache(cache_dir)
    writable.get(offline_renderer, "A slow line.", "f-us", 0.8)
    _, sidecar = writable.paths(offline_renderer, "A slow line.", "f-us", 0.8)
    meta = json.loads(sidecar.read_text())
    meta.pop("speed")
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(PreflightError, match="cached speed mismatch"):
        VerifiedCache(cache_dir).get(CacheOnlyRenderer(), "A slow line.", "f-us", 0.8)
