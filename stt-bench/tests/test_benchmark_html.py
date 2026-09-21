"""Offline report contract tests; synthetic observations, never provider calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_benchmark_html.py"
spec = importlib.util.spec_from_file_location("benchmark_html", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def inputs():
    counts = dict(substitutions=1, insertions=0, deletions=0, reference_words=10)
    rows = []
    for index in range(1000):
        rows.append(dict(clip_id=f"clip-{index}", reference="same frozen reference", accuracy_usable=True,
            word_errors=deepcopy(counts),
            deadlines=[dict(deadline_ms=t, word_errors=deepcopy(counts), pacing_valid=True) for t in report.DEADLINES],
            finalize_latency_ms=100., completion_latency_ms=1000., exclusion_reasons=[], model_versions=[],
            attempts=[dict(attempt=1, valid=True, sent_audio_seconds=2., completion_timed_out=False,
                           transport_failed=False, pacing=dict(valid=True))]))
    manifest=dict(dataset_id="pipecat-stt-benchmark", source_revision=report.REVISION,
                  clips=[dict(clip_id=r["clip_id"]) for r in rows])
    summaries=[]
    for model,*_ in report.MODELS:
        summaries.append(dict(model=model, identity=dict(model=model,dataset="pipecat-stt-benchmark",full_manifest_sha256="frozen"),
            schema_version=4, measurement_version=4, status="complete",full_coverage=True,
            planned_clips=1000, completed_clips=1000, scored_clips=1000, normalization={"name":"frozen-normalizer"},
            shared_measurement_contract=dict(reference_manifest="frozen",deadlines_ms=list(report.DEADLINES)),
            clips=deepcopy(rows), eventual_word_errors=report.aggregate([r["word_errors"] for r in rows]),
            deadlines=report.deadlines(rows),finalize_latency=report.percentiles([100.]*1000),
            completion_latency=report.percentiles([1000.]*1000), failures=0,retries=0,
            smoke_cost_included=False,estimated_cost_usd=None,
            provider_contract=dict(finalization="manual_at_speech_end",completion_basis="done_after_close",sample_rate=16000)))
    sources=[dict(path=f"model-{i}/summary.json",sha256="saved-hash") for i in range(10)]
    return summaries,manifest,"frozen",sources


def test_weighted_wer_and_empty():
    result=report.aggregate([dict(substitutions=1,insertions=0,deletions=0,reference_words=1),
                             dict(substitutions=0,insertions=0,deletions=0,reference_words=99)])
    assert result["wer"] == .01  # Averaging clip percentages would incorrectly give 50%.
    assert report.aggregate([])["wer"] is None


def test_percentiles_match_linear_interpolation_and_missing():
    assert report.percentiles([100.,None,200.]) == dict(n=2,p50_ms=150.,p90_ms=190.)
    assert report.percentiles([None]) == dict(n=0,p50_ms=None,p90_ms=None)


def test_excludes_speechmatics_and_uses_replacement(inputs):
    data=report.build_data(*inputs)
    assert len(data["models"])==10
    assert not any(m["id"].startswith("speechmatics") for m in data["models"])
    assert data["models"][0]["run_id"].endswith("-unformatted")
    assert data["models"][0]["cost"] is None


@pytest.mark.parametrize("change,match",[
    (lambda s:s.update(status="running"),"incomplete"),
    (lambda s:s["clips"][1].update(clip_id="clip-0"),"duplicate"),
    (lambda s:s["identity"].update(full_manifest_sha256="wrong"),"identity differs"),
    (lambda s:s.update(normalization={"name":"changed"}),"normalization differs"),
    (lambda s:s["shared_measurement_contract"].update(version=99),"measurement_contract differs"),
    (lambda s:s.update(model="speechmatics-standard"),"Wrong model"),
    (lambda s:s["eventual_word_errors"].update(wer=.5),"differs from saved"),
    (lambda s:s["finalize_latency"].update(p90_ms=200.),"differs from saved"),
])
def test_rejects_invalid_input(inputs,change,match):
    change(inputs[0][1])
    with pytest.raises(ValueError,match=match):report.build_data(*inputs)


def test_missing_observation_stays_unavailable_and_fixed_intersections(inputs):
    summaries=inputs[0]
    # Distinct exclusions for final accuracy and deadline observations.
    s=summaries[0];s["clips"][0]["accuracy_usable"]=False;s["scored_clips"]=999
    s["eventual_word_errors"]=report.aggregate([r["word_errors"] for r in s["clips"][1:]])
    s["finalize_latency"]=report.percentiles([100.]*999);s["completion_latency"]=report.percentiles([1000.]*999)
    s["clips"][1]["deadlines"][2]["word_errors"]=None;s["deadlines"]=report.deadlines(s["clips"])
    summaries[1]["clips"][2]["deadlines"][0]["pacing_valid"]=False
    summaries[1]["deadlines"]=report.deadlines(summaries[1]["clips"])
    data=report.build_data(*inputs)
    assert len(data["shared_final_ids"])==999
    assert len(data["shared_deadline_ids"])==998
    assert "clip-0" not in data["shared_final_ids"]
    assert all(m["shared_final"]["n"]==999 for m in data["models"])
    assert all(d["measured_clips"]==998 for m in data["models"] for d in m["shared_deadlines"])
    # Filtering embedded models cannot alter precomputed shared counts.
    filtered=data["models"][2:]
    assert all(m["shared_final"]["n"]==999 for m in filtered)
    one=deepcopy(s["clips"][:1]);one[0]["deadlines"][0]["word_errors"]=None
    assert report.deadlines(one)[0]["wer"] is None


def test_retry_attempts_and_retried_clips_are_separate(inputs):
    s=inputs[0][0];row=s["clips"][0];first=row["attempts"][0];first["valid"]=False
    second=deepcopy(first);second.update(attempt=2,valid=True);row["attempts"].append(second)
    s.update(failures=1,retries=1)
    data=report.build_data(*inputs);m=data["models"][0]
    assert (m["failures"],m["attempts"],m["retried_clips"],m["retry_attempts"])==(1,1001,1,1)


def test_embedding_cannot_close_script_tag():
    data={"text":"</script><img src=x onerror=alert(1)>&"}
    template='<script type="application/json">__BENCHMARK_DATA__</script>'
    html=report.render(data,template)
    assert html.count("</script>")==1
    assert "<img" not in html
    assert json.loads(html.split(">",1)[1].rsplit("<",1)[0])==data
