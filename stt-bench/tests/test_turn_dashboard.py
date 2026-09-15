from copy import deepcopy
import pytest
from scripts.turn_dashboard import validate_model


def sample():
    errors=dict(substitutions=0,insertions=0,deletions=0,reference_words=1)
    rows=[dict(clip_id=str(i),attempted=True,valid=True,word_errors=errors,turn_timing={'ttft_ms':i,'ttfs_ms':i}) for i in range(1,207)]
    timing=dict(n=206,p50_ms=103.5,p90_ms=185.5,p95_ms=195.75,status_counts={'measured':206})
    model=dict(clips=rows,counts=dict(planned=206,attempted=206,valid=206,failed=0,not_run=0),accuracy={**errors,'reference_words':206,'wer':0},ttft=timing,ttfs=deepcopy(timing),deadlines=[],recovery_summary={})
    return model,deepcopy(model)


def test_checked_turn_summary_is_accepted():
    model,saved=sample();validate_model(model,saved)


def test_changed_percentile_or_word_counts_are_rejected():
    model,saved=sample();model['ttft']['p95_ms']=200;saved['ttft']['p95_ms']=200
    with pytest.raises(ValueError,match='percentile'):validate_model(model,saved)
    model,saved=sample();model['clips'][0]['word_errors']={**model['clips'][0]['word_errors'],'insertions':1}
    with pytest.raises(ValueError,match='word counts'):validate_model(model,saved)


def test_duplicate_clips_and_incomplete_attempts_are_rejected():
    model,saved=sample();model['clips'][0]['clip_id']=model['clips'][1]['clip_id']
    with pytest.raises(ValueError,match='coverage'):validate_model(model,saved)
    model,saved=sample();model['clips'][0]['attempted']=False
    with pytest.raises(ValueError,match='incomplete'):validate_model(model,saved)
