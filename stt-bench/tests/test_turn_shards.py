import asyncio
import json
from pathlib import Path
import pytest

from stt_bench.turn_runner import run_turns, smoke_selection
from stt_bench.turns import prepare_turns, freeze_turns
from stt_bench.data import write_json, sha256
from test_turns import inputs, approved


def test_selected_turns_use_same_manifest_and_resume_cannot_expand(tmp_path):
    d=prepare_turns(inputs(tmp_path/'source'),tmp_path/'draft')
    manifest=freeze_turns(d,approved(d),tmp_path/'frozen')
    ids=[c['clip_id'] for c in json.loads(manifest.read_text())['clips']]
    config=Path('config/profiles/private-turns-v1/deepgram-nova-3.json')
    out=tmp_path/'run'
    result=asyncio.run(run_turns(manifest,config,out,dry_run=True,selected_clip_ids=ids[:1]))
    assert [r['clip_id'] for r in result]==ids[:1]
    before=sha256(out/'raw'/f'{ids[0]}--attempt-1.jsonl')
    asyncio.run(run_turns(manifest,config,out,dry_run=True,resume=True,selected_clip_ids=ids[:1]))
    assert sha256(out/'raw'/f'{ids[0]}--attempt-1.jsonl')==before
    with pytest.raises(ValueError,match='assignment'):
        asyncio.run(run_turns(manifest,config,out,dry_run=True,resume=True,selected_clip_ids=ids))


def test_full_shard_cannot_bypass_smoke(tmp_path):
    d=prepare_turns(inputs(tmp_path/'source',segments=6),tmp_path/'draft')
    manifest=freeze_turns(d,approved(d),tmp_path/'frozen')
    m=json.loads(manifest.read_text());smoke=smoke_selection(m)
    full=[c['clip_id'] for c in m['clips'] if c['clip_id'] not in smoke]
    config=Path('config/profiles/private-turns-v1/deepgram-nova-3.json')
    with pytest.raises(ValueError,match='smoke evidence'):
        asyncio.run(run_turns(manifest,config,tmp_path/'run',selected_clip_ids=full))
    receipt=dict(manifest_sha256=sha256(manifest),config_sha256=sha256(config),
        rows=[dict(clip_id=cid,valid=True,turn_timing=dict(ttft_ms=400,ttfs_ms=None)) for cid in smoke])
    with pytest.raises(ValueError,match='passing model smoke'):
        asyncio.run(run_turns(manifest,config,tmp_path/'run',selected_clip_ids=full,smoke_receipt=receipt))


def test_explicit_completed_empty_smoke_preserves_missing_metrics():
    from stt_bench.turn_runner import smoke_row_passed
    profile={'turn_finalization_class':'controlled'}
    row={'valid':True,'turn_timing':{'ttft_ms':None,'ttfs_ms':None,'ttft_status':'no_text','ttfs_status':'no_final_text'}}
    assert not smoke_row_passed(row, profile)
    assert smoke_row_passed(row, profile, 'completed-empty-allowed-v1')
    assert row['turn_timing']['ttft_ms'] is None
    assert not smoke_row_passed(dict(row,valid=False), profile, 'completed-empty-allowed-v1')
    assert not smoke_row_passed(dict(row,turn_timing=dict(row['turn_timing'],ttfs_status='incomplete_stream')), profile, 'completed-empty-allowed-v1')
