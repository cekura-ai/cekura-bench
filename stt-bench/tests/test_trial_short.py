import asyncio
import importlib
import json
from pathlib import Path
import pytest
from stt_bench.data import sha256
from stt_bench.score import score


def test_one_attempt_and_private_authorization_are_enforced(tmp_path,monkeypatch):
    runner=importlib.import_module('stt_bench.run')
    source=Path('reports/trial-short-20260914/dataset/manifest.json')
    # This private integration fixture is local only; unit tests below remain portable.
    if not source.exists():pytest.skip('Local private selection not present')
    monkeypatch.setattr(runner,'validate_preflight',lambda _: {})
    monkeypatch.setattr(runner,'require_credential',lambda _: 'fixture')
    config=Path('config/models/soniox-stt-rt-v5.json')
    with pytest.raises(ValueError,match='authorization'):
        asyncio.run(runner.run(source,config,tmp_path/'denied',False,max_attempts=1))
    calls=[]
    async def failed(*args):
        calls.append(1);raise RuntimeError('fixture')
    monkeypatch.setattr(runner,'transcribe',failed)
    asyncio.run(runner.run(source,config,tmp_path/'run',False,max_attempts=1,
        authorized_private_manifest_sha256=sha256(source),stop_on_provider_failure=True))
    assert len(calls)==1
    outcomes=json.loads((tmp_path/'run/outcomes.json').read_text())
    assert len(outcomes)==1 and len(outcomes[0]['attempts'])==1
    report=score(tmp_path/'run',tmp_path/'report')
    assert not report['completeness']['all_planned_clips_have_attempts']
    assert len(report['public_private_gap'])==1
    assert report['public_private_gap'][0]['condition']=='private_short'


def test_attempt_budget_validation(tmp_path,monkeypatch):
    runner=importlib.import_module('stt_bench.run')
    monkeypatch.setattr(runner,'load_manifest',lambda _: {'clips':[]})
    for value in (0,3,-1):
        with pytest.raises(ValueError,match='one or two'):
            asyncio.run(runner.run(Path('unused'),Path('config/models/soniox-stt-rt-v5.json'),tmp_path,True,max_attempts=value))
