"""Synthetic preparation tests; no synthetic data is presented as reviewed audio."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from stt_bench.data import write_json, sha256
from stt_bench.turn_auto import automatic_preparation, candidate_bounds, MODE, SETTINGS
from stt_bench.turns import prepare_turns, freeze_turns, verify_turn_manifest, validate_review_data
from stt_bench.turn_runner import run_turns, prepare_run_plan
from stt_bench.score import score
from test_turns import inputs


class FixtureDetector:
    identity = dict(engine='silero-vad', version='6.2.1',model_sha256='0'*64,
                    runtime='synthetic-test-only', settings=SETTINGS)

    def __init__(self):
        self.inputs=[]

    def __call__(self, x):
        self.inputs.append(x.copy())
        # Multiple islands must remain ONE turn including its thinking pause.
        return [dict(start=0,end=min(1600,len(x))),dict(start=3200,end=len(x))]


def prepared(tmp_path, detector=None, stereo=False):
    src=inputs(tmp_path/'source',stereo=stereo)
    draft=prepare_turns(src,tmp_path/'draft')
    evidence=automatic_preparation(draft,tmp_path/'automatic.json',detector=detector or FixtureDetector())
    return draft,evidence


def test_auto_freezes_without_approvals_preserves_pause_and_channels(tmp_path):
    detector=FixtureDetector()
    draft,evidence=prepared(tmp_path,detector,True)
    np.testing.assert_allclose(detector.inputs[0],-detector.inputs[1],atol=1e-7)
    m=freeze_turns(draft,evidence,tmp_path/'frozen')
    frozen=verify_turn_manifest(m)
    assert frozen['preparation_mode']==MODE and frozen['turn_schema_version']==2
    assert frozen['listening_review_verified'] is False
    assert len(frozen['clips'])==2  # not four islands
    for c in frozen['clips']:
        assert c['review']['boundary_approved'] is False
        assert c['review']['transcript_approved'] is False
        assert c['review']['reviewed_by'] is None
        assert c['reviewed_speech_end_seconds'] is None
        assert c['estimated_speech_end_seconds'] > .5
        assert c['reference']=='yes please'


@pytest.mark.parametrize('mutation', ['fake_approval','changed_boundary','changed_reference','changed_units','missing_detector','wrong_settings','stale_hash'])
def test_auto_evidence_cannot_bypass_validation(tmp_path,mutation):
    draft,evidence=prepared(tmp_path)
    r=json.loads(evidence.read_text())
    if mutation=='fake_approval':r['turns'][0]['boundary_approved']=True
    if mutation=='changed_boundary':r['turns'][0]['end']+=.1
    if mutation=='changed_reference':r['turns'][0]['reference']='different words'
    if mutation=='changed_units':r['turns'][0]['unit_ids']=r['turns'][1]['unit_ids']
    if mutation=='missing_detector':r.pop('detector')
    if mutation=='wrong_settings':r['detector']['settings']['speech_pad_ms']=0
    if mutation=='stale_hash':r['draft_sha256']='0'*64
    write_json(evidence,r)
    with pytest.raises(ValueError):freeze_turns(draft,evidence,tmp_path/'frozen')


def test_missing_speech_excluded_and_accounted_for(tmp_path):
    class OneEmpty(FixtureDetector):
        def __call__(self,x):
            return super().__call__(x) if not self.inputs else []
    draft,evidence=prepared(tmp_path,OneEmpty())
    r=json.loads(evidence.read_text())
    assert r['counts']==dict(candidates=2,included=1,excluded=1)
    assert r['turns'][1]['reason']=='no_speech_detected'
    m=json.loads(freeze_turns(draft,evidence,tmp_path/'frozen').read_text())
    assert m['counts']['excluded_turns']==1
    assert len({u for t in r['turns'] for u in t['unit_ids']})==4


def test_vad_cannot_trim_off_a_reference_word(tmp_path):
    class OneEarly(FixtureDetector):
        def __call__(self,x):
            if not self.inputs:return super().__call__(x)
            return [dict(start=0,end=2000)]
    draft,evidence=prepared(tmp_path,OneEarly())
    r=json.loads(evidence.read_text())
    assert r['turns'][1]['reason']=='vad_end_before_reference_word_end'
    assert r['turns'][1]['reference']=='yes please'


def test_invalid_timestamps_are_excluded_without_fabricated_corrections(tmp_path):
    src=inputs(tmp_path/'source')
    p=src/'A.json';s=json.loads(p.read_text());s['segments'][0]['words'][0]['end']=-1;write_json(p,s)
    draft=prepare_turns(src,tmp_path/'draft')
    evidence=automatic_preparation(draft,tmp_path/'auto.json',detector=FixtureDetector())
    r=json.loads(evidence.read_text());assert r['turns'][0]['reason']=='invalid_word_timing'
    assert r['turns'][0]['word_corrections']=={}


def test_short_reply_and_word_outside_segment_are_eligible(tmp_path):
    src=inputs(tmp_path/'source')
    p=src/'A.json';s=json.loads(p.read_text());s.update(transcript='yes')
    s['segments']=[dict(start=.1,end=.18,text='yes',words=[dict(word='yes',start=.1,end=.2)])];write_json(p,s)
    draft=prepare_turns(src,tmp_path/'draft');d=json.loads(draft.read_text())
    bounds,reason=candidate_bounds(d,d['turns'][0]);assert reason is None and bounds['word_end']==.2


def test_automatic_dataset_dry_run_and_report(tmp_path,monkeypatch):
    draft,evidence=prepared(tmp_path)
    manifest=freeze_turns(draft,evidence,tmp_path/'frozen')
    config=Path('config/profiles/private-turns-v1/deepgram-nova-3.json')
    # Real shared runner, synthetic audio, actual local pacing; no provider calls.
    out=tmp_path/'run'
    asyncio.run(run_turns(manifest,config,out,dry_run=True))
    report=score(out,tmp_path/'report')
    assert report['counts']['attempted']==2 and report['listening_review_verified'] is False
    assert all(not r['human_review_verified'] for r in report['clips'])
    assert report['ttfs']['n']==0 and report['ttft']['n']==0
    assert 'automatically estimated' in (tmp_path/'report/index.html').read_text()
    plan=prepare_run_plan(manifest,[config],tmp_path/'plan.json')
    assert plan['planned_sessions']==2


def test_manifest_cannot_relabel_automatic_as_reviewed(tmp_path):
    draft,evidence=prepared(tmp_path)
    manifest=freeze_turns(draft,evidence,tmp_path/'frozen')
    m=json.loads(manifest.read_text());m['listening_review_verified']=True;write_json(manifest,m)
    with pytest.raises(ValueError):verify_turn_manifest(manifest)


def test_automatic_cli_freezes_without_review_argument(tmp_path,monkeypatch,capsys):
    import sys
    from stt_bench.cli import main
    draft=prepare_turns(inputs(tmp_path/'source'),tmp_path/'draft')
    monkeypatch.setattr('stt_bench.turn_auto.SileroDetector',FixtureDetector)
    monkeypatch.setattr(sys,'argv',['stt-bench','auto-prepare-turns','--draft',str(draft),'--out',str(tmp_path/'frozen')])
    main()
    m=verify_turn_manifest(tmp_path/'frozen/manifest.json')
    assert m['counts']['turns']==2 and m['listening_review_verified'] is False
    assert str(tmp_path/'frozen/manifest.json') in capsys.readouterr().out


def test_installed_silero_returns_no_speech_for_silence():
    pytest.importorskip('silero_vad')
    from stt_bench.turn_auto import SileroDetector
    detector=SileroDetector()
    assert detector(np.zeros(16000,dtype=np.float32))==[]
    assert detector.identity['runtime']=='torch-cpu'
