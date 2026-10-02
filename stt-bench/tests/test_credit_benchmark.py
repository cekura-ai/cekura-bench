"""Offline tests for the hard audio budget and frozen public selection."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from scripts.prepare_credit_benchmark import select
from scripts.credit_benchmark_run import validate_plan
from stt_bench.data import sha256, write_json


def fixture_plan(root):
    groups=[]
    for label in ('pipecat','fleurs-general','fleurs-entities'):
        directory=root/label;directory.mkdir()
        sf.write(directory/'clip.wav',np.zeros(55*320,dtype='int16'),16000,subtype='PCM_16')
        clip=dict(clip_id=label,condition='public_anchor',reference='hello',audio='clip.wav',
                  audio_sha256=sha256(directory/'clip.wav'),total_frames=55,speech_frames=5,submitted_seconds=1.1)
        write_json(directory/'manifest.json',dict(schema_version=2,clips=[clip]))
        groups.append(dict(id=label,manifest=label+'/manifest.json',manifest_sha256=sha256(directory/'manifest.json'),
                           clips=1,total_frames=55))
    plan=dict(max_audio_seconds_per_provider=300,max_attempts_per_clip=1,
              models=[dict(model='gradium-default'),dict(model='reson8-realtime')],
              datasets=groups,planned_clips=3,planned_audio_seconds_per_provider=3.3)
    write_json(root/'plan.json',plan)
    return root/'plan.json',plan


def test_valid_frozen_plan(tmp_path):
    path,plan=fixture_plan(tmp_path)
    assert validate_plan(path,sha256(path))==plan


@pytest.mark.parametrize('field,value', [('max_audio_seconds_per_provider',301),('max_attempts_per_clip',2),
                                        ('planned_audio_seconds_per_provider',1),('planned_clips',9)])
def test_cannot_expand_or_misreport_budget(tmp_path,field,value):
    path,plan=fixture_plan(tmp_path);plan[field]=value;write_json(path,plan)
    with pytest.raises(ValueError):validate_plan(path,sha256(path))


def test_changed_manifest_or_nonpublic_audio_rejected(tmp_path):
    path,plan=fixture_plan(tmp_path)
    manifest=tmp_path/plan['datasets'][0]['manifest'];m=json.loads(manifest.read_text())
    m['clips'][0]['condition']='private_short';write_json(manifest,m)
    with pytest.raises(ValueError):validate_plan(path,sha256(path))
    plan['datasets'][0]['manifest_sha256']=sha256(manifest);write_json(path,plan)
    with pytest.raises(ValueError,match='nonpublic'):validate_plan(path,sha256(path))


def test_selection_is_stable_whole_clip_and_result_independent():
    clips=[dict(clip_id=str(i),condition='public_anchor',total_frames=51+i*20) for i in range(20)]
    a=select(clips,'public_anchor',1000)
    b=select(list(reversed(clips)),'public_anchor',1000)
    assert a==b and sum(c['total_frames'] for c in a)<=1000
    changed=copy.deepcopy(clips)
    for c in changed:c['hypothetical_provider_wer']=1
    assert [c['clip_id'] for c in a]==[c['clip_id'] for c in select(changed,'public_anchor',1000)]
