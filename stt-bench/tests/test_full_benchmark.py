import copy
import json
from pathlib import Path
import pytest
from stt_bench.full_benchmark import MODELS, combine, classify, final_snapshots, load_plan, replay_archive
from stt_bench.score import word_errors
from stt_bench.measurement import deadline_observations

def row(cid,number,valid,text='apple pear',cohort='public',completion=10):
    return dict(clip_id=cid,attempt=number,valid=valid,cohort=cohort,word_errors=word_errors('apple pear',text),
        failure_class=None,pacing={'valid':valid},sent_audio_seconds=1,completion_latency_ms=completion,
        deadlines=deadline_observations([], '', None, None, 'live'),private_latency=None)

def state(rows):
    return {'verified':True,'assignment':{'model':MODELS[0]},'rows':rows}

def test_partitioned_merge_reproduces_counts_and_percentiles():
    plan={'run_id':'test','items':[{'clip_id':str(i),'cohort':'public'} for i in range(4)],'configs':dict.fromkeys(MODELS,{})}
    rows=[row(str(i),1,True,'apple' if i%2 else 'apple pear',completion=i*100) for i in range(4)]
    a=combine(plan,[state(rows)])['models'];b=combine(plan,[state(rows[:1]),state(rows[1:])])['models']
    assert a==b
    assert a[MODELS[0]]['groups']['combined']['final_wer']['wer']==.25
    assert a[MODELS[0]]['groups']['public']['completion_first_attempt']['p50_ms']==150
    with pytest.raises(ValueError,match='Duplicate'): combine(plan,[state(rows),state(rows)])

def test_recovery_does_not_rescue_original_latency_or_deadlines():
    plan={'run_id':'test','items':[{'clip_id':'a','cohort':'public'}],'configs':dict.fromkeys(MODELS,{})}
    first=row('a',1,False,'');second=row('a',2,True)
    report=combine(plan,[state([first,second])])['models'][MODELS[0]]
    assert report['groups']['combined']['recovered']==1
    assert report['groups']['public']['completion_first_attempt']['n']==0
    assert report['groups']['public']['deadlines'][0]['measured_clips']==0
    assert report['groups']['combined']['final_wer']['wer']==0
    with pytest.raises(ValueError,match='Recovery without'):combine(plan,[state([second])])

@pytest.mark.parametrize('message,expected',[('HTTP 429','concurrency'),('insufficient credits','credits'),('HTTP 402','credits'),('HTTP 401','authentication'),('Provider selected a different model','model_identity'),('connection reset','transient')])
def test_failure_classification(message,expected):
    assert classify([dict(kind='error',message=message)])==expected

@pytest.mark.parametrize('model',MODELS)
def test_long_final_sequence_keeps_repeated_words_and_receipt_times(model):
    c=json.loads(Path(f'config/models/{model}.json').read_text())
    events=[]
    for i in range(500):
        if c['provider']=='gradium':
            messages=[dict(type='text',text='yes',start_s=i),dict(type='end_text',stop_s=i+1)]
        elif c['provider']=='reson8': messages=[dict(type='transcript',text='yes',is_final=True)]
        elif c['provider']=='smallest': messages=[dict(type='transcription',transcript='yes',is_final=True,is_last=False)]
        else: messages=[{'result':{'transcription':{'transcript':'yes','isFinal':True}}}]
        events += [dict(kind='provider_message',time_seconds=i+1,message=m) for m in messages]
    snaps=final_snapshots(events,c)
    assert len(snaps[-1]['text'].split())==500
    assert snaps[-1]['time_seconds']==500

def test_full_manifest_rejects_duplicates_and_wrong_scope(tmp_path):
    p=tmp_path/'plan.json'
    data=dict(models=list(MODELS),max_attempts=2,workers_per_model=10,
        items=[dict(clip_id=str(i),cohort='public' if i<1000 else 'private') for i in range(1008)])
    p.write_text(json.dumps(data));assert len(load_plan(p)['items'])==1008
    data['items'][-1]['clip_id']='0';p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='coverage'):load_plan(p)

def test_replay_rejects_tampered_archive_before_reading(tmp_path):
    p=tmp_path/'evidence.tar.gz';p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):replay_archive({},p,'incorrect',tmp_path/'verified.json')

def test_unverified_results_never_enter_report():
    with pytest.raises(ValueError,match='Unverified'):combine({},[{'verified':False}])
