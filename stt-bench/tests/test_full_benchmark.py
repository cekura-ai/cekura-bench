import copy
import json
from pathlib import Path
import pytest
from stt_bench.full_benchmark import MODELS, combine, classify, final_snapshots, load_plan, replay_archive, evaluate
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

def test_text_exception_does_not_break_deadline_scoring():
    config=json.loads(Path('config/models/gradium-default.json').read_text())
    events=[dict(kind='clip_start',time_seconds=0),
        dict(kind='provider_message',time_seconds=.1,message={'type':'error','message':'Concurrency limit exceeded: 3 active sessions'}),
        dict(kind='error',time_seconds=.2,error_type='ProviderError',message='Provider rejected request'),
        dict(kind='clip_end',time_seconds=.3)]
    result=evaluate(events,dict(clip_id='a',cohort='public',reference='hello'),config,1,'hash')
    assert result['failure_class']=='concurrency' and not result['valid']
    assert result['deadlines'][0]['status']=='failed_before_speech_end'
    assert events[2]['message']=='Provider rejected request'

def test_failure_codes_ignore_timestamp_digits():
    for stamp in (.0014017,.002403,.429402):
        assert classify([dict(kind='error',time_seconds=stamp,error_message='Temporary failure in name resolution')])=='transient'

def test_consecutive_sessions_preserve_every_source_frame(monkeypatch):
    import asyncio
    from stt_bench import full_benchmark as full
    captures=[];events=[]
    class Log:
        origin=0
        def now(self):return 0
        def emit(self,kind,**fields):events.append(dict(kind=kind,**fields))
    async def fake(pcm,frames,config,key,log):captures.append((pcm,frames))
    monkeypatch.setattr(full.gradium,'transcribe',fake)
    source=b'\x01\x00'*(13501*480)
    asyncio.run(full.gradium_sessions(source+bytes(48000),13501,{'sample_rate':24000},'fixture',Log()))
    assert [n for _,n in captures]==[13500,1]
    assert b''.join(pcm[:n*960] for pcm,n in captures)==source
    assert all(pcm[n*960:]==bytes(48000) for pcm,n in captures)
    assert events[0]['additional_tail_seconds']==1
    assert events[-1]['kind']=='longform_complete'

def test_nested_credit_error_is_terminal():
    assert classify([{'kind':'provider_message','message':{'error':{'code':7,'message':'You have no credits remaining. Please add credits to continue using the service.'}}}])=='credits'

def test_missing_provider_scopes_are_terminal_authentication_failure():
    assert classify([{'kind':'provider_message','message':{'error':{'code':7,'message':'api key does not have required scopes','details':[]}}}])=='authentication'

def test_consecutive_sessions_use_real_stream_clock_and_terminal_events(tmp_path,monkeypatch):
    import asyncio
    from websockets.asyncio.server import serve
    from stt_bench import full_benchmark as full
    from stt_bench.streaming import EventLog,read_events
    async def check():
        config=json.loads(Path('config/models/gradium-default.json').read_text())
        sessions=[]
        async def server(ws):
            received=[];sessions.append(received)
            async for raw in ws:
                m=json.loads(raw);received.append(m)
                if m['type']=='setup':await ws.send(json.dumps(dict(type='ready',model_name='55966eda@500',sample_rate=24000,delay_in_frames=10)))
                elif m['type']=='flush':
                    for reply in [dict(type='text',text='hello'),dict(type='end_text'),dict(type='flushed',flush_id=1)]:await ws.send(json.dumps(reply))
                elif m['type']=='end_of_stream':
                    await ws.send(json.dumps(dict(type='end_of_stream')));await ws.close();return
        async with serve(server,'127.0.0.1',0) as local:
            monkeypatch.setattr(full,'GRADIUM_SESSION_FRAMES',2)
            monkeypatch.setattr(full.gradium,'connection',lambda c,k:(f'ws://127.0.0.1:{local.sockets[0].getsockname()[1]}',{}))
            path=tmp_path/'raw.jsonl';log=EventLog(path)
            try:await full.gradium_sessions(bytes(54*960),4,config,'fixture',log)
            finally:log.close()
        events=read_events(path)
        clip=dict(clip_id='private',cohort='private',speech_frames=4,reference='hello hello',words=[])
        result,snaps,frames=full.session_assessment(events,clip,config)
        assert result['transcript_complete'] and result['model_verified']
        assert result['transcript']=='hello hello' and result['context_resets']==1
        assert set(frames)=={0,1,2,3}
        assert len(sessions)==2 and all(sum(m['type']=='audio' for m in s)==52 for s in sessions)
        assert len(result['pacing']['transition_gaps_seconds'])==1
        incomplete=events[:-1]
        assert not full.session_assessment(incomplete,clip,config)[0]['transcript_complete']
    asyncio.run(check())
