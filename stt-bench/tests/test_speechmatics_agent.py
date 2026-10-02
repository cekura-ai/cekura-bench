import asyncio
import json
from pathlib import Path
import pytest
from stt_bench import speechmatics_agent as agent
from stt_bench import provider_protocol as shared
from stt_bench.providers import validate, reduce_events, transcript_at
from stt_bench.linden_full_benchmark import combine, MODELS, final_snapshots, classify
from stt_bench.streaming import EventLog, read_events
from stt_bench.score import word_errors


def config():
    return json.loads(Path('config/models/speechmatics-linden-1.json').read_text())


def event(kind, at, **fields):
    return dict(kind=kind, time_seconds=at, **fields)


def segment(text, partial=False):
    return {'message':'AddPartialSegment' if partial else 'AddSegment', 'segment':{'transcript':text}}


def test_agent_wire_contract_and_explicit_identity():
    c=validate(config());p=agent.Protocol(c)
    assert p.setup()['turn_config']=={'turn_detection_mode':'external'}
    assert p.setup()['transcription_config']['model']=='linden-1'
    for _ in range(7):p.audio(bytes(640))
    assert p.finalize()=={'message':'ForceEndOfUtterance','timestamp':.14}
    assert p.finish(57)=={'message':'EndOfStream','last_seq_no':57}
    with pytest.raises(ValueError):validate({**c,'endpoint':'wss://us.rt.speechmatics.com/v2'})
    with pytest.raises(ValueError):validate({**c,'turn_detection_mode':'vad'})
    p.feed({'message':'RecognitionStarted','model':'enhanced'})
    assert p.model_mismatch


def test_segments_replace_partial_preserve_repetitions_ignore_rt_duplicates():
    c=config()
    events=[event('provider_message',.1,message=segment('ye',True)),
            event('provider_message',.2,message=segment('yes')),
            event('provider_message',.3,message={'message':'AddTranscript','metadata':{'transcript':'yes'}}),
            event('provider_message',.4,message=segment('yes')),
            event('provider_message',.5,message=segment('please',True)),
            event('provider_message',.6,message=segment('please.'))]
    assert transcript_at(events,.5,c)['text']=='yes yes please'
    assert transcript_at(events,.5,c)['final_text']=='yes yes'
    assert transcript_at(events,1,c)['final_text']=='yes yes please.'
    assert final_snapshots(events,c)[-1]==dict(text='yes yes please.',time_seconds=.6)


def test_completion_requires_terminal_audio_and_no_partial():
    c=config()
    events=[event('model_accepted',0,model='linden-1'),event('speech_end',1),
            event('provider_message',1.1,message=segment('hello')),
            event('audio_complete',2),event('close_stream_requested',2),
            event('provider_message',2.1,message={'message':'EndOfTranscript'}),
            event('provider_terminal',2.1)]
    r=reduce_events(events,c)
    assert r['transcript_complete'] and r['model_verified']
    assert r['finalize_latency_status']=='unsupported' and r['finalize_latency_ms'] is None
    assert not reduce_events(events[:-2],c)['transcript_complete']
    events.insert(3,event('provider_message',1.2,message=segment('pending',True)))
    assert not reduce_events(events,c)['transcript_complete']
    assert classify([event('provider_message',0,message={'message':'Error','reason':'Concurrent Quota Exceeded'})])=='concurrency'


def test_exchange_sends_timestamp_and_waits_for_end_of_transcript(tmp_path):
    class Socket:
        def __init__(self):self.queue=asyncio.Queue();self.sent=[]
        async def send(self,raw):
            if isinstance(raw,bytes):return
            m=json.loads(raw);self.sent.append(m)
            if m['message']=='StartRecognition':await self.queue.put({'message':'RecognitionStarted'})
            elif m['message']=='ForceEndOfUtterance':await self.queue.put(segment('hello'))
            elif m['message']=='EndOfStream':await self.queue.put({'message':'EndOfTranscript'})
        def __aiter__(self):return self
        async def __anext__(self):return json.dumps(await self.queue.get())
    async def streamer(pcm,n,send,finalize,log,**kwargs):
        await send(bytes(640));log.emit('speech_end');await finalize(log.now())
        for _ in range(50):await send(bytes(640))
        log.emit('audio_complete')
    async def run():
        ws=Socket();log=EventLog(tmp_path/'events.jsonl')
        try:await shared.exchange(ws,b'',1,config(),log,protocol_factory=agent.Protocol,streamer=streamer)
        finally:log.close()
        assert ws.sent[1]=={'message':'ForceEndOfUtterance','timestamp':.02}
        assert ws.sent[-1]=={'message':'EndOfStream','last_seq_no':51}
        assert reduce_events(read_events(tmp_path/'events.jsonl'),config())['transcript_complete']
    asyncio.run(run())


def test_partitioned_merge_pools_words_and_rejects_duplicate_attempts():
    from test_full_benchmark import row
    m=MODELS[0];rows=[row('a',1,True),row('b',1,True,'apple')]
    plan=dict(run_id='test',items=[dict(clip_id=x,cohort='public') for x in ('a','b')],configs={m:config()})
    def state(rs):return dict(verified=True,assignment={'model':m},rows=rs)
    one=combine(plan,[state(rows)])['models'];split=combine(plan,[state(rows[:1]),state(rows[1:])])['models']
    assert one==split
    assert one[m]['groups']['combined']['final_wer']['wer']==.25
    with pytest.raises(ValueError,match='Duplicate'):combine(plan,[state(rows),state(rows)])
