import asyncio
import json
from pathlib import Path
import pytest
from stt_bench.assemblyai import Protocol, connection, replay
from stt_bench.providers import validate
from stt_bench.credentials import credential

CONFIG=json.loads(Path('config/models/assemblyai-universal-3-5-pro.json').read_text())

def test_auth_alias_and_model():
    validate(CONFIG)
    assert credential('assemblyai',{'ASSEMBLY_API_KEY':'secret'},None)==('secret','ASSEMBLY_API_KEY')
    url,headers=connection(CONFIG,'secret')
    assert headers=={'Authorization':'secret'} and 'secret' not in url
    assert 'speech_model=universal-3-5-pro' in url

def test_turn_order_final_revisions_and_late_partial():
    p=Protocol(CONFIG)
    for order,text,final in [(1,'world',False),(0,'hello',True),(1,'world',True),(0,'Hello,',True),(0,'bad',False)]:
        p.feed(dict(type='Turn',turn_order=order,transcript=text,end_of_turn=final))
    assert p.snapshot()['final_text']=='Hello, world'
    assert not p.snapshot()['provisional']
    assert p.finalize() is None and p.finish(10)=={'type':'Terminate'}

def test_termination_requires_close_and_complete_audio():
    def e(kind,**kw):return dict(kind=kind,time_seconds=1,**kw)
    events=[e('model_accepted',model=CONFIG['model']),e('provider_message',message=dict(type='Begin')),
        e('provider_message',message=dict(type='Turn',turn_order=0,transcript='hello',end_of_turn=True))]
    assert not replay(events,CONFIG)[1]['transcript_complete']
    events += [e('audio_complete'),e('close_stream_requested'),
        e('provider_message',message=dict(type='Termination')),e('provider_terminal')]
    assert replay(events,CONFIG)[1]['transcript_complete']
    assert not replay(events+[e('error')],CONFIG)[1]['transcript_complete']

def test_unfinalized_turn_is_incomplete():
    events=[dict(kind=k,time_seconds=i,**v) for i,(k,v) in enumerate([
      ('model_accepted',dict(model=CONFIG['model'])),('audio_complete',{}),('close_stream_requested',{}),
      ('provider_message',dict(message=dict(type='Turn',turn_order=0,transcript='pending',end_of_turn=False))),
      ('provider_message',dict(message=dict(type='Termination'))),('provider_terminal',{})])]
    assert not replay(events,CONFIG)[1]['transcript_complete']

def test_provider_error():
    with pytest.raises(RuntimeError):Protocol(CONFIG).feed(dict(type='Error',error='failure'))

@pytest.mark.parametrize('failure',[None,'disconnect','timeout'])
def test_exchange_lifecycle(tmp_path,monkeypatch,failure):
    from stt_bench import provider_protocol as shared
    from stt_bench.streaming import EventLog,read_events
    async def scenario():
        class Socket:
            def __init__(self):
                self.queue=asyncio.Queue(); self.sent=[]
                self.queue.put_nowait(json.dumps(dict(type='Begin')))
            def __aiter__(self):return self
            async def __anext__(self):
                v=await self.queue.get()
                if v is None:raise StopAsyncIteration
                return v
            async def send(self,value):
                self.sent.append(value)
                if isinstance(value,str) and json.loads(value)['type']=='Terminate':
                    if failure=='disconnect':self.queue.put_nowait(None)
                    elif failure!='timeout':
                        self.queue.put_nowait(json.dumps(dict(type='Turn',turn_order=0,transcript='hello',end_of_turn=True)))
                        self.queue.put_nowait(json.dumps(dict(type='Termination')))
                        self.queue.put_nowait(None)
        async def audio(pcm,frames,send,finalize,log,**kwargs):
            await send(b'pcm');log.emit('speech_end');await finalize(log.now());log.emit('audio_complete')
        monkeypatch.setattr(shared,'stream_audio',audio)
        socket=Socket();log=EventLog(tmp_path/'raw.jsonl')
        cfg=dict(CONFIG,close_timeout_seconds=.01)
        try:
            if failure:
                with pytest.raises((RuntimeError,TimeoutError)):
                    await shared.exchange(socket,b'pcm',1,cfg,log,protocol_factory=Protocol)
            else:await shared.exchange(socket,b'pcm',1,cfg,log,protocol_factory=Protocol)
        finally:log.close()
        assert json.loads(socket.sent[-1])=={'type':'Terminate'}
        result=replay(read_events(tmp_path/'raw.jsonl'),cfg)[1]
        assert result['transcript_complete']==(failure is None)
    asyncio.run(scenario())

@pytest.mark.parametrize('count',range(3,19))
def test_wire_packets_preserve_every_sample_and_silence(count,monkeypatch):
    from stt_bench import assemblyai_pacing as ap
    from stt_bench.streaming import pacing_metrics
    class Log:
        origin=0
        clock=0
        def __init__(self):self.events=[]
        def now(self):return self.clock
        def emit(self,kind,at=None,**fields):self.events.append(dict(kind=kind,time_seconds=self.clock if at is None else at,**fields))
    log=Log();sent=[]
    async def wait(deadline,*args,**kwargs):log.clock=max(log.clock,deadline)
    async def send(data):sent.append(data);log.clock+=.0001
    async def finalize(t0):pass
    monkeypatch.setattr(ap,'wait_until',wait)
    pcm=b'\x01\x00'*(count*320)+b'\x00'*(50*640)
    asyncio.run(ap.stream_audio(pcm,count,send,finalize,log))
    assert b''.join(sent)==pcm
    assert all(50<=len(p)/32<=1000 for p in sent)
    assert pacing_metrics(log.events)['valid']
    bad=[dict(e) for e in log.events]
    bad[-1]['kind']='missing_complete'
    assert not pacing_metrics(bad)['valid']
    packet=next(e for e in log.events if e['kind']=='audio_sent')
    packet['send_completed_seconds']+=.05
    assert not pacing_metrics(log.events)['valid']

def test_server_model_mismatch():
    p=Protocol(CONFIG);p.feed(dict(type='Begin',configuration={'model':'different'}))
    assert p.model_mismatch
