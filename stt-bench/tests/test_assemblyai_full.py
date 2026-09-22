import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import pytest
from stt_bench import assemblyai_full_benchmark as full, assemblyai_min_latency as adapter
from stt_bench.full_private_metrics import latency
from stt_bench.providers import validate

CONFIG=json.loads(Path('config/models/assemblyai-universal-3-5-pro-min-latency.json').read_text())

def test_exact_wire_configuration():
    validate(CONFIG)
    url,headers=adapter.connection(CONFIG,'secret')
    params=parse_qs(urlparse(url).query)
    assert params['speech_model']==['universal-3-5-pro']
    assert params['mode']==['min_latency']
    assert json.loads(params['language_codes'][0])==['en']
    assert not any(k in params for k in ('prompt','keyterms_prompt','min_turn_silence','max_turn_silence','agent_context'))
    assert headers=={'Authorization':'secret'} and 'secret' not in url
    with pytest.raises(ValueError):adapter.connection(dict(CONFIG,mode='balanced'),'secret')

@pytest.mark.parametrize('actual',[{}, {'model':'other','mode':'min_latency'}, {'model':CONFIG['model'],'mode':'balanced'}])
def test_configuration_confirmation_is_required(actual):
    with pytest.raises(RuntimeError,match='Model mismatch'):
        adapter.Protocol(CONFIG).feed({'type':'Begin','configuration':actual})

def test_long_final_revisions_and_private_source_word_delivery():
    p=adapter.Protocol(CONFIG)
    p.feed({'type':'Begin','configuration':{'model':CONFIG['model'],'mode':CONFIG['mode']}})
    events=[]
    for i in range(1000):
        message={'type':'Turn','turn_order':i,'end_of_turn':True,'transcript':'hello'}
        p.feed(message);events.append({'kind':'provider_message','time_seconds':i+1.,'message':message})
    revision={'type':'Turn','turn_order':999,'end_of_turn':True,'transcript':'world'}
    p.feed(revision);events.append({'kind':'provider_message','time_seconds':1001.,'message':revision})
    assert len(p.snapshot()['final_text'].split())==1000
    assert full.final_snapshots(events,CONFIG)[-1]['text']==p.snapshot()['final_text']
    packets=[{'kind':'audio_sent','source_frames':3,'bytes':1920,'send_completed_seconds':.06},
             {'kind':'audio_sent','source_frames':5,'bytes':3200,'send_completed_seconds':.16}]
    frames=full.packet_source_frames(packets)
    assert frames==dict(zip(range(8),[.06]*3+[.16]*5))
    clip={'reference':'hello','words':[{'text':'hello','end':.04,'timing_valid':True}]}
    result=latency(clip,[{'text':'hello','time_seconds':.5}],frames,True)
    assert result['words'][0]['delay_ms']==440

def test_provider_rate_limit_is_not_bad_credentials():
    assert full.classify([{'kind':'error','error_message':'Unauthorized connection: Too many concurrent sessions'}])=='concurrency'
    assert full.classify([{'kind':'error','http_status':401,'error_message':'invalid api key'}])=='authentication'

def test_no_model_leaks_into_existing_run():
    from stt_bench.full_benchmark import MODELS
    assert len(MODELS)==4 and len(full.MODELS)==1 and 'assemblyai' in full.MODELS[0]

# Use the same merge invariants against this isolated model profile.
from stt_bench.score import word_errors

def test_merge_retains_original_delays_and_word_weights():
    model=full.MODELS[0]
    plan={'run_id':'test','items':[{'clip_id':c,'cohort':'private'} for c in ['a','b']], 'configs':{model:CONFIG}}
    def row(cid,n,valid,delays):
        return dict(clip_id=cid,cohort='private',attempt=n,valid=valid,word_errors=word_errors('one two','one two'),failure_class=None,
          pacing={'valid':valid},sent_audio_seconds=1,completion_latency_ms=100,
          private_latency=None if n==2 else {'reference_words':2,'words':[{'delay_ms':d} for d in delays], 'exclusions':{} if valid else {'invalid_first_attempt':2}})
    rows=[row('a',1,True,[10,90]),row('b',1,False,[]),row('b',2,True,[])]
    def state(rows):return {'verified':True,'assignment':{'model':model},'rows':rows}
    combined=full.combine(plan,[state(rows)])['models']
    assert combined==full.combine(plan,[state(rows[:1]),state(rows[1:])])['models']
    g=combined[model]['groups']['private']
    assert g['word_finalization']['n']==2 and g['word_finalization']['p50_ms']==50
    assert g['completion_first_attempt']['n']==1 and g['recovered']==1
