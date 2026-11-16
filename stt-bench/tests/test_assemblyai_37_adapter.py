import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import pytest
from stt_bench import assemblyai_min_latency as adapter
from stt_bench.providers import validate

CONFIG=json.loads(Path('config/models/assemblyai-stt-3-7-realtime-agent-min-latency.json').read_text())
TURN=json.loads(Path('config/profiles/private-turns-v1/assemblyai-stt-3-7-realtime-agent-min-latency.json').read_text())
# The Begin event the live API returned for this request on 2026-11-16.
LIVE_BEGIN={'type':'Begin','id':'2c79f999-b7f8-402d-a556-41aa49edcec1','expires_at':1794855619,
    'configuration':{'model':'assemblyai-stt-3-7-realtime-agent','mode':'min_latency','api_version':'2025-05-12',
    'speaker_labels':False,'redact_pii':False,'filter_profanity':False,'domain':None,'voice_focus':None}}

@pytest.mark.parametrize('config',[CONFIG,TURN])
def test_exact_wire_configuration(config):
    validate(config)
    url,headers=adapter.connection(config,'secret')
    params=parse_qs(urlparse(url).query)
    assert params['speech_model']==['assemblyai-stt-3-7-realtime-agent']
    assert params['mode']==['min_latency']
    assert json.loads(params['language_codes'][0])==['en']
    # Audio only, as for every other provider: no agent context, prompts or silence overrides.
    assert not any(k in params for k in ('prompt','keyterms_prompt','min_turn_silence','max_turn_silence','agent_context','previous_context_n_turns'))
    assert headers=={'Authorization':'secret'} and 'secret' not in url
    with pytest.raises(ValueError):adapter.connection(dict(config,mode='balanced'),'secret')

def test_turn_profile_differs_from_full_config_only_in_turn_fields():
    extra={k:v for k,v in TURN.items() if k not in CONFIG or CONFIG[k]!=v}
    # As for 3.6, the turn profile carries its own caveat text.
    assert set(extra)=={'finalization','measurement_profile','profile_version','force_endpoint','turn_finalization_class','base_config','protocol_caveat'}
    assert TURN['base_config']=='config/models/assemblyai-stt-3-7-realtime-agent-min-latency.json'

def test_live_begin_is_accepted():
    protocol=adapter.Protocol(CONFIG)
    protocol.feed(LIVE_BEGIN)
    assert protocol.ready and not protocol.model_mismatch

@pytest.mark.parametrize('actual',[{}, {'model':'universal-3-6-pro','mode':'min_latency'},
    {'model':'assemblyai-stt-3-7-realtime','mode':'min_latency'}, {'model':CONFIG['model'],'mode':'balanced'}])
def test_configuration_confirmation_is_required(actual):
    with pytest.raises(RuntimeError,match='Model mismatch'):
        adapter.Protocol(CONFIG).feed({'type':'Begin','configuration':actual})

def test_full_run_plan_accepts_this_model(tmp_path):
    from stt_bench import assemblyai_full_benchmark as full
    p=tmp_path/'plan.json'
    items=[dict(clip_id=str(i),cohort='public' if i<1000 else 'private') for i in range(1008)]
    p.write_text(json.dumps(dict(models=[CONFIG['model_id']],max_attempts=2,workers_per_model=20,items=items)))
    assert full.load_plan(p)['models']==[CONFIG['model_id']]

def test_earlier_assemblyai_profiles_are_unchanged():
    old=json.loads(Path('config/models/assemblyai-universal-3-6-pro-min-latency.json').read_text())
    params=parse_qs(urlparse(adapter.connection(old,'secret')[0]).query)
    assert params['speech_model']==['universal-3-6-pro']
