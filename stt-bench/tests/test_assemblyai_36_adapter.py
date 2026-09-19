import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import pytest
from stt_bench import assemblyai_min_latency as adapter
from stt_bench.providers import validate

CONFIG=json.loads(Path('config/models/assemblyai-universal-3-6-pro-min-latency.json').read_text())

def test_exact_wire_configuration():
    validate(CONFIG)
    url,headers=adapter.connection(CONFIG,'secret')
    params=parse_qs(urlparse(url).query)
    assert params['speech_model']==['universal-3-6-pro']
    assert params['mode']==['min_latency']
    assert json.loads(params['language_codes'][0])==['en']
    assert not any(k in params for k in ('prompt','keyterms_prompt','min_turn_silence','max_turn_silence','agent_context'))
    assert headers=={'Authorization':'secret'} and 'secret' not in url
    with pytest.raises(ValueError):adapter.connection(dict(CONFIG,mode='balanced'),'secret')

@pytest.mark.parametrize('actual',[{}, {'model':'other','mode':'min_latency'}, {'model':CONFIG['model'],'mode':'balanced'}])
def test_configuration_confirmation_is_required(actual):
    with pytest.raises(RuntimeError,match='Model mismatch'):
        adapter.Protocol(CONFIG).feed({'type':'Begin','configuration':actual})
