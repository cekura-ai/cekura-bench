import asyncio
import base64
import json
import re

import pytest

from stt_bench import nvidia_batch

CONFIG = {'model': 'm', 'temperature': 0, 'min_max_tokens': 256, 'max_tokens_per_audio_second': 12,
          'chat_template_kwargs': {'enable_thinking': False}, 'audio': {'format': 'wav'}, 'prompt': 'p',
          'request_timeout_seconds': 5}


def clip(tmp_path, reference='hello there'):
    path = tmp_path / 'a.wav'
    path.write_bytes(b'RIFF' + bytes(64))
    return dict(clip_id='c', cohort='public', reference=reference, audio=str(path),
                audio_sha256=nvidia_batch.sha256(path), seconds=2.0)


def reply(text='hello there', reason='stop', model='m', status=200):
    return lambda url, body, timeout: (status, json.dumps({'model': model, 'choices': [{'finish_reason': reason, 'message': {'content': text}}]}))


def test_request_sends_the_audio_then_the_prompt_with_thinking_off_and_scales_the_token_budget():
    body = nvidia_batch.request_body(CONFIG, b'RIFF', 100.0)
    audio, prompt = body['messages'][0]['content']
    assert audio == {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(b'RIFF').decode(), 'format': 'wav'}}
    assert prompt == {'type': 'text', 'text': 'p'}
    assert body['chat_template_kwargs'] == {'enable_thinking': False}
    assert body['max_tokens'] == 1200
    assert nvidia_batch.request_body(CONFIG, b'RIFF', 2.0)['max_tokens'] == 256


def test_a_complete_reply_is_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia_batch, 'post', reply('Hello there.'))
    row = asyncio.run(nvidia_batch.transcribe_one('u', CONFIG, clip(tmp_path), 1))
    assert row['valid'] and row['hypothesis'] == 'Hello there.'
    assert row['word_errors']['reference_words'] == 2
    assert sum(row['word_errors'][k] for k in ('substitutions', 'insertions', 'deletions')) == 0


@pytest.mark.parametrize('fake, failure', [
    (reply(reason='length'), 'incomplete'),
    (reply(model='other'), 'model_identity'),
    (reply(status=503), 'server'),
    (reply(status=403), 'authentication'),
])
def test_a_cut_off_relabelled_or_refused_reply_is_not_scored(tmp_path, monkeypatch, fake, failure):
    monkeypatch.setattr(nvidia_batch, 'post', fake)
    row = asyncio.run(nvidia_batch.transcribe_one('u', CONFIG, clip(tmp_path), 1))
    assert not row['valid'] and row['failure_class'] == failure


def test_changed_audio_is_refused_before_any_request(tmp_path, monkeypatch):
    item = clip(tmp_path)
    (tmp_path / 'a.wav').write_bytes(b'RIFF' + bytes(65))
    monkeypatch.setattr(nvidia_batch, 'post', lambda *args: pytest.fail('sent a request'))
    with pytest.raises(ValueError, match='Frozen audio changed'):
        asyncio.run(nvidia_batch.transcribe_one('u', CONFIG, item, 1))


def test_the_summary_scores_the_first_valid_attempt_and_counts_the_failures_before_it(tmp_path):
    plan = [dict(clip_id='c', cohort='public')]
    retried = {'valid': False, 'failure_class': 'server'}
    first = {'valid': True, 'round_trip_ms': 10, 'word_errors': {'substitutions': 1, 'insertions': 0, 'deletions': 0, 'reference_words': 4}}
    later = {'valid': True, 'round_trip_ms': 20, 'word_errors': {'substitutions': 0, 'insertions': 0, 'deletions': 0, 'reference_words': 4}}
    cohort = nvidia_batch.summarize(tmp_path, plan, {'c': [retried, first, later]})['cohorts']['public']
    assert cohort['usable'] == 1 and cohort['first_attempt_valid'] == 0
    assert cohort['failures'] == {'server': 1}
    assert cohort['errors']['substitutions'] == 1 and cohort['round_trip_ms_p50'] == 10


def test_the_endpoint_comes_only_from_the_environment():
    config = nvidia_batch.load_config()
    with pytest.raises(ValueError, match=config['endpoint_env']):
        nvidia_batch.endpoint(config, environ={})
    assert nvidia_batch.endpoint(config, environ={config['endpoint_env']: ' https://example.test/v1 '}) == 'https://example.test/v1'


def test_the_shipped_config_names_no_endpoint():
    # The deployment needs no key, so its address is the credential. The only links allowed are public pages.
    config = nvidia_batch.load_config()
    text = nvidia_batch.Path(f'config/batch/{nvidia_batch.MODEL}.json').read_text()
    assert set(re.findall(r'https?://[^"\s]+', text)) == {config['model_card'], config['pricing']['source']}
