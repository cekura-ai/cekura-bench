"""Offline trial-provider fixtures. No tests connect to a provider host."""
import asyncio
import base64
import copy
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from stt_bench import trial_providers as wire
from stt_bench.catalog import model_config
from stt_bench.credentials import command_environment, redact
from stt_bench.providers import validate, transcript_at, reduce_events
from stt_bench.streaming import EventLog, read_events, transmitted_silence_frames

NAMES = ['soniox-stt-rt-v5', 'smallest-pulse', 'sarvam-saaras-v3-realtime', 'inworld-stt-1']


def config(name):
    return json.loads(model_config(name).read_text())


def message(payload, at):
    return dict(kind='provider_message', time_seconds=at, message=payload)


def final(p, text='hello'):
    return {'soniox': {'tokens': [{'text': text, 'is_final': True}]},
            'smallest': {'type': 'transcription', 'transcript': text, 'is_final': True},
            'sarvam': {'event': 'transcript.final', 'text': text},
            'inworld': {'result': {'transcription': {'transcript': text, 'isFinal': True}}}}[p]


def partial(p, text):
    m = final(p, text)
    if p == 'soniox':
        m['tokens'][0]['is_final'] = False
    elif p == 'smallest':
        m['is_final'] = False
    elif p == 'sarvam':
        m['event'] = 'transcript.partial'
    else:
        m['result']['transcription']['isFinal'] = False
    return m


def terminal(p):
    return {'soniox': {'tokens': [], 'finished': True},
            'smallest': {'type': 'transcription', 'transcript': '', 'is_final': True, 'is_last': True},
            'sarvam': {'event': 'session.end', 'audio_duration_s': 1.2},
            'inworld': {'result': {'usage': {'modelId': 'inworld/inworld-stt-1', 'transcribedAudioMs': 1200}}}}[p]


@pytest.mark.parametrize('name', NAMES)
def test_config_auth_audio_and_secret_isolation(name):
    c = config(name)
    validate(c)
    p = c['provider']
    from stt_bench.credentials import NAMES as credentials
    keyname = credentials[p][0]
    assert command_environment(p, environ={keyname: 'fixture', 'OPENAI_API_KEY': 'unrelated'}, env_file=None) == {keyname: 'fixture'}
    url, headers = wire.connection(c, 'fixture')
    q = parse_qs(urlsplit(url).query)
    protocol = wire.Protocol(c)
    setup = protocol.setup('fixture')
    audio = protocol.audio(b'\x01\x00' * 320)
    if p == 'soniox':
        assert not headers and setup['api_key'] == 'fixture'
        assert setup['audio_format'] == 'pcm_s16le' and setup['model'] == 'stt-rt-v5'
        assert isinstance(audio, bytes) and protocol.finish(60) == b''
    elif p == 'smallest':
        assert headers == {'Authorization': 'Bearer fixture'}
        assert q['word_timestamps'] == ['true'] and q['sample_rate'] == ['16000']
        assert isinstance(audio, bytes)
    elif p == 'sarvam':
        assert headers == {'api-subscription-key': 'fixture'}
        assert q['endpointing'] == ['manual'] and q['language_code'] == ['en-IN']
        assert len(base64.b64decode(audio['audio'])) == 640
        assert not protocol.ready and protocol.start() == {'event': 'speech_start'}
    else:
        assert headers == {'Authorization': 'Basic fixture'}
        assert setup['transcribeConfig']['voiceProfileConfig'] == {'enableVoiceProfile': True, 'topN': 3}
        assert 'inworldConfig' not in setup['transcribeConfig']
        assert len(base64.b64decode(audio['audioChunk']['content'])) == 640
    assert 'fixture' not in json.dumps(redact({'headers': headers, 'setup': setup}, 'fixture'))
    for key, value in [('frame_ms', 100), ('channels', 2), ('sample_rate', 24000), ('finalize_ack_supported', p != 'soniox')]:
        changed = dict(c, **{key: value})
        with pytest.raises(ValueError):
            validate(changed)


@pytest.mark.parametrize('name', NAMES)
def test_replay_partial_replacement_finals_and_terminal_requirement(name):
    c = config(name); p = c['provider']
    events = [dict(kind='model_accepted', time_seconds=0, model=c['model']),
              message(partial(p, 'wrong'), .1), message(partial(p, 'hello'), .2),
              dict(kind='speech_end', time_seconds=.2), dict(kind='finalize_requested', time_seconds=.2),
              message(final(p), .5), dict(kind='audio_complete', time_seconds=1.2),
              dict(kind='close_stream_requested', time_seconds=1.2), message(terminal(p), 1.3),
              dict(kind='provider_terminal', time_seconds=1.3)]
    assert transcript_at(events, .2, c)['text'] == 'hello'
    assert transcript_at(events, .2, c)['final_text'] == ''
    assert reduce_events(events, c)['transcript'] == 'hello'
    assert reduce_events(events, c)['transcript_complete']
    assert not reduce_events(events[:7], c)['transcript_complete']
    # A locally logged terminal event cannot manufacture provider completion.
    assert not reduce_events([e for e in events if e != events[-2]], c)['transcript_complete']
    if p != 'soniox':
        r = reduce_events(events, c)
        assert r['finalize_latency_ms'] is None and r['finalize_latency_status'] == 'unsupported'
    protocol = wire.Protocol(c)
    protocol.feed(final(p, 'yes ')); protocol.feed(final(p, 'yes'))
    assert protocol.snapshot()['final_text'] == 'yes yes'


def test_soniox_tokens_partial_revisions_and_control_markers():
    protocol = wire.Protocol(config(NAMES[0]))
    protocol.feed({'tokens': [{'text': 'Hello', 'is_final': True}, {'text': ' wrong', 'is_final': False}]})
    protocol.feed({'tokens': [{'text': ' world', 'is_final': False}]})
    assert protocol.snapshot()['text'] == 'Hello world'
    protocol.requested = True
    protocol.feed({'tokens': [{'text': ' world', 'is_final': True}, {'text': '<end>', 'is_final': True},
                              {'text': '<fin>', 'is_final': True}]})
    assert protocol.ack and not protocol.terminal
    assert protocol.snapshot()['text'] == 'Hello world'


def test_inworld_profile_shapes_missing_fields_and_model_mismatch():
    c = config(NAMES[-1]); protocol = wire.Protocol(c)
    profile = {'pitch': [{'label': 'low', 'confidence': .8}], 'vocal_style': [{'label': 'normal', 'confidence': .9}]}
    protocol.feed({'result': {'transcription': {'transcript': 'hello', 'is_final': True, 'voice_profile': profile}}})
    protocol.feed({'result': {'voiceProfile': {'age': []}}})
    assert protocol.snapshot()['text'] == 'hello'
    saved = protocol.snapshot()['provider_metadata']['voice_profiles']
    assert saved[0]['voice_profile'] == profile and saved[1]['voice_profile'] == {'age': []}
    protocol.closing = True
    protocol.feed({'result': {'usage': {'model_id': 'different', 'transcribed_audio_ms': 3}}})
    assert protocol.model_mismatch


@pytest.mark.parametrize('name', NAMES)
def test_empty_completion_and_unresolved_partial(name):
    c = config(name); p = c['provider']
    events = [dict(kind='model_accepted', time_seconds=0, model=c['model']),
              dict(kind='audio_complete', time_seconds=1), dict(kind='close_stream_requested', time_seconds=1),
              message(terminal(p), 1.1), dict(kind='provider_terminal', time_seconds=1.1)]
    r = reduce_events(events, c)
    assert r['transcript_complete'] and r['transcript'] == ''
    # Partial text after the terminal response is never final text.
    events.append(message(partial(p, 'unresolved'), 1.2))
    assert not reduce_events(events, c)['transcript_complete']


@pytest.mark.parametrize('payload', [{'error_code': 401}, {'event': 'error', 'is_fatal': True},
                                    {'error': {'code': 3}}, {'status': 'error'}, ['invalid']])
def test_errors_fail_without_leaking_details(payload):
    with pytest.raises(wire.shared.ProviderError):
        wire.Protocol(config(NAMES[0])).feed(payload)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('failure', [None, 'disconnect', 'missing_terminal', 'error'])
def test_loopback_exchange_completion_and_failures(tmp_path, name, failure):
    async def check():
        c = config(name); p = c['provider']
        c['close_timeout_seconds'] = .08
        received = []
        async def server(ws):
            if p == 'sarvam':
                await ws.send(json.dumps({'event': 'session.begin'}))
            async for raw in ws:
                value = json.loads(raw) if isinstance(raw, str) else raw
                received.append(value)
                if value == wire.Protocol(c).finalize():
                    if failure == 'disconnect':
                        await ws.close(); return
                    if failure == 'error':
                        await ws.send(json.dumps({'error': {'message': 'fixture'}})); return
                    await ws.send(json.dumps(final(p)))
                    if p == 'soniox':
                        await ws.send(json.dumps({'tokens': [{'text': '<fin>', 'is_final': True}]}))
                if value == wire.Protocol(c).finish(60):
                    if failure == 'missing_terminal':
                        continue
                    await ws.send(json.dumps(terminal(p)))
                    await ws.close(); return
        async with serve(server, '127.0.0.1', 0) as local:
            port = local.sockets[0].getsockname()[1]
            log = EventLog(tmp_path / 'events.jsonl')
            try:
                async with connect(f'ws://127.0.0.1:{port}') as ws:
                    if failure:
                        with pytest.raises((wire.shared.ProviderError, TimeoutError)):
                            await wire.exchange(ws, bytes(640 * 60), 10, c, log, 'fixture-secret')
                    else:
                        await wire.exchange(ws, bytes(640 * 60), 10, c, log, 'fixture-secret')
            finally:
                log.close()
        events = read_events(tmp_path / 'events.jsonl')
        assert 'fixture-secret' not in (tmp_path / 'events.jsonl').read_text()
        if not failure:
            assert reduce_events(events, c)['transcript_complete']
            assert len([e for e in events if e['kind'] == 'audio_sent']) == 10 + transmitted_silence_frames(c)
            assert received[-1] == wire.Protocol(c).finish(60)
            assert not any(e['kind'] == 'model_accepted' for e in events[:1])
    asyncio.run(check())
