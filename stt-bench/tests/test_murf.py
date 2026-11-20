"""Murf Finch fixtures and local WebSocket tests. No provider hosts."""
import asyncio
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import serve

from stt_bench import full_benchmark, murf
from stt_bench.credentials import credential
from stt_bench.providers import reduce_events, transcript_at, validate
from stt_bench.streaming import EventLog, read_events
from stt_bench.turn_batch import OPT_IN
from stt_bench.turn_metrics import transcript_timeline
from stt_bench.turn_runner import validate_profile


def config():
    return json.loads(Path('config/models/murf-finch-global.json').read_text())


def message(kind, text, turn=0, sequence=0, trigger='manual'):
    return {'type': kind, 'transcript': text, 'rawEvent': {
        'type': 'TurnInfo', 'request_id': 'request-fixture',
        'event': 'EndOfTurn' if kind == 'final' else 'Update',
        'turn_index': turn, 'sequence_id': sequence, 'transcript': text,
        **({'trigger': trigger} if kind == 'final' else {}),
    }}


def event(value, at):
    return {'kind': 'provider_message', 'time_seconds': at, 'message': value}


def session_started():
    return {'type': 'SessionStarted', 'session_id': 'session-fixture',
            'request_id': 'request-fixture', 'language': 'en'}


def test_config_connection_and_credentials():
    c = validate(config())
    assert credential('murf', {'MURF_API_KEY': 'fixture'}, None) == ('fixture', 'MURF_API_KEY')
    url, headers = murf.connection(c, 'fixture')
    assert headers == {'api-key': 'fixture'} and 'fixture' not in url
    assert parse_qs(urlsplit(url).query) == {'model': ['finch_global_1.0'], 'language': ['en']}
    for key, value in [('language', 'hi'), ('sample_rate', 8000), ('channels', 2),
                       ('finalization', 'stream_end_after_tail')]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key: value}))


def test_full_benchmark_and_turn_reconstruction_use_the_murf_protocol():
    c = config()
    events = [event(session_started(), .05),
              event(message('partial', 'hello', sequence=1), .1),
              dict(kind='finalize_requested', time_seconds=.2),
              event(message('final', 'hello world', sequence=2), .4),
              event(message('final', '', turn=1, sequence=3, trigger='vad'), 1.0)]
    assert [s['text'] for s in full_benchmark.final_snapshots(events, c)] == ['', 'hello world']
    assert transcript_timeline(events, c)[-1]['text'] == 'hello world'


def test_turn_profile_is_the_model_config_with_controlled_turn_fields():
    profile = json.loads(Path('config/profiles/private-turns-v1/murf-finch-global.json').read_text())
    assert {k: v for k, v in profile.items() if k in config()} == config()
    assert profile['turn_finalization_class'] == 'controlled'
    assert profile['base_config'] == 'config/models/murf-finch-global.json'
    validate_profile(profile)
    assert 'murf-finch-global' in OPT_IN and 'murf-finch-global' in full_benchmark.FULL_MODELS


def test_replay_replaces_partials_and_requires_finalize_ack_then_clean_close():
    c = config()
    events = [dict(kind='model_accepted', time_seconds=0, model=c['model']),
              event(session_started(), .05),
              event(message('partial', 'hello', sequence=1), .1),
              event(message('partial', 'hello world', sequence=2), .2),
              dict(kind='speech_end', time_seconds=.2),
              dict(kind='finalize_requested', time_seconds=.2),
              event(message('final', 'hello world', sequence=3), .4),
              dict(kind='audio_complete', time_seconds=1.2),
              dict(kind='close_stream_requested', time_seconds=1.2),
              event(message('partial', '', turn=1, sequence=4), 1.25),
              event(message('final', '', turn=1, sequence=5, trigger='vad'), 1.26),
              dict(kind='provider_terminal', time_seconds=1.3)]
    assert transcript_at(events, .2, c) == {
        'text': 'hello world', 'final_text': '', 'partial_text': 'hello world',
        'provisional': True, 'reconstruction_status': 'supported'}
    result = reduce_events(events, c)
    assert result['transcript'] == 'hello world' and result['transcript_complete']
    assert result['finalize_latency_ms'] == pytest.approx(200)
    assert not reduce_events(events[:-1], c)['transcript_complete']
    premature = [events[0], event(message('final', 'hello', sequence=1), .1),
                 dict(kind='finalize_requested', time_seconds=.2), dict(kind='audio_complete', time_seconds=1),
                 dict(kind='close_stream_requested', time_seconds=1),
                 dict(kind='provider_terminal', time_seconds=1.1)]
    assert not reduce_events(premature, c)['transcript_complete']


@pytest.mark.parametrize('payload', [
    {'type': 'Error', 'message': 'rejected'},
    {'type': 'final', 'transcript': 'bad', 'rawEvent': {'event': 'EndOfTurn'}},
    {'type': 'partial', 'transcript': 'bad', 'rawEvent': {'type': 'TurnInfo', 'event': 'EndOfTurn',
     'turn_index': 0, 'sequence_id': 0, 'transcript': 'bad'}},
])
def test_error_and_malformed_messages_fail(payload):
    with pytest.raises(murf.shared.ProviderError):
        murf.Protocol(config()).feed(payload)


def test_transcribe_loopback_uses_header_audio_config_finalize_and_clean_close(tmp_path, monkeypatch):
    async def check():
        c = config()
        received, headers = [], []

        async def server(ws):
            headers.append(ws.request.headers['api-key'])
            await ws.send(json.dumps(session_started()))
            async for raw in ws:
                received.append(json.loads(raw) if isinstance(raw, str) else raw)
                if isinstance(raw, str) and json.loads(raw).get('type') == 'Finalize':
                    await ws.send(json.dumps(message('final', 'hello world', sequence=1)))
                elif isinstance(raw, str) and json.loads(raw).get('type') == 'CloseStream':
                    await ws.send(json.dumps(message('final', '', turn=1, sequence=2, trigger='vad')))
                    await ws.close(code=1000)
                    return

        async def fast_audio(pcm, speech_frames, send, finalize, log, **kwargs):
            await send(b'speech')
            t0 = log.now()
            log.emit('speech_end', at=t0)
            await finalize(t0)
            await send(b'tail')
            log.emit('audio_complete', t0_seconds=t0)

        async with serve(server, '127.0.0.1', 0) as local:
            monkeypatch.setattr(murf, 'connection', lambda _config, key: (
                f'ws://127.0.0.1:{local.sockets[0].getsockname()[1]}?model=finch_global_1.0&language=en',
                {'api-key': key}))
            monkeypatch.setattr(murf, 'stream_audio', fast_audio)
            log = EventLog(tmp_path / 'events.jsonl')
            try:
                await murf.transcribe(b'', 1, c, 'fixture-secret', log)
            finally:
                log.close()
        assert headers == ['fixture-secret']
        assert received == [
            {'type': 'AudioConfig', 'codec': 'pcm_s16le', 'sample_rate': 16000, 'channels': 1},
            b'speech', {'type': 'Finalize'}, b'tail', {'type': 'CloseStream'},
        ]
        events = read_events(tmp_path / 'events.jsonl')
        assert reduce_events(events, c)['transcript_complete']
        assert 'fixture-secret' not in (tmp_path / 'events.jsonl').read_text()

    asyncio.run(check())


def test_finalize_reply_labelled_vad_after_a_silence_ended_turn_completes():
    # Live smoke: Murf ended the speech turn on silence just before Finalize,
    # then answered Finalize with an empty final labelled vad.
    c = config()
    events = [dict(kind='model_accepted', time_seconds=0, model=c['model']),
              event(session_started(), .05),
              event(message('partial', 'concert tickets', sequence=1), .5),
              event(message('final', 'concert tickets', sequence=2, trigger='vad'), 1.0),
              dict(kind='speech_end', time_seconds=1.2),
              dict(kind='finalize_requested', time_seconds=1.2),
              event(message('partial', '', turn=1, sequence=3), 1.4),
              event(message('final', '', turn=1, sequence=4, trigger='vad'), 1.4),
              dict(kind='audio_complete', time_seconds=2.2),
              dict(kind='close_stream_requested', time_seconds=2.2),
              dict(kind='provider_terminal', time_seconds=2.3)]
    result = reduce_events(events, c)
    assert result['transcript'] == 'concert tickets' and result['transcript_complete']
    assert result['finalize_latency_ms'] == pytest.approx(200)


def test_finalize_reply_with_speech_labelled_vad_completes():
    c = config()
    events = [dict(kind='model_accepted', time_seconds=0, model=c['model']),
              event(message('partial', 'hello', sequence=1), .1),
              dict(kind='speech_end', time_seconds=.2),
              dict(kind='finalize_requested', time_seconds=.2),
              event(message('final', 'hello there', sequence=2, trigger='vad'), .45),
              dict(kind='audio_complete', time_seconds=1.2),
              dict(kind='close_stream_requested', time_seconds=1.2),
              dict(kind='provider_terminal', time_seconds=1.3)]
    result = reduce_events(events, c)
    assert result['transcript'] == 'hello there' and result['transcript_complete']


def test_pending_speech_in_a_later_turn_blocks_the_acknowledgment():
    protocol = murf.Protocol(config())
    protocol.feed(message('partial', 'first part', sequence=1))
    protocol.feed(message('partial', 'second', turn=1, sequence=2))
    protocol.requested = True
    protocol.feed(message('final', 'first part', sequence=3, trigger='vad'))
    assert not protocol.ack
    protocol.feed(message('final', 'second part', turn=1, sequence=4))
    assert protocol.ack and protocol.snapshot()['final_text'] == 'first part second part'
