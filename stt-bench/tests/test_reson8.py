"""Reson8 offline fixtures and local WebSocket tests. No provider hosts."""
import asyncio
import json
from urllib.parse import parse_qs, urlsplit
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from stt_bench import reson8 as wire
from stt_bench.catalog import model_config
from stt_bench.credentials import command_environment
from stt_bench.providers import validate, reduce_events, transcript_at
from stt_bench.streaming import EventLog, read_events


def config():
    return json.loads(model_config('reson8-realtime').read_text())


def msg(value, at):
    return dict(kind='provider_message', time_seconds=at, message=value)


def test_config_audio_and_credential_isolation():
    c = validate(config())
    assert command_environment('reson8', environ={'RESON_API_KEY': 'fixture', 'GRADIUM_API_KEY': 'other'},
                               env_file=None) == {'RESON_API_KEY': 'fixture'}
    url, headers = wire.connection(c, 'fixture')
    assert headers == {'Authorization': 'ApiKey fixture'} and 'fixture' not in url
    q = parse_qs(urlsplit(url).query)
    assert q['sample_rate'] == ['16000'] and q['include_interim'] == ['true']
    assert q['encoding'] == ['pcm_s16le'] and q['filler_mode'] == ['verbatim']
    assert not any(k in q for k in ('model', 'custom_model_id', 'phrases', 'patterns'))
    p = wire.Protocol(c)
    assert p.ready and p.setup() is None and p.audio(bytes(640)) == bytes(640)
    for key, value in [('sample_rate', 24000), ('channels', 2), ('frame_ms', 80),
                       ('include_interim', False), ('filler_mode', 'clean'),
                       ('endpoint', 'wss://other.example'), ('model', 'other')]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key: value}))


def test_replay_partial_replacement_repeated_finals_and_correlated_completion():
    c = config()
    events = [dict(kind='model_accepted', time_seconds=0, model='realtime'),
              msg(dict(type='transcript', text='wrong', is_final=False), .1),
              msg(dict(type='transcript', text='yes', is_final=False), .2),
              dict(kind='speech_end', time_seconds=.2), dict(kind='finalize_requested', time_seconds=.2),
              msg(dict(type='transcript', text='yes', is_final=True), .3),
              msg(dict(type='flush_confirmation', id='unrelated'), .35),
              msg(dict(type='flush_confirmation', id=wire.SPEECH_FLUSH), .4),
              dict(kind='audio_complete', time_seconds=1.2),
              dict(kind='close_stream_requested', time_seconds=1.2),
              msg(dict(type='transcript', text='yes', is_final=True), 1.25),
              msg(dict(type='flush_confirmation', id=wire.COMPLETE_FLUSH), 1.3),
              dict(kind='provider_terminal', time_seconds=1.3)]
    assert transcript_at(events, .2, c)['text'] == 'yes'
    assert transcript_at(events, .2, c)['final_text'] == ''
    r = reduce_events(events, c)
    assert r['transcript'] == 'yes yes' and r['transcript_complete']
    assert r['finalize_latency_ms'] == pytest.approx(200)
    assert not reduce_events(events[:-2] + [events[-1]], c)['transcript_complete']
    assert not reduce_events(events[:8], c)['transcript_complete']
    assert not reduce_events(events + [msg(dict(type='transcript', text='pending', is_final=False), 1.4)], c)['transcript_complete']


def test_empty_final_flush_can_complete_but_never_promotes_partial():
    c = config()
    events = [dict(kind='audio_complete', time_seconds=1), dict(kind='close_stream_requested', time_seconds=1),
              msg(dict(type='flush_confirmation', id=wire.COMPLETE_FLUSH), 1.1),
              dict(kind='provider_terminal', time_seconds=1.1)]
    assert reduce_events(events,c)['transcript_complete']
    events.insert(0,msg(dict(type='transcript', text='pending', is_final=False), .5))
    assert not reduce_events(events,c)['transcript_complete']
    p = wire.Protocol(c)
    p.feed(dict(type='flush_confirmation', id=wire.COMPLETE_FLUSH))
    p.feed(dict(type='flush_confirmation', id=wire.SPEECH_FLUSH))
    assert not p.terminal and not p.ack


@pytest.mark.parametrize('payload', [dict(type='error', message='secret'), [],
                                    dict(type='transcript', text='unspecified'),
                                    dict(type='transcript', text='invalid', is_final='false')])
def test_malformed_messages_fail(payload):
    with pytest.raises(wire.shared.ProviderError):
        wire.Protocol(config()).feed(payload)


@pytest.mark.parametrize('failure', [None, 'disconnect', 'error', 'missing_terminal', 'wrong_final_flush', 'wrong_speech_flush'])
def test_loopback_exchange(tmp_path, failure):
    async def check():
        c = config(); c['close_timeout_seconds'] = .08; c['finalize_timeout_seconds'] = 1.1
        received = []
        async def server(ws):
            async for raw in ws:
                value = json.loads(raw) if isinstance(raw,str) else raw
                received.append(value)
                if value == wire.Protocol(c).finalize():
                    if failure == 'disconnect':
                        await ws.close(); return
                    if failure == 'error':
                        await ws.send(json.dumps(dict(type='error'))); return
                    for m in [dict(type='transcript', text='hello', is_final=False),
                              dict(type='transcript', text='hello', is_final=True),
                              dict(type='flush_confirmation', id='wrong' if failure=='wrong_speech_flush' else wire.SPEECH_FLUSH)]:
                        await ws.send(json.dumps(m))
                if value == wire.Protocol(c).finish(55) and failure != 'missing_terminal':
                    await ws.send(json.dumps(dict(type='flush_confirmation',
                        id='wrong' if failure=='wrong_final_flush' else wire.COMPLETE_FLUSH)))
        async with serve(server, '127.0.0.1', 0) as local:
            log = EventLog(tmp_path/'events.jsonl')
            try:
                async with connect(f'ws://127.0.0.1:{local.sockets[0].getsockname()[1]}') as ws:
                    if failure:
                        with pytest.raises((wire.shared.ProviderError,TimeoutError)):
                            await wire.exchange(ws,bytes(640*55),5,c,log)
                    else:
                        await wire.exchange(ws,bytes(640*55),5,c,log)
            finally:
                log.close()
        if not failure:
            assert reduce_events(read_events(tmp_path/'events.jsonl'),c)['transcript_complete']
            assert sum(isinstance(v,bytes) for v in received)==55
            assert received[5] == wire.Protocol(c).finalize()
            assert received[-1] == wire.Protocol(c).finish(55)
    asyncio.run(check())
