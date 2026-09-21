"""Offline protocol and local WebSocket tests; no Gradium API calls."""
import asyncio
import base64
import json
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from stt_bench import gradium as wire
from stt_bench.catalog import model_config
from stt_bench.credentials import command_environment
from stt_bench.providers import validate, reduce_events, transcript_at
from stt_bench.streaming import EventLog, read_events


def config():
    # These fixtures preserve the original interpretation for historical replay.
    return dict(json.loads(model_config('gradium-default').read_text()),
                transcript_reconstruction='gradium-segment-finality-v1')


def msg(value, at):
    return dict(kind='provider_message', time_seconds=at, message=value)


def test_config_and_credentials():
    c = validate(config())
    assert command_environment('gradium', environ={'GRADIUM_API_KEY': 'fixture', 'OPENAI_API_KEY': 'other'},
                               env_file=None) == {'GRADIUM_API_KEY': 'fixture'}
    assert wire.connection(c, 'fixture') == (c['endpoint'], {'x-api-key': 'fixture'})
    p = wire.Protocol(c)
    assert p.setup()['input_format'] == 'pcm'
    assert p.setup()['json_config'] == dict(language='en', delay_in_frames=10)
    assert base64.b64decode(p.audio(bytes(640))['audio']) == bytes(640)
    for key, value in [('sample_rate', 16000), ('channels', 2), ('frame_ms', 80),
                       ('input_format', 'pcm_16000'), ('delay_in_frames', True), ('delay_in_frames', 56),
                       ('endpoint', 'wss://other.example'), ('model', 'other')]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key: value}))


def test_replay_receipt_cutoffs_flush_and_terminal():
    c = config()
    events = [dict(kind='model_accepted', time_seconds=0, model='default'),
              msg(dict(type='text', text='yes', start_s=0, stream_id=0), .1),
              msg(dict(type='end_text', stop_s=.1, stream_id=0), .2),
              msg(dict(type='text', text='yes', start_s=.2, stream_id=0), .3),
              dict(kind='speech_end', time_seconds=.3), dict(kind='finalize_requested', time_seconds=.3),
              msg(dict(type='end_text', stop_s=.3, stream_id=0), .4),
              msg(dict(type='flushed', flush_id=99), .45),
              msg(dict(type='flushed', flush_id=1), .5),
              dict(kind='audio_complete', time_seconds=1.3),
              dict(kind='close_stream_requested', time_seconds=1.3),
              msg(dict(type='end_of_stream'), 1.4), dict(kind='provider_terminal', time_seconds=1.4)]
    assert transcript_at(events, .3, c)['text'] == 'yes yes'
    assert transcript_at(events, .3, c)['final_text'] == 'yes'
    result = reduce_events(events, c)
    assert result['transcript'] == 'yes yes' and result['transcript_complete']
    assert result['finalize_latency_ms'] == pytest.approx(200)
    assert not reduce_events(events[:-2] + [events[-1]], c)['transcript_complete']
    assert not reduce_events(events[:9], c)['transcript_complete']
    broken = events + [msg(dict(type='text', text='unfinished'), 1.5)]
    assert not reduce_events(broken, c)['transcript_complete']


def test_unexpected_stream_and_settings_fail_closed():
    c = config(); p = wire.Protocol(c)
    p.feed(dict(type='text', text='other', stream_id=1))
    assert p.unsupported
    with pytest.raises(wire.shared.ProviderError):
        wire.Protocol(c).feed(dict(type='ready', sample_rate=16000, delay_in_frames=10))
    with pytest.raises(wire.shared.ProviderError):
        wire.Protocol(c).feed(dict(type='error', message='sensitive server detail'))
    p = wire.Protocol(c)
    p.feed(dict(type='ready', model_name='other', sample_rate=24000, delay_in_frames=10))
    assert p.model_mismatch


def test_terminal_closes_last_segment_without_backdating_finality():
    # Shape captured from the 2026-09-14 live synthetic-phrase smoke.
    events = [dict(kind='model_accepted', time_seconds=0, model='default'),
              msg(dict(type='text', text='test.', stream_id=0), .5),
              dict(kind='audio_complete', time_seconds=1),
              dict(kind='close_stream_requested', time_seconds=1),
              msg(dict(type='end_of_stream'), 1.2), dict(kind='provider_terminal', time_seconds=1.2)]
    c = config()
    assert transcript_at(events, 1.1, c)['final_text'] == ''
    assert transcript_at(events, 1.1, c)['text'] == 'test.'
    assert reduce_events(events,c)['transcript'] == 'test.'
    assert reduce_events(events,c)['transcript_complete']
    assert reduce_events(events,c)['final_transcript_received_seconds'] == 1.2


@pytest.mark.parametrize('failure', [None, 'disconnect', 'error', 'missing_terminal', 'wrong_flush'])
def test_loopback_exchange(tmp_path, failure):
    async def check():
        c = config(); c['close_timeout_seconds'] = .05; c['finalize_timeout_seconds'] = 1.1
        received = []
        async def server(ws):
            async for raw in ws:
                value = json.loads(raw); received.append(value)
                if value['type'] == 'setup':
                    await ws.send(json.dumps(dict(type='ready', model_name='55966eda@500',
                                                 sample_rate=24000, delay_in_frames=10)))
                elif value['type'] == 'flush':
                    if failure == 'disconnect':
                        await ws.close(); return
                    if failure == 'error':
                        await ws.send(json.dumps(dict(type='error'))); return
                    for m in [dict(type='text', text='hello', stream_id=0), dict(type='end_text', stream_id=0),
                              dict(type='flushed', flush_id=99 if failure == 'wrong_flush' else 1)]:
                        await ws.send(json.dumps(m))
                elif value['type'] == 'end_of_stream' and failure != 'missing_terminal':
                    await ws.send(json.dumps(dict(type='end_of_stream')))
                    await ws.close(); return
        async with serve(server, '127.0.0.1', 0) as local:
            log = EventLog(tmp_path / 'events.jsonl')
            try:
                async with connect(f'ws://127.0.0.1:{local.sockets[0].getsockname()[1]}') as ws:
                    if failure:
                        with pytest.raises((wire.shared.ProviderError, TimeoutError)):
                            await wire.exchange(ws, bytes(960 * 55), 5, c, log)
                    else:
                        await wire.exchange(ws, bytes(960 * 55), 5, c, log)
            finally:
                log.close()
        if not failure:
            assert reduce_events(read_events(tmp_path / 'events.jsonl'), c)['transcript_complete']
            assert sum(v['type'] == 'audio' for v in received) == 55
            assert received[-1]['type'] == 'end_of_stream'
    asyncio.run(check())
