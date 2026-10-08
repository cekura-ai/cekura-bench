"""Zoom Scribe live offline fixtures and local WebSocket tests. No provider hosts."""
import asyncio
import json
import pytest
from websockets.asyncio.server import serve
from test_providers import fast_audio
from stt_bench import zoom as wire
from stt_bench import full_benchmark, provider_protocol
from stt_bench.catalog import model_config
from stt_bench.credentials import command_environment
from stt_bench.provider_protocol import ProviderError
from stt_bench.providers import validate, reduce_events, transcript_at
from stt_bench.streaming import EventLog, read_events
from stt_bench.turn_metrics import transcript_timeline

UPDATED = {'type': 'session.updated', 'model': 'zoom-asr-en-v1', 'language': 'en-US',
           'diarization_enabled': False, 'word_time_offsets': False}


def config():
    return json.loads(model_config('zoom-scribe-live').read_text())


def msg(value, at):
    return dict(kind='provider_message', time_seconds=at, message=value)


def started(item, ms):
    return {'type': 'input_audio_buffer.speech_started', 'item_id': item, 'audio_start_ms': ms}


def stopped(item, ms):
    return {'type': 'input_audio_buffer.speech_stopped', 'item_id': item, 'audio_end_ms': ms}


def completed(item, text):
    return {'type': 'transcription.completed', 'item_id': item, 'transcript': text, 'transcription_latency_ms': 120}


def stream():
    # Zoom's detector closes the first segment mid-clip; its completion lands
    # after the second segment has started. The commit at speech end closes the second.
    return [dict(kind='model_accepted', time_seconds=0, model='zoom-asr-en-v1'),
            msg({'type': 'session.created', 'session_id': 'ls_fixture'}, .01), msg(UPDATED, .02),
            msg(started('a', 800), .9), msg(stopped('a', 3000), 3.3), msg(started('b', 3000), 3.31),
            msg(completed('a', 'Hello there.'), 3.4),
            dict(kind='speech_end', time_seconds=5), dict(kind='finalize_requested', time_seconds=5),
            msg(stopped('b', 5000), 5.3), msg(completed('b', 'How are you?'), 5.36),
            dict(kind='audio_complete', time_seconds=6), dict(kind='close_stream_requested', time_seconds=6),
            msg({'type': 'session.closed', 'reason': 'client_requested'}, 6.2),
            dict(kind='provider_terminal', time_seconds=6.2)]


def test_config_credentials_and_setup():
    c = validate(config())
    assert command_environment('zoom', environ={'ZOOM_API_KEY': 'fixture', 'RESON_API_KEY': 'x'},
                               env_file=None) == {'ZOOM_API_KEY': 'fixture'}
    assert wire.connection(c, 'fixture') == (wire.ENDPOINT, {'Authorization': 'Bearer fixture'})
    assert wire.Protocol(c).setup() == {'type': 'session.update', 'language': 'en-US', 'audio': {'format': 'pcm16'}}
    assert wire.Protocol(c).finalize() == {'type': 'input_audio_buffer.commit'}
    assert wire.Protocol(c).finish(0) == {'type': 'session.close'}
    for key, value in [('language', 'en'), ('model', 'other'), ('sample_rate', 24000),
                       ('finalization', 'stream_end_after_tail'), ('finalize_ack_supported', True),
                       ('endpoint', 'wss://example.test/live')]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key: value}))


def test_segments_join_in_speech_order_and_close_completes():
    c, events = config(), stream()
    r = reduce_events(events, c)
    assert r['transcript'] == 'Hello there. How are you?'
    assert r['transcript_complete'] and r['model_verified']
    assert r['final_transcript_received_seconds'] == 5.36
    assert r['finalize_latency_status'] == 'unsupported'
    assert transcript_at(events, 5, c)['final_text'] == 'Hello there.'
    # A late completion of an earlier segment cannot reorder the transcript.
    swapped = stream()
    swapped[6], swapped[10] = msg(completed('b', 'How are you?'), 3.4), msg(completed('a', 'Hello there.'), 5.36)
    assert reduce_events(swapped, c)['transcript'] == 'Hello there. How are you?'


def test_completion_requires_session_closed_after_our_close():
    c = config()
    missing = [e for e in stream() if e.get('message', {}).get('type') != 'session.closed']
    assert not reduce_events(missing, c)['transcript_complete']
    # A server-side close before we asked for one (idle timeout) is not completion.
    early = missing[:9] + [msg({'type': 'session.closed', 'reason': 'idle_timeout'}, 5.2)] + missing[9:]
    assert not reduce_events(early, c)['transcript_complete']


@pytest.mark.parametrize('update', [dict(UPDATED, model='zoom-asr-multi-v1'), dict(UPDATED, language='es-ES')])
def test_server_reported_model_and_language_must_match(update):
    events = stream()
    events[2] = msg(update, .02)
    assert not reduce_events(events, config())['model_verified']


def test_errors_and_conflicting_finals_fail_closed():
    c = config()
    events = stream()
    events.insert(9, msg({'type': 'error', 'error': {'code': 'internal', 'message': 'x', 'fatal': False}}, 5.1))
    r = reduce_events(events, c)
    assert r['transport_failed'] and not r['transcript_complete']
    events = stream()
    events.insert(11, msg(completed('b', 'Something else.'), 5.4))
    assert not reduce_events(events, c)['transcript_complete']
    with pytest.raises(ProviderError):
        wire.Protocol(c).feed({'type': 'transcription.completed', 'transcript': 'no item'})


def test_unfinished_delta_is_provisional_and_blocks_completion():
    c = config()
    events = stream()
    events.insert(10, msg({'type': 'transcription.delta', 'item_id': 'c', 'delta': 'And'}, 5.35))
    snap = transcript_at(events, 5.35, c)
    assert snap['partial_text'] == 'And' and snap['final_text'] == 'Hello there.'
    assert not reduce_events(events, c)['transcript_complete']
    events.insert(11, msg(completed('c', 'And then.'), 5.37))
    assert reduce_events(events, c)['transcript'] == 'Hello there. How are you? And then.'


def test_full_benchmark_and_turn_reconstruction_use_the_zoom_protocol():
    c, events = config(), stream()
    texts = [s['text'] for s in full_benchmark.final_snapshots(events, c)]
    assert texts == ['', 'Hello there.', 'Hello there. How are you?']
    assert transcript_timeline(events, c)[-1]['text'] == 'Hello there. How are you?'


async def fake_zoom(ws, text, fault=None):
    await ws.send(json.dumps({'type': 'session.created', 'session_id': 'ls_fixture'}))
    frames, open_item = 0, False
    async for raw in ws:
        if isinstance(raw, bytes):
            frames += 1
            if text and frames == 2:
                open_item = True
                await ws.send(json.dumps(started('a', 0)))
            continue
        m = json.loads(raw)
        if m['type'] == 'session.update':
            await ws.send(json.dumps(dict(UPDATED, model='other') if fault == 'model' else UPDATED))
        elif m['type'] == 'input_audio_buffer.commit' and open_item:
            open_item = False
            await ws.send(json.dumps(stopped('a', frames * 20)))
            await ws.send(json.dumps(completed('a', text)))
        elif m['type'] == 'session.close':
            if fault == 'disconnect':
                await ws.close(); return
            await ws.send(json.dumps({'type': 'session.closed', 'reason': 'client_requested'}))


def run_local(tmp_path, monkeypatch, c, handler, *, subprotocols=('live-asr',)):
    monkeypatch.setattr(provider_protocol, 'stream_audio', fast_audio)
    async def scenario():
        async with serve(handler, '127.0.0.1', 0, subprotocols=list(subprotocols) or None) as server:
            url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
            log = EventLog(tmp_path / 'events.jsonl'); log.emit('clip_start')
            monkeypatch.setattr(wire, 'connection', lambda config, key: (url, {}))
            try:
                await wire.transcribe(bytes(640 * 60), 10, c, 'fixture-secret', log)
            except Exception:
                log.emit('error', error_type='FixtureFailure'); raise
            finally:
                log.emit('clip_end'); log.close()
    asyncio.run(scenario())
    assert 'fixture-secret' not in (tmp_path / 'events.jsonl').read_text()
    return reduce_events(read_events(tmp_path / 'events.jsonl'), c)


@pytest.mark.parametrize('text', ['hello world', ''])
def test_local_websocket_success(text, tmp_path, monkeypatch):
    c = config()
    r = run_local(tmp_path, monkeypatch, c, lambda ws: fake_zoom(ws, text))
    assert r['transcript_complete'] and r['model_verified'] and r['transcript'] == text


@pytest.mark.parametrize('fault', ['model', 'disconnect'])
def test_local_websocket_failure_never_becomes_success(fault, tmp_path, monkeypatch):
    c = dict(config(), finalize_timeout_seconds=.05, close_timeout_seconds=.05, ready_timeout_seconds=.05)
    with pytest.raises(Exception):
        run_local(tmp_path, monkeypatch, c, lambda ws: fake_zoom(ws, 'hello', fault))
    assert 'fixture-secret' not in (tmp_path / 'events.jsonl').read_text()
    assert not reduce_events(read_events(tmp_path / 'events.jsonl'), c)['transcript_complete']


def test_connection_without_the_live_asr_subprotocol_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ProviderError, match='subprotocol'):
        run_local(tmp_path, monkeypatch, config(), lambda ws: fake_zoom(ws, 'hello'),
                  subprotocols=())
