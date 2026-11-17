"""AirCaps Live 1 offline fixtures and local WebSocket tests. No provider hosts."""
import asyncio
import io
import json
import urllib.error
import urllib.request
import pytest
from websockets.asyncio.server import serve
from test_providers import fast_audio
from stt_bench import aircaps as wire
from stt_bench import full_benchmark, provider_protocol
from stt_bench.catalog import model_config
from stt_bench.credentials import command_environment
from stt_bench.provider_protocol import ProviderError
from stt_bench.providers import validate, reduce_events, transcript_at
from stt_bench.streaming import EventLog, read_events
from stt_bench.turn_metrics import transcript_timeline

READY = {'type': 'ready', 'providers': ['aircaps-live-1'], 'failed_providers': [], 'sample_rate': 16000,
         'encoding': 'pcm_s16le', 'language': 'en-US', 'session_limit_ms': 3600000, 'model': 'aircaps-live-1'}
STATUS = {'type': 'provider_status', 'message': 'Ready', 'provider': 'aircaps-live-1'}
FINISHED = {'type': 'finished', 'audio_ms': 3000, 'billed_ms': 3000}


def config():
    return json.loads(model_config('aircaps-live-1').read_text())


def msg(value, at):
    return dict(kind='provider_message', time_seconds=at, message=value)


def text(value, settled, final=False):
    return {'type': 'transcript', 'text': value, 'final_chars': settled, 'is_final': final,
            'provider': 'aircaps-live-1'}


def stream():
    return [dict(kind='model_accepted', time_seconds=0, model='aircaps-live-1'),
            msg(STATUS, .01), msg(READY, .02),
            msg(text('Hello there', 0), 1.0), msg(text('Hello there, how are', 12), 2.0),
            dict(kind='speech_end', time_seconds=3), dict(kind='finalize_requested', time_seconds=3),
            msg(text('Hello there, how are you?', 25, True), 3.2),
            dict(kind='audio_complete', time_seconds=4), dict(kind='close_stream_requested', time_seconds=4),
            msg(text('Hello there, how are you?', 25, True), 4.2), msg(FINISHED, 4.3),
            dict(kind='provider_terminal', time_seconds=4.3)]


def test_config_credentials_and_messages():
    c = validate(config())
    assert command_environment('aircaps', environ={'AIRCAPS_API_KEY': 'fixture', 'ZOOM_API_KEY': 'x'},
                               env_file=None) == {'AIRCAPS_API_KEY': 'fixture'}
    p = wire.Protocol(c)
    assert p.setup() == {'type': 'start', 'model': 'aircaps-live-1', 'encoding': 'pcm_s16le',
                         'sample_rate': 16000, 'channels': 1}
    assert p.finalize() == {'type': 'finalize'} and p.finish(0) == {'type': 'finish'}
    for key, value in [('language', 'en'), ('model', 'other'), ('sample_rate', 24000), ('finalize_ack_supported', True),
                       ('completion_basis', 'session_closed_after_close'), ('endpoint', 'https://example.test/endpoint'),
                       ('socket_hosts', ['*.example.test'])]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key: value}))


def test_settled_prefix_is_final_and_finished_completes():
    c, events = config(), stream()
    r = reduce_events(events, c)
    assert r['transcript'] == 'Hello there, how are you?'
    assert r['transcript_complete'] and r['model_verified']
    assert r['final_transcript_received_seconds'] == 3.2
    assert r['finalize_latency_status'] == 'unsupported'
    snap = transcript_at(events, 2.0, c)
    assert (snap['final_text'], snap['partial_text']) == ('Hello there,', 'how are')
    assert transcript_at(events, 1.0, c)['final_text'] == ''


def test_completion_requires_finished_after_our_finish():
    c = config()
    missing = [e for e in stream() if e.get('message', {}).get('type') != 'finished']
    assert not reduce_events(missing, c)['transcript_complete']
    early = missing[:7] + [msg(FINISHED, 3.1)] + missing[7:]
    assert not reduce_events(early, c)['transcript_complete']


@pytest.mark.parametrize('ready', [dict(READY, model='aircaps-live-0'), dict(READY, language='es-ES')])
def test_server_reported_model_and_language_must_match(ready):
    events = stream()
    events[2] = msg(ready, .02)
    assert not reduce_events(events, config())['model_verified']


@pytest.mark.parametrize('revision', [text('Hello their, how are', 12), text('Hello there, how are', 5)])
def test_settled_text_that_changes_or_shrinks_fails_closed(revision):
    events = stream()
    events.insert(5, msg(revision, 2.5))
    assert not reduce_events(events, config())['transcript_complete']


def test_stale_final_holding_older_settled_text_is_skipped():
    events = stream()
    events.insert(5, msg(text('Hello', 5, True), 2.5))
    events.insert(6, msg(text('Hello there, how are', 12), 2.6))
    r = reduce_events(events, config())
    assert r['transcript_complete'] and r['transcript'] == 'Hello there, how are you?'
    assert transcript_at(events, 2.5, config())['final_text'] == 'Hello there,'


def test_errors_and_malformed_transcripts_fail_closed():
    c = config()
    events = stream()
    events.insert(5, msg({'type': 'error', 'code': 'quota_unavailable', 'message': 'x'}, 2.5))
    r = reduce_events(events, c)
    assert r['transport_failed'] and not r['transcript_complete']
    for bad in [{'type': 'transcript', 'text': 'hi'}, text('hi', 3), {'type': 'transcript', 'text': None, 'final_chars': 0}]:
        with pytest.raises(ProviderError):
            wire.Protocol(c).feed(bad)


def test_unsettled_text_is_provisional_and_blocks_completion():
    c = config()
    events = stream()
    events.insert(10, msg(text('Hello there, how are you? And', 25), 4.25))
    assert transcript_at(events, 4.25, c)['partial_text'] == 'And'
    assert not reduce_events(events, c)['transcript_complete']


def test_full_benchmark_and_turn_reconstruction_use_the_aircaps_protocol():
    c, events = config(), stream()
    texts = [s['text'] for s in full_benchmark.final_snapshots(events, c)]
    assert texts == ['', 'Hello there,', 'Hello there, how are you?']
    assert transcript_timeline(events, c)[-1]['text'] == 'Hello there, how are you?'


def lookup_with(monkeypatch, body, status=None):
    seen = {}
    def urlopen(request, timeout):
        seen.update(url=request.full_url, auth=request.get_header('Authorization'))
        if status:
            raise urllib.error.HTTPError(request.full_url, status, 'refused', {}, io.BytesIO(body.encode()))
        return io.BytesIO(json.dumps(body).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    return seen


def test_lookup_returns_only_an_aircaps_socket(monkeypatch):
    c = config()
    seen = lookup_with(monkeypatch, {'websocket_url': 'wss://rt1.aircaps.ai/v1/realtime', 'model': 'aircaps-live-1'})
    assert wire.socket_url(c, 'fixture') == 'wss://rt1.aircaps.ai/v1/realtime'
    assert seen == {'url': wire.ENDPOINT + '?model=aircaps-live-1', 'auth': 'Bearer fixture'}
    for body in [{'websocket_url': 'wss://rt1.example.test/v1/realtime', 'model': 'aircaps-live-1'},
                 {'websocket_url': 'ws://rt1.aircaps.ai/v1/realtime', 'model': 'aircaps-live-1'},
                 {'websocket_url': 'wss://aircaps.ai.example.test/v1', 'model': 'aircaps-live-1'},
                 {'websocket_url': 'wss://rt1.aircaps.ai/v1/realtime', 'model': 'other'}, {}]:
        lookup_with(monkeypatch, body)
        with pytest.raises(ProviderError):
            wire.socket_url(c, 'fixture')


@pytest.mark.parametrize('status,body,failure', [(402, '{"code":"payment_required"}', 'credits'),
                                                 (429, '{"code":"quota_unavailable"}', 'concurrency')])
def test_lookup_http_errors_are_classified(status, body, failure, monkeypatch):
    lookup_with(monkeypatch, body, status)
    with pytest.raises(ProviderError) as exc:
        wire.socket_url(config(), 'fixture')
    assert full_benchmark.classify([{'kind': 'error', 'error_message': str(exc.value)}]) == failure


async def fake_aircaps(ws, said, fault=None):
    start = json.loads(await ws.recv())
    assert start == wire.Protocol(config()).setup()
    await ws.send(json.dumps(STATUS))
    await ws.send(json.dumps(dict(READY, model='other' if fault == 'model' else start['model'])))
    frames = 0
    async for raw in ws:
        if isinstance(raw, bytes):
            frames += 1
            if said and frames == 2:
                await ws.send(json.dumps(text(said.split()[0], 0)))
            continue
        kind = json.loads(raw)['type']
        if kind == 'finalize' and said:
            await ws.send(json.dumps(text(said, len(said), True)))
        elif kind == 'finish':
            if fault == 'disconnect':
                await ws.close(); return
            if said:
                await ws.send(json.dumps(text(said, len(said), True)))
            await ws.send(json.dumps(FINISHED))


def run_local(tmp_path, monkeypatch, handler):
    c = config()
    monkeypatch.setattr(provider_protocol, 'stream_audio', fast_audio)
    async def scenario():
        async with serve(handler, '127.0.0.1', 0) as server:
            url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
            monkeypatch.setattr(wire, 'socket_url', lambda config, key: url)
            log = EventLog(tmp_path / 'events.jsonl'); log.emit('clip_start')
            try:
                await wire.transcribe(bytes(640 * 60), 10, c, 'fixture-secret', log)
            except Exception:
                log.emit('error', error_type='FixtureFailure'); raise
            finally:
                log.emit('clip_end'); log.close()
    asyncio.run(scenario())
    assert 'fixture-secret' not in (tmp_path / 'events.jsonl').read_text()
    return reduce_events(read_events(tmp_path / 'events.jsonl'), c)


@pytest.mark.parametrize('said', ['hello world', ''])
def test_local_websocket_success(said, tmp_path, monkeypatch):
    r = run_local(tmp_path, monkeypatch, lambda ws: fake_aircaps(ws, said))
    assert r['transcript_complete'] and r['model_verified'] and r['transcript'] == said


@pytest.mark.parametrize('fault', ['model', 'disconnect'])
def test_local_websocket_faults_never_become_success(fault, tmp_path, monkeypatch):
    with pytest.raises(ProviderError):
        run_local(tmp_path, monkeypatch, lambda ws: fake_aircaps(ws, 'hello', fault))
    r = reduce_events(read_events(tmp_path / 'events.jsonl'), config())
    assert not r['transcript_complete']
