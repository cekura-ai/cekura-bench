"""Amazon Transcribe streaming offline fixtures and local WebSocket tests. No provider hosts."""
import asyncio
import datetime
import json
from urllib.parse import parse_qs, urlsplit
import pytest
from websockets.asyncio.server import serve
from test_providers import fast_audio
from stt_bench import amazon as wire
from stt_bench import full_benchmark, provider_protocol
from stt_bench.catalog import model_config
from stt_bench.credentials import command_environment, credential
from stt_bench.provider_protocol import ProviderError
from stt_bench.providers import validate, reduce_events, transcript_at
from stt_bench.streaming import EventLog, read_events
from stt_bench.turn_metrics import transcript_timeline

KEY = json.dumps({'access_key_id': 'AKIDFIXTURE', 'secret_access_key': 'fixture-secret'})


def config():
    return json.loads(model_config('amazon-transcribe-turbo').read_text())


def msg(value, at):
    return dict(kind='provider_message', time_seconds=at, message=value)


def result(rid, text, partial):
    return {'type': 'TranscriptEvent', 'payload': {'Transcript': {'Results': [
        {'ResultId': rid, 'IsPartial': partial, 'StartTime': 0.1, 'EndTime': 1.0,
         'Alternatives': [{'Transcript': text}]}]}}}


def stream():
    return [dict(kind='model_accepted', time_seconds=0, model='transcribe-turbo'),
            msg(result('a', 'Hello', True), .8), msg(result('a', 'Hello there.', False), 1.5),
            msg(result('b', 'How', True), 4.6),
            dict(kind='speech_end', time_seconds=5), msg(result('b', 'How are you?', False), 5.4),
            dict(kind='audio_complete', time_seconds=6), dict(kind='close_stream_requested', time_seconds=6),
            msg({'type': 'stream_closed'}, 6.3), dict(kind='provider_terminal', time_seconds=6.3)]


def test_config_credentials_and_commands():
    c = validate(config())
    env = {'AWS_ACCESS_KEY_ID': 'AKIDFIXTURE', 'AWS_SECRET_ACCESS_KEY': 'fixture-secret', 'RESON_API_KEY': 'x'}
    key, name = credential('amazon', environ=env, env_file=None)
    assert json.loads(key) == {'access_key_id': 'AKIDFIXTURE', 'secret_access_key': 'fixture-secret'}
    assert name == 'AWS_ACCESS_KEY_ID'
    assert command_environment('amazon', environ=dict(env, AWS_SESSION_TOKEN='tok'), env_file=None) == {
        'AWS_ACCESS_KEY_ID': 'AKIDFIXTURE', 'AWS_SECRET_ACCESS_KEY': 'fixture-secret', 'AWS_SESSION_TOKEN': 'tok'}
    # Half a key pair is no credential.
    assert credential('amazon', environ={'AWS_ACCESS_KEY_ID': 'AKIDFIXTURE'}, env_file=None) == ('', None)
    assert wire.Protocol(c).ready and wire.Protocol(c).finalize() is None
    assert wire.Protocol(c).finish(0) == {'type': 'end_of_stream'}
    for key_name, value in [('language', 'en'), ('region', 'us-east-1'), ('sample_rate', 8000),
                            ('finalization', 'manual_at_speech_end'), ('endpoint', 'wss://example.test/x')]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key_name: value}))


def test_presigned_url_matches_aws_sigv4_reference():
    url = wire.presigned_url(config(), KEY, now=datetime.datetime(2026, 11, 28, 8, 0, tzinfo=datetime.timezone.utc))
    parts = urlsplit(url)
    assert f'{parts.scheme}://{parts.netloc}{parts.path}' == wire.ENDPOINT
    q = {k: v[0] for k, v in parse_qs(parts.query).items()}
    assert q['X-Amz-Credential'] == 'AKIDFIXTURE/20261128/us-east-2/transcribe/aws4_request'
    assert (q['language-code'], q['media-encoding'], q['sample-rate']) == ('en-US', 'pcm', '16000')
    assert 'X-Amz-Security-Token' not in q


@pytest.mark.parametrize('token', [None, 'TOKEN/with+chars='])
def test_presigned_signature_matches_botocore(token):
    pytest.importorskip('botocore')
    from botocore.auth import SigV4QueryAuth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials
    request = AWSRequest(method='GET', url=wire.ENDPOINT.replace('wss://', 'https://'),
                         params={'language-code': 'en-US', 'media-encoding': 'pcm', 'sample-rate': '16000'})
    SigV4QueryAuth(Credentials('AKIDFIXTURE', 'fixture-secret', token), 'transcribe', 'us-east-2', expires=300).add_auth(request)
    reference = {k: v[0] for k, v in parse_qs(urlsplit(request.url).query).items()}
    now = datetime.datetime.strptime(reference['X-Amz-Date'], '%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc)
    key = dict(json.loads(KEY), **({'session_token': token} if token else {}))
    ours = {k: v[0] for k, v in parse_qs(urlsplit(wire.presigned_url(config(), json.dumps(key), now=now)).query).items()}
    assert ours['X-Amz-Signature'] == reference['X-Amz-Signature']


def test_event_stream_round_trip_and_corruption():
    frame = wire.audio_event(b'\x01\x02' * 320)
    assert wire.decode_events(frame)[0]['type'] == 'AudioEvent'
    payload = json.dumps(result('r1', 'hi', False)['payload']).encode()
    event = wire.encode_event({':event-type': 'TranscriptEvent', ':message-type': 'event'}, payload)
    both = wire.decode_events(event + event)
    assert [e['type'] for e in both] == ['TranscriptEvent'] * 2 and both[0]['payload'] == json.loads(payload)
    exception = wire.encode_event({':exception-type': 'BadRequestException', ':message-type': 'exception'},
                                  b'{"Message":"bad"}')
    assert wire.decode_events(exception) == [{'type': 'error', 'exception_type': 'BadRequestException',
                                              'payload': {'Message': 'bad'}}]
    with pytest.raises(ProviderError):
        wire.decode_events(event[:-1] + bytes([event[-1] ^ 1]))
    assert len(wire.audio_event(b'')) == len(wire.encode_event(
        {':content-type': 'application/octet-stream', ':event-type': 'AudioEvent', ':message-type': 'event'}, b''))


def test_results_join_in_order_and_close_completes():
    c, events = config(), stream()
    r = reduce_events(events, c)
    assert r['transcript'] == 'Hello there. How are you?'
    assert r['transcript_complete'] and r['model_verified']
    assert r['final_transcript_received_seconds'] == 5.4
    assert r['finalize_latency_status'] == 'unsupported'
    assert transcript_at(events, 5, c)['final_text'] == 'Hello there.'
    assert transcript_at(events, 5, c)['partial_text'] == 'How'


def test_completion_requires_close_after_end_of_stream_and_no_open_partial():
    c = config()
    missing = [e for e in stream() if e.get('message', {}).get('type') != 'stream_closed']
    assert not reduce_events(missing, c)['transcript_complete']
    early = missing[:4] + [msg({'type': 'stream_closed'}, 4.9)] + missing[4:]
    assert not reduce_events(early, c)['transcript_complete']
    open_partial = stream()
    open_partial.insert(6, msg(result('c', 'And', True), 5.6))
    assert not reduce_events(open_partial, c)['transcript_complete']


def test_errors_and_conflicting_finals_fail_closed():
    c = config()
    events = stream()
    events.insert(5, msg({'type': 'error', 'exception_type': 'LimitExceededException', 'payload': {}}, 5.1))
    r = reduce_events(events, c)
    assert r['transport_failed'] and not r['transcript_complete']
    events = stream()
    events.insert(6, msg(result('b', 'Something else.', False), 5.5))
    assert not reduce_events(events, c)['transcript_complete']
    with pytest.raises(ProviderError):
        wire.Protocol(c).feed({'type': 'TranscriptEvent', 'payload': {}})


def test_full_benchmark_and_turn_reconstruction_use_the_amazon_protocol():
    c, events = config(), stream()
    texts = [s['text'] for s in full_benchmark.final_snapshots(events, c)]
    assert texts == ['', 'Hello there.', 'Hello there. How are you?']
    assert transcript_timeline(events, c)[-1]['text'] == 'Hello there. How are you?'


async def fake_transcribe(ws, text, fault=None):
    frames = 0
    async for raw in ws:
        assert isinstance(raw, bytes)
        event = wire.decode_events(raw)[0]
        assert event['type'] == 'AudioEvent'
        body = raw[12 + int.from_bytes(raw[4:8], 'big'):-4]
        if fault == 'auth':
            await ws.send(wire.encode_event({':exception-type': 'BadRequestException', ':message-type': 'exception'},
                                            b'{"Message":"denied"}'))
            return
        if body:
            frames += 1
            if text and frames == 3:
                await ws.send(wire.encode_event({':event-type': 'TranscriptEvent', ':message-type': 'event'},
                                                json.dumps(result('r1', text.split()[0], True)['payload']).encode()))
            continue
        # Empty AudioEvent: flush the last result, then close.
        if fault == 'disconnect':
            await ws.close(code=1011); return
        if text:
            await ws.send(wire.encode_event({':event-type': 'TranscriptEvent', ':message-type': 'event'},
                                            json.dumps(result('r1', text, False)['payload']).encode()))
        await ws.close()
        return


def run_local(tmp_path, monkeypatch, c, handler):
    monkeypatch.setattr(provider_protocol, 'stream_audio', fast_audio)
    async def scenario():
        async with serve(handler, '127.0.0.1', 0) as server:
            url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
            log = EventLog(tmp_path / 'events.jsonl'); log.emit('clip_start')
            monkeypatch.setattr(wire, 'presigned_url', lambda config, key: url)
            try:
                await wire.transcribe(bytes(640 * 60), 10, c, KEY, log)
            except Exception:
                log.emit('error', error_type='FixtureFailure'); raise
            finally:
                log.emit('clip_end'); log.close()
    asyncio.run(scenario())
    assert 'fixture-secret' not in (tmp_path / 'events.jsonl').read_text()
    return reduce_events(read_events(tmp_path / 'events.jsonl'), c)


@pytest.mark.parametrize('text', ['hello world', ''])
def test_local_websocket_success(text, tmp_path, monkeypatch):
    r = run_local(tmp_path, monkeypatch, config(), lambda ws: fake_transcribe(ws, text))
    assert r['transcript_complete'] and r['model_verified'] and r['transcript'] == text


@pytest.mark.parametrize('fault', ['auth', 'disconnect'])
def test_local_websocket_failure_never_becomes_success(fault, tmp_path, monkeypatch):
    c = dict(config(), close_timeout_seconds=.5)
    with pytest.raises(Exception):
        run_local(tmp_path, monkeypatch, c, lambda ws: fake_transcribe(ws, 'hello', fault))
    assert not reduce_events(read_events(tmp_path / 'events.jsonl'), c)['transcript_complete']
