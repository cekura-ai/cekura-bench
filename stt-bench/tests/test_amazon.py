"""Amazon Transcribe streaming offline fixtures and a fake SDK stream. No provider hosts."""
import asyncio
import json
from types import SimpleNamespace
import pytest
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
                            ('finalization', 'manual_at_speech_end'), ('endpoint', 'https://example.test')]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key_name: value}))


def test_results_join_in_order_and_close_completes():
    c, events = config(), stream()
    r = reduce_events(events, c)
    assert r['transcript'] == 'Hello there. How are you?'
    assert r['transcript_complete'] and r['model_verified']
    assert r['final_transcript_received_seconds'] == 5.4
    assert r['finalize_latency_status'] == 'unsupported'
    assert transcript_at(events, 5, c)['final_text'] == 'Hello there.'
    assert transcript_at(events, 5, c)['partial_text'] == 'How'


def test_completion_requires_end_after_end_stream_and_no_open_partial():
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
    events.insert(5, msg({'type': 'error', 'exception_type': 'LimitExceededException'}, 5.1))
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


class FakeSdkStream:
    """The SDK's stream surface: input_stream.send_audio_event/end_stream and an async output_stream."""

    def __init__(self, text, fault=None):
        self.text, self.fault, self.frames, self.queue = text, fault, 0, asyncio.Queue()
        self.input_stream = SimpleNamespace(send_audio_event=self.send_audio_event, end_stream=self.end_stream)
        self.output_stream = self.events()

    def event(self, text, partial):
        r = SimpleNamespace(result_id='r1', start_time=0.0, end_time=0.2, is_partial=partial,
                            alternatives=[SimpleNamespace(transcript=text)])
        return SimpleNamespace(transcript=SimpleNamespace(results=[r]))

    async def send_audio_event(self, audio_chunk):
        self.frames += 1
        if self.fault == 'auth':
            await self.queue.put(RuntimeError('BadRequestException'))
        elif self.text and self.frames == 3:
            await self.queue.put(self.event(self.text.split()[0], True))

    async def end_stream(self):
        if self.fault == 'disconnect':
            await self.queue.put(ConnectionError('reset'))
            return
        if self.text:
            await self.queue.put(self.event(self.text, False))
        await self.queue.put(None)

    async def events(self):
        while (item := await self.queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item


def run_local(tmp_path, monkeypatch, c, fake):
    monkeypatch.setattr(provider_protocol, 'stream_audio', fast_audio)
    async def opened(config, key):
        return fake
    monkeypatch.setattr(wire, 'open_stream', opened)
    async def scenario():
        log = EventLog(tmp_path / 'events.jsonl'); log.emit('clip_start')
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
def test_local_stream_success(text, tmp_path, monkeypatch):
    fake = FakeSdkStream(text)
    r = run_local(tmp_path, monkeypatch, config(), fake)
    assert r['transcript_complete'] and r['model_verified'] and r['transcript'] == text
    assert fake.frames == 60  # 10 speech frames plus the 50-frame silence tail


@pytest.mark.parametrize('fault', ['auth', 'disconnect'])
def test_local_stream_failure_never_becomes_success(fault, tmp_path, monkeypatch):
    c = dict(config(), close_timeout_seconds=.5)
    with pytest.raises(Exception):
        run_local(tmp_path, monkeypatch, c, FakeSdkStream('hello', fault))
    assert not reduce_events(read_events(tmp_path / 'events.jsonl'), c)['transcript_complete']
