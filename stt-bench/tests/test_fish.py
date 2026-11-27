"""Fish Audio realtime ASR fixtures and local WebSocket tests. No provider hosts."""
import asyncio
import json
from pathlib import Path

import msgpack
import pytest
from websockets.asyncio.server import serve

from stt_bench import fish, full_benchmark
from stt_bench.credentials import credential
from stt_bench.providers import reduce_events, transcript_at, validate
from stt_bench.streaming import EventLog, read_events
from stt_bench.turn_batch import OPT_IN
from stt_bench.turn_metrics import transcript_timeline
from stt_bench.turn_runner import validate_profile


def config():
    return json.loads(Path('config/models/fish-transcribe-stream-2.json').read_text())


def started(seq=0, **overrides):
    applied = {'protocol_version': 1, 'model': 'transcribe-stream-2',
               'audio': {'format': 'pcm_s16le', 'sample_rate': 16000, 'channels': 1},
               'turn_detection': {'mode': 'manual', 'eager': False, 'silence_timeout_ms': 60000},
               'results': {'drafts': True, 'ignore_timestamps': True, 'diarize': False, 'tag_audio_events': False,
                           'speaker_turns': False, 'text_format': 'written', 'vad_events': False, 'eot_events': False}}
    for path, value in overrides.items():
        section, _, key = path.partition('.')
        if key:
            applied[section] = {**applied[section], key: value}
        else:
            applied[section] = value
    return {'type': 'session_started', 'seq': seq, 'session_id': 's_fixture', 'request_id': 'request-fixture',
            'config': applied, 'limits': {'max_session_ms': 3600000, 'max_turn_ms': 60000}}


def draft(seq, text, turn=0):
    return {'type': 'draft', 'seq': seq, 'turn_index': turn, 'text': text, 'settled_text': ''}


def committed(seq, text, turn=0, reason='finalize'):
    return {'type': 'committed', 'seq': seq, 'turn_index': turn, 'text': text, 'reason': reason,
            'user_turn': True, 'turn_kind': 'user', 'degraded': False}


def done(seq):
    return {'type': 'session_done', 'seq': seq, 'stats': {'audio_ms': 2000, 'turns': 1}}


def event(value, at):
    return {'kind': 'provider_message', 'time_seconds': at, 'message': value}


def session(*messages, finalize_at=.2, close_at=1.2):
    """A complete attempt: messages are (time, message) pairs around speech end and audio_done."""
    c = config()
    events = [dict(kind='model_accepted', time_seconds=.06, model=c['model']),
              dict(kind='speech_end', time_seconds=finalize_at),
              dict(kind='finalize_requested', time_seconds=finalize_at),
              dict(kind='audio_complete', time_seconds=close_at),
              dict(kind='close_stream_requested', time_seconds=close_at)]
    events += [event(m, at) for at, m in messages]
    return sorted(events, key=lambda e: e['time_seconds'])


def test_config_connection_and_credentials():
    c = validate(config())
    assert credential('fish', {'FISH_API_KEY': 'fixture'}, None) == ('fixture', 'FISH_API_KEY')
    url, headers = fish.connection(c, 'fixture')
    assert url == 'wss://api.fish.audio/v1/asr/realtime' and 'fixture' not in url
    assert headers == {'Authorization': 'Bearer fixture', 'model': 'transcribe-stream-2'}
    assert fish.Protocol(c).setup() == {
        'type': 'config', 'audio': {'format': 'pcm_s16le', 'sample_rate': 16000, 'channels': 1},
        'turn_detection': {'mode': 'manual'},
        'results': {'drafts': True, 'ignore_timestamps': True, 'diarize': False, 'tag_audio_events': False,
                    'text_format': 'written'}}
    for key, value in [('turn_detection_mode', 'auto'), ('text_format', 'spoken'), ('tag_audio_events', True),
                       ('diarize', True), ('sample_rate', 24000), ('finalize_ack_supported', True),
                       ('model', 'transcribe-stream-1'), ('transmitted_silence_frames', 0)]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key: value}))


def test_turn_profile_is_the_model_config_with_controlled_turn_fields():
    profile = json.loads(Path('config/profiles/private-turns-v1/fish-transcribe-stream-2.json').read_text())
    assert {k: v for k, v in profile.items() if k in config()} == config()
    assert profile['turn_finalization_class'] == 'controlled'
    assert profile['base_config'] == 'config/models/fish-transcribe-stream-2.json'
    validate_profile(profile)
    assert 'fish-transcribe-stream-2' in OPT_IN and 'fish-transcribe-stream-2' in full_benchmark.FULL_MODELS


def test_replay_times_the_finalize_commit_and_completes_on_session_done():
    c = config()
    events = session((.05, started()), (.1, draft(1, 'hello')), (.15, {'type': 'speech_start', 'seq': 2, 'turn_index': 0}),
                     (.45, committed(3, 'Hello world.')), (1.3, done(4)))
    events.append(dict(kind='provider_terminal', time_seconds=1.31))
    assert transcript_at(events, .2, c) == {
        'text': 'hello', 'final_text': '', 'partial_text': 'hello',
        'provisional': True, 'reconstruction_status': 'supported'}
    result = reduce_events(events, c)
    assert result['transcript'] == 'Hello world.' and result['transcript_complete'] and result['model_verified']
    assert result['finalize_latency_ms'] == pytest.approx(250)
    assert result['finalize_latency_status'] == 'observed'
    # Without session_done (and so without provider_terminal) nothing is complete.
    assert not reduce_events(events[:-2], c)['transcript_complete']
    assert [s['text'] for s in full_benchmark.final_snapshots(events, c)] == ['', 'Hello world.']
    assert transcript_timeline(events, c)[-1]['text'] == 'Hello world.'


def test_forced_sixty_second_commits_join_in_turn_order():
    # Manual mode commits a turn after 60 s; the next turn starts where it ended.
    c = config()
    events = session((.05, started()), (30, draft(1, 'first part')),
                     (60.3, committed(2, 'First part of it,', reason='max_turn_duration')),
                     (61, draft(3, 'second', turn=1)), (75.2, committed(4, 'second part.', turn=1)), (76.3, done(5)),
                     finalize_at=75, close_at=76)
    events.append(dict(kind='provider_terminal', time_seconds=76.31))
    result = reduce_events(events, c)
    assert result['transcript'] == 'First part of it, second part.' and result['transcript_complete']
    assert result['finalize_latency_ms'] == pytest.approx(200)


def test_finalize_without_an_open_turn_needs_no_reply():
    # Live: finalize with no open turn is ignored; audio_done is still answered by session_done.
    c = config()
    events = session((.05, started()), (1.3, done(1)))
    events.append(dict(kind='provider_terminal', time_seconds=1.31))
    result = reduce_events(events, c)
    assert result['transcript'] == '' and result['transcript_complete']
    assert result['finalize_latency_ms'] is None and result['finalize_latency_status'] == 'no_reply_to_finalize'


def test_a_turn_left_provisional_is_incomplete():
    c = config()
    events = session((.05, started()), (.1, draft(1, 'hello')), (1.3, done(2)))
    events.append(dict(kind='provider_terminal', time_seconds=1.31))
    assert not reduce_events(events, c)['transcript_complete']


@pytest.mark.parametrize('payloads', [
    [started(), {'type': 'error', 'seq': 1, 'code': 429, 'error_type': 'rate_limit', 'message': 'busy'}],
    [started(), draft(2, 'skipped a sequence number')],
    [started(), {'type': 'mystery', 'seq': 1}],
    [started(), done(1)],  # session_done before audio_done
    [draft(0, 'before session_started')],
    [started(**{'turn_detection.mode': 'auto'})],
    [started(**{'results.text_format': 'spoken'})],
    [started(**{'results.tag_audio_events': True})],
    [started(), committed(1, 'no reason', reason=None)],
    [started(), {**draft(1, 'x'), 'turn_index': -1}],
])
def test_errors_and_malformed_or_unapplied_settings_fail(payloads):
    protocol = fish.Protocol(config())
    with pytest.raises(fish.shared.ProviderError):
        for payload in payloads:
            protocol.feed(payload)


def test_a_different_model_in_the_session_echo_is_a_mismatch():
    protocol = fish.Protocol(config())
    protocol.feed(started(model='transcribe-stream-1'))
    assert protocol.model_mismatch


@pytest.mark.parametrize('raw', ['{"type": "session_started"}', b'\xc1', msgpack.packb([1, 2]),
                                 msgpack.packb({'type': 'draft', 'text': b'bytes'}, use_bin_type=True)])
def test_frames_must_be_msgpack_maps_of_json_values(raw):
    with pytest.raises(fish.shared.ProviderError):
        fish.unpack(raw)


def test_transcribe_loopback_sends_msgpack_config_audio_finalize_and_audio_done(tmp_path, monkeypatch):
    async def check():
        c = config()
        received, headers = [], []

        async def server(ws):
            headers.append((ws.request.headers['Authorization'], ws.request.headers['model']))
            seq = 0
            async for raw in ws:
                assert isinstance(raw, bytes)
                message = msgpack.unpackb(raw, raw=False)
                received.append(message)
                kind = message['type']
                if kind == 'config':
                    await ws.send(msgpack.packb(started(seq), use_bin_type=True)); seq += 1
                elif kind == 'audio' and message['data'] == b'speech':
                    await ws.send(msgpack.packb(draft(seq, 'hello'), use_bin_type=True)); seq += 1
                elif kind == 'finalize':
                    await ws.send(msgpack.packb(committed(seq, 'Hello world.'), use_bin_type=True)); seq += 1
                elif kind == 'audio_done':
                    await ws.send(msgpack.packb(done(seq), use_bin_type=True))
                    await ws.close(code=1000)
                    return

        async def fast_audio(pcm, speech_frames, send, finalize, log, **kwargs):
            await send(b'speech')
            t0 = log.now()
            log.emit('speech_end', at=t0)
            await finalize(t0)
            await send(b'\0' * 4)
            log.emit('audio_complete', t0_seconds=t0)

        async with serve(server, '127.0.0.1', 0) as local:
            monkeypatch.setattr(fish, 'connection', lambda config, key: (
                f'ws://127.0.0.1:{local.sockets[0].getsockname()[1]}',
                {'Authorization': f'Bearer {key}', 'model': config['model']}))
            monkeypatch.setattr(fish, 'stream_audio', fast_audio)
            log = EventLog(tmp_path / 'events.jsonl')
            try:
                await fish.transcribe(b'', 1, c, 'fixture-secret', log)
            finally:
                log.close()
        assert headers == [('Bearer fixture-secret', 'transcribe-stream-2')]
        assert received == [fish.Protocol(c).setup(), {'type': 'audio', 'data': b'speech'}, {'type': 'finalize'},
                            {'type': 'audio', 'data': b'\0' * 4}, {'type': 'audio_done'}]
        events = read_events(tmp_path / 'events.jsonl')
        result = reduce_events(events, c)
        assert result['transcript'] == 'Hello world.' and result['transcript_complete'] and result['model_verified']
        assert 'fixture-secret' not in (tmp_path / 'events.jsonl').read_text()

    asyncio.run(check())
