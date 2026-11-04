"""Mistral Voxtral realtime fixtures and local WebSocket tests. No provider hosts.

Message shapes are copied from live sessions recorded on 2026-11-04.
"""
import asyncio
import base64
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import serve

from stt_bench import catalog, full_benchmark, mistral
from stt_bench.credentials import credential
from stt_bench.providers import reduce_events, transcript_at, validate
from stt_bench.streaming import EventLog, read_events
from stt_bench.turn_batch import OPT_IN
from stt_bench.turn_metrics import measure
from stt_bench.turn_runner import validate_profile

MODEL_ID = 'mistral-voxtral-realtime-3'
BUILD = 'voxtral-mini-transcribe-realtime-3-0-4'
PROFILE = Path('config/profiles/private-turns-v1/mistral-voxtral-realtime-3.json')


def config():
    return json.loads(Path('config/models/mistral-voxtral-realtime-3.json').read_text())


def state(**changes):
    return {'request_id': 'ws-fixture', 'model': BUILD,
            'audio_format': {'encoding': 'pcm_s16le', 'sample_rate': 16000},
            'target_streaming_delay_ms': None, 'diarize': False, 'verbatim': False, 'context_bias': [],
            'language': None, 'speaker_references': None, 'turn_detection': None, **changes}


def created(**changes):
    return {'type': 'session.created', 'session': state(**changes)}


def updated(**changes):
    return {'type': 'session.updated', 'session': state(**{'language': 'en', 'target_streaming_delay_ms': 480, **changes})}


def delta(text):
    return {'type': 'transcription.text.delta', 'text': text}


def done(text, model=BUILD):
    return {'model': model, 'text': text, 'language': None, 'segments': [], 'type': 'transcription.done',
            'usage': {'prompt_audio_seconds': 2, 'prompt_tokens': 40, 'total_tokens': 86, 'completion_tokens': 46}}


def event(value, at):
    return {'kind': 'provider_message', 'time_seconds': at, 'message': value}


def clip_events(*, flush_reply='hello world', tail=''):
    """One clip as the live service answers it; None leaves a reply out."""
    events = [dict(kind='model_accepted', time_seconds=0, model=config()['model']),
              event(created(), .01), event(updated(), .02),
              event(delta('hello'), .1), event(delta(' world'), .2),
              dict(kind='speech_end', time_seconds=.3), dict(kind='finalize_requested', time_seconds=.3)]
    if flush_reply is not None:
        events.append(event(done(flush_reply), .9))
    events += [dict(kind='audio_complete', time_seconds=1.3), dict(kind='close_stream_requested', time_seconds=1.3)]
    if tail is not None:
        events.append(event(done(tail), 2.0))
    return events + [dict(kind='provider_terminal', time_seconds=2.0)]


def test_config_connection_and_credentials():
    c = validate(config())
    assert credential('mistral', {'MISTRAL_API_KEY': 'canonical'}, None) == ('canonical', 'MISTRAL_API_KEY')
    assert credential('mistral', {'MISTRAL': 'alias'}, None) == ('alias', 'MISTRAL')
    assert credential('mistral', {'MISTRAL': 'alias', 'MISTRAL_API_KEY': 'canonical'}, None)[1] == 'MISTRAL_API_KEY'
    url, headers = mistral.connection(c, 'fixture')
    assert headers == {'Authorization': 'Bearer fixture'} and 'fixture' not in url
    assert parse_qs(urlsplit(url).query) == {'model': ['voxtral-mini-transcribe-realtime-3-0']}
    for key, value in [('language', None), ('language', 'fr'), ('sample_rate', 24000),
                       ('target_streaming_delay_ms', None), ('target_streaming_delay_ms', 2400),
                       ('finalization', 'stream_end_after_tail'), ('finalize_ack_supported', False),
                       ('served_model', None), ('served_model', 'voxtral-mini-transcribe-realtime-2602'),
                       ('model', 'voxtral-mini-transcribe-realtime-3')]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key: value}))


def test_turn_profile_and_run_lists_include_the_model():
    profile = json.loads(PROFILE.read_text())
    assert {k: v for k, v in profile.items() if k in config()} == config()
    assert profile['turn_finalization_class'] == 'controlled'
    assert profile['base_config'] == 'config/models/mistral-voxtral-realtime-3.json'
    validate_profile(profile)
    assert MODEL_ID in catalog.MODELS and MODEL_ID in OPT_IN and MODEL_ID in full_benchmark.FULL_MODELS


def test_clip_completes_on_the_flush_reply_then_the_end_reply():
    c = config()
    events = clip_events()
    # Deadline text at speech end is the streamed, not yet final, text.
    assert transcript_at(events, .3, c) == {
        'text': 'hello world', 'final_text': '', 'partial_text': 'hello world',
        'provisional': True, 'reconstruction_status': 'supported'}
    result = reduce_events(events, c)
    assert result['transcript'] == 'hello world'
    assert result['transcript_complete'] and result['model_verified']
    assert result['finalize_latency_status'] == 'observed'
    assert result['finalize_latency_ms'] == pytest.approx(600)
    assert result['completion_latency_ms'] == pytest.approx(1700)
    assert result['model_versions'] == [BUILD]


def test_completion_requires_the_reply_to_input_audio_end():
    result = reduce_events(clip_events(tail=None), config())
    assert result['transcript'] == 'hello world' and not result['transcript_complete']


def test_a_done_before_flush_is_a_segment_and_not_the_acknowledgment():
    # Long audio may close a segment on its own; only the done after flush acknowledges it.
    c = config()
    events = [dict(kind='model_accepted', time_seconds=0, model=c['model']), event(created(), .01), event(updated(), .02),
              event(delta('first part'), .1), event(done('first part'), .2), event(delta(' second'), .3),
              dict(kind='speech_end', time_seconds=.4), dict(kind='finalize_requested', time_seconds=.4),
              event(delta(' part'), .5), event(done('second part'), .7),
              dict(kind='audio_complete', time_seconds=1.4), dict(kind='close_stream_requested', time_seconds=1.4),
              event(done(''), 2.0), dict(kind='provider_terminal', time_seconds=2.0)]
    result = reduce_events(events, c)
    assert result['transcript'] == 'first part second part' and result['transcript_complete']
    assert result['finalize_latency_ms'] == pytest.approx(300)


@pytest.mark.parametrize('tail, doubled', [
    ('hello world', True),               # a whole-session transcript would count the speech twice
    ('Hello, world. Goodbye.', True),    # the same words, punctuated, plus more
    ('', False),
    ('helloworld', False),               # word boundaries, not a character prefix
    ('world hello', False),
])
def test_an_end_reply_repeating_earlier_segments_fails_closed(tail, doubled):
    snap, result = mistral.replay(clip_events(tail=tail), config())
    assert result['transcript_complete'] is not doubled
    assert (snap['reconstruction_status'] == 'unsupported_order_or_overlap') is doubled


@pytest.mark.parametrize('tail, ttfs_ms, snapshots', [
    ('', 600, ['', 'hello world']),
    # Text invented from the silence tail is scored and delays the final text.
    ('Thank you.', 1700, ['', 'hello world', 'hello world Thank you.']),
])
def test_turn_timing_and_full_run_snapshots_use_the_last_final_text(tail, ttfs_ms, snapshots):
    profile = json.loads(PROFILE.read_text())
    events = clip_events(tail=tail)
    timing = measure(events, profile, {'pacing': {'valid': True}, 'valid': True})
    assert timing['ttfs_status'] == 'measured' and timing['ttfs_ms'] == pytest.approx(ttfs_ms)
    assert timing['final_text'] == snapshots[-1]
    assert [s['text'] for s in full_benchmark.final_snapshots(events, profile)] == snapshots


@pytest.mark.parametrize('index, message', [(1, created(model='voxtral-mini-transcribe-realtime-3-0-5')),
                                            (2, updated(model='voxtral-mini-transcribe-realtime-3-0-5')),
                                            (7, done('hello world', model='voxtral-mini-transcribe-realtime-3-0-5'))])
def test_a_different_served_build_is_never_verified(index, message):
    events = clip_events()
    events[index] = event(message, events[index]['time_seconds'])
    assert not reduce_events(events, config())['model_verified']


@pytest.mark.parametrize('changes', [
    dict(language=None), dict(target_streaming_delay_ms=None),
    dict(audio_format={'encoding': 'pcm_s16le', 'sample_rate': 24000}),
    dict(turn_detection={'end_threshold': .5}), dict(diarize=True),
])
def test_session_settings_must_be_echoed_before_audio(changes):
    protocol = mistral.Protocol(config())
    protocol.feed(created())
    with pytest.raises(mistral.shared.ProviderError):
        protocol.feed(updated(**changes))
    assert not protocol.ready
    fresh = mistral.Protocol(config())
    with pytest.raises(mistral.shared.ProviderError):
        fresh.feed(updated())  # session.updated before session.created


@pytest.mark.parametrize('payload', [
    {'type': 'error', 'error': {'message': "Model 'voxtral-mini-transcribe-realtime-2602' does not support "
                                           'language forcing.', 'code': 3051}},
    {'type': 'turn.end', 'time_s': 4.4, 'end_reason': 'threshold'},
    {'type': 'transcription.text.delta'},
    {'type': 'transcription.done', 'model': BUILD},
    {'type': 'transcription.done', 'text': 'hello'},
    {'type': 'session.created'},
])
def test_errors_and_unexpected_events_fail(payload):
    with pytest.raises(mistral.shared.ProviderError):
        mistral.Protocol(config()).feed(payload)


def test_language_events_are_informational_and_session_start_happens_once():
    protocol = mistral.Protocol(config())
    protocol.feed(created())
    protocol.feed({'type': 'transcription.language', 'audio_language': 'en'})
    with pytest.raises(mistral.shared.ProviderError):
        protocol.feed(created())


SPEECH, TAIL = bytes(range(256)) * 2 + bytes(128), bytes(640)


async def fast_audio(pcm, speech_frames, send, finalize, log, **kwargs):
    log.emit('audio_start')
    await send(SPEECH)
    t0 = log.now()
    log.emit('speech_end', at=t0)
    await finalize(t0)
    await send(TAIL)
    log.emit('audio_complete', t0_seconds=t0)


async def loopback(tmp_path, monkeypatch, fault=None):
    """Run the real adapter against a local server that answers like the live service."""
    record = dict(early=[], received=[], headers=[], paths=[], closes=[])

    async def server(ws):
        record['headers'].append(ws.request.headers['Authorization'])
        record['paths'].append(ws.request.path)
        try:
            record['early'].append(await asyncio.wait_for(ws.recv(), .2))
        except TimeoutError:
            pass
        build = 'voxtral-mini-transcribe-realtime-3-0-5' if fault == 'build_change' else BUILD
        await ws.send(json.dumps(created(model=build)))
        async for raw in ws:
            m = json.loads(raw)
            record['received'].append(m)
            if m['type'] == 'session.update':
                await ws.send(json.dumps(updated(**m['session'])))
            elif m['type'] == 'input_audio.append' and fault == 'error_event':
                await ws.send(json.dumps({'type': 'error', 'error': {'message': 'rejected', 'code': 3000}}))
            elif m['type'] == 'input_audio.flush' and fault != 'no_flush_reply':
                await ws.send(json.dumps(delta('hello')))
                await ws.send(json.dumps(done('hello')))
            elif m['type'] == 'input_audio.end':
                if fault == 'drop_before_end_reply':
                    await ws.close(code=1011, reason='Upstream connection error')
                    return
                await ws.send(json.dumps(done('')))
        record['closes'].append(ws.close_code)

    monkeypatch.setattr(mistral.shared, 'stream_audio', fast_audio)
    async with serve(server, '127.0.0.1', 0) as local:
        port = local.sockets[0].getsockname()[1]
        c = dict(config(), endpoint=f'ws://127.0.0.1:{port}/v1/audio/transcriptions/realtime',
                 ready_timeout_seconds=1, finalize_timeout_seconds=.2, close_timeout_seconds=.2)
        log = EventLog(tmp_path / 'events.jsonl')
        try:
            await mistral.transcribe(SPEECH + TAIL, 1, c, 'fixture-secret', log)
            record['raised'] = None
        except Exception as exc:
            # The runner records any adapter exception as an attempt error.
            log.emit('error', error_type=type(exc).__name__)
            record['raised'] = exc
        finally:
            log.close()
    record['events'] = read_events(tmp_path / 'events.jsonl')
    return record


def test_loopback_waits_for_session_created_sends_flush_then_end_and_closes_cleanly(tmp_path, monkeypatch):
    record = asyncio.run(loopback(tmp_path, monkeypatch))
    assert record['raised'] is None and record['early'] == []
    assert record['headers'] == ['Bearer fixture-secret']
    assert parse_qs(urlsplit(record['paths'][0]).query) == {'model': ['voxtral-mini-transcribe-realtime-3-0']}
    received = record['received']
    assert [m['type'] for m in received] == ['session.update', 'input_audio.append', 'input_audio.flush',
                                              'input_audio.append', 'input_audio.end']
    assert received[0]['session'] == {'audio_format': {'encoding': 'pcm_s16le', 'sample_rate': 16000},
                                      'language': 'en', 'target_streaming_delay_ms': 480}
    assert [base64.b64decode(m['audio']) for m in received if m['type'] == 'input_audio.append'] == [SPEECH, TAIL]
    assert record['closes'] == [1000]
    result = reduce_events(record['events'], config())
    assert result['transcript'] == 'hello' and result['transcript_complete'] and result['model_verified']
    assert 'fixture-secret' not in (tmp_path / 'events.jsonl').read_text()


@pytest.mark.parametrize('fault', ['no_flush_reply', 'drop_before_end_reply', 'error_event', 'build_change'])
def test_loopback_failures_never_become_a_verified_complete_transcript(fault, tmp_path, monkeypatch):
    record = asyncio.run(loopback(tmp_path, monkeypatch, fault))
    assert record['raised'] is not None
    result = reduce_events(record['events'], config())
    assert not (result['transcript_complete'] and result['model_verified'])
    if fault == 'build_change':
        # The adapter stops before sending settings or audio to a different build.
        assert record['received'] == []
