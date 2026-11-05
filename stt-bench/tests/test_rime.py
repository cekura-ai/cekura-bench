"""Rime offline fixtures and local WebSocket tests. No provider hosts."""
import asyncio
import base64
from http import HTTPStatus
import json
from pathlib import Path

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed, InvalidStatus

from stt_bench import full_benchmark as full
from stt_bench import rime as wire
from stt_bench.catalog import model_config
from stt_bench.credentials import command_environment
from stt_bench.providers import validate, reduce_events, transcript_at
from stt_bench.run import assess
from stt_bench.streaming import EventLog, read_events, transmitted_silence_frames
from stt_bench.turn_runner import validate_profile

PROFILE = Path('config/profiles/private-turns-v1/rime-stt.json')
LANGUAGE = {'tag': 'en', 'source': 'LANGUAGE_SOURCE_SELECTED'}
STARTED = {'started': {'requestId': 'req-1', 'accepted': {'outputContract': wire.OUTPUT_CONTRACT, 'language': LANGUAGE}}}
SPEECH = b'\x01\x00' * 320  # one nonzero 20 ms frame at 16 kHz


def config():
    return json.loads(model_config('rime-stt').read_text())


def msg(value, at):
    return dict(kind='provider_message', time_seconds=at, message=value)


def hyp(text, revision):
    return {'hypothesis': {'text': text, 'revision': str(revision)}}


def done(text, revision):
    body = {'language': LANGUAGE}
    if text:
        body['text'] = text  # proto3 JSON omits defaults: empty text and revision 0
    if revision:
        body['revision'] = str(revision)
    return {'done': body}


def audio(frames):
    """Frozen-clip shape: speech frames, then the 50 zero frames the bench never sends to Rime."""
    return SPEECH * frames + bytes(640 * 50)


def test_config_contract_credentials_and_wire_messages():
    c = validate(config())
    assert command_environment('rime', environ={'RIME_API_KEY': 'fixture', 'RESON_API_KEY': 'other'},
                               env_file=None) == {'RIME_API_KEY': 'fixture'}
    url, headers = wire.connection(c, 'fixture')
    assert url == 'wss://stt.api.rime.ai/ws' and headers == {'Authorization': 'Bearer fixture'}
    p = wire.Protocol(c)
    assert not p.ready
    assert p.setup() == {'start': {'outputContract': 'STREAMING_OUTPUT_CONTRACT_REVISED_HYPOTHESES',
                                   'language': 'en', 'mode': 'TRANSCRIPTION_MODE_WRITTEN'}}
    assert base64.b64decode(p.audio(SPEECH)['audio']) == SPEECH and list(p.audio(SPEECH)) == ['audio']
    assert p.finalize() == {'end': {}} and p.finish(55) is None
    for key, value in [('sample_rate', 24000), ('channels', 2), ('frame_ms', 80), ('model', 'other'),
                       ('endpoint', 'wss://other.example/ws'), ('language', 'es'), ('mode', 'TRANSCRIPTION_MODE_VERBATIM'),
                       ('output_contract', 'STREAMING_OUTPUT_CONTRACT_APPEND_FRAGMENTS'), ('transmitted_silence_frames', 50),
                       ('private_variant', None), ('ready_timeout_seconds', 0), ('finalization', 'stream_end_after_tail'),
                       ('requires_final_only_completion', False)]:
        with pytest.raises(ValueError):
            validate(dict(c, **{key: value}))
    without_tail_setting = dict(c)
    del without_tail_setting['transmitted_silence_frames']
    with pytest.raises(ValueError):
        validate(without_tail_setting)  # Absent means the historical 50-frame tail.


def test_no_tail_exception_is_rime_only():
    assert transmitted_silence_frames(config()) == 0
    with pytest.raises(ValueError, match='Only Inworld'):
        transmitted_silence_frames(dict(config(), provider='deepgram'))


def session_events():
    return [dict(kind='model_accepted', time_seconds=0, model='default'), msg(STARTED, .05),
            msg(hyp('Hi.', 1), .5), msg(hyp('Hi. I am', 2), 1.0),
            dict(kind='speech_end', time_seconds=1.2), dict(kind='finalize_requested', time_seconds=1.2),
            dict(kind='audio_complete', time_seconds=1.21),
            msg(hyp('Hi. I am calling.', 3), 1.5), msg(done('Hi. I am calling.', 3), 1.6),
            dict(kind='close_stream_requested', time_seconds=1.61), dict(kind='provider_terminal', time_seconds=1.61)]


def test_replay_replaces_partials_and_completes_only_on_done():
    c, events = config(), session_events()
    assert transcript_at(events, 1.1, c)['text'] == 'Hi. I am'
    assert transcript_at(events, 1.1, c)['final_text'] == '' and transcript_at(events, 1.1, c)['provisional']
    r = reduce_events(events, c)
    assert r['transcript'] == 'Hi. I am calling.' and r['transcript_complete'] and r['model_verified']
    assert r['finalize_latency_ms'] == pytest.approx(400) and r['completion_latency_ms'] == pytest.approx(410)
    assert r['first_partial_after_t0']['latency_ms'] == pytest.approx(300)
    assert r['final_transcript_received_seconds'] == 1.6
    assert 'model_version_not_exposed' in r['model_verification_basis']
    # Without done, or without the terminal record, nothing is complete.
    assert not reduce_events([e for e in events if 'done' not in e.get('message', {})], c)['transcript_complete']
    assert not reduce_events(events[:-1], c)['transcript_complete']
    # done before end, or anything after done, is a protocol failure.
    early = [e for e in events if e['kind'] != 'finalize_requested']
    assert not reduce_events(early, c)['transcript_complete']
    assert not reduce_events(events + [msg(hyp('late', 4), 1.7)], c)['transcript_complete']


def test_empty_utterance_completes_with_empty_text():
    events = [dict(kind='model_accepted', time_seconds=0, model='default'), msg(STARTED, .05),
              dict(kind='speech_end', time_seconds=.5), dict(kind='finalize_requested', time_seconds=.5),
              dict(kind='audio_complete', time_seconds=.5), msg(done('', 0), .7),
              dict(kind='close_stream_requested', time_seconds=.71), dict(kind='provider_terminal', time_seconds=.71)]
    r = reduce_events(events, config())
    assert r['transcript_complete'] and r['transcript'] == ''


@pytest.mark.parametrize('messages,requested', [
    ([{'error': {'kind': 'resource_exhausted', 'message': 'limit', 'requestId': 'r'}}], True),
    ([STARTED, {'cancelled': {}}], True),
    ([STARTED, {'transcript': {'text': 'x'}}], True),
    ([[]], True),
    ([{'started': STARTED['started'], 'hypothesis': {}}], True),
    ([STARTED, {'hypothesis': 'text'}], True),
    ([hyp('early', 1)], True),
    ([STARTED, STARTED], True),
    ([{'started': {'requestId': 'r', 'accepted': {'outputContract': wire.OUTPUT_CONTRACT,
                                                  'language': {'tag': 'es'}}}}], True),
    ([{'started': {'requestId': 'r', 'accepted': {'outputContract': 'STREAMING_OUTPUT_CONTRACT_APPEND_FRAGMENTS',
                                                  'language': LANGUAGE}}}], True),
    ([STARTED, hyp('skipped', 2)], True),
    ([STARTED, hyp('a', 1), hyp('a', 1)], True),
    ([STARTED, {'hypothesis': {'text': 'a', 'revision': 'x'}}], True),
    ([STARTED, {'hypothesis': {'text': 5, 'revision': '1'}}], True),
    ([STARTED, hyp('a', 1), done('a', 2)], True),
    ([STARTED, hyp('a', 1), done('b', 1)], True),
    ([STARTED, hyp('a', 1), {'done': {'text': 'a', 'revision': '1', 'language': {'tag': 'es'}}}], True),
    ([STARTED, done('', 0)], False),
    ([STARTED, hyp('a', 1), done('a', 1), hyp('a b', 2)], True),
])
def test_malformed_or_out_of_order_events_fail(messages, requested):
    p = wire.Protocol(config())
    p.requested = requested
    with pytest.raises(wire.shared.ProviderError):
        for m in messages:
            p.feed(m)


def fake_rime(log, failure=None, every=3):
    """A Rime-shaped server: start, then audio until end, then hypothesis and done, then close."""
    async def server(ws):
        log.append(('authorization', ws.request.headers.get('Authorization')))
        try:
            first = json.loads(await ws.recv())
            log.append(first)
            if failure == 'no_started':
                await asyncio.sleep(1)
                return
            if failure == 'reject_start':
                await ws.send(json.dumps({'error': {'kind': 'invalid_input', 'message': 'bad start', 'requestId': 'r'}}))
                return
            await ws.send(json.dumps(STARTED))
            text, revision, frames, ended = '', 0, 0, False
            async for raw in ws:
                m = json.loads(raw)
                log.append(m)
                if 'audio' in m:
                    if ended:  # Rime refuses audio after end.
                        await ws.send(json.dumps({'error': {'kind': 'invalid_input', 'message': 'audio after end',
                                                            'requestId': 'r'}}))
                        return
                    assert len(base64.b64decode(m['audio'])) == 640
                    frames += 1
                    if frames % every == 0:
                        text, revision = (text + ' word').strip(), revision + 1
                        await ws.send(json.dumps(hyp(text, revision)))
                elif 'end' in m:
                    ended = True
                    if failure == 'close_before_done':
                        return
                    if failure == 'error_after_end':
                        await ws.send(json.dumps({'error': {'kind': 'resource_exhausted', 'message': 'no capacity',
                                                            'requestId': 'r'}}))
                        return
                    if failure == 'slow_done':
                        await asyncio.sleep(1)
                    text, revision = (text + ' final').strip(), revision + 1
                    await ws.send(json.dumps(hyp(text, revision)))
                    await ws.send(json.dumps(done(text + (' changed' if failure == 'mismatched_done' else ''), revision)))
                    return
        except ConnectionClosed:
            return
    return server


def run_loopback(tmp_path, c, frames, failure=None, every=3):
    received = []

    async def check():
        async with serve(fake_rime(received, failure, every), '127.0.0.1', 0) as local:
            local_config = dict(c, endpoint=f'ws://127.0.0.1:{local.sockets[0].getsockname()[1]}')
            log = EventLog(tmp_path / 'events.jsonl')
            log.emit('clip_start')
            try:
                await wire.transcribe(audio(frames), frames, local_config, 'fixture', log)
            except Exception as exc:
                log.emit('error', error_type=type(exc).__name__, error_message=str(exc))
                raise
            finally:
                log.emit('clip_end')
                log.close()
    asyncio.run(check())
    return received, read_events(tmp_path / 'events.jsonl')


def test_loopback_session_sends_speech_then_end_and_scores_complete(tmp_path):
    c = config()
    received, events = run_loopback(tmp_path, c, 7)
    assert received[0] == ('authorization', 'Bearer fixture')
    assert received[1] == wire.Protocol(c).setup()
    sent = received[2:]
    # Exactly the seven speech frames, then end, and no audio after it: no silence tail.
    assert [list(m) for m in sent] == [['audio']] * 7 + [['end']]
    assert b''.join(base64.b64decode(m['audio']) for m in sent[:7]) == SPEECH * 7
    r = reduce_events(events, c)
    assert r['transcript'] == 'word word final' and r['transcript_complete'] and r['model_verified']
    assert r['finalize_latency_status'] == 'observed' and r['finalize_latency_ms'] >= 0
    accepted = next(e for e in events if e['kind'] == 'model_accepted')
    assert accepted['request_id'] == 'req-1'
    assert 'fixture' not in (tmp_path / 'events.jsonl').read_text()
    # The full-benchmark row for a public clip: valid, paced, scored, no tail sent.
    row = full.evaluate(events, dict(clip_id='p', cohort='public', reference='word word final'), c, 1, 'hash')
    assert row['valid'] and row['pacing']['valid'] and row['pacing']['silence_frames'] == 0
    assert row['word_errors']['hypothesis_normalized'] == 'word word final' and row['failure_class'] is None
    assert row['word_errors']['substitutions'] == row['word_errors']['deletions'] == row['word_errors']['insertions'] == 0
    assert row['sent_audio_seconds'] == pytest.approx(7 * .02)
    assert any(d['status'] != 'failed_before_speech_end' for d in row['deadlines'])


@pytest.mark.parametrize('failure', ['no_started', 'reject_start', 'close_before_done', 'error_after_end',
                                     'slow_done', 'mismatched_done'])
def test_loopback_failures_raise_and_never_score_complete(tmp_path, failure):
    c = dict(config(), ready_timeout_seconds=.3, finalize_timeout_seconds=.3)
    with pytest.raises((wire.shared.ProviderError, TimeoutError)):
        run_loopback(tmp_path, c, 5, failure)
    events = read_events(tmp_path / 'events.jsonl')
    assert not reduce_events(events, c)['transcript_complete']
    row = full.evaluate(events, dict(clip_id='p', cohort='public', reference='word final'), c, 1, 'hash')
    assert not row['valid']
    if failure == 'error_after_end':
        assert row['failure_class'] == 'concurrency'  # resource_exhausted is retried with backoff


def test_refused_upgrade_surfaces_http_status_for_failure_classes(tmp_path):
    async def check():
        async def refuse(connection, request):
            return connection.respond(HTTPStatus.UNAUTHORIZED, '')

        async def never_called(ws):
            raise AssertionError('a refused upgrade never reaches the handler')
        async with serve(never_called, '127.0.0.1', 0, process_request=refuse) as local:
            c = dict(config(), endpoint=f'ws://127.0.0.1:{local.sockets[0].getsockname()[1]}')
            log = EventLog(tmp_path / 'events.jsonl')
            try:
                with pytest.raises(InvalidStatus) as caught:
                    await wire.transcribe(audio(3), 3, c, 'fixture', log)
            finally:
                log.close()
        assert caught.value.response.status_code == 401  # The worker records this as http_status.
    asyncio.run(check())
    assert full.classify([dict(kind='error', time_seconds=0, http_status=401, error_message='rejected')]) == 'authentication'


def words_between(start, end, step=.5, length=.4, gaps=()):
    """Words every `step` seconds, with the given (start, end) pauses left empty."""
    out, t = [], start
    while t + length <= end:
        if not any(a - length < t < b for a, b in gaps):
            out.append(dict(text='w', start=round(t, 3), end=round(t + length, 3), timing_valid=True))
        t += step
    return out


def inside_word(words, frame):
    return any(w['start'] * 50 < frame < w['end'] * 50 for w in words)


def test_windows_cut_mid_pause_within_30_to_60_seconds():
    words = words_between(0, 180, gaps=[(52, 54), (100, 101)])
    windows = full.rime_windows(words, 9000)
    # The longest pause in 30-60 s is 51.9-54.0 s; in the next range it is 99.9-101.0 s. Cut mid-pause.
    assert windows[0]['offset'] == 0
    assert abs(windows[1]['offset'] - (51.9 + 54.0) / 2 * 50) <= 1
    assert abs(windows[2]['offset'] - (99.9 + 101.0) / 2 * 50) <= 1
    assert windows[0]['cut_pause_seconds'] == pytest.approx(2.1, abs=.01)
    assert windows[1]['cut_pause_seconds'] == pytest.approx(1.1, abs=.01)
    assert sum(w['frames'] for w in windows) == 9000
    assert all(1500 <= w['frames'] <= 3000 for w in windows[:-1]) and windows[-1]['frames'] <= 3000
    assert not any(w['forced_cut'] for w in windows)
    assert not any(inside_word(words, w['offset']) for w in windows[1:])
    assert [w['index'] for w in windows] == list(range(len(windows)))


def test_windows_use_wordless_stretches_and_never_trust_unknown_timing():
    # No words between 20 s and 120 s: cut in the middle of the allowed range.
    words = words_between(0, 20) + words_between(120, 150)
    assert abs(full.rime_windows(words, 7500)[1]['offset'] - 2250) <= 1
    # A long pause next to a word without valid timing is not a known pause.
    words = words_between(0, 150, gaps=[(40, 50)])
    middle = next(i for i, w in enumerate(words) if w['start'] >= 50)
    words.insert(middle, dict(text='?', start=None, end=None, timing_valid=False))
    window = full.rime_windows(words, 7500)[0]
    assert window['cut_pause_seconds'] == pytest.approx(.1) and not window['forced_cut']


def test_windows_force_a_cut_only_without_any_pause_and_keep_a_final_session():
    contiguous = [dict(text='w', start=i * .5, end=(i + 1) * .5, timing_valid=True) for i in range(300)]
    windows = full.rime_windows(contiguous, 7500)
    assert windows[0]['forced_cut'] and windows[0]['frames'] == 3000
    # A pause at the very end never leaves a final session shorter than one second.
    windows = full.rime_windows(words_between(0, 59.5) + [dict(text='w', start=60.3, end=60.4, timing_valid=True)], 3001)
    assert windows[-1]['frames'] >= 50 and sum(w['frames'] for w in windows) == 3001
    assert full.rime_windows([], 3000) == [dict(index=0, offset=0, frames=3000, cut_pause_seconds=None, forced_cut=False)]


@pytest.mark.skipif(not full.PRIVATE_MANIFEST.exists(), reason='private recordings are not in this checkout')
def test_windows_on_the_private_recordings_cut_only_at_real_pauses():
    for clip in json.loads(full.PRIVATE_MANIFEST.read_text())['clips']:
        windows = full.rime_windows(clip['words'], clip['speech_frames'])
        assert sum(w['frames'] for w in windows) == clip['speech_frames']
        assert all(w['frames'] <= full.RIME_SESSION_MAX_FRAMES for w in windows)
        cuts = windows[:-1]
        assert not any(w['forced_cut'] for w in cuts)
        assert min(w['cut_pause_seconds'] for w in cuts) >= .3
        assert not any(inside_word(clip['words'], w['offset']) for w in windows[1:])


class Log:
    origin = 0

    def __init__(self):
        self.events = []

    def now(self):
        return 0

    def emit(self, kind, **fields):
        self.events.append(dict(kind=kind, **fields))


def test_rime_sessions_send_every_source_frame_once_in_order(monkeypatch):
    captures = []

    async def fake(pcm, frames, config, key, log):
        captures.append((pcm, frames))
    monkeypatch.setattr(full.rime, 'transcribe', fake)
    words = words_between(0, 180, gaps=[(52, 54), (100, 101)])
    source = bytes(range(256)) * (9000 * 640 // 256)
    log = Log()
    asyncio.run(full.rime_sessions(source + bytes(32000), 9000, config(), 'fixture', log, words=words))
    windows = full.rime_windows(words, 9000)
    assert [n for _, n in captures] == [w['frames'] for w in windows]
    assert b''.join(pcm[:n * 640] for pcm, n in captures) == source
    assert all(pcm[n * 640:] == bytes(32000) for pcm, n in captures)  # The zero tail is dropped by the adapter.
    header = log.events[0]
    assert header['kind'] == 'longform_sessions' and header['variant'] == wire.PRIVATE_VARIANT
    assert header['windows'] == windows and header['additional_tail_seconds'] == 0
    assert log.events[-1]['kind'] == 'longform_complete'
    with pytest.raises(ValueError, match='pause-session'):
        asyncio.run(full.rime_sessions(source, 9000, dict(config(), private_variant=None), 'fixture', Log(), words=words))


def test_private_recording_runs_as_live_sessions_and_scores_once(tmp_path, monkeypatch):
    monkeypatch.setattr(full, 'RIME_SESSION_MAX_FRAMES', 6)
    monkeypatch.setattr(full, 'RIME_SESSION_MIN_FRAMES', 3)
    monkeypatch.setattr(full, 'RIME_FINAL_SESSION_MIN_FRAMES', 1)
    sessions = []

    async def check():
        async def server(ws):
            received = []
            sessions.append(received)
            await fake_rime(received, every=100)(ws)
        async with serve(server, '127.0.0.1', 0) as local:
            c = dict(config(), endpoint=f'ws://127.0.0.1:{local.sockets[0].getsockname()[1]}')
            log = EventLog(tmp_path / 'raw.jsonl')
            log.emit('clip_start')
            try:
                await full.rime_sessions(audio(10), 10, c, 'fixture', log, words=[])
            finally:
                log.emit('clip_end')
                log.close()
            return c
    c = asyncio.run(check())
    events = read_events(tmp_path / 'raw.jsonl')
    clip = dict(clip_id='private', cohort='private', speech_frames=10, reference='final final', words=[])
    result, snapshots, frames = full.session_assessment(events, clip, c)
    assert result['transcript_complete'] and result['valid'] and result['model_verified']
    assert result['transcript'] == 'final final' and result['context_resets'] == 1
    assert result['transport_variant'] == wire.PRIVATE_VARIANT
    assert set(frames) == set(range(10)) and len(result['pacing']['transition_gaps_seconds']) == 1
    # Each session got its own window of speech, then end, and never the zero tail.
    expected = [w['frames'] for w in full.rime_windows([], 10)]
    assert len(expected) == 2 and sum(expected) == 10
    assert [sum('audio' in m for m in s if isinstance(m, dict)) for s in sessions] == expected
    assert all([list(m) for m in s[-1:]] == [['end']] for s in sessions)
    assert snapshots[-1]['text'] == 'final final'
    row = full.evaluate(events, clip, c, 1, 'hash')
    assert row['valid'] and row['word_errors']['hypothesis_normalized'] == 'final final'
    assert row['private_latency'] is not None
    # Removing the completion marker means the recording was not fully covered.
    assert not full.session_assessment([e for e in events if e['kind'] != 'longform_complete'], clip, c)[0]['transcript_complete']


def test_transport_routing():
    c = config()
    public = dict(clip_id='p', cohort='public')
    private = dict(clip_id='q', cohort='private', words=[{'text': 'w', 'start': 0, 'end': .1, 'timing_valid': True}])
    assert full.transport_for({}, public, c) == (full.transcribe, c)
    transport, used = full.transport_for({}, private, c)
    assert transport.func is full.rime_sessions and transport.keywords == {'words': private['words']} and used is c
    assert full.transport_for({'private_variant': wire.PRIVATE_VARIANT}, private, c)[0].func is full.rime_sessions
    with pytest.raises(ValueError):
        full.transport_for({'private_variant': 'gradium-consecutive-v1'}, private, c)
    gradium = json.loads(model_config('gradium-default').read_text())
    transport, used = full.transport_for({'private_variant': 'gradium-consecutive-v2'}, private, gradium)
    assert transport is full.gradium_sessions and used['private_variant'] == 'gradium-consecutive-v2'
    assert full.transport_for({}, private, gradium) == (full.transcribe, gradium)
    assert 'rime-stt' in full.FULL_MODELS


def test_final_snapshots_follow_done_only():
    snapshots = full.final_snapshots(session_events(), config())
    # Partials never change the final text; only done does.
    assert snapshots == [dict(text='', time_seconds=.05), dict(text='Hi. I am calling.', time_seconds=1.6)]


def test_turn_profile_measures_first_text_and_final_after_speech_end(tmp_path):
    profile = validate_profile(json.loads(PROFILE.read_text()))
    assert profile['turn_finalization_class'] == 'controlled'
    assert {k: v for k, v in profile.items() if k not in ('measurement_profile', 'profile_version',
            'turn_finalization_class', 'base_config')} == config()
    _, events = run_loopback(tmp_path, profile, 7)
    result = assess(events, profile)
    timing = result['turn_timing']
    assert result['valid'] and timing['ttft_status'] == 'measured' and timing['ttfs_status'] == 'measured'
    assert timing['first_text'] == 'word' and timing['first_text_kind'] == 'partial'
    assert timing['final_text'] == 'word word final' and timing['ttfs_ms'] >= 0
    assert timing['finalization_class'] == 'controlled' and not timing['provisional_final']
