"""Protocol tests use loopback servers and fixture credentials, never provider hosts."""
import asyncio
import base64
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from stt_bench.catalog import MODELS, model_config
from stt_bench import provider_protocol as wire
from stt_bench.providers import validate, is_nova, reduce_events, transcript_at
from stt_bench.credentials import credential, command_environment, redact
from stt_bench.audio_formats import derivative, resample_24k
from stt_bench.streaming import EventLog, read_events, pacing_metrics
from stt_bench.data import sha256


def config(name):
    return json.loads(model_config(name).read_text())


@pytest.mark.parametrize('name', list(MODELS))
def test_all_models_have_explicit_valid_contract(name):
    c = config(name)
    validate(c)
    c['endpoint'] = 'wss://example.org/steal'
    with pytest.raises(ValueError, match='endpoint'):
        validate(c)


def test_credentials_alias_precedence_isolation_and_no_interpolation(tmp_path):
    env = tmp_path / '.env'
    env.write_text('OpenAI=file-key\nGoogle=google-key\nElevenLabs=eleven-key\nCartesia=${OpenAI}\n')
    assert credential('openai', {'OpenAI': 'process-key'}, env) == ('process-key', 'OpenAI')
    assert credential('openai', {'OpenAI': 'alias', 'OPENAI_API_KEY': 'canonical'}, env)[0] == 'canonical'
    assert credential('cartesia', {}, env)[0] == '${OpenAI}'
    assert command_environment('gemini', environ={}, env_file=env) == {'GEMINI_API_KEY': 'google-key'}
    assert 'file-key' not in json.dumps(redact({'url': 'key=file-key', 'token': 'anything'}, 'file-key'))


def test_resampling_duration_boundary_tail_hash_and_tamper(tmp_path):
    speech = (np.sin(np.arange(3200) * 2*np.pi*440/16000)*20000).astype('int16')
    pcm = np.concatenate([speech, np.zeros(16000, dtype='int16')])
    source = tmp_path / 'source.wav'; sf.write(source, pcm, 16000, subtype='PCM_16')
    before = sha256(source)
    converted, record = derivative(source, {'speech_frames': 10}, tmp_path / 'cache')
    out = np.frombuffer(converted, dtype='<i2')
    assert len(out) == 28800 and not np.any(out[-24000:])
    assert record['duration_seconds'] == len(pcm)/16000
    assert sha256(source) == before
    assert derivative(source, {'speech_frames': 10}, tmp_path / 'cache') == (converted, record)
    assert np.array_equal(out, resample_24k(pcm, 10))
    # The resampler preserves pitch and amplitude rather than merely changing a header.
    assert abs(np.max(out[:4800]) - 20000) < 200
    peak = np.argmax(abs(np.fft.rfft(out[:4800]))) * 24000 / 4800
    assert peak == 440
    path = next((tmp_path / 'cache').glob('*.wav')); path.write_bytes(path.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='changed'):
        derivative(source, {'speech_frames': 10}, tmp_path / 'cache')


@pytest.mark.parametrize('rate', [16000, 24000])
def test_rate_aware_pacing_accepts_only_consistent_20ms_frames(rate):
    events = [dict(kind='audio_sent', index=i, time_seconds=i*.02, ideal_seconds=i*.02,
                   sample_rate=rate, bytes=rate//50*2, phase='speech' if i < 10 else 'silence') for i in range(60)]
    assert pacing_metrics(events)['valid']
    events[20]['sample_rate'] = 8000
    assert not pacing_metrics(events)['valid']


def event(m, at):
    return {'kind': 'provider_message', 'message': m, 'time_seconds': at}


def test_openai_out_of_order_items_deltas_and_duplicates():
    c = config('openai-gpt-realtime-whisper')
    events = [event({'type': 'input_audio_buffer.committed', 'item_id': 'a', 'previous_item_id': None}, 0),
              event({'type': 'input_audio_buffer.committed', 'item_id': 'b', 'previous_item_id': 'a'}, .1),
              event({'type': 'conversation.item.input_audio_transcription.completed', 'item_id': 'b', 'transcript': 'world'}, .2),
              event({'type': 'conversation.item.input_audio_transcription.delta', 'item_id': 'a', 'delta': 'hel', 'event_id': '1'}, .3),
              event({'type': 'conversation.item.input_audio_transcription.delta', 'item_id': 'a', 'delta': 'hel', 'event_id': '1'}, .4),
              event({'type': 'conversation.item.input_audio_transcription.delta', 'item_id': 'a', 'delta': 'lo', 'event_id': '2'}, .5),
              event({'type': 'conversation.item.input_audio_transcription.completed', 'item_id': 'a', 'transcript': 'hello'}, .6)]
    assert transcript_at(events, .5, c)['text'] == 'hello world'
    assert transcript_at(events, 1, c)['final_text'] == 'hello world'
    assert transcript_at(events, .25, c)['text'] == 'world'


def test_flux_cumulative_turns_do_not_repeat_words_and_preserve_turn_order():
    c = config('deepgram-flux-en')
    events = [event({'type': 'TurnInfo', 'turn_index': 0, 'event': 'Update', 'transcript': 'yes'}, .1),
              event({'type': 'TurnInfo', 'turn_index': 0, 'event': 'EndOfTurn', 'transcript': 'yes yes'}, .2),
              event({'type': 'TurnInfo', 'turn_index': 1, 'event': 'EndOfTurn', 'transcript': 'yes'}, .3)]
    assert transcript_at(events, .3, c)['text'] == 'yes yes yes'


def test_cartesia_preserves_delta_spacing_and_deduplicates_word_timestamps():
    c = config('cartesia-ink-2')
    a = {'type': 'transcript', 'is_final': True, 'text': 'hello', 'words': [{'start': 0, 'end': .1}]}
    b = {'type': 'transcript', 'is_final': True, 'text': ' world', 'words': [{'start': .1, 'end': .2}]}
    assert transcript_at([event(a, .1), event(a, .2), event(b, .3)], 1, c)['text'] == 'hello world'


def test_speechmatics_revisions_final_order_and_gemini_partial_replacement():
    c = config('speechmatics-standard')
    def msg(text, start, end, final):
        return {'message': 'AddTranscript' if final else 'AddPartialTranscript',
                'metadata': {'transcript': text, 'start_time': start, 'end_time': end}}
    events = [event(msg('hello', 0, 1, False), .1), event(msg('hello world', 0, 2, False), .2),
              event(msg('hello world', 0, 2, True), .3), event(msg('again', 2, 3, True), .4)]
    assert transcript_at(events, .2, c)['text'] == 'hello world'
    assert transcript_at(events, 1, c)['text'] == 'hello world again'
    c = config('gemini-3.5-transcribe-live')
    events = [event({'serverContent': {'interimInputTranscription': {'text': 'wrong'}}}, .1),
              event({'serverContent': {'interimInputTranscription': {'text': 'right'}}}, .2),
              event({'serverContent': {'inputTranscription': {'text': 'right'}}}, .3)]
    assert transcript_at(events, .2, c)['text'] == 'right'
    assert transcript_at(events, .4, c)['final_text'] == 'right'


async def fast_audio(pcm, speech_frames, send, finalize, log, *, sample_rate=16000):
    frame = sample_rate//50*2
    for i in range(speech_frames+50):
        await send(pcm[i*frame:(i+1)*frame])
        log.emit('audio_sent', bytes=frame, sample_rate=sample_rate)
        if i == speech_frames-1:
            t0 = log.now(); log.emit('speech_end', at=t0); await finalize(t0)
        await asyncio.sleep(0)
    log.emit('audio_complete')


async def fake_provider(ws, c, *, text='hello world', fault=None):
    p = c['provider']
    async def send(m): await ws.send(json.dumps(m))
    if fault == 'auth':
        await send({'type': 'error', 'error': {'message': 'fixture-secret rejected'}})
        await ws.close(); return
    if p == 'deepgram' and not is_nova(c): await send({'type': 'Connected'})
    if p == 'elevenlabs': await send({'message_type': 'session_started', 'config': {'model_id': c['model']}})
    async for raw in ws:
        if fault == 'disconnect': await ws.close(); return
        if isinstance(raw, bytes): continue
        if raw in ('finalize', 'close'): m = {'command': raw}
        else: m = json.loads(raw)
        kind = m.get('type', m.get('message', ''))
        if kind == 'session.update': await send({'type': 'session.updated', 'session': m['session']})
        if kind == 'StartRecognition': await send({'message': 'RecognitionStarted'})
        if 'setup' in m: await send({'setupComplete': {}})
        if fault == 'timeout': continue
        if kind == 'input_audio_buffer.commit':
            await send({'type': 'input_audio_buffer.committed', 'item_id': 'a', 'previous_item_id': None})
            await send({'type': 'conversation.item.input_audio_transcription.completed', 'item_id': 'a', 'transcript': text})
        if m.get('commit'):
            await send({'message_type': 'committed_transcript_with_timestamps', 'text': text})
        if kind == 'ForceEndTurn': await send({'type': 'TurnInfo', 'turn_index': 0, 'event': 'EndOfTurn', 'transcript': text})
        if m.get('command') == 'finalize':
            await send({'type': 'transcript', 'is_final': True, 'text': text})
            await send({'type': 'flush_done'})
        if m.get('command') == 'close':
            await send({'type': 'done'}); await ws.close(); return
        if kind == 'EndOfStream':
            await send({'message': 'AddTranscript', 'metadata': {'transcript': text, 'start_time': 0, 'end_time': .2}})
            await send({'message': 'EndOfTranscript'}); await ws.close(); return
        if 'activityEnd' in m.get('realtimeInput', {}):
            await send({'serverContent': {'inputTranscription': {'text': text}}})
            await send({'serverContent': {'generationComplete': True}})
        if kind == 'Finalize':
            await send({'type': 'Results', 'is_final': True, 'from_finalize': True, 'start': 0, 'duration': .2,
                'channel': {'alternatives': [{'transcript': text}]},
                'metadata': {'model_info': {'version': c['version']}, 'model_uuid': c['expected_model_uuid']}})
        if kind == 'CloseStream':
            await send({'type': 'Metadata', 'model_info': {c['expected_model_uuid']: {'version': c['version']}}})
            await ws.close(); return


@pytest.mark.parametrize('name', [n for n in MODELS if config(n)['provider'] not in ('google', 'soniox', 'smallest', 'sarvam', 'inworld', 'gradium', 'reson8', 'assemblyai')])
@pytest.mark.parametrize('text', ['hello world', ''])
def test_local_websocket_success_and_empty_completed_transcript(name, text, tmp_path, monkeypatch):
    c = config(name)
    from stt_bench import deepgram
    monkeypatch.setattr(wire, 'stream_audio', fast_audio)
    monkeypatch.setattr(deepgram, 'stream_audio', fast_audio)
    async def scenario():
        async with serve(lambda ws: fake_provider(ws, c, text=text), '127.0.0.1', 0) as server:
            log = EventLog(tmp_path / 'events.jsonl'); log.emit('clip_start')
            rate = c.get('sample_rate', 16000)
            async with connect(f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}') as ws:
                try:
                    await (deepgram if is_nova(c) else wire).exchange(ws, bytes(rate//50*2*60), 10, c, log)
                finally:
                    log.emit('clip_end'); log.close()
        reduced = reduce_events(read_events(tmp_path / 'events.jsonl'), c)
        assert reduced['transcript_complete'] and reduced['model_verified']
        assert reduced['transcript'] == text
        if c['provider'] == 'speechmatics': assert reduced['finalize_latency_ms'] is None
    asyncio.run(scenario())


@pytest.mark.parametrize('name', [n for n in MODELS if 'nova' not in n and config(n)['provider'] not in ('google', 'soniox', 'smallest', 'sarvam', 'inworld', 'gradium', 'reson8', 'assemblyai')])
@pytest.mark.parametrize('fault', ['auth', 'disconnect', 'timeout'])
def test_local_websocket_failure_never_becomes_success(name, fault, tmp_path, monkeypatch):
    c = config(name); c.update(finalize_timeout_seconds=.05, close_timeout_seconds=.05, ready_timeout_seconds=.05)
    monkeypatch.setattr(wire, 'stream_audio', fast_audio)
    async def scenario():
        async with serve(lambda ws: fake_provider(ws, c, fault=fault), '127.0.0.1', 0) as server:
            log = EventLog(tmp_path / 'events.jsonl'); log.emit('clip_start')
            async with connect(f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}') as ws:
                with pytest.raises((wire.ProviderError, TimeoutError, ConnectionClosed)):
                    await wire.exchange(ws, bytes(c['sample_rate']//50*2*60), 10, c, log, secret='fixture-secret')
                log.emit('error', error_type='FixtureFailure')
                log.emit('clip_end'); log.close()
        raw = (tmp_path / 'events.jsonl').read_text()
        assert 'fixture-secret' not in raw
        assert not reduce_events(read_events(tmp_path / 'events.jsonl'), c)['transcript_complete']
    asyncio.run(scenario())


def test_model_mismatch_cannot_be_verified():
    c = config('openai-gpt-4o-transcribe')
    events = [dict(kind='model_accepted', model=c['model'], time_seconds=0),
              event({'type': 'session.updated', 'session': {'audio': {'input': {'transcription': {'model': 'wrong'}}}}}, .1)]
    assert not reduce_events(events, c)['model_verified']
