"""PolyAI Owl speaks OpenAI's Realtime transcription protocol under its own host, model and key.

Loopback servers and fixture credentials only, never the provider host.
"""
import asyncio
import json
from pathlib import Path
import pytest
from websockets.asyncio.server import serve
from stt_bench import full_benchmark, provider_protocol as wire
from stt_bench.catalog import model_config
from stt_bench.credentials import command_environment, credential
from stt_bench.providers import reduce_events, transcript_at, validate
from stt_bench.streaming import EventLog, read_events
from stt_bench.turn_batch import OPT_IN
from stt_bench.turn_metrics import transcript_timeline
from stt_bench.turn_runner import validate_profile

MODEL_ID = 'polyai-dialog-owl-1-0'


def config(name=MODEL_ID):
    return json.loads(model_config(name).read_text())


def event(message, at):
    return {'kind': 'provider_message', 'message': message, 'time_seconds': at}


async def fast_audio(pcm, speech_frames, send, finalize, log, *, sample_rate=16000):
    frame = sample_rate // 50 * 2
    for i in range(speech_frames + 50):
        await send(pcm[i * frame:(i + 1) * frame])
        log.emit('audio_sent', bytes=frame, sample_rate=sample_rate)
        if i == speech_frames - 1:
            t0 = log.now(); log.emit('speech_end', at=t0); await finalize(t0)
        await asyncio.sleep(0)
    log.emit('audio_complete')


def test_owl_opens_a_transcription_session_with_its_own_bearer_key_and_model(tmp_path, monkeypatch):
    c = config()
    seen = {'kinds': []}

    async def owl(ws):
        seen.update(path=ws.request.path, authorization=ws.request.headers.get('Authorization'))
        async for raw in ws:
            message = json.loads(raw)
            seen['kinds'].append(message['type'])
            if message['type'] == 'session.update':
                seen['session'] = message['session']
                await ws.send(json.dumps({'type': 'session.updated', 'session': message['session']}))
            if message['type'] == 'input_audio_buffer.commit':
                await ws.send(json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'a', 'previous_item_id': None}))
                await ws.send(json.dumps({'type': 'conversation.item.input_audio_transcription.completed',
                                          'item_id': 'a', 'transcript': 'hello world'}))

    monkeypatch.setattr(wire, 'stream_audio', fast_audio)

    async def scenario():
        async with serve(owl, '127.0.0.1', 0) as server:
            local = dict(c, endpoint=f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime')
            log = EventLog(tmp_path / 'events.jsonl'); log.emit('clip_start')
            try:
                await wire.transcribe(bytes(c['sample_rate'] // 50 * 2 * 60), 10, local, 'fixture-owl-key', log)
            finally:
                log.emit('clip_end'); log.close()

    asyncio.run(scenario())
    assert seen['path'] == '/v1/realtime?intent=transcription'
    assert seen['authorization'] == 'Bearer fixture-owl-key'
    assert seen['session']['type'] == 'transcription'
    assert seen['session']['audio']['input']['transcription']['model'] == 'dialog-owl-1.0'
    assert seen['session']['audio']['input']['format'] == {'type': 'audio/pcm', 'rate': 24000}
    assert seen['kinds'][0] == 'session.update' and seen['kinds'].count('input_audio_buffer.commit') == 1
    reduced = reduce_events(read_events(tmp_path / 'events.jsonl'), c)
    assert reduced['transcript'] == 'hello world' and reduced['transcript_complete'] and reduced['model_verified']
    assert 'fixture-owl-key' not in (tmp_path / 'events.jsonl').read_text()


def test_owl_and_openai_never_share_a_key_or_a_host(tmp_path):
    env = tmp_path / '.env'
    env.write_text('OPENAI_API_KEY=openai-key\nOpenAI=openai-alias\n')
    assert credential('polyai', {}, env) == ('', None)
    with pytest.raises(ValueError, match='POLY_AI_OWL_STT_API_KEY'):
        command_environment('polyai', environ={}, env_file=env)
    env.write_text('OPENAI_API_KEY=openai-key\nPOLY_AI_OWL_STT_API_KEY=owl-key\n')
    assert command_environment('polyai', environ={}, env_file=env) == {'POLY_AI_OWL_STT_API_KEY': 'owl-key'}
    assert credential('openai', {}, env) == ('openai-key', 'OPENAI_API_KEY')
    owl, openai = config(), config('openai-gpt-4o-transcribe')
    for c, other_host in ((owl, openai['endpoint']), (openai, owl['endpoint'])):
        with pytest.raises(ValueError, match='endpoint'):
            validate(dict(c, endpoint=other_host))
    with pytest.raises(ValueError, match='provider/model'):
        validate(dict(owl, model=openai['model']))


def test_early_partials_give_way_to_the_completed_transcript():
    """Owl's live shape: a few append-only deltas early in the utterance, none until the commit, then
    a completed item that can differ from the partial text. The completed item is the transcript."""
    c = config()
    events = [dict(kind='model_accepted', model=c['model'], time_seconds=.8),
              event({'type': 'session.updated', 'session': {'audio': {'input': {'transcription': {'model': c['model']}}}}}, .8),
              event({'type': 'conversation.item.input_audio_transcription.delta', 'item_id': 'i', 'delta': 'Hello'}, 2.6),
              event({'type': 'conversation.item.input_audio_transcription.delta', 'item_id': 'i', 'delta': ' word'}, 2.9),
              dict(kind='speech_end', time_seconds=8.7),
              dict(kind='finalize_requested', time_seconds=8.7),
              event({'type': 'input_audio_buffer.committed', 'item_id': 'i', 'previous_item_id': None}, 8.9),
              event({'type': 'conversation.item.input_audio_transcription.completed', 'item_id': 'i',
                     'transcript': 'Hello wide world.'}, 8.95),
              dict(kind='audio_complete', time_seconds=9.7),
              dict(kind='close_stream_requested', time_seconds=9.7),
              dict(kind='provider_terminal', time_seconds=9.7)]
    assert transcript_at(events, 5, c)['partial_text'] == 'Hello word'
    final = transcript_at(events, 10, c)
    assert final['final_text'] == 'Hello wide world.' and not final['partial_text']
    reduced = reduce_events(events, c)
    assert reduced['transcript'] == 'Hello wide world.'
    assert reduced['transcript_complete'] and reduced['model_verified'] and not reduced['completion_timed_out']
    assert reduced['finalize_latency_ms'] == pytest.approx(250)
    # Turn timing reads first text of any kind, so it starts at the early partial, as for every row.
    first = next(s for s in transcript_timeline(events, c) if s['text'])
    assert (first['time_seconds'], first['text']) == (2.6, 'Hello')


def test_turn_profile_is_the_model_config_with_controlled_turn_fields():
    profile = json.loads(Path(f'config/profiles/private-turns-v1/{MODEL_ID}.json').read_text())
    assert {k: v for k, v in profile.items() if k in config()} == config()
    assert profile['turn_finalization_class'] == 'controlled'
    assert profile['base_config'] == f'config/models/{MODEL_ID}.json'
    validate_profile(profile)
    assert MODEL_ID in OPT_IN and MODEL_ID in full_benchmark.FULL_MODELS
