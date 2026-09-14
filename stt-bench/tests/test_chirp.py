"""Chirp tests use a local gRPC server; no Google requests or real credentials."""
import asyncio
import json
from pathlib import Path

import grpc
import pytest
from google.cloud.speech_v2.types import cloud_speech as s
from google.cloud.speech_v2.services.speech.transports.grpc_asyncio import SpeechGrpcAsyncIOTransport

from stt_bench import chirp
from stt_bench.credentials import command_environment, credential
from stt_bench.providers import reduce_events, transcript_at, validate
from stt_bench.streaming import EventLog, read_events


def config(version=3):
    return json.loads(Path(f'config/models/google-chirp-{version}.json').read_text())


def event(kind, at, **fields):
    return dict(kind=kind, time_seconds=at, **fields)


def result(text, end, final=True):
    return dict(alternatives=[dict(transcript=text)], result_end_offset=f'{end}s', is_final=final)


def completed_events(results):
    c = config()
    return [event('speech_end', 1), *results, event('audio_complete', 2),
        event('close_stream_requested', 2.01),
        event('model_accepted', 2.1, model=c['model'], verification='requested_model_rpc_succeeded'),
        event('provider_terminal', 2.1, basis=c['completion_basis'])]


def test_deadlines_replacements_multiple_results_and_repeated_words():
    events = completed_events([
        event('provider_message', 1.1, message={'results': [result('hello', .5, False), result('w', 1, False)]}),
        event('provider_message', 1.3, message={'results': [result('hello', .5), result('hello', 1, False)]}),
        event('provider_message', 1.6, message={'results': [result('hello', 1)]}),
        event('provider_message', 1.7, message={'results': [result('hello', 1)]})])
    c = config()
    assert transcript_at(events, 1, c)['text'] == ''
    assert transcript_at(events, 1.25, c)['text'] == 'hello w'
    assert transcript_at(events, 1.5, c)['text'] == 'hello hello'
    r = reduce_events(events, c)
    assert r['transcript'] == 'hello hello' and r['transcript_complete']
    assert r['final_transcript_received_seconds'] == 1.6
    assert r['completion_latency_ms'] == pytest.approx(1100)
    assert r['finalize_latency_ms'] is None and r['finalize_latency_status'] == 'unsupported'


@pytest.mark.parametrize('bad', ['changed_final', 'backward_final', 'missing_offset', 'partial_at_eof', 'error', 'no_audio', 'no_identity'])
def test_incomplete_or_ambiguous_evidence_is_excluded(bad):
    results = [result('hello', 1)]
    extra = {'changed_final': result('different', 1), 'backward_final': result('old', .5),
             'missing_offset': {'is_final': True, 'alternatives': [{'transcript': 'x'}]},
             'partial_at_eof': result('unfinished', 2, False)}
    events = completed_events([event('provider_message', 1.1, message={'results': results})])
    if bad in extra:
        events.append(event('provider_message', 1.2, message={'results': [extra[bad]]}))
    if bad == 'error': events.append(event('error', 2.2))
    if bad == 'no_audio': events = [e for e in events if e['kind'] != 'audio_complete']
    if bad == 'no_identity': events = [e for e in events if e['kind'] != 'model_accepted']
    assert not reduce_events(events, config())['transcript_complete']


def test_empty_success_is_valid():
    r = reduce_events(completed_events([]), config())
    assert r['transcript_complete'] and r['transcript'] == ''


def test_credentials_reject_api_keys_without_logging_them(tmp_path):
    value = 'AIza-secret-test'
    with pytest.raises(ValueError) as error:
        command_environment('google', environ={'GCP': value}, env_file=None)
    assert value not in str(error.value)
    assert 'service account JSON' in str(error.value)
    path = tmp_path / 'creds.json'
    path.write_text('{"type":"service_account"}')
    assert credential('google', {'GOOGLE_APPLICATION_CREDENTIALS': str(path)}, None)[0] == path.read_text()
    assert credential('google', {'VERTEX_CREDS':'{"type":"override"}', 'GCP':value}, None)[1] == 'VERTEX_CREDS'


def test_project_is_explicit_not_taken_from_credential():
    c = config()
    assert configuration_project(c) == 'gen-lang-client-0085530523'
    for field,value in [('project_id',''), ('location','europe-west4'), ('finalization','manual_at_speech_end')]:
        invalid = {**c, field:value}
        with pytest.raises(ValueError): validate(invalid)


def configuration_project(c):
    return chirp.configuration(c).recognizer.split('/')[1]


async def fast_audio(pcm, speech_frames, send, finalize, log):
    for index in range(speech_frames + 50):
        await send(pcm[index*640:(index+1)*640])
        log.emit('audio_sent', index=index, bytes=640)
        if index == speech_frames-1:
            t0 = log.now(); log.emit('speech_end', at=t0); await finalize(t0)
        await asyncio.sleep(.001)
    log.emit('audio_complete')


@pytest.mark.parametrize('version', [2, 3])
@pytest.mark.parametrize('mode', ['success','empty','denied','early_eof','timeout'])
def test_actual_grpc_transport_write_and_half_close(version, mode, tmp_path, monkeypatch):
    monkeypatch.setattr(chirp, 'stream_audio', fast_audio)
    c = config(version); c['close_timeout_seconds'] = .15
    received = []
    async def recognize(requests, context):
        first = await anext(requests)
        assert first.recognizer.endswith(f'/locations/{c["location"]}/recognizers/_')
        assert first.streaming_config.config.model == c['model']
        assert not first.audio
        if mode == 'denied':
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, 'fixture denial')
        if mode == 'early_eof': return
        async for request in requests:
            received.append(request.audio)
            if len(received) == 1 and mode != 'empty':
                yield s.StreamingRecognizeResponse(results=[s.StreamingRecognitionResult(
                    alternatives=[s.SpeechRecognitionAlternative(transcript='interim')],
                    result_end_offset={'seconds':0, 'nanos':20000000})])
        if mode == 'timeout': await asyncio.sleep(1)
        if mode != 'empty':
            yield s.StreamingRecognizeResponse(results=[s.StreamingRecognitionResult(
                alternatives=[s.SpeechRecognitionAlternative(transcript='hello world')],
                is_final=True, result_end_offset={'seconds':1})])
    async def scenario():
        server = grpc.aio.server()
        handler = grpc.stream_stream_rpc_method_handler(recognize,
            request_deserializer=s.StreamingRecognizeRequest.deserialize,
            response_serializer=s.StreamingRecognizeResponse.serialize)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            'google.cloud.speech.v2.Speech', {'StreamingRecognize':handler}),))
        port = server.add_insecure_port('127.0.0.1:0')
        await server.start()
        channel = grpc.aio.insecure_channel(f'127.0.0.1:{port}')
        transport = SpeechGrpcAsyncIOTransport(channel=channel)
        log = EventLog(tmp_path / 'events.jsonl')
        try:
            call = transport.streaming_recognize(timeout=3)
            if mode in ('success','empty'):
                await chirp.exchange(call, bytes(640*60), 10, c, log)
            else:
                with pytest.raises((grpc.RpcError, chirp.ProviderError, TimeoutError, asyncio.InvalidStateError)):
                    await chirp.exchange(call, bytes(640*60), 10, c, log)
                log.emit('error', error_type='FixtureFailure')
        finally:
            log.close(); await transport.close(); await server.stop(None)
    asyncio.run(scenario())
    events = read_events(tmp_path / 'events.jsonl')
    r = reduce_events(events, c)
    assert r['transcript_complete'] == (mode in ('success','empty'))
    if mode in ('success','empty'):
        assert len(received) == 60 and all(len(frame)==640 for frame in received)
        assert r['transcript'] == ('hello world' if mode=='success' else '')
        assert r['finalize_latency_ms'] is None
