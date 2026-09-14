"""Cloud Speech V2: paced gRPC writes and receipt-time transcript reconstruction.

No SDK retry, request queue or invented force-finalize acknowledgment. Half-close
after all 50 tail frames; successful RPC EOF is the completion diagnostic.
"""
import asyncio
import contextlib
import json
import re

from .provider_protocol import ProviderError
from .streaming import stream_audio


def credentials_from_json(value):
    from google.oauth2 import service_account
    try:
        info = json.loads(value)
        if (info.get('type') != 'service_account' or
                info.get('token_uri') != 'https://oauth2.googleapis.com/token' or
                info.get('universe_domain', 'googleapis.com') != 'googleapis.com'):
            raise ValueError()
        return service_account.Credentials.from_service_account_info(
            info, scopes=['https://www.googleapis.com/auth/cloud-platform'])
    except (ValueError, TypeError, AttributeError, KeyError):
        raise ValueError('Chirp requires valid service account JSON credentials, not an API key') from None


def validate(c):
    if not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]', c.get('project_id', '')):
        raise ValueError('Explicit Google Cloud project_id required')
    region = {'chirp_2': 'us-central1', 'chirp_3': 'us'}.get(c['model'])
    if c.get('location') != region or c['endpoint'] != f'https://{region}-speech.googleapis.com':
        raise ValueError('Chirp model/location/endpoint mismatch')
    if (c.get('language') != 'en-US' or c.get('channels') != 1 or
            c.get('encoding') != 'pcm_s16le' or c.get('frame_ms') != 20 or
            c.get('finalization') != 'stream_end_after_tail' or
            c.get('completion_basis') != 'grpc_ok_after_half_close_after_tail'):
        raise ValueError('Unsupported Chirp audio or completion contract')


def configuration(c):
    from google.cloud.speech_v2.types import cloud_speech as s
    return s.StreamingRecognizeRequest(
        recognizer=f'projects/{c["project_id"]}/locations/{c["location"]}/recognizers/_',
        streaming_config=s.StreamingRecognitionConfig(
            config=s.RecognitionConfig(model=c['model'], language_codes=[c['language']],
                explicit_decoding_config=s.ExplicitDecodingConfig(
                    encoding=s.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000, audio_channel_count=1)),
            streaming_features=s.StreamingRecognitionFeatures(interim_results=True)))


async def exchange(call, pcm, speech_frames, config, log):
    from google.cloud.speech_v2.types import cloud_speech as s
    closing = False

    async def receive():
        async for response in call:
            at = log.now()
            log.emit('provider_message', at=at,
                     message=s.StreamingRecognizeResponse.to_dict(response))
        if not closing:
            raise ProviderError('Google ended the RPC before the silence tail and half-close')
        log.emit('model_accepted', model=config['model'], verification='requested_model_rpc_succeeded')
        log.emit('provider_terminal', basis=config['completion_basis'])

    async def send(frame):
        # Await the actual gRPC write, not enqueueing in an application buffer.
        await call.write(s.StreamingRecognizeRequest(audio=frame))

    async def speech_end(_):
        pass  # Google has no force-finalize command at this boundary.

    receiver = sender = None
    try:
        await asyncio.wait_for(call.write(configuration(config)), config['ready_timeout_seconds'])
        receiver = asyncio.create_task(receive())
        sender = asyncio.create_task(stream_audio(pcm, speech_frames, send, speech_end, log))
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        if receiver in done:
            await receiver
            raise ProviderError('Google stream ended before all audio was sent')
        await sender
        log.emit('close_stream_requested')
        closing = True
        await asyncio.wait_for(call.done_writing(), config['close_timeout_seconds'])
        await asyncio.wait_for(receiver, config['close_timeout_seconds'])
    except TimeoutError:
        log.emit('close_stream_timeout')
        raise
    finally:
        call.cancel()
        for task in (sender, receiver):
            if task is not None and not task.done():
                task.cancel()
        for task in (sender, receiver):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task


async def transcribe(pcm, speech_frames, config, key, log):
    from google.cloud import speech_v2
    client = speech_v2.SpeechAsyncClient(credentials=credentials_from_json(key),
        client_options={'api_endpoint': config['endpoint'].removeprefix('https://')})
    try:
        log.emit('connection_requested', url=config['endpoint'])
        await asyncio.wait_for(client.transport.grpc_channel.channel_ready(), config['ready_timeout_seconds'])
        log.emit('connection_open')
        request = configuration(config)
        # Raw generated transport callable retains explicit write completion and
        # bypasses GAPIC's retry/request-iterator wrapper.
        call = client.transport.streaming_recognize(
            metadata=(('x-goog-request-params', 'recognizer=' + request.recognizer),),
            timeout=len(pcm) / 32000 + config['ready_timeout_seconds'] + config['close_timeout_seconds'] + 5)
        await exchange(call, pcm, speech_frames, config, log)
    finally:
        await client.transport.close()


def replay(events, config, cutoff=float('inf')):
    finals, partials = {}, []
    unsupported = errored = accepted = audio_complete = closing = False
    t0 = terminal_at = last_text = first_partial = None
    timed_out = False
    for e in sorted(events, key=lambda e: e['time_seconds']):
        at, kind = e['time_seconds'], e['kind']
        if at > cutoff:
            break
        if kind == 'speech_end':
            t0 = at
        elif kind == 'audio_complete':
            audio_complete = True
        elif kind == 'close_stream_requested':
            closing = audio_complete
        elif kind == 'error':
            errored = True
        elif kind == 'close_stream_timeout':
            timed_out = True
        elif kind == 'model_accepted':
            accepted = e.get('model') == config['model'] and e.get('verification') == 'requested_model_rpc_succeeded'
        elif kind == 'provider_terminal' and closing and e.get('basis') == config['completion_basis']:
            terminal_at = at
        elif kind == 'provider_message':
            results = e['message'].get('results', [])
            if not results:
                continue
            partials = []
            for r in results:
                alternatives = r.get('alternatives', [])
                text = alternatives[0].get('transcript', '') if alternatives else ''
                if r.get('is_final'):
                    try:
                        end = float(r['result_end_offset'].removesuffix('s'))
                        if not 0 <= end < float('inf'):
                            raise ValueError()
                    except (KeyError, ValueError, TypeError, AttributeError):
                        unsupported = True
                        continue
                    if end in finals:
                        unsupported |= finals[end] != text
                    elif finals and end < max(finals):
                        unsupported = True
                    else:
                        finals[end] = text
                        if text:
                            last_text = at
                else:
                    partials.append(text)
            partial = ' '.join(partials).strip()
            if first_partial is None and t0 is not None and at >= t0 and partial:
                first_partial = dict(text=partial, received_seconds=at, latency_ms=(at-t0)*1000)
    final = ' '.join(finals.values()).strip()
    partial = ' '.join(partials).strip()
    snap = dict(text=' '.join(filter(None, [final, partial])), final_text=final,
        partial_text=partial, provisional=bool(partial),
        reconstruction_status='unsupported_order_or_overlap' if unsupported else 'supported')
    complete = bool(audio_complete and terminal_at is not None and accepted and
                    not (errored or unsupported or timed_out or partial))
    reduced = dict(transcript=final, transcript_complete=complete, model_verified=accepted,
        model_verification_basis='requested_model_rpc_succeeded; immutable version unavailable',
        model_versions=[], model_uuids=[], t0_seconds=t0, first_partial_after_t0=first_partial,
        final_transcript_received_seconds=last_text, finalize_ack_received_seconds=None,
        finalize_latency_ms=None, finalize_latency_status='unsupported',
        completion_received_seconds=terminal_at,
        completion_latency_ms=(terminal_at-t0)*1000 if complete and t0 is not None else None,
        completion_timed_out=timed_out, transport_failed=errored,
        transcript_completion_basis=config['completion_basis'] if complete else 'unconfirmed',
        transcript_at_finalize=None, additional_final_segments_after_ack=[])
    return snap, reduced
