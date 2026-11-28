"""Amazon Transcribe streaming over HTTP/2 with AWS's `amazon-transcribe` SDK.

AWS also offers a WebSocket endpoint, but only on port 8443, which common
egress policies block; the HTTP/2 endpoint on 443 carries the same event
stream. The SDK signs the request and frames events; the bench still owns
audio pacing, and every result is logged in the same JSON shape as the
service's `TranscriptEvent` payload.

A result keeps its `ResultId` while it is partial and is final once
`IsPartial` is false. There is no finalize command mid-stream, so the bench's
common tail of silence precedes `end_stream()`. The service then sends its
remaining results and ends the response stream, which is the completion
signal. Transcribe Turbo needs no model parameter: AWS routes supported
languages in us-east-2 to it, so the model is not echoed back and identity
rests on the configured endpoint and region.
"""
import json
from . import provider_protocol as shared

REGION = 'us-east-2'
ENDPOINT = f'https://transcribestreaming.{REGION}.amazonaws.com'
END_OF_STREAM = 'end_of_stream'


def validate(config):
    expected = dict(language='en-US', sample_rate=16000, encoding='pcm_s16le', channels=1, frame_ms=20,
                    finalization='stream_end_after_tail', completion_basis='response_stream_ended_after_end_stream',
                    requires_final_only_completion=True, region=REGION)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Amazon Transcribe requires {key}={value!r}')


def credentials(key):
    value = json.loads(key)
    if not value.get('access_key_id') or not value.get('secret_access_key'):
        raise ValueError('Amazon Transcribe needs an access key ID and secret access key')
    return value


def result_object(result):
    return {'ResultId': result.result_id, 'StartTime': result.start_time, 'EndTime': result.end_time,
            'IsPartial': result.is_partial,
            'Alternatives': [{'Transcript': a.transcript} for a in result.alternatives or []]}


class Stream:
    """Presents an SDK stream to the shared exchange: audio and an end marker in, JSON events out."""

    def __init__(self, stream):
        self.stream = stream

    async def send(self, value):
        if isinstance(value, (bytes, bytearray)):
            await self.stream.input_stream.send_audio_event(audio_chunk=bytes(value))
        elif json.loads(value).get('type') == END_OF_STREAM:
            await self.stream.input_stream.end_stream()
        else:
            raise ValueError('Amazon Transcribe accepts only audio and the end-of-stream marker')

    async def __aiter__(self):
        try:
            async for event in self.stream.output_stream:
                results = [result_object(r) for r in event.transcript.results]
                yield json.dumps({'type': 'TranscriptEvent', 'payload': {'Transcript': {'Results': results}}})
        except Exception as exc:
            yield json.dumps({'type': 'error', 'exception_type': type(exc).__name__})
            return
        yield json.dumps({'type': 'stream_closed'})


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        self.ready = True  # The service starts transcribing on the first audio event.

    def finalize(self):
        return None

    def finish(self, frames):
        return {'type': END_OF_STREAM}

    def feed(self, m):
        super().feed(m)
        kind = m.get('type')
        if kind == 'TranscriptEvent':
            results = (m.get('payload') or {}).get('Transcript', {}).get('Results')
            if not isinstance(results, list):
                raise shared.ProviderError('Transcribe event lacks Transcript.Results')
            for result in results:
                alternatives = result.get('Alternatives') or []
                if not result.get('ResultId') or not alternatives:
                    raise shared.ProviderError('Transcribe result lacks an ID or alternatives')
                self.put(result['ResultId'], alternatives[0].get('Transcript', ''), result.get('IsPartial') is False)
        elif kind == 'stream_closed' and self.closing:
            self.terminal = True


async def open_stream(config, key):
    from amazon_transcribe.auth import StaticCredentialResolver
    from amazon_transcribe.client import TranscribeStreamingClient
    creds = credentials(key)
    client = TranscribeStreamingClient(region=config['region'], credential_resolver=StaticCredentialResolver(
        creds['access_key_id'], creds['secret_access_key'], creds.get('session_token')))
    return await client.start_stream_transcription(language_code=config['language'],
                                                   media_sample_rate_hz=config['sample_rate'], media_encoding='pcm')


async def transcribe(pcm, speech_frames, config, key, log):
    log.emit('connection_requested', url=config['endpoint'])
    stream = await open_stream(config, key)
    log.emit('connection_open')
    await shared.exchange(Stream(stream), pcm, speech_frames, config, log,
                          credentials(key)['secret_access_key'], protocol_factory=Protocol)


def replay(events, config, cutoff=float('inf')):
    snap, result = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    result['model_verification_basis'] = 'implicit_turbo_routing_in_us_east_2; model_not_echoed'
    return snap, result
