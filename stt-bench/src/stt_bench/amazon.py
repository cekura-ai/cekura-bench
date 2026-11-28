"""Amazon Transcribe streaming over its WebSocket API: SigV4-presigned URL, AWS event-stream frames.

Each audio frame travels as an `AudioEvent` message in AWS's binary
event-stream encoding; results come back as `TranscriptEvent` messages whose
JSON payload carries `Transcript.Results`. A result keeps its `ResultId` while
it is partial and is final once `IsPartial` is false. An `AudioEvent` with an
empty payload ends the stream; the service then sends its remaining results
and closes the socket, which is the completion signal.

There is no finalize command mid-stream, so the bench's common tail of silence
precedes the empty event. Transcribe Turbo needs no model parameter: AWS routes
supported languages in us-east-2 to it, so the model is not echoed back and
identity rests on the configured endpoint and region.
"""
import binascii
import datetime
import hashlib
import hmac
import json
import struct
from urllib.parse import quote
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK
from . import provider_protocol as shared

REGION = 'us-east-2'
HOST = f'transcribestreaming.{REGION}.amazonaws.com:8443'
ENDPOINT = f'wss://{HOST}/stream-transcription-websocket'
END_OF_STREAM = 'end_of_stream'


def validate(config):
    expected = dict(language='en-US', sample_rate=16000, encoding='pcm_s16le', channels=1, frame_ms=20,
                    finalization='stream_end_after_tail', completion_basis='stream_closed_after_empty_audio_event',
                    requires_final_only_completion=True, region=REGION)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Amazon Transcribe requires {key}={value!r}')


def credentials(key):
    value = json.loads(key)
    if not value.get('access_key_id') or not value.get('secret_access_key'):
        raise ValueError('Amazon Transcribe needs an access key ID and secret access key')
    return value


def _sign(key, message):
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def presigned_url(config, key, now=None):
    """SigV4 query-string signature over the `host` header, as AWS documents for this endpoint."""
    creds = credentials(key)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    stamp, day = now.strftime('%Y%m%dT%H%M%SZ'), now.strftime('%Y%m%d')
    scope = f'{day}/{REGION}/transcribe/aws4_request'
    params = {'X-Amz-Algorithm': 'AWS4-HMAC-SHA256', 'X-Amz-Credential': f'{creds["access_key_id"]}/{scope}',
              'X-Amz-Date': stamp, 'X-Amz-Expires': '300', 'X-Amz-SignedHeaders': 'host',
              'language-code': config['language'], 'media-encoding': 'pcm',
              'sample-rate': str(config['sample_rate'])}
    if creds.get('session_token'):
        params['X-Amz-Security-Token'] = creds['session_token']
    query = '&'.join(f'{quote(k, safe="-_.~")}={quote(v, safe="-_.~")}' for k, v in sorted(params.items()))
    canonical = '\n'.join(['GET', '/stream-transcription-websocket', query, f'host:{HOST}\n', 'host',
                           hashlib.sha256(b'').hexdigest()])
    to_sign = '\n'.join(['AWS4-HMAC-SHA256', stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    signing = _sign(_sign(_sign(_sign(('AWS4' + creds['secret_access_key']).encode(), day), REGION),
                          'transcribe'), 'aws4_request')
    signature = hmac.new(signing, to_sign.encode(), hashlib.sha256).hexdigest()
    return f'{ENDPOINT}?{query}&X-Amz-Signature={signature}'


def _crc(data):
    return binascii.crc32(data) & 0xFFFFFFFF


def encode_event(headers, payload):
    """One event-stream message: prelude, CRC, string headers, payload, CRC."""
    encoded = b''.join(struct.pack('>B', len(name)) + name.encode() + struct.pack('>BH', 7, len(value)) + value.encode()
                       for name, value in headers.items())
    prelude = struct.pack('>II', 16 + len(encoded) + len(payload), len(encoded))
    message = prelude + struct.pack('>I', _crc(prelude)) + encoded + payload
    return message + struct.pack('>I', _crc(message))


def audio_event(frame):
    return encode_event({':content-type': 'application/octet-stream', ':event-type': 'AudioEvent',
                         ':message-type': 'event'}, frame)


def _header_value(kind, data, at):
    if kind in (0, 1):
        return kind == 0, at
    sizes = {2: ('>b', 1), 3: ('>h', 2), 4: ('>i', 4), 5: ('>q', 8), 8: ('>q', 8)}
    if kind in sizes:
        fmt, size = sizes[kind]
        return struct.unpack_from(fmt, data, at)[0], at + size
    if kind in (6, 7):
        length = struct.unpack_from('>H', data, at)[0]
        raw = data[at + 2:at + 2 + length]
        return (raw.decode() if kind == 7 else raw.hex()), at + 2 + length
    if kind == 9:
        return data[at:at + 16].hex(), at + 16
    raise shared.ProviderError('Unknown event-stream header type')


def decode_events(data):
    """Split one WebSocket message into event-stream messages, checking both CRCs."""
    events, offset = [], 0
    while offset < len(data):
        if len(data) - offset < 16:
            raise shared.ProviderError('Truncated event-stream message')
        total, header_len, prelude_crc = struct.unpack_from('>III', data, offset)
        message = data[offset:offset + total]
        if (len(message) != total or _crc(message[:8]) != prelude_crc
                or _crc(message[:-4]) != struct.unpack_from('>I', message, total - 4)[0]):
            raise shared.ProviderError('Corrupt event-stream message')
        headers, at = {}, 12
        while at < 12 + header_len:
            length = message[at]
            name = message[at + 1:at + 1 + length].decode()
            headers[name], at = _header_value(message[at + 1 + length], message, at + 2 + length)
        body = message[12 + header_len:total - 4]
        try:
            payload = json.loads(body) if body else {}
        except ValueError:
            payload = {'raw': body.decode(errors='replace')}
        events.append(event_object(headers, payload))
        offset += total
    return events


def event_object(headers, payload):
    """Logged form of a service message. Exceptions become `error` so the shared checks fail them."""
    if headers.get(':message-type') == 'exception':
        return {'type': 'error', 'exception_type': headers.get(':exception-type'), 'payload': payload}
    return {'type': headers.get(':event-type'), 'payload': payload}


class Socket:
    """Adapts the shared JSON exchange to AWS's binary framing in both directions."""

    def __init__(self, ws):
        self.ws = ws

    async def send(self, value):
        if isinstance(value, (bytes, bytearray)):
            await self.ws.send(audio_event(bytes(value)))
        elif json.loads(value).get('type') == END_OF_STREAM:
            await self.ws.send(audio_event(b''))
        else:
            raise ValueError('Amazon Transcribe accepts only audio and the end-of-stream marker')

    async def __aiter__(self):
        try:
            async for raw in self.ws:
                if isinstance(raw, str):
                    raise shared.ProviderError('Expected binary event-stream messages')
                for event in decode_events(raw):
                    yield json.dumps(event)
        except ConnectionClosedOK:
            pass
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


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    await shared.exchange(Socket(ws), pcm, speech_frames, config, log, secret, protocol_factory=Protocol)


async def transcribe(pcm, speech_frames, config, key, log):
    url = presigned_url(config, key)
    log.emit('connection_requested', url=config['endpoint'])  # The presigned query carries the key ID and signature.
    async with connect(url, open_timeout=15, close_timeout=5, max_size=8 * 1024 * 1024) as ws:
        log.emit('connection_open')
        await exchange(ws, pcm, speech_frames, config, log, secret=credentials(key)['secret_access_key'])


def replay(events, config, cutoff=float('inf')):
    snap, result = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    result['model_verification_basis'] = 'implicit_turbo_routing_in_us_east_2; model_not_echoed'
    return snap, result
