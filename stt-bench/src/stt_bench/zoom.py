"""Zoom Scribe live mode: binary PCM in, server-segmented final transcripts out.

Zoom segments speech with its own voice activity detection and sends one
`transcription.completed` per segment; no partials were observed. The session
also honors `input_audio_buffer.commit`, which ends the open segment at the
commit point. A commit with no open segment produces no event at all, so it
has no acknowledgment to wait for. `session.close` flushes every open segment
before `session.closed`, which is the completion signal.
"""
from websockets.asyncio.client import connect
from . import provider_protocol as shared

ENDPOINT = 'wss://api.zoom.us/v2/aiservices/scribe/live'
SUBPROTOCOL = 'live-asr'


def validate(config):
    expected = dict(endpoint=ENDPOINT, model='zoom-asr-en-v1', language='en-US', sample_rate=16000,
                    encoding='pcm_s16le', channels=1, frame_ms=20, finalization='manual_at_speech_end',
                    completion_basis='session_closed_after_close', finalize_ack_supported=False,
                    requires_final_only_completion=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Zoom Scribe requires {key}={value!r}')


def connection(config, key):
    # An API key is accepted directly as a bearer token; no JWT is minted.
    return config['endpoint'], {'Authorization': f'Bearer {key}'}


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        self.order = []

    def setup(self):
        return {'type': 'session.update', 'language': self.config['language'], 'audio': {'format': 'pcm16'}}

    def finalize(self):
        return {'type': 'input_audio_buffer.commit'}

    def finish(self, frames):
        return {'type': 'session.close'}

    def segment(self, item):
        if item is None:
            raise shared.ProviderError('Zoom transcript event lacks an item')
        if item not in self.order:
            self.order.append(item)
        return item

    def feed(self, m):
        super().feed(m)
        kind = m.get('type')
        if kind == 'session.updated':
            # The server names the model it selected from the language; the
            # request cannot name one, so the echo is the identity check.
            self.model_mismatch |= (m.get('model') != self.config['model']
                                    or m.get('language') != self.config['language'])
            self.ready = True
        elif kind == 'input_audio_buffer.speech_started':
            self.segment(m.get('item_id'))
        elif kind == 'transcription.delta':
            item = self.segment(m.get('item_id'))
            if item not in self.finals:
                self.partials[item] = self.partials.get(item, '') + m.get('delta', '')
        elif kind == 'transcription.completed':
            if not isinstance(m.get('transcript'), str):
                raise shared.ProviderError('Zoom completion lacks a transcript')
            self.put(self.segment(m.get('item_id')), m['transcript'], True)
        elif kind == 'session.closed' and self.closing:
            self.terminal = True

    def snapshot(self):
        keys = [k for k in self.order if k in self.finals or k in self.partials]
        final = ' '.join(self.finals[k].strip() for k in keys if self.finals.get(k, '').strip())
        partial = ' '.join(self.partials[k].strip() for k in keys if self.partials.get(k, '').strip())
        text = ' '.join(self.finals.get(k, self.partials.get(k, '')).strip() for k in keys)
        return dict(text=' '.join(text.split()), final_text=final, partial_text=partial, provisional=bool(partial),
                    reconstruction_status='unsupported_order_or_overlap' if self.unsupported else 'supported')


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    await shared.exchange(ws, pcm, speech_frames, config, log, secret, protocol_factory=Protocol)


async def transcribe(pcm, speech_frames, config, key, log):
    url, headers = connection(config, key)
    log.emit('connection_requested', url=url)
    async with connect(url, subprotocols=[SUBPROTOCOL], additional_headers=headers, open_timeout=15,
                       close_timeout=5, max_size=8 * 1024 * 1024) as ws:
        if ws.subprotocol != SUBPROTOCOL:
            raise shared.ProviderError('Zoom did not accept the live-asr subprotocol')
        log.emit('connection_open')
        await exchange(ws, pcm, speech_frames, config, log, secret=key)


def replay(events, config, cutoff=float('inf')):
    snap, result = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    result['model_verification_basis'] = 'server_reported_model_in_session_updated; model_version_not_exposed'
    return snap, result
