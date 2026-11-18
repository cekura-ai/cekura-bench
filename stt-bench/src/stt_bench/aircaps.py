"""AirCaps Live 1: binary PCM in, one growing session transcript out.

Each session starts with an HTTPS lookup that names the WebSocket host to use.
Every transcript message carries the whole session text; its first
`final_chars` characters never change, and `is_final` makes all of it final.
`finalize` returns a final at once when there is speech, but gets no reply
when there is none, so it has no acknowledgment to wait for. `finish` sends a
last final and then `finished`, which is the completion signal. Without a
`finalize` the server also settles all text by itself about every 20 seconds.
Such a final can carry an older, shorter copy of text already settled; the
next message restores the full text, so that stale final is skipped.
"""
import asyncio
import json
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlsplit

from websockets.asyncio.client import connect

from . import provider_protocol as shared

ENDPOINT = 'https://api.aircaps.com/v1/realtime/endpoint'
SOCKET_DOMAIN = '.aircaps.ai'


def validate(config):
    expected = dict(language='en-US', sample_rate=16000, encoding='pcm_s16le', channels=1, frame_ms=20,
                    finalization='manual_at_speech_end', completion_basis='finished_after_finish',
                    finalize_ack_supported=False, requires_final_only_completion=True,
                    socket_hosts=['*' + SOCKET_DOMAIN])
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'AirCaps requires {key}={value!r}')


def headers(key):
    return {'Authorization': f'Bearer {key}'}


def socket_url(config, key):
    request = urllib.request.Request(config['endpoint'] + '?' + urlencode({'model': config['model']}),
                                     headers=headers(key))
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:500]
        raise shared.ProviderError(f'AirCaps endpoint lookup failed: HTTP {exc.code} {detail}') from exc
    url = body.get('websocket_url')
    parts = urlsplit(url if isinstance(url, str) else '')
    if parts.scheme != 'wss' or not (parts.hostname or '').endswith(SOCKET_DOMAIN):
        raise shared.ProviderError('AirCaps returned an unexpected WebSocket address')
    if body.get('model') != config['model']:
        raise shared.ProviderError('Provider selected a different model')
    return url


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        self.text = ''
        self.final_chars = 0

    def setup(self):
        c = self.config
        return {'type': 'start', 'model': c['model'], 'encoding': c['encoding'],
                'sample_rate': c['sample_rate'], 'channels': c['channels']}

    def finalize(self):
        return {'type': 'finalize'}

    def finish(self, frames):
        return {'type': 'finish'}

    def feed(self, m):
        super().feed(m)
        kind = m.get('type')
        if kind == 'ready':
            self.model_mismatch |= (m.get('model') != self.config['model']
                                    or m.get('language') != self.config['language'])
            self.ready = True
        elif kind == 'transcript':
            text, settled = m.get('text'), m.get('final_chars')
            if not isinstance(text, str) or type(settled) is not int or not 0 <= settled <= len(text):
                raise shared.ProviderError('AirCaps transcript lacks text or final_chars')
            if m.get('is_final') is True:
                if len(text) < self.final_chars and self.text.startswith(text):
                    return
                settled = len(text)
            if settled < self.final_chars or not text.startswith(self.text[:self.final_chars]):
                self.unsupported = True
            self.text, self.final_chars = text, settled
        elif kind == 'finished' and self.closing:
            self.terminal = True

    def snapshot(self):
        final, partial = self.text[:self.final_chars].strip(), self.text[self.final_chars:].strip()
        status = 'unsupported_order_or_overlap' if self.unsupported else 'supported'
        return dict(text=self.text.strip(), final_text=final, partial_text=partial,
                    provisional=bool(partial), reconstruction_status=status)


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    await shared.exchange(ws, pcm, speech_frames, config, log, secret, protocol_factory=Protocol)


async def transcribe(pcm, speech_frames, config, key, log):
    log.emit('connection_requested', url=config['endpoint'])
    url = await asyncio.to_thread(socket_url, config, key)
    log.emit('socket_resolved', url=urlsplit(url)._replace(query='', fragment='').geturl())
    async with connect(url, additional_headers=headers(key), open_timeout=15, close_timeout=5,
                       max_size=8 * 1024 * 1024) as ws:
        log.emit('connection_open')
        await exchange(ws, pcm, speech_frames, config, log, secret=key)


def replay(events, config, cutoff=float('inf')):
    snap, result = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    result['model_verification_basis'] = 'server_reported_model_in_lookup_and_ready; model_version_not_exposed'
    return snap, result
