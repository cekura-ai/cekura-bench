"""Mistral Voxtral realtime STT: base64 JSON audio and flush-acknowledged segments.

Wire contract, probed live on 2026-11-04:

- The server opens with `session.created`; the documentation asks clients to
  wait for it. `session.update` is answered by `session.updated`, which echoes
  the applied settings. Both name the served build: the requested `-3-0` name
  resolves to a build such as `-3-0-4`, and requesting a build directly is
  refused, so the echo is the identity check.
- `transcription.text.delta` appends text to the open segment. Mistral documents
  each delta as final when received and never revised, so deltas count as final
  text. Every `transcription.done` repeated its segment's deltas exactly in the
  live runs; a done that differs fails closed rather than revising final text.
- `input_audio.flush` closes the open segment and is answered by one
  `transcription.done` holding that segment's text alone. Server turn detection
  stays off unless requested, so segments end only on flush or end. Mistral's
  SDK sends flush; the current documentation describes `done` only after end.
- `input_audio.end` closes the remaining segment (normally the silence tail)
  with one more `transcription.done`. A client close after it completes with
  code 1000; left open, the server drops the connection without a close frame.
"""
import asyncio
import base64
import json
import re
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from . import provider_protocol as shared
from .credentials import redact

ENDPOINT = 'wss://api.mistral.ai/v1/audio/transcriptions/realtime'


def validate(config):
    # The provider registry already pins the endpoint and requested model.
    expected = dict(language='en', sample_rate=16000, encoding='pcm_s16le', channels=1, frame_ms=20,
                    target_streaming_delay_ms=480, finalization='manual_at_speech_end',
                    completion_basis='transcription_done_after_input_audio_end',
                    finalize_ack_supported=True, requires_final_only_completion=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Mistral Voxtral requires {key}={value!r}')
    served = config.get('served_model')
    if not isinstance(served, str) or not served.startswith(config['model'] + '-'):
        raise ValueError('Mistral Voxtral requires the served build of the requested model')
    return config


def connection(config, key):
    return (config['endpoint'] + '?' + urlencode({'model': config['model']}),
            {'Authorization': f'Bearer {key}'})


def session(config):
    """The settings sent in session.update, which session.updated must echo."""
    return {'audio_format': {'encoding': config['encoding'], 'sample_rate': config['sample_rate']},
            'language': config['language'], 'target_streaming_delay_ms': config['target_streaming_delay_ms']}


def words(text):
    # Case and punctuation differ between a segment and a repeat of it.
    return re.findall(r"[\w']+", text.casefold())


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        self.created = False

    def setup(self):
        return {'type': 'session.update', 'session': session(self.config)}

    def audio(self, frame):
        return {'type': 'input_audio.append', 'audio': base64.b64encode(frame).decode('ascii')}

    def finalize(self):
        return {'type': 'input_audio.flush'}

    def finish(self, frames):
        return {'type': 'input_audio.end'}

    def served(self, model):
        if not isinstance(model, str):
            raise shared.ProviderError('Mistral event lacks the served model')
        self.model_mismatch |= model != self.config['served_model']

    def feed(self, m):
        super().feed(m)
        kind = m.get('type')
        if kind in ('session.created', 'session.updated'):
            state = m.get('session')
            if not isinstance(state, dict):
                raise shared.ProviderError('Mistral session event lacks a session')
            self.served(state.get('model'))
            if kind == 'session.created':
                if self.created:
                    raise shared.ProviderError('Repeated Mistral session.created')
                self.created = True
            else:
                # Audio is sent only after the server confirms every setting,
                # so a silently ignored language or delay cannot be measured.
                if (not self.created or any(state.get(k) != v for k, v in session(self.config).items())
                        or state.get('turn_detection') is not None or state.get('diarize')):
                    raise shared.ProviderError('Mistral did not apply the requested session settings')
                self.ready = True
        elif kind == 'transcription.text.delta':
            if not isinstance(m.get('text'), str):
                raise shared.ProviderError('Invalid Mistral text delta')
            # Final on receipt: Mistral never revises a delta.
            self.finals[self.sequence] = self.finals.get(self.sequence, '') + m['text']
        elif kind == 'transcription.done':
            text = m.get('text')
            if not isinstance(text, str):
                raise shared.ProviderError('Invalid Mistral transcription.done')
            self.served(m.get('model'))
            streamed = self.finals.get(self.sequence, '')
            earlier = words(' '.join(self.finals[k] for k in sorted(self.finals) if k != self.sequence))
            if earlier and words(text)[:len(earlier)] == earlier:
                # Live segments hold only their own text. A done that repeats
                # every earlier segment is a whole-session transcript; joining
                # it would count the speech twice, so fail closed instead.
                self.unsupported = True
            if ' '.join(text.split()) != ' '.join(streamed.split()):
                # The done must confirm the words already counted as final.
                self.unsupported = True
            self.finals[self.sequence] = text
            self.sequence += 1
            if self.closing:
                self.terminal = True
            elif self.requested:
                self.ack = True
        elif kind == 'transcription.language':
            return
        else:
            raise shared.ProviderError('Unexpected Mistral provider event')

    def snapshot(self):
        keys = sorted(set(self.finals) | set(self.partials))
        join = lambda texts: ' '.join(' '.join(texts[k] for k in keys if k in texts).split())
        final, partial = join(self.finals), join(self.partials)
        status = 'unsupported_order_or_overlap' if self.unsupported else 'supported'
        return dict(text=join({**self.partials, **self.finals}), final_text=final, partial_text=partial,
                    provisional=bool(partial), reconstruction_status=status)


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    protocol = Protocol(config)
    # Wait for session.created before sending session.update, as documented.
    raw = await asyncio.wait_for(ws.recv(), config['ready_timeout_seconds'])
    at = log.now()
    message = json.loads(raw)
    log.emit('provider_message', at=at, message=redact(message, secret))
    protocol.feed(message)
    if not protocol.created:
        raise shared.ProviderError('Mistral did not start with session.created')
    if protocol.model_mismatch:
        raise shared.ProviderError('Provider selected a different model')
    await shared.exchange(ws, pcm, speech_frames, config, log, secret, protocol_factory=lambda _config: protocol)


async def transcribe(pcm, speech_frames, config, key, log):
    url, headers = connection(config, key)
    log.emit('connection_requested', url=url)
    async with connect(url, additional_headers=headers, open_timeout=15, close_timeout=5,
                       max_size=8 * 1024 * 1024) as ws:
        log.emit('connection_open')
        await exchange(ws, pcm, speech_frames, config, log, secret=key)


def replay(events, config, cutoff=float('inf')):
    snap, result = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    served = ((e['message'].get('session') or {}).get('model') or e['message'].get('model')
              for e in events if e['kind'] == 'provider_message' and e['time_seconds'] <= cutoff
              and e['message'].get('type') in ('session.created', 'session.updated', 'transcription.done'))
    result['model_versions'] = list(dict.fromkeys(m for m in served if isinstance(m, str)))
    result['model_verification_basis'] = 'served_build_echo_matches_pinned_served_model'
    return snap, result
