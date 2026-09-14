"""Streaming contracts for trial providers. Imports and replay never open a socket.

See TRIAL_STT_SETUP.md for documentation sources and measurement caveats.
The benchmark owns pacing and retries; adapters never retry a request themselves.
"""
import asyncio
import base64
import contextlib
import json
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from . import provider_protocol as shared
from .credentials import redact
from .streaming import stream_audio

MODELS = {'soniox': {'stt-rt-v5'}, 'smallest': {'pulse'},
          'sarvam': {'saaras:v3-realtime'}, 'inworld': {'inworld/inworld-stt-1'}}
ENDPOINTS = {
    'soniox': {'wss://stt-rt.soniox.com/transcribe-websocket'},
    'smallest': {'wss://api.smallest.ai/waves/v1/pulse/get_text'},
    'sarvam': {'wss://api.sarvam.ai/speech-to-text-realtime/ws'},
    'inworld': {'wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional'},
}
COMPLETION = {'soniox': 'finished_after_empty_audio', 'smallest': 'is_last_after_close_stream',
              'sarvam': 'session_end_after_end', 'inworld': 'usage_after_close_stream'}


def validate(config):
    p = config['provider']
    expected = dict(sample_rate=16000, channels=1, encoding='pcm_s16le', frame_ms=20,
                    finalization='manual_at_speech_end', completion_basis=COMPLETION[p],
                    finalize_ack_supported=p == 'soniox', requires_final_only_completion=True)
    expected['language'] = {'soniox': 'en', 'smallest': 'en', 'sarvam': 'en-IN', 'inworld': 'en-US'}[p]
    if p == 'sarvam':
        expected.update(stream_type='balanced', mode='transcribe', endpointing='manual')
    if p == 'smallest':
        expected.update(word_timestamps=True)
    if p == 'inworld':
        expected.update(voice_profile={'enableVoiceProfile': True, 'topN': 3}, vad_threshold=0)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Trial provider contract mismatch: {key}')
    return config


def connection(config, key):
    p = config['provider']
    query = {}
    headers = {'Authorization': f'Bearer {key}'}
    if p == 'soniox':
        # The initial configuration message authenticates this protocol.
        headers = {}
    elif p == 'smallest':
        query = dict(language=config['language'], encoding='linear16', sample_rate=16000,
                     word_timestamps='true')
    elif p == 'sarvam':
        headers = {'api-subscription-key': key}
        query = dict(model=config['model'], language_code=config['language'],
                     stream_type=config['stream_type'], mode=config['mode'],
                     endpointing=config['endpointing'], encoding='linear16', sample_rate=16000)
    elif p == 'inworld':
        headers = {'Authorization': f'Basic {key}'}
    return config['endpoint'] + ('?' + urlencode(query) if query else ''), headers


def field(obj, camel, snake):
    return obj.get(camel, obj.get(snake))


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        # These three APIs have no documented configuration acknowledgment.
        # Readiness to SEND is not proof that the requested model was accepted.
        self.ready = self.provider != 'sarvam'
        self.accepted = False
        self.voice_profiles = []
        self.usage = None

    def setup(self, secret=''):
        c = self.config
        if self.provider == 'soniox':
            return dict(api_key=secret, model=c['model'], audio_format='pcm_s16le',
                        sample_rate=16000, num_channels=1, language_hints=[c['language']],
                        enable_endpoint_detection=False, enable_speaker_diarization=False,
                        enable_language_identification=False)
        if self.provider == 'inworld':
            return {'transcribeConfig': dict(modelId=c['model'], audioEncoding='LINEAR16',
                sampleRateHertz=16000, numberOfChannels=1, language=c['language'],
                voiceProfileConfig=c['voice_profile'], inworldSttV1Config={'vadThreshold': c['vad_threshold']})}
        return None

    def start(self):
        return {'event': 'speech_start'} if self.provider == 'sarvam' else None

    def audio(self, frame):
        if self.provider == 'sarvam':
            return {'event': 'audio_input', 'audio': base64.b64encode(frame).decode('ascii')}
        if self.provider == 'inworld':
            return {'audioChunk': {'content': base64.b64encode(frame).decode('ascii')}}
        return frame

    def finalize(self):
        return {'soniox': {'type': 'finalize'}, 'smallest': {'type': 'finalize'},
                'sarvam': {'event': 'speech_end'}, 'inworld': {'endTurn': {}}}[self.provider]

    def finish(self, frames):
        return {'soniox': b'', 'smallest': {'type': 'close_stream'},
                'sarvam': {'event': 'end'}, 'inworld': {'closeStream': {}}}[self.provider]

    def segment(self, text, final):
        if not isinstance(text, str):
            raise shared.ProviderError('Transcript must be a string')
        if self.provider != 'soniox':
            text = text.strip()
        self.put(self.sequence, text, final)
        if final:
            self.sequence += 1

    def profile(self, value):
        if value is None:
            return
        if not isinstance(value, dict):
            raise shared.ProviderError('Voice Profile must be an object')
        profile = {k: value[k] for k in ('age', 'accent', 'emotion', 'pitch') if k in value}
        style = field(value, 'vocalStyle', 'vocal_style')
        if style is not None:
            profile['vocal_style'] = style
        # Retain optional fields and confidence ordering as returned, without
        # synthesizing speaker identity or turning classifications into text.
        self.voice_profiles.append({'segment': self.sequence, 'voice_profile': profile})

    def feed(self, m):
        if not isinstance(m, dict):
            raise shared.ProviderError('Expected a provider event object')
        if (m.get('error') or m.get('error_code') or m.get('event') == 'error'
                or m.get('type') == 'error' or m.get('status') in ('error', 'failed')):
            raise shared.ProviderError('Provider rejected request; see redacted raw evidence')
        event_id = m.get('event_id')
        if event_id is not None:
            if event_id in self.seen:
                return
            self.seen.add(event_id)
        p = self.provider
        if p == 'soniox':
            if 'tokens' in m:
                self.accepted = True
                self.partials.clear()
                provisional = []
                for token in m['tokens']:
                    text = token.get('text', '')
                    if text in ('<fin>', '<end>'):
                        if text == '<fin>' and token.get('is_final') and self.requested:
                            self.ack = True
                        continue
                    if token.get('is_final'):
                        self.segment(text, True)
                    else:
                        provisional.append(text)
                if provisional:
                    self.put(self.sequence, ''.join(provisional), False)
            if m.get('finished') is True and self.closing:
                self.accepted = self.terminal = True
                self.usage = {k: m[k] for k in ('final_audio_proc_ms', 'total_audio_proc_ms') if k in m}
        elif p == 'smallest':
            if m.get('type') == 'transcription':
                self.accepted = True
                self.segment(m.get('transcript', ''), m.get('is_final') is True)
                if m.get('is_last') is True and m.get('is_final') is True and self.closing:
                    self.terminal = True
                # An ordinary final cannot be correlated with our finalize request.
        elif p == 'sarvam':
            kind = m.get('event')
            if kind == 'session.begin':
                self.ready = self.accepted = True
            elif kind in ('transcript.partial', 'transcript.final'):
                self.accepted = True
                self.segment(m.get('text', ''), kind == 'transcript.final')
            elif kind == 'session.end' and self.closing:
                self.accepted = self.terminal = True
                self.usage = {k: m[k] for k in ('audio_duration_s',) if k in m}
        elif p == 'inworld':
            result = m.get('result') or {}
            transcription = result.get('transcription')
            if transcription is not None:
                self.accepted = True
                self.profile(field(transcription, 'voiceProfile', 'voice_profile'))
                self.segment(transcription.get('transcript', ''),
                             field(transcription, 'isFinal', 'is_final') is True)
            self.profile(field(result, 'voiceProfile', 'voice_profile'))
            if 'usage' in result:
                self.usage = result['usage']
                model = field(self.usage, 'modelId', 'model_id')
                self.model_mismatch |= bool(model and model != self.config['model'])
                if self.closing:
                    self.accepted = self.terminal = True

    def snapshot(self):
        if self.provider == 'soniox':
            # Soniox tokens include their own spaces and punctuation.
            final = ''.join(self.finals.values())
            partial = ''.join(self.partials.values())
            result = dict(text=(final + partial).strip(), final_text=final.strip(),
                          partial_text=partial.strip(), provisional=bool(partial),
                          reconstruction_status='supported')
        else:
            result = super().snapshot()
        result['provider_metadata'] = {'voice_profiles': list(self.voice_profiles), 'usage': self.usage}
        return result


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    protocol = Protocol(config)
    ready, ack, terminal = asyncio.Event(), asyncio.Event(), asyncio.Event()
    if protocol.ready:
        ready.set()

    async def send(value):
        if value is not None:
            await ws.send(json.dumps(value) if isinstance(value, dict) else value)

    async def receive():
        accepted = False
        async for raw in ws:
            at = log.now()
            message = json.loads(raw)
            log.emit('provider_message', at=at, message=redact(message, secret))
            protocol.feed(message)
            if protocol.model_mismatch:
                raise shared.ProviderError('Provider selected a different model')
            if protocol.accepted and not accepted:
                log.emit('model_accepted', model=config['model'],
                         verification='requested_alias_response_accepted')
                accepted = True
            for value, signal in ((protocol.ready, ready), (protocol.ack, ack), (protocol.terminal, terminal)):
                if value:
                    signal.set()
        if not protocol.terminal:
            raise shared.ProviderError('Provider disconnected before terminal completion')

    async def wait_signal(signal, timeout):
        waiter = asyncio.create_task(signal.wait())
        try:
            done, _ = await asyncio.wait({waiter, receiver}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                await receiver
            if not signal.is_set():
                if not done:
                    raise TimeoutError('Provider completion timed out')
                raise shared.ProviderError('Provider closed before expected response')
        finally:
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter

    receiver = asyncio.create_task(receive())
    sender = None
    speech_end = None
    try:
        await send(protocol.setup(secret))
        await wait_signal(ready, config['ready_timeout_seconds'])
        await send(protocol.start())

        async def send_audio(frame):
            await send(protocol.audio(frame))

        async def finalize(t0):
            nonlocal speech_end
            speech_end = t0
            protocol.requested = True
            log.emit('finalize_requested', t0_seconds=t0)
            await send(protocol.finalize())
            log.emit('finalize_sent', t0_seconds=t0)

        sender = asyncio.create_task(stream_audio(pcm, speech_frames, send_audio, finalize, log,
                                                 sample_rate=config['sample_rate']))
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        if receiver in done:
            await receiver
            raise shared.ProviderError('Connection ended before all audio was sent')
        await sender
        # Only Soniox supplies a documented, unambiguous finalize acknowledgment.
        if config['finalize_ack_supported']:
            await wait_signal(ack, max(0, speech_end + config['finalize_timeout_seconds'] - log.now()))
        protocol.closing = True
        log.emit('close_stream_requested')
        await send(protocol.finish(speech_frames + 50))
        await wait_signal(terminal, config['close_timeout_seconds'])
        if protocol.snapshot()['partial_text']:
            raise shared.ProviderError('Stream ended with unresolved partial text')
        log.emit('provider_terminal', basis=config['completion_basis'])
    except TimeoutError:
        log.emit('close_stream_timeout')
        raise
    finally:
        for task in (sender, receiver):
            if task is not None and not task.done():
                task.cancel()
        for task in (sender, receiver):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task


async def transcribe(pcm, speech_frames, config, key, log):
    url, headers = connection(config, key)
    log.emit('connection_requested', url=config['endpoint'])
    try:
        async with connect(url, additional_headers=headers, open_timeout=15, close_timeout=5,
                           max_size=8 * 1024 * 1024) as ws:
            log.emit('connection_open')
            await exchange(ws, pcm, speech_frames, config, log, secret=key)
    except Exception as exc:
        response = getattr(exc, 'response', None)
        body = getattr(response, 'body', b'')
        if isinstance(body, bytes):
            body = body.decode('utf-8', errors='replace')
        log.emit('connection_diagnostic', error_type=type(exc).__name__,
                 http_status=getattr(response, 'status_code', None),
                 detail=redact(str(exc), key), response_body=redact(body, key))
        raise


def replay(events, config, cutoff=float('inf')):
    snapshot, reduced = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    reduced['provider_metadata'] = snapshot['provider_metadata']
    return snapshot, reduced
