"""Explicit streaming protocols and offline reconstruction from receipt-time evidence.

Wire contracts checked against vendor docs and Coval d62a5d22 (see MODEL_READINESS.md).
No provider SDK controls audio pacing or retries.
"""
import asyncio
import base64
import contextlib
import json
from urllib.parse import urlencode
from websockets.asyncio.client import connect
from .credentials import redact
from .streaming import stream_audio


class ProviderError(RuntimeError):
    pass


class Protocol:
    def __init__(self, config):
        self.config = config
        self.provider = config['provider']
        self.ready = self.provider in ('cartesia',)
        self.requested = False
        self.closing = False
        self.ack = False
        self.terminal = False
        self.model_mismatch = False
        self.finals = {}
        self.partials = {}
        self.previous = {}
        self.completed = set()
        self.commit_id = None
        self.seen = set()
        self.sequence = 0
        self.unsupported = False

    def setup(self):
        c, p = self.config, self.provider
        if p == 'openai':
            return {'type': 'session.update', 'session': {'type': 'transcription', 'audio': {'input': {
                'format': {'type': 'audio/pcm', 'rate': 24000},
                'transcription': {'model': c['model'], 'language': 'en'}, 'turn_detection': None}}}}
        if p == 'speechmatics':
            return {'message': 'StartRecognition', 'audio_format': {
                'type': 'raw', 'encoding': 'pcm_s16le', 'sample_rate': 16000},
                'transcription_config': {'language': 'en', 'enable_partials': True,
                    'model': c['model'], 'enable_entities': False,
                    **{k: c[k] for k in ('max_delay', 'max_delay_mode') if k in c}}}
        if p == 'gemini':
            return {'setup': {'model': 'models/' + c['model'],
                'generationConfig': {'responseModalities': ['TEXT']},
                'inputAudioTranscription': {'languageCodes': ['en-US'], 'mode': 'VERBATIM'},
                'realtimeInputConfig': {'automaticActivityDetection': {'disabled': True}}}}
        return None

    def audio(self, frame):
        encoded = lambda: base64.b64encode(frame).decode('ascii')
        if self.provider == 'openai':
            return {'type': 'input_audio_buffer.append', 'audio': encoded()}
        if self.provider == 'elevenlabs':
            return {'message_type': 'input_audio_chunk', 'audio_base_64': encoded(), 'sample_rate': 16000}
        if self.provider == 'gemini':
            return {'realtimeInput': {'audio': {'data': encoded(), 'mimeType': 'audio/pcm;rate=16000'}}}
        return frame

    def finalize(self):
        if self.provider == 'speechmatics' and self.config.get('force_end_of_utterance'):
            return {'message': 'ForceEndOfUtterance'}
        return {'openai': {'type': 'input_audio_buffer.commit'},
                'elevenlabs': {'message_type': 'input_audio_chunk', 'audio_base_64': '',
                               'commit': True, 'sample_rate': 16000},
                'cartesia': 'finalize', 'deepgram': {'type': 'ForceEndTurn'},
                'gemini': {'realtimeInput': {'activityEnd': {}}}}.get(self.provider)

    def finish(self, frames):
        if self.provider == 'speechmatics':
            return {'message': 'EndOfStream', 'last_seq_no': frames}
        if self.provider == 'cartesia':
            return 'close'
        return None

    def put(self, key, text, final):
        if final:
            if key in self.finals and self.finals[key] != text:
                self.unsupported = True
            self.finals[key] = text
            self.partials.pop(key, None)
        elif key not in self.finals:
            self.partials[key] = text

    def feed(self, m):
        """Update transcript state. Duplicate IDs are ignored, repeated words are not."""
        if not isinstance(m, dict):
            raise ProviderError('Expected a provider event object')
        event_id = m.get('event_id')
        if event_id:
            if event_id in self.seen:
                return
            self.seen.add(event_id)
        kind = m.get('type', m.get('message_type', m.get('message', '')))
        if kind in ('error', 'Error', 'conversation.item.input_audio_transcription.failed') or m.get('error'):
            raise ProviderError('Provider rejected the request; see redacted raw evidence')
        p = self.provider
        if p == 'elevenlabs' and (str(kind).endswith('_error') or kind in (
                'auth_error', 'quota_exceeded', 'rate_limited', 'unaccepted_terms', 'queue_overflow',
                'resource_exhausted', 'session_time_limit_exceeded', 'input_error', 'chunk_size_exceeded',
                'insufficient_audio_activity')):
            raise ProviderError('ElevenLabs request failed')
        if p == 'openai':
            if kind == 'session.updated':
                self.ready = True
                session = m.get('session', {})
                model = session.get('audio', {}).get('input', {}).get('transcription', {}).get('model')
                self.model_mismatch |= bool(model and model != self.config['model'])
            item = m.get('item_id', 'unknown')
            if kind == 'input_audio_buffer.committed':
                self.commit_id = item
                self.previous[item] = m.get('previous_item_id')
            elif kind.endswith('.input_audio_transcription.delta'):
                self.put(item, self.partials.get(item, '') + m.get('delta', ''), False)
            elif kind.endswith('.input_audio_transcription.completed'):
                self.put(item, m.get('transcript', ''), True)
                self.completed.add(item)
            if self.requested and self.commit_id in self.completed:
                self.ack = True
        elif p == 'elevenlabs':
            if kind == 'session_started':
                self.ready = True
                model = m.get('config', {}).get('model_id')
                self.model_mismatch |= bool(model and model != self.config['model'])
            if kind == 'partial_transcript':
                self.put('utterance', m.get('text', ''), False)
            elif kind in ('committed_transcript', 'committed_transcript_with_timestamps'):
                # This adapter sends exactly one manual commit per clip.
                self.put('utterance', m.get('text', ''), True)
                if self.requested:
                    self.ack = True
        elif p == 'deepgram':
            if kind == 'Connected':
                self.ready = True
            if kind == 'TurnInfo':
                self.put(m.get('turn_index', 0), m.get('transcript', ''), m.get('event') == 'EndOfTurn')
                if self.requested and m.get('event') == 'EndOfTurn':
                    self.ack = True
            if kind == 'Warning' and m.get('code') == 'FORCE_END_TURN_NO_ACTIVE_TURN':
                # No-active-turn isn't a final transcript; never treat it as one.
                self.unsupported = True
        elif p == 'speechmatics':
            if kind == 'RecognitionStarted':
                self.ready = True
            if (self.requested and kind == 'EndOfUtterance' and m.get('forced') is True):
                self.ack = True
            if kind in ('AddTranscript', 'AddPartialTranscript'):
                meta = m.get('metadata', {})
                key = (meta.get('start_time'), meta.get('end_time'))
                if None in key:
                    self.unsupported = True
                elif kind == 'AddTranscript' and any(
                        old != key and None not in old and key[0] < old[1] and old[0] < key[1]
                        for old in self.finals):
                    self.unsupported = True
                text = meta.get('transcript', '')
                self.partials.clear()
                self.put(key, text, kind == 'AddTranscript')
            if kind == 'EndOfTranscript' and self.closing:
                self.terminal = True
        elif p == 'cartesia':
            if kind == 'transcript':
                self.partials.clear()
                words = m.get('words') or []
                key = (words[0].get('start'), words[-1].get('end')) if words else ('sequence', self.sequence)
                if m.get('is_final'):
                    self.partials.clear()
                    self.sequence += 1
                self.put(key, m.get('text', ''), bool(m.get('is_final')))
            if kind == 'flush_done' and self.requested:
                self.ack = True
            if kind == 'done' and self.closing:
                self.terminal = True
        elif p == 'gemini':
            if 'setupComplete' in m:
                self.ready = True
            content = m.get('serverContent') or {}
            interim = content.get('interimInputTranscription')
            if interim is not None:
                self.put(self.sequence, interim.get('text', ''), False)
            final = content.get('inputTranscription')
            if final is not None:
                self.put(self.sequence, final.get('text', ''), True)
                self.sequence += 1
            if self.requested and (content.get('generationComplete') or content.get('turnComplete')):
                self.ack = True

    def snapshot(self):
        keys = list(dict.fromkeys([*self.finals, *self.partials]))
        if self.provider in ('deepgram', 'speechmatics'):
            keys.sort(key=lambda k: repr(k) if self.unsupported else k)
        if self.provider == 'openai' and len(keys) > 1:
            ordered = []
            pending = set(keys)
            while pending:
                candidates = [k for k in keys if k in pending and k in self.previous
                              and (self.previous[k] is None or self.previous[k] in ordered)]
                if len(candidates) != 1:
                    self.unsupported = True
                    break
                ordered.append(candidates[0]); pending.remove(candidates[0])
            if not pending:
                keys = ordered
        sep = '' if self.provider == 'cartesia' else ' '
        final = sep.join(self.finals[k] for k in keys if k in self.finals).strip()
        partial = sep.join(self.partials[k] for k in keys if k in self.partials).strip()
        text = sep.join(self.finals.get(k, self.partials.get(k, '')) for k in keys).strip()
        return dict(text=text, final_text=final, partial_text=partial, provisional=bool(partial),
                    reconstruction_status='unsupported_order_or_overlap' if self.unsupported else 'supported')


def connection(config, key):
    p, model, url = config['provider'], config['model'], config['endpoint']
    query = {}
    headers = {'Authorization': f'Bearer {key}'}
    if p == 'openai':
        query = {'intent': 'transcription'}
    elif p == 'gemini':
        query = {'key': key}
        headers = {}
    elif p == 'elevenlabs':
        query = {'model_id': model, 'audio_format': 'pcm_16000', 'language_code': 'en',
                 'commit_strategy': 'manual', 'include_timestamps': 'true'}
        headers = {'xi-api-key': key}
    elif p == 'cartesia':
        query = {'model': model, 'encoding': 'pcm_s16le', 'sample_rate': 16000, 'language': 'en'}
        headers['cartesia-version'] = config['api_version']
    elif p == 'deepgram':
        headers = {'Authorization': f'Token {key}'}
        query = {'model': model, 'encoding': 'linear16', 'sample_rate': 16000,
                 'eot_threshold': 1.0, 'eot_timeout_ms': 60000}
    return url + ('?' + urlencode(query) if query else ''), headers


async def exchange(ws, pcm, speech_frames, config, log, secret='', *, protocol_factory=Protocol, streamer=None):
    protocol = protocol_factory(config)
    ready, ack, terminal = asyncio.Event(), asyncio.Event(), asyncio.Event()
    if protocol.ready:
        ready.set()
    async def send(value):
        if value is not None:
            await ws.send(json.dumps(value) if isinstance(value, dict) else value)
    async def receive():
        async for raw in ws:
            at = log.now()
            message = json.loads(raw)
            log.emit('provider_message', at=at, message=redact(message, secret))
            protocol.feed(message)
            if protocol.model_mismatch:
                raise ProviderError('Provider selected a different model')
            if protocol.ready:
                ready.set()
            if protocol.ack:
                ack.set()
            if protocol.terminal:
                terminal.set()
        if not protocol.terminal:
            raise ProviderError('Provider disconnected before terminal completion')
    async def wait_signal(signal, timeout):
        waiter = asyncio.create_task(signal.wait())
        try:
            done, _ = await asyncio.wait({waiter, receiver}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                await receiver
            if not signal.is_set():
                if not done:
                    raise TimeoutError('Provider completion timed out')
                raise ProviderError('Provider closed before expected acknowledgment')
        finally:
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter
    receiver = asyncio.create_task(receive())
    sender = None
    speech_end = None
    try:
        await send(protocol.setup())
        await wait_signal(ready, config.get('ready_timeout_seconds', 15))
        log.emit('model_accepted', model=config['model'], verification='requested_alias_session_accepted')
        if protocol.provider == 'gemini':
            await send({'realtimeInput': {'activityStart': {}}})
        async def send_audio(frame):
            await send(protocol.audio(frame))
        async def finalize(t0):
            nonlocal speech_end
            speech_end = t0
            command = protocol.finalize()
            if command is not None:
                protocol.requested = True
                log.emit('finalize_requested', t0_seconds=t0, message=command)
                await send(command)
                log.emit('finalize_sent', t0_seconds=t0, message=command)
        sender = asyncio.create_task((streamer or stream_audio)(pcm, speech_frames, send_audio, finalize, log,
                                                 sample_rate=config['sample_rate']))
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        if receiver in done:
            await receiver
            raise ProviderError('Connection ended before all audio was sent')
        await sender
        if protocol.finalize() is not None and config.get('finalize_ack_supported') is not False:
            await wait_signal(ack, max(0, speech_end + config['finalize_timeout_seconds'] - log.now()))
        protocol.closing = True
        log.emit('close_stream_requested')
        finish = protocol.finish(speech_frames + 50)
        if finish is not None:
            await send(finish)
            await wait_signal(terminal, config['close_timeout_seconds'])
        # Providers without a terminal message have acknowledged the single speech
        # commit. The 50 uncommitted silence frames are retained in sent-audio evidence.
        log.emit('provider_terminal', basis=config['completion_basis'])
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver
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
    log.emit('connection_requested', url=config['endpoint'])  # Never log auth query parameters.
    async with connect(url, additional_headers=headers, open_timeout=15, close_timeout=5,
                       max_size=8 * 1024 * 1024) as ws:
        log.emit('connection_open')
        await exchange(ws, pcm, speech_frames, config, log, secret=key)


def replay(events, config, cutoff=float('inf'), *, protocol_factory=Protocol):
    protocol = protocol_factory(config)
    t0 = ack_at = terminal_at = last_text = first_partial = None
    errored = accepted = audio_complete = False
    finalize_snapshot = None
    for e in sorted(events, key=lambda e: e['time_seconds']):
        at, kind = e['time_seconds'], e['kind']
        if at > cutoff:
            break
        if kind == 'speech_end':
            t0 = at
        elif kind == 'finalize_requested':
            protocol.requested = True
        elif kind == 'close_stream_requested':
            protocol.closing = True
        elif kind == 'audio_complete':
            audio_complete = True
        elif kind == 'error':
            errored = True
        elif kind == 'model_accepted':
            accepted = e.get('model') == config['model']
        elif kind == 'provider_terminal':
            if protocol.terminal or (protocol.ack and protocol.provider not in (
                    'speechmatics', 'cartesia', 'soniox', 'smallest', 'sarvam', 'inworld', 'gradium', 'reson8')):
                terminal_at = at
        elif kind == 'provider_message':
            before = protocol.snapshot()['final_text']
            try:
                protocol.feed(e['message'])
            except ProviderError:
                errored = True
            snap = protocol.snapshot()
            if snap['final_text'] != before:
                last_text = at
            if first_partial is None and t0 is not None and at >= t0 and snap['partial_text']:
                first_partial = {'text': snap['partial_text'], 'received_seconds': at, 'latency_ms': (at-t0)*1000}
            if protocol.ack and ack_at is None:
                ack_at, finalize_snapshot = at, snap['final_text']
    snap = protocol.snapshot()
    complete = bool(audio_complete and terminal_at is not None and not errored and not protocol.unsupported)
    if config.get('requires_final_only_completion'):
        complete = complete and not bool(snap['partial_text'])
    reduced = dict(transcript=snap['final_text'], transcript_complete=complete,
        model_verified=bool(accepted and not protocol.model_mismatch),
        model_verification_basis='requested_alias_session_accepted; immutable version unavailable',
        model_versions=[], model_uuids=[], t0_seconds=t0, first_partial_after_t0=first_partial,
        final_transcript_received_seconds=last_text, finalize_ack_received_seconds=ack_at,
        finalize_latency_ms=(ack_at-t0)*1000 if ack_at is not None and t0 is not None else None,
        finalize_latency_status='unsupported' if config['finalization'] == 'stream_end_after_tail' or
                                config.get('finalize_ack_supported') is False else
                                'observed' if ack_at is not None else 'missing_finalize_ack',
        completion_received_seconds=terminal_at,
        completion_latency_ms=(terminal_at-t0)*1000 if complete and t0 is not None else None,
        completion_timed_out=any(e['kind'] == 'close_stream_timeout' for e in events),
        transport_failed=errored, transcript_completion_basis=config['completion_basis'] if complete else 'unconfirmed',
        transcript_at_finalize=finalize_snapshot, additional_final_segments_after_ack=[])
    return snap, reduced
