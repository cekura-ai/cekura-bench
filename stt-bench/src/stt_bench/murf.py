"""Murf Finch realtime STT: binary PCM, explicit turns, and clean close evidence."""
import asyncio
import contextlib
import json
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from . import provider_protocol as shared
from .credentials import redact
from .streaming import stream_audio


def validate(config):
    expected = dict(model='finch_global_1.0', language='en', sample_rate=16000,
                    encoding='pcm_s16le', channels=1, frame_ms=20,
                    finalization='manual_finalize_at_speech_end',
                    completion_basis='final_EndOfTurn_after_Finalize_then_CloseStream',
                    finalize_ack_supported=True, requires_final_only_completion=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Murf Finch requires {key}={value!r}')


def connection(config, key):
    return config['endpoint'] + '?' + urlencode({
        'model': config['model'], 'language': config['language'],
    }), {'api-key': key}


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        # Murf sends no ready message. A successful WebSocket upgrade is ready.
        self.ready = True

    def setup(self):
        return {'type': 'AudioConfig', 'codec': self.config['encoding'],
                'sample_rate': self.config['sample_rate'], 'channels': self.config['channels']}

    def finalize(self):
        return {'type': 'Finalize'}

    def finish(self):
        return {'type': 'CloseStream'}

    @staticmethod
    def turn(message, final):
        raw = message.get('rawEvent')
        if (not isinstance(message.get('transcript'), str) or not isinstance(raw, dict)
                or raw.get('type') != 'TurnInfo' or type(raw.get('turn_index')) is not int
                or raw['turn_index'] < 0 or type(raw.get('sequence_id')) is not int
                or raw['sequence_id'] < 0 or raw.get('transcript') != message['transcript']):
            raise shared.ProviderError('Invalid Murf transcript event')
        expected_events = {'EndOfTurn'} if final else {'StartOfTurn', 'Update'}
        if raw.get('event') not in expected_events:
            raise shared.ProviderError('Invalid Murf transcript event')
        if final and raw.get('trigger') != 'manual':
            raise shared.ProviderError('Murf final was not manually finalized')
        return raw['turn_index']

    def feed(self, message):
        super().feed(message)
        kind = message.get('type')
        if kind == 'partial':
            self.put(self.turn(message, final=False), message['transcript'], False)
        elif kind == 'final':
            self.put(self.turn(message, final=True), message['transcript'], True)
            if self.requested:
                self.ack = True
        elif kind == 'QuotaWarning':
            return
        else:
            raise shared.ProviderError('Unexpected Murf provider event')

    def snapshot(self):
        keys = sorted(set(self.finals) | set(self.partials))
        final = ' '.join(self.finals[k] for k in keys if k in self.finals).strip()
        partial = ' '.join(self.partials[k] for k in keys if k in self.partials).strip()
        return dict(text=' '.join(self.finals.get(k, self.partials.get(k, '')) for k in keys).strip(),
                    final_text=final, partial_text=partial, provisional=bool(partial),
                    reconstruction_status='unsupported_order_or_overlap' if self.unsupported else 'supported')


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    protocol = Protocol(config)
    finalized = asyncio.Event()
    receiver = sender = None
    close_sent = False

    async def send(value):
        await ws.send(json.dumps(value) if isinstance(value, dict) else value)

    async def receive():
        async for raw in ws:
            if not isinstance(raw, str):
                raise shared.ProviderError('Murf sent a non-text provider event')
            at = log.now()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise shared.ProviderError('Murf sent invalid JSON') from exc
            log.emit('provider_message', at=at, message=redact(message, secret))
            protocol.feed(message)
            if protocol.ack:
                finalized.set()
        if getattr(ws, 'close_code', None) != 1000:
            raise shared.ProviderError('Murf closed without normal completion')

    async def wait_for(signal, timeout, purpose):
        waiter = asyncio.create_task(signal.wait())
        try:
            done, _ = await asyncio.wait({waiter, receiver}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                await receiver
                raise shared.ProviderError(f'Murf closed before {purpose}')
            if not signal.is_set():
                raise TimeoutError(f'Murf {purpose} timed out')
        finally:
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter

    try:
        receiver = asyncio.create_task(receive())
        await send(protocol.setup())
        log.emit('model_accepted', model=config['model'], verification='requested_alias_session_accepted')

        async def finalize(t0):
            protocol.requested = True
            command = protocol.finalize()
            log.emit('finalize_requested', t0_seconds=t0, message=command)
            await send(command)
            log.emit('finalize_sent', t0_seconds=t0, message=command)

        sender = asyncio.create_task(stream_audio(pcm, speech_frames, send, finalize, log,
                                                   sample_rate=config['sample_rate']))
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        if receiver in done:
            await receiver
            raise shared.ProviderError('Murf closed before all audio was sent')
        await sender

        try:
            await wait_for(finalized, config['finalize_timeout_seconds'], 'finalization')
        except TimeoutError:
            log.emit('finalize_ack_missing')
            raise

        log.emit('close_stream_requested')
        await send(protocol.finish())
        close_sent = True
        try:
            await asyncio.wait_for(receiver, config['close_timeout_seconds'])
        except TimeoutError:
            log.emit('close_stream_timeout')
            raise
        log.emit('provider_terminal', basis=config['completion_basis'])
    finally:
        if not close_sent and receiver is not None and not receiver.done():
            with contextlib.suppress(Exception):
                await send(protocol.finish())
        for task in (sender, receiver):
            if task is not None and not task.done():
                task.cancel()
        for task in (sender, receiver):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task


async def transcribe(pcm, speech_frames, config, key, log):
    url, headers = connection(config, key)
    log.emit('connection_requested', url=url)
    async with connect(url, additional_headers=headers, open_timeout=15, close_timeout=5,
                       max_size=8 * 1024 * 1024) as ws:
        log.emit('connection_open')
        await exchange(ws, pcm, speech_frames, config, log, secret=key)


def replay(events, config, cutoff=float('inf')):
    snap, result = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    result['model_verification_basis'] = 'requested_alias_session_accepted; model_version_not_exposed'
    return snap, result
