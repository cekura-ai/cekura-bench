"""Rime speech-to-text over its direct WebSocket: JSON envelopes, one utterance per session.

Rime does not detect the end of speech: the client ends each utterance with
`end`, and the session closes after `done`. Audio sent after `end` is refused,
so the artificial silence tail is never transmitted and `end` follows the last
speech frame immediately. Under the revised-hypotheses contract each
`hypothesis` carries the whole transcript so far, revisions count up by one,
and `done` repeats the last hypothesis as the final transcript.

A stream ends at 120 seconds and its context fills after roughly 90-110
seconds of speech, so the full benchmark streams each private recording as
consecutive pause-aligned sessions (see `full_benchmark.rime_sessions`).

Reference: https://docs.rime.ai/api-reference/stt/websockets
"""
import asyncio
import base64
import contextlib
import json

from websockets.asyncio.client import connect

from . import provider_protocol as shared
from .credentials import redact
from .streaming import stream_audio, transmitted_silence_frames

OUTPUT_CONTRACT = 'STREAMING_OUTPUT_CONTRACT_REVISED_HYPOTHESES'
# The references are word-for-word, so ask for word-for-word output where a provider
# offers it, as Gemini (VERBATIM) and Reson8 (filler_mode verbatim) are configured.
MODE = 'TRANSCRIPTION_MODE_VERBATIM'
PRIVATE_VARIANT = 'rime-pause-sessions-v1'
UTTERANCE = 'utterance'


def validate(config):
    expected = dict(model='default', language='en', sample_rate=16000, encoding='pcm_s16le', channels=1,
                    frame_ms=20, output_contract=OUTPUT_CONTRACT, mode=MODE, transmitted_silence_frames=0,
                    private_variant=PRIVATE_VARIANT, finalization='manual_at_speech_end',
                    completion_basis='done_after_end_at_speech_end', finalize_ack_supported=True,
                    requires_final_only_completion=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Rime requires {key}={value!r}')
    if transmitted_silence_frames(config) != 0:
        raise ValueError('Rime refuses audio after end; the silence tail must be omitted')
    timeout = config.get('ready_timeout_seconds')
    # Rime closes a session whose start message arrives more than 5 seconds after the upgrade.
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 30:
        raise ValueError('Rime requires a ready timeout between 0 and 30 seconds')


def connection(config, key):
    # The key travels only in the upgrade header; the URL takes no query parameters.
    return config['endpoint'], {'Authorization': f'Bearer {key}'}


def revision(value, field):
    """uint64 as a decimal string (proto3 JSON); omitted means zero."""
    if value is None:
        return 0
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if type(value) is int and value >= 0:
        return value
    raise shared.ProviderError(f'Rime sent an invalid {field} revision')


def text_of(body, field):
    value = body.get('text', '')
    if not isinstance(value, str):
        raise shared.ProviderError(f'Rime sent non-text {field} content')
    return value


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        self.revision = 0
        self.hypothesis = ''
        self.request_id = None

    def setup(self):
        return {'start': {'outputContract': self.config['output_contract'],
                          'language': self.config['language'], 'mode': self.config['mode']}}

    def audio(self, frame):
        return {'audio': base64.b64encode(frame).decode('ascii')}

    def finalize(self):
        return {'end': {}}

    def finish(self, frames):
        # `done` ends the session and Rime closes the socket; nothing else is sent.
        return None

    def feed(self, message):
        super().feed(message)  # Shared rejection of error envelopes; never logs secrets.
        if len(message) != 1:
            raise shared.ProviderError('Unexpected Rime event envelope')
        (kind, body), = message.items()
        if not isinstance(body, dict):
            raise shared.ProviderError('Unexpected Rime event body')
        if self.terminal:
            raise shared.ProviderError('Rime sent an event after done')
        if kind == 'started':
            accepted = body.get('accepted')
            if self.ready or not isinstance(body.get('requestId'), str) or not isinstance(accepted, dict):
                raise shared.ProviderError('Invalid Rime start acknowledgment')
            language = accepted.get('language') if isinstance(accepted.get('language'), dict) else {}
            if (accepted.get('outputContract') != self.config['output_contract']
                    or language.get('tag') != self.config['language']):
                raise shared.ProviderError('Rime accepted different recognition settings')
            self.ready, self.request_id = True, body['requestId']
        elif kind == 'hypothesis':
            if not self.ready:
                raise shared.ProviderError('Rime sent a hypothesis before started')
            number = revision(body.get('revision'), 'hypothesis')
            # Each revision changes the text and counts up by exactly one.
            if number != self.revision + 1:
                raise shared.ProviderError('Rime hypothesis revisions are out of order')
            self.revision, self.hypothesis = number, text_of(body, 'hypothesis')
            self.put(UTTERANCE, self.hypothesis, False)
        elif kind == 'done':
            if not self.ready or not self.requested:
                raise shared.ProviderError('Rime sent done before end')
            final = text_of(body, 'done')
            language = body.get('language') if isinstance(body.get('language'), dict) else {}
            # Documented guarantee: done repeats the last hypothesis and its revision.
            if (revision(body.get('revision'), 'done') != self.revision or final != self.hypothesis
                    or language.get('tag') != self.config['language']):
                raise shared.ProviderError('Rime final transcript does not match its last hypothesis')
            self.put(UTTERANCE, final, True)
            self.ack = self.terminal = True
        elif kind == 'cancelled':
            raise shared.ProviderError('Rime cancelled the utterance')
        else:
            raise shared.ProviderError('Unexpected Rime provider event')


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    protocol = Protocol(config)
    ready, done = asyncio.Event(), asyncio.Event()
    receiver = sender = None
    speech_end = None

    async def send(value):
        await ws.send(json.dumps(value))

    async def receive():
        async for raw in ws:
            at = log.now()
            if not isinstance(raw, str):
                raise shared.ProviderError('Rime sent a binary event; JSON was requested')
            try:
                message = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise shared.ProviderError('Rime sent invalid JSON') from exc
            log.emit('provider_message', at=at, message=redact(message, secret))
            protocol.feed(message)
            if protocol.ready:
                ready.set()
            if protocol.terminal:
                done.set()
        if not protocol.terminal:
            raise shared.ProviderError('Rime closed before its final transcript')

    async def wait_for(signal, timeout, purpose):
        waiter = asyncio.create_task(signal.wait())
        try:
            finished, _ = await asyncio.wait({waiter, receiver}, timeout=timeout,
                                             return_when=asyncio.FIRST_COMPLETED)
            if receiver in finished:
                await receiver  # Re-raises a provider error from the socket.
            if not signal.is_set():
                if not finished:
                    raise TimeoutError(f'Rime {purpose} timed out')
                raise shared.ProviderError(f'Rime closed before {purpose}')
        finally:
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter

    async def finalize(t0):
        nonlocal speech_end
        speech_end = t0
        protocol.requested = True
        command = protocol.finalize()
        log.emit('finalize_requested', t0_seconds=t0, message=command)
        await send(command)
        log.emit('finalize_sent', t0_seconds=t0, message=command)

    async def send_audio(frame):
        await send(protocol.audio(frame))

    try:
        receiver = asyncio.create_task(receive())
        await send(protocol.setup())
        await wait_for(ready, config['ready_timeout_seconds'], 'start acknowledgment')
        log.emit('model_accepted', model=config['model'], verification='stt_endpoint_session_accepted',
                 request_id=protocol.request_id)
        sender = asyncio.create_task(stream_audio(pcm, speech_frames, send_audio, finalize, log,
                                                  sample_rate=config['sample_rate'],
                                                  transmitted_silence_frames=transmitted_silence_frames(config)))
        finished, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        if receiver in finished and not sender.done():
            await receiver  # A provider error ends the attempt here.
            # A clean done can only follow end, after which the sender has nothing left to send.
            if not protocol.requested:
                raise shared.ProviderError('Rime closed before all audio was sent')
        await sender
        await wait_for(done, max(0, speech_end + config['finalize_timeout_seconds'] - log.now()),
                       'final transcript')
        protocol.closing = True
        log.emit('close_stream_requested')
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
    log.emit('connection_requested', url=url)
    async with connect(url, additional_headers=headers, open_timeout=15, close_timeout=5,
                       max_size=8 * 1024 * 1024) as ws:
        log.emit('connection_open', request_id=ws.response.headers.get('x-request-id'))
        await exchange(ws, pcm, speech_frames, config, log, secret=key)


def replay(events, config, cutoff=float('inf')):
    snap, result = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    result['model_verification_basis'] = 'stt_endpoint_session_accepted; model_version_not_exposed'
    return snap, result
