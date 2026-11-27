"""Fish Audio realtime ASR: binary MessagePack frames and explicit (manual) turns.

Wire contract, from Fish's transcribe-stream-2 test client and guidance and
probed live on 2026-11-27 (docs: https://docs.fish.audio/api-reference/endpoint/websocket/asr-realtime):

- Every frame in both directions is a binary WebSocket frame holding one
  MessagePack map. Audio is raw pcm_s16le bytes in `data` (MessagePack bin),
  never base64.
- The model is chosen by the `model` upgrade header. The client sends
  `config`; `session_started` echoes the applied config, including the model
  name. Fish exposes no build or version, so that echo is the identity check,
  and every requested setting must be echoed before audio is sent.
- Fish's benchmark guidance for transcribe-stream-2 is manual turn detection
  with `finalize` right after the last audio frame, written text, and no
  diarization or audio-event tags. In manual mode a turn opens when Fish
  detects speech (`speech_start`) and ends on `finalize`, on `audio_done`, or
  after 60 seconds (reason `max_turn_duration`), when the next turn starts
  where it ended.
- `draft` carries the open turn's provisional text and `committed` its final
  text, keyed by `turn_index`. Every server message carries `seq`, counting up
  from 0.
- `finalize` with no open turn is ignored without a reply, so completion does
  not wait for one. The 1 s silence tail opens no turn; `audio_done` commits
  any open turn and is answered by `session_done`, after which Fish closes the
  socket with code 1000.
- An `error` message fails the attempt.
"""
import asyncio
import contextlib

import msgpack
from websockets.asyncio.client import connect

from . import provider_protocol as shared
from .credentials import redact
from .streaming import stream_audio

ENDPOINT = 'wss://api.fish.audio/v1/asr/realtime'
INFORMATIONAL = ('speech_start', 'user_turn_taken')


def validate(config):
    expected = dict(model='transcribe-stream-2', endpoint=ENDPOINT, language='en', sample_rate=16000,
                    encoding='pcm_s16le', channels=1, frame_ms=20, turn_detection_mode='manual',
                    text_format='written', diarize=False, tag_audio_events=False,
                    finalization='manual_at_speech_end', completion_basis='session_done_after_audio_done',
                    finalize_ack_supported=False, requires_final_only_completion=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Fish realtime ASR requires {key}={value!r}')
    return config


def connection(config, key):
    return config['endpoint'], {'Authorization': f'Bearer {key}', 'model': config['model']}


def session(config):
    """The settings sent in `config`, which `session_started` must echo."""
    return {'audio': {'format': config['encoding'], 'sample_rate': config['sample_rate'],
                      'channels': config['channels']},
            'turn_detection': {'mode': config['turn_detection_mode']},
            'results': {'drafts': True, 'ignore_timestamps': True, 'diarize': config['diarize'],
                        'tag_audio_events': config['tag_audio_events'], 'text_format': config['text_format']}}


def pack(value):
    return msgpack.packb(value, use_bin_type=True)


def plain(value):
    """Server events are logged as JSON evidence, so they may hold only JSON types."""
    if isinstance(value, dict):
        return all(isinstance(k, str) and plain(v) for k, v in value.items())
    if isinstance(value, list):
        return all(plain(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


def unpack(raw):
    if not isinstance(raw, (bytes, bytearray)):
        raise shared.ProviderError('Fish sent a text frame; MessagePack binary frames are expected')
    try:
        message = msgpack.unpackb(raw, raw=False)
    except Exception as exc:
        raise shared.ProviderError('Fish sent an invalid MessagePack frame') from exc
    if not isinstance(message, dict) or not plain(message):
        raise shared.ProviderError('Fish sent an unexpected event shape')
    return message


def turn_text(message):
    turn, text = message.get('turn_index'), message.get('text')
    if type(turn) is not int or turn < 0 or not isinstance(text, str):
        raise shared.ProviderError('Invalid Fish transcript event')
    return turn, text


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        self.sequence = 0
        self.request_id = None

    def setup(self):
        return {'type': 'config', **session(self.config)}

    def audio(self, frame):
        return {'type': 'audio', 'data': frame}

    def finalize(self):
        return {'type': 'finalize'}

    def finish(self, frames):
        return {'type': 'audio_done'}

    def feed(self, message):
        super().feed(message)  # Shared rejection of error events.
        kind = message.get('type')
        if message.get('seq') != self.sequence or type(message.get('seq')) is not int:
            raise shared.ProviderError('Fish events are out of sequence')
        self.sequence += 1
        if self.terminal:
            raise shared.ProviderError('Fish sent an event after session_done')
        if kind == 'session_started':
            applied = message.get('config')
            if self.ready or not isinstance(applied, dict) or not isinstance(message.get('request_id'), str):
                raise shared.ProviderError('Invalid Fish session start')
            self.model_mismatch |= applied.get('model') != self.config['model']
            # Audio is sent only after Fish confirms every setting, so a silently
            # ignored turn mode or text format cannot be measured.
            if any(not isinstance(applied.get(section), dict)
                   or any(applied[section].get(k) != v for k, v in settings.items())
                   for section, settings in session(self.config).items()):
                raise shared.ProviderError('Fish did not apply the requested session settings')
            self.ready, self.request_id = True, message['request_id']
            return
        if not self.ready:
            raise shared.ProviderError('Fish sent an event before session_started')
        if kind in INFORMATIONAL:
            return
        if kind == 'draft':
            self.put(*turn_text(message), False)
        elif kind == 'committed':
            turn, text = turn_text(message)
            if not isinstance(message.get('reason'), str):
                raise shared.ProviderError('Invalid Fish commit reason')
            self.put(turn, text, True)
            if self.requested and message['reason'] == 'finalize':
                self.ack = True
        elif kind == 'session_done':
            if not self.closing:
                raise shared.ProviderError('Fish ended the session before audio_done')
            self.terminal = True
        else:
            raise shared.ProviderError('Unexpected Fish provider event')

    def snapshot(self):
        keys = sorted(set(self.finals) | set(self.partials))
        join = lambda texts: ' '.join(' '.join(texts[k] for k in keys if k in texts).split())
        final, partial = join(self.finals), join(self.partials)
        return dict(text=join({**self.partials, **self.finals}), final_text=final, partial_text=partial,
                    provisional=bool(partial),
                    reconstruction_status='unsupported_order_or_overlap' if self.unsupported else 'supported')


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    protocol = Protocol(config)
    ready, terminal = asyncio.Event(), asyncio.Event()
    receiver = sender = None

    async def send(value):
        await ws.send(pack(value))

    async def receive():
        async for raw in ws:
            at = log.now()
            message = unpack(raw)
            log.emit('provider_message', at=at, message=redact(message, secret))
            protocol.feed(message)
            if protocol.model_mismatch:
                raise shared.ProviderError('Provider selected a different model')
            if protocol.ready:
                ready.set()
            if protocol.terminal:
                terminal.set()
        if not protocol.terminal:
            raise shared.ProviderError('Fish closed before session_done')

    async def wait_for(signal, timeout, purpose):
        waiter = asyncio.create_task(signal.wait())
        try:
            finished, _ = await asyncio.wait({waiter, receiver}, timeout=timeout,
                                             return_when=asyncio.FIRST_COMPLETED)
            if receiver in finished and not signal.is_set():
                await receiver  # Re-raises a provider error from the socket.
                raise shared.ProviderError(f'Fish closed before {purpose}')
            if not signal.is_set():
                raise TimeoutError(f'Fish {purpose} timed out')
        finally:
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter

    async def send_audio(frame):
        await send(protocol.audio(frame))

    async def finalize(t0):
        protocol.requested = True
        command = protocol.finalize()
        log.emit('finalize_requested', t0_seconds=t0, message=command)
        await send(command)
        log.emit('finalize_sent', t0_seconds=t0, message=command)

    try:
        receiver = asyncio.create_task(receive())
        await send(protocol.setup())
        await wait_for(ready, config['ready_timeout_seconds'], 'session start')
        log.emit('model_accepted', model=config['model'], verification='requested_model_echoed_by_session_started',
                 request_id=protocol.request_id)
        sender = asyncio.create_task(stream_audio(pcm, speech_frames, send_audio, finalize, log,
                                                  sample_rate=config['sample_rate']))
        finished, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        if receiver in finished:
            await receiver
            raise shared.ProviderError('Fish closed before all audio was sent')
        await sender
        # finalize gets no reply when no turn is open, so completion is audio_done -> session_done.
        protocol.closing = True
        log.emit('close_stream_requested')
        await send(protocol.finish(speech_frames + 50))
        await wait_for(terminal, config['close_timeout_seconds'], 'session_done')
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
    result['model_verification_basis'] = 'requested_model_echoed_by_session_started; model_version_not_exposed'
    # The reply to finalize is timed when there is one; completion never waits for it.
    result['finalize_latency_status'] = ('observed' if result['finalize_ack_received_seconds'] is not None
                                         else 'no_reply_to_finalize')
    return snap, result
