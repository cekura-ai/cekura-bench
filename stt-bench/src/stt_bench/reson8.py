"""Reson8 realtime API: binary PCM and correlated flush confirmations."""
from urllib.parse import urlencode
from functools import partial
from websockets.asyncio.client import connect
from . import provider_protocol as shared
from .streaming import stream_audio, transmitted_silence_frames

SPEECH_FLUSH = 'speech-end'
COMPLETE_FLUSH = 'audio-complete'
# The US/global deployment serves only rumble-2, so it is always requested by name.
GLOBAL_ENDPOINT = 'wss://api.global.reson8.dev/v1/speech-to-text/realtime'
GLOBAL_MODEL = 'rumble-2'
# 'realtime' is the legacy local label for the unversioned EU default; it is never sent.
LEGACY_MODEL_LABEL = 'realtime'


def validate(config):
    tail = transmitted_silence_frames(config)
    profile = config.get('transport_profile')
    if profile not in (None, 'reson8-stop-after-flush-v2'):
        raise ValueError('Unknown Reson8 transport profile')
    if profile == 'reson8-stop-after-flush-v2' and tail != 0:
        raise ValueError('Reson8 stop-after-flush profile must omit the artificial tail')
    expected = dict(language='en', sample_rate=16000, encoding='pcm_s16le',
                    channels=1, frame_ms=20, include_interim=True, include_timestamps=True,
                    filler_mode='verbatim', finalization='manual_at_speech_end',
                    completion_basis='correlated_flush_after_all_audio',
                    finalize_ack_supported=True, requires_final_only_completion=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Reson8 requires {key}={value!r}')
    if config.get('endpoint') == GLOBAL_ENDPOINT and config.get('model') != GLOBAL_MODEL:
        raise ValueError(f'Reson8 global endpoint serves only {GLOBAL_MODEL}')


def connection(config, key):
    query = dict(encoding=config['encoding'], sample_rate=config['sample_rate'],
                 channels=config['channels'], language=config['language'],
                 include_interim='true', include_timestamps='true',
                 filler_mode=config['filler_mode'], diarize='false')
    if config['model'] != LEGACY_MODEL_LABEL:
        query['model'] = config['model']
    return config['endpoint'] + '?' + urlencode(query), {'Authorization': f'ApiKey {key}'}


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        # The successful HTTP upgrade is readiness; no setup/ready JSON exists.
        self.ready = True

    def finalize(self):
        return dict(type='flush_request', id=SPEECH_FLUSH)

    def finish(self, frames):
        return dict(type='flush_request', id=COMPLETE_FLUSH)

    def feed(self, message):
        super().feed(message)
        kind = message.get('type')
        if kind == 'transcript':
            # is_final must be present because include_interim=true was requested.
            if type(message.get('is_final')) is not bool or not isinstance(message.get('text'), str):
                raise shared.ProviderError('Reson8 transcript lacks finality or text')
            self.put(self.sequence, message['text'], message['is_final'])
            if message['is_final']:
                self.sequence += 1
        elif kind == 'flush_confirmation':
            if self.requested and message.get('id') == SPEECH_FLUSH:
                self.ack = True
            if self.closing and message.get('id') == COMPLETE_FLUSH:
                self.terminal = True


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    # A versioned opt-in keeps saved legacy configurations reproducible. Never
    # send new silence audio while the provider is processing the speech flush.
    streamer = partial(stream_audio, transmitted_silence_frames=transmitted_silence_frames(config))
    await shared.exchange(ws, pcm, speech_frames, config, log, secret,
                          protocol_factory=Protocol, streamer=streamer)


async def transcribe(pcm, speech_frames, config, key, log):
    url, headers = connection(config, key)
    log.emit('connection_requested', url=url)
    async with connect(url, additional_headers=headers, open_timeout=15, close_timeout=5,
                       max_size=8 * 1024 * 1024) as ws:
        log.emit('connection_open')
        await exchange(ws, pcm, speech_frames, config, log, secret=key)


def replay(events, config, cutoff=float('inf')):
    snap, result = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    basis = ('realtime_endpoint_session_accepted' if config['model'] == LEGACY_MODEL_LABEL
             else 'requested_model_session_accepted')
    result['model_verification_basis'] = basis + '; model_version_not_exposed'
    return snap, result
