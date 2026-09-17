"""Gradium's direct WebSocket contract; no SDK retries or hidden audio buffering."""
import base64
from functools import partial
from .streaming import stream_audio, transmitted_silence_frames
from websockets.asyncio.client import connect
from . import provider_protocol as shared


def validate(config):
    expected = dict(language='en', input_format='pcm', sample_rate=24000,
                    encoding='pcm_s16le', channels=1, frame_ms=20,
                    finalization='manual_at_speech_end', finalize_ack_supported=True,
                    completion_basis=('end_of_stream_after_speech' if transmitted_silence_frames(config) == 0
                                      else 'end_of_stream_after_tail'), requires_final_only_completion=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Gradium requires {key}={value!r}')
    if config.get('transcript_reconstruction', 'gradium-segment-finality-v1') not in (
            'gradium-segment-finality-v1', 'gradium-append-only-text-v2'):
        raise ValueError('Unknown Gradium transcript reconstruction')
    # Conservative intersection of the current guide and settings reference.
    delay = config.get('delay_in_frames')
    if type(delay) is not int or not 7 <= delay <= 55:
        raise ValueError('Gradium delay must be an integer from 7 through 55')
    if not config.get('expected_resolved_model'):
        raise ValueError('Gradium requires the observed server model identifier')


def connection(config, key):
    return config['endpoint'], {'x-api-key': key}


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        self.active = None

    def setup(self):
        return dict(type='setup', model_name=self.config['model'],
                    input_format=self.config['input_format'], retry_for_s=0,
                    json_config=dict(language=self.config['language'],
                                     delay_in_frames=self.config['delay_in_frames']))

    def audio(self, frame):
        return dict(type='audio', audio=base64.b64encode(frame).decode('ascii'))

    def finalize(self):
        return dict(type='flush', flush_id=1)

    def finish(self, frames):
        return dict(type='end_of_stream')

    def feed(self, message):
        super().feed(message)  # Shared error handling; never includes server secrets.
        kind = message.get('type')
        if kind == 'ready':
            self.ready = True
            self.model_mismatch |= message.get('model_name') != self.config['expected_resolved_model']
            if message.get('sample_rate') != 24000 or message.get('delay_in_frames') != self.config['delay_in_frames']:
                raise shared.ProviderError('Gradium accepted different audio or delay settings')
        elif kind in ('text', 'end_text'):
            if message.get('stream_id', 0) != 0:
                self.unsupported = True
                return
            if kind == 'text':
                if self.active is not None:
                    self.unsupported = True  # Never silently replace an unclosed segment.
                self.active = self.sequence
                self.sequence += 1
                # text carries an appended segment; end_text supplies its audio stop time.
                # Legacy configurations preserve the previous finality interpretation.
                self.put(self.active, message.get('text', ''),
                         self.config.get('transcript_reconstruction') == 'gradium-append-only-text-v2')
            elif self.active is None:
                self.unsupported = True
            else:
                if self.active in self.partials:
                    self.put(self.active, self.partials[self.active], True)
                self.active = None
        elif kind == 'flushed' and self.requested and message.get('flush_id') == 1:
            self.ack = True
        elif kind == 'end_of_stream' and self.closing:
            # Live Gradium evidence: the final text can lack end_text. A successful
            # terminal response closes that segment; do not backdate its finality.
            if self.active is not None:
                if self.active in self.partials:
                    self.put(self.active, self.partials[self.active], True)
                self.active = None
            self.terminal = True


async def exchange(ws, pcm, speech_frames, config, log, secret=''):
    sender = partial(stream_audio, transmitted_silence_frames=transmitted_silence_frames(config))
    await shared.exchange(ws, pcm, speech_frames, config, log, secret,
                          protocol_factory=Protocol, streamer=sender)


async def transcribe(pcm, speech_frames, config, key, log):
    url, headers = connection(config, key)
    log.emit('connection_requested', url=url)
    async with connect(url, additional_headers=headers, open_timeout=15, close_timeout=5,
                       max_size=8 * 1024 * 1024) as ws:
        log.emit('connection_open')
        await exchange(ws, pcm, speech_frames, config, log, secret=key)


def replay(events, config, cutoff=float('inf')):
    snap, result = shared.replay(events, config, cutoff, protocol_factory=Protocol)
    from .timing_observations import observations
    result['timing_observations'] = observations(events, config, result, cutoff=cutoff)
    result['model_versions'] = list(dict.fromkeys(e['message']['model_name'] for e in events
        if e['time_seconds'] <= cutoff and e['kind'] == 'provider_message'
        and e['message'].get('type') == 'ready' and e['message'].get('model_name')))
    result['model_verification_basis'] = 'ready_model_name_matches_expected_resolved_model'
    return snap, result
