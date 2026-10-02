"""Speechmatics Linden-1 Agent STT, with externally signalled speech end.

Wire contract: speechmatics-python-sdk e93433ce2a0f98248c9fc0bbce9bd652b1a43b45,
sdk/agent_stt. Only Agent STT segment messages contribute transcript text.
"""
from websockets.asyncio.client import connect
from . import provider_protocol as shared


def validate(config):
    if (config.get('model') != 'linden-1' or
            config.get('endpoint') != 'wss://global.rt.speechmatics.com/v2/agent' or
            config.get('turn_detection_mode') != 'external'):
        raise ValueError('Linden-1 requires the Agent endpoint and external turn detection')


class Protocol(shared.Protocol):
    def __init__(self, config):
        super().__init__(config)
        self.audio_seconds = 0
        self.delimiter = ' '

    def setup(self):
        return {
            'message': 'StartRecognition',
            'audio_format': {'type': 'raw', 'encoding': 'pcm_s16le', 'sample_rate': 16000},
            'transcription_config': {'language': 'en', 'model': 'linden-1', 'enable_partials': True},
            'turn_config': {'turn_detection_mode': 'external'},
        }

    def audio(self, frame):
        self.audio_seconds += len(frame) / 32000
        return frame

    def finalize(self):
        return {'message': 'ForceEndOfUtterance', 'timestamp': round(self.audio_seconds, 6)}

    def finish(self, frames):
        return {'message': 'EndOfStream', 'last_seq_no': frames}

    def feed(self, message):
        kind = message.get('message')
        if kind == 'Error':
            raise shared.ProviderError('Speechmatics Agent error: ' + str(message))
        if kind == 'RecognitionStarted':
            self.ready = True
            self.delimiter = (message.get('language_pack_info') or {}).get('word_delimiter', ' ')
            selected = message.get('model') or (message.get('transcription_config') or {}).get('model')
            self.model_mismatch |= bool(selected and selected != self.config['model'])
        elif kind in ('AddSegment', 'AddPartialSegment'):
            text = (message.get('segment') or {}).get('transcript')
            if not isinstance(text, str):
                raise shared.ProviderError('Agent segment lacks transcript text')
            # SDK semantics: append each final segment, replace the live partial.
            # RT AddTranscript messages may also be passed through: ignore them.
            if kind == 'AddSegment':
                self.finals[len(self.finals)] = text
                self.partials.clear()
            else:
                self.partials = {0: text} if text else {}
        elif kind == 'EndOfTranscript' and self.closing:
            self.terminal = True

    def snapshot(self):
        final = self.delimiter.join(t for t in self.finals.values() if t).strip()
        partial = self.partials.get(0, '')
        return dict(text=self.delimiter.join(t for t in (final, partial) if t),
                    final_text=final, partial_text=partial, provisional=bool(partial),
                    reconstruction_status='supported')


async def transcribe(pcm, speech_frames, config, key, log):
    validate(config)
    log.emit('connection_requested', url=config['endpoint'])
    async with connect(config['endpoint'], additional_headers={'Authorization': f'Bearer {key}'},
                       open_timeout=15, close_timeout=5, max_size=8*1024*1024) as ws:
        log.emit('connection_open')
        log.emit('session_configuration', message=Protocol(config).setup())
        await shared.exchange(ws, pcm, speech_frames, config, log, secret=key, protocol_factory=Protocol)


def replay(events, config, cutoff=float('inf')):
    return shared.replay(events, config, cutoff, protocol_factory=Protocol)
