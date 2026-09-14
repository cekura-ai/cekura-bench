"""AssemblyAI v3 streaming, native turns, and explicit session termination."""
import asyncio
import contextlib
import json
from urllib.parse import urlencode
from websockets.asyncio.client import connect
from . import provider_protocol as shared
from .assemblyai_pacing import stream_audio


class Protocol(shared.Protocol):
    def finalize(self):
        # Preserve native endpointing throughout long conversations.
        return None

    def finish(self, frames):
        return {'type': 'Terminate'}

    def feed(self, message):
        super().feed(message)
        kind = message.get('type')
        if kind == 'Begin':
            self.ready = True
            selected = message.get("configuration", {}).get("model")
            self.model_mismatch |= bool(selected and selected != self.config["model"])
        elif kind == 'Turn':
            order = message.get('turn_order')
            if (type(order) is not int or order < 0 or
                    type(message.get('end_of_turn')) is not bool or
                    not isinstance(message.get('transcript'), str)):
                raise shared.ProviderError('Invalid AssemblyAI turn')
            # Formatted revisions replace the same turn; they never append twice.
            if message['end_of_turn']:
                self.finals[order] = message['transcript']
                self.partials.pop(order, None)
            elif order not in self.finals:
                self.partials[order] = message['transcript']
        elif kind == 'Termination' and self.closing:
            self.terminal = True

    def snapshot(self):
        keys = sorted(set(self.finals) | set(self.partials))
        return dict(text=' '.join(self.finals.get(k, self.partials.get(k, '')) for k in keys).strip(),
                    final_text=' '.join(self.finals[k] for k in keys if k in self.finals).strip(),
                    partial_text=' '.join(self.partials[k] for k in keys if k in self.partials).strip(),
                    provisional=bool(self.partials), reconstruction_status='supported')


def connection(config, key):
    return config['endpoint'] + '?' + urlencode(dict(sample_rate=16000,
        speech_model=config['model'], encoding='pcm_s16le', inactivity_timeout=20)), {'Authorization': key}


async def transcribe(pcm, speech_frames, config, key, log):
    url, headers = connection(config, key)
    log.emit('connection_requested', url=url)
    async with connect(url, additional_headers=headers, open_timeout=15,
                       close_timeout=5, max_size=8 * 1024 * 1024) as ws:
        log.emit('connection_open')
        try:
            await shared.exchange(ws, pcm, speech_frames, config, log, key, protocol_factory=Protocol, streamer=stream_audio)
        finally:
            # Also close billed sessions on timeout, cancellation, or sender failure.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.send(json.dumps({'type': 'Terminate'})), 2)


def replay(events, config, cutoff=float('inf')):
    return shared.replay(events, config, cutoff, protocol_factory=Protocol)
