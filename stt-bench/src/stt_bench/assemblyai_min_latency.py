"""Explicit user-requested AssemblyAI profile; baseline adapter stays frozen."""
import asyncio
import contextlib
import json
from pathlib import Path
from datetime import datetime, timezone
from .data import write_json
from urllib.parse import urlencode
from websockets.asyncio.client import connect
from . import provider_protocol as shared
from .assemblyai_pacing import stream_audio


class Protocol(shared.Protocol):
    def finalize(self):
        # Opt-in only; historical profiles retain native endpointing.
        return {'type': 'ForceEndpoint'} if self.config.get('force_endpoint') else None

    def finish(self, frames):
        return {'type': 'Terminate'}

    def feed(self, message):
        super().feed(message)
        kind = message.get('type')
        if kind == 'Begin':
            actual = message.get('configuration', {})
            if actual.get('model') != self.config['model'] or actual.get('mode') != self.config['mode']:
                raise shared.ProviderError('Model mismatch: requested model/mode not confirmed')
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
    if config.get('model') != 'universal-3-5-pro' or config.get('mode') != 'min_latency' or config.get('language_codes') != ['en']:
        raise ValueError('Unexpected AssemblyAI benchmark configuration')
    params = dict(sample_rate=16000, speech_model=config['model'], encoding='pcm_s16le',
                  inactivity_timeout=20, mode=config['mode'], language_codes=json.dumps(config['language_codes']))
    return config['endpoint'] + '?' + urlencode(params), {'Authorization': key}


async def transcribe(pcm, speech_frames, config, key, log):
    receipt_path = config.get('rate_receipt_path')
    marked = False
    def mark(stage):
        nonlocal marked
        if receipt_path and not marked:
            write_json(Path(receipt_path), {'batch_id':config['batch_id'], 'stage':stage,
                       'wall_time':datetime.now(timezone.utc).isoformat()})
            marked = True
    try:
        url, headers = connection(config, key)
        log.emit('connection_requested', url=url)
        async with connect(url, additional_headers=headers, open_timeout=15,
                           close_timeout=5, max_size=8 * 1024 * 1024) as ws:
            log.emit('connection_open')
            mark('handshake_complete')
            try:
                await shared.exchange(ws, pcm, speech_frames, config, log, key, protocol_factory=Protocol, streamer=stream_audio)
            finally:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.send(json.dumps({'type': 'Terminate'})), 2)
    finally:
        mark('connection_attempt_finished')


def replay(events, config, cutoff=float('inf')):
    return shared.replay(events, config, cutoff, protocol_factory=Protocol)
