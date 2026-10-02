"""Offline checks for opt-in future profiles, including the published AssemblyAI mode."""
import asyncio
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import pytest
from stt_bench import assemblyai, assemblyai_min_latency, provider_protocol, trial_providers, providers
from stt_bench.streaming import EventLog, read_events

PROFILE = Path('config/profiles/finalization-v1')
MODELS = [p.stem for p in sorted(PROFILE.glob('*.json'))]

def load(model):
    return json.loads((PROFILE / f'{model}.json').read_text())

@pytest.mark.parametrize('model', MODELS)
def test_future_profile_preserves_other_settings(model):
    c = load(model)
    old = json.loads(Path(c['base_config']).read_text())
    providers.validate(c)
    allowed = {'profile_version', 'base_config'}
    if c['provider'] == 'assemblyai':
        allowed |= {'force_endpoint', 'finalization'}
        assert providers.assembly_adapter(c).Protocol(c).finalize() == {'type': 'ForceEndpoint'}
        assert providers.assembly_adapter(old).Protocol(old).finalize() is None
        if c.get('mode') == 'min_latency':
            url, _ = providers.assembly_adapter(c).connection(c, 'fixture')
            q = parse_qs(urlsplit(url).query)
            assert q['mode'] == ['min_latency'] and json.loads(q['language_codes'][0]) == ['en']
            assert c['frame_ms'] == 60
    elif c['provider'] == 'speechmatics':
        allowed |= {'force_end_of_utterance', 'max_delay', 'max_delay_mode', 'finalization', 'finalize_ack_supported'}
        p = provider_protocol.Protocol(c)
        cfg = p.setup()['transcription_config']
        assert cfg['max_delay'] == 1.0 and cfg['max_delay_mode'] == 'flexible'
        p.feed({'message': 'EndOfUtterance', 'forced': True})
        assert not p.ack
        p.requested = True
        p.feed({'message': 'EndOfUtterance'})
        assert not p.ack
        p.feed({'message': 'EndOfUtterance', 'forced': True})
        assert p.ack
        assert provider_protocol.Protocol(old).finalize() is None
    else:
        allowed |= {'voice_profile'}
        assert c['voice_profile'] == {'enableVoiceProfile': False}
    assert {k for k in c.keys() | old.keys() if c.get(k) != old.get(k)} == allowed

@pytest.mark.parametrize('model', MODELS)
def test_force_request_before_tail_and_late_text_retained(tmp_path, model):
    config = load(model)
    factory = (assemblyai_min_latency.Protocol if config.get('mode') == 'min_latency' else assemblyai.Protocol) if config['provider'] == 'assemblyai' else trial_providers.Protocol if config['provider'] == 'inworld' else provider_protocol.Protocol
    is_inworld = config['provider'] == 'inworld'
    is_assembly = config['provider'] == 'assemblyai'

    async def scenario():
        class Socket:
            def __init__(self):
                self.queue = asyncio.Queue()
                self.sent = []
                self.queue.put_nowait(json.dumps({'type': 'Begin', 'configuration': {'model': config['model'], 'mode': config.get('mode')}} if is_assembly else {'message': 'RecognitionStarted'}))

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await self.queue.get()

            async def send(self, value):
                self.sent.append(value)
                if isinstance(value, bytes):
                    return
                m = json.loads(value)
                if m.get('message') == 'ForceEndOfUtterance':
                    self.queue.put_nowait(json.dumps({'message': 'EndOfUtterance', 'forced': True}))
                if m.get('type') == 'Terminate' or m.get('message') == 'EndOfStream' or 'closeStream' in m:
                    messages = ([{'type': 'Turn', 'turn_order': 0, 'transcript': 'late words', 'end_of_turn': True},
                                 {'type': 'Termination'}] if is_assembly else [{'result': {'transcription': {'transcript': 'late words', 'isFinal': True}}}, {'result': {'usage': {'modelId': config['model'], 'transcribedAudioMs': 1200}}}] if is_inworld else [
                                 {'message': 'AddTranscript', 'metadata': {'start_time': 0, 'end_time': 1, 'transcript': 'late words'}},
                                 {'message': 'EndOfTranscript'}])
                    for message in messages:
                        self.queue.put_nowait(json.dumps(message))

        async def stream(pcm, frames, send, finalize, log, **kwargs):
            await send(b'speech')
            t0 = log.now()
            log.emit('speech_end', at=t0)
            await finalize(t0)
            await send(b'tail')
            log.emit('audio_complete')

        ws = Socket()
        log = EventLog(tmp_path / 'events.jsonl')
        try:
            await asyncio.wait_for(provider_protocol.exchange(ws, b'', 1, config, log,
                                    protocol_factory=factory, streamer=stream), 1)
        finally:
            log.close()
        assert json.loads(ws.sent[0])['transcribeConfig']['voiceProfileConfig'] == {'enableVoiceProfile': False} if is_inworld else ws.sent[0 if is_assembly else 1] == b'speech'
        force = {'type': 'ForceEndpoint'} if is_assembly else {'endTurn': {}} if is_inworld else {'message': 'ForceEndOfUtterance'}
        tail = json.dumps(trial_providers.Protocol(config).audio(b'tail')) if is_inworld else b'tail'
        assert ws.sent.index(json.dumps(force)) < ws.sent.index(tail)
        events = read_events(tmp_path / 'events.jsonl')
        snap, reduced = provider_protocol.replay(events, config, protocol_factory=factory)
        assert snap['final_text'] == 'late words'
        assert reduced['transcript_complete']
        assert reduced['finalize_latency_ms'] is None if is_assembly or is_inworld else reduced['finalize_latency_ms'] is not None

    asyncio.run(scenario())
