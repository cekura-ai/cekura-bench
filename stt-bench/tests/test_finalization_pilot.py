import asyncio
import json
from pathlib import Path

import pytest

from scripts.finalization_pilot import MODELS, profiles, select_clips, summarize
from stt_bench import assemblyai, provider_protocol, trial_providers
from stt_bench.streaming import EventLog, read_events


def test_profiles_preserve_defaults_and_limit_changes():
    expected = {
        MODELS[0]: {'force_endpoint', 'finalization'},
        MODELS[1]: {'force_end_of_utterance', 'max_delay', 'finalization', 'finalize_ack_supported'},
        MODELS[2]: {'force_end_of_utterance', 'max_delay', 'finalization', 'finalize_ack_supported'},
        MODELS[3]: {'voice_profile'},
    }
    for model in MODELS:
        pair = profiles(model)
        original = json.loads(Path(f'config/models/{model}.json').read_text())
        assert pair['baseline'] == original
        assert {k for k in original.keys() | pair['candidate'].keys()
                if original.get(k) != pair['candidate'].get(k)} == expected[model]
    p = trial_providers.Protocol(profiles(MODELS[3])['candidate'])
    assert p.setup()['transcribeConfig']['voiceProfileConfig'] == {'enableVoiceProfile': False}
    assert p.finalize() == {'endTurn': {}}


def test_speechmatics_configuration_and_forced_ack_only():
    pair = profiles(MODELS[1])
    baseline = provider_protocol.Protocol(pair['baseline'])
    assert baseline.finalize() is None
    assert 'max_delay' not in baseline.setup()['transcription_config']
    p = provider_protocol.Protocol(pair['candidate'])
    assert p.setup()['transcription_config']['max_delay'] == 1.0
    assert p.finalize() == {'message': 'ForceEndOfUtterance'}
    p.feed({'message': 'EndOfUtterance', 'forced': True})
    assert not p.ack
    p.requested = True
    p.feed({'message': 'EndOfUtterance'})
    assert not p.ack
    p.feed({'message': 'EndOfUtterance', 'forced': True})
    assert p.ack


@pytest.mark.parametrize('model', MODELS[:3])
def test_force_request_before_tail_and_late_text_retained(tmp_path, model):
    config = profiles(model)['candidate']
    factory = assemblyai.Protocol if config['provider'] == 'assemblyai' else provider_protocol.Protocol
    is_assembly = config['provider'] == 'assemblyai'

    async def scenario():
        class Socket:
            def __init__(self):
                self.queue = asyncio.Queue()
                self.sent = []
                self.queue.put_nowait(json.dumps({'type': 'Begin'} if is_assembly else {'message': 'RecognitionStarted'}))

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
                if m.get('type') == 'Terminate' or m.get('message') == 'EndOfStream':
                    messages = ([{'type': 'Turn', 'turn_order': 0, 'transcript': 'late words', 'end_of_turn': True},
                                 {'type': 'Termination'}] if is_assembly else [
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
        assert ws.sent[0 if is_assembly else 1] == b'speech'
        force = {'type': 'ForceEndpoint'} if is_assembly else {'message': 'ForceEndOfUtterance'}
        assert ws.sent.index(json.dumps(force)) < ws.sent.index(b'tail')
        events = read_events(tmp_path / 'events.jsonl')
        snap, reduced = provider_protocol.replay(events, config, protocol_factory=factory)
        assert snap['final_text'] == 'late words'
        assert reduced['transcript_complete']
        assert reduced['finalize_latency_ms'] is None if is_assembly else reduced['finalize_latency_ms'] is not None

    asyncio.run(scenario())


def test_selection_is_fixed_public_only():
    source = Path('datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/full/manifest.json')
    manifest = json.loads(source.read_text())
    clips = select_clips(manifest)
    assert len(clips) == len({c['clip_id'] for c in clips}) == 20
    assert clips == select_clips({**manifest, 'clips': list(reversed(manifest['clips']))})
    assert all(c['condition'] == 'public_anchor' and 1 < c['submitted_seconds'] <= 20 for c in clips)


def test_empty_pilot_keeps_planned_failures_visible():
    summary = summarize([])
    for model in summary['models'].values():
        assert model['paired_clips'] == 0
        for variant in model['variants'].values():
            assert variant['not_run'] == variant['planned'] == 20
            assert variant['paired_wer']['wer'] is None
