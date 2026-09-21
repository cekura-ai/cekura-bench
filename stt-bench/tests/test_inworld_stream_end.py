"""Exercise the real sender against a deterministic, in-process provider socket."""
import asyncio
import base64
import json
from pathlib import Path

import pytest

from stt_bench import trial_providers as wire
from stt_bench.providers import validate, reduce_events
from stt_bench.streaming import EventLog, read_events, pacing_metrics, transmitted_silence_frames


def config(tail):
    c = json.loads(Path('config/models/inworld-stt-1.json').read_text())
    return dict(c, transmitted_silence_frames=tail, close_timeout_seconds=.05)


def test_default_inworld_configuration_uses_no_tail():
    c = json.loads(Path('config/models/inworld-stt-1.json').read_text())
    validate(c)
    assert transmitted_silence_frames(c) == 0


class Socket:
    def __init__(self, missing_usage=False):
        self.queue = asyncio.Queue()
        self.sent = []
        self.ended = False
        self.tail_audio = False
        self.missing_usage = missing_usage

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.queue.get()

    def final(self, text):
        self.queue.put_nowait(json.dumps({'result': {'transcription': {
            'transcript': text, 'isFinal': True}}}))

    async def send(self, raw):
        m = json.loads(raw)
        self.sent.append(m)
        if 'audioChunk' in m and self.ended:
            self.tail_audio = True
        if 'endTurn' in m:
            self.ended = True
            self.final('hello')
        if 'closeStream' in m:
            # A legitimate late segment must survive in BOTH configurations.
            self.final('world')
            if self.tail_audio:
                self.final("I'm not sure.")
            if not self.missing_usage:
                self.queue.put_nowait(json.dumps({'result': {'usage': {
                    'modelId': 'inworld/inworld-stt-1'}}}))


@pytest.mark.parametrize('tail', [0, 50])
def test_no_audio_after_end_turn_and_every_final_retained(tmp_path, tail):
    c = config(tail)
    socket = Socket()
    # Nonzero speech followed by exactly the prepared one-second zero tail.
    speech = b'\x01\x00' * 320 * 3
    log = EventLog(tmp_path / 'raw.jsonl')
    try:
        asyncio.run(wire.exchange(socket, speech + bytes(640 * 50), 3, c, log))
    finally:
        log.close()
    audio = [base64.b64decode(m['audioChunk']['content']) for m in socket.sent if 'audioChunk' in m]
    assert b''.join(audio) == speech + bytes(640 * tail)
    end = socket.sent.index({'endTurn': {}})
    assert len(socket.sent[end + 1:-1]) == tail
    assert socket.sent[-1] == {'closeStream': {}}
    events = read_events(tmp_path / 'raw.jsonl')
    assert len([e for e in events if e['kind'] == 'audio_sent']) == 3 + tail
    result = reduce_events(events, c)
    assert result['transcript_complete']
    assert result['transcript'] == ('hello world' if tail == 0 else "hello world I'm not sure.")
    assert result['finalize_latency_ms'] is None  # No invented finalize acknowledgment.


def test_no_tail_still_requires_provider_completion(tmp_path):
    log = EventLog(tmp_path / 'raw.jsonl')
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(wire.exchange(Socket(missing_usage=True), bytes(640 * 53), 3, config(0), log))
    finally:
        log.close()
    assert not reduce_events(read_events(tmp_path / 'raw.jsonl'), config(0))['transcript_complete']


def test_cannot_silently_trim_nonzero_audio(tmp_path):
    socket = Socket()
    log = EventLog(tmp_path / 'raw.jsonl')
    try:
        with pytest.raises(ValueError, match='nonzero audio tail'):
            asyncio.run(wire.exchange(socket, bytes(640 * 53 - 1) + b'\x01', 3, config(0), log))
    finally:
        log.close()
    assert not any('audioChunk' in m for m in socket.sent)


def test_pacing_contract_is_explicit_and_legacy_remains_strict():
    events = [dict(kind='audio_sent', index=i, bytes=640, phase='speech',
                   time_seconds=(i + 1) * .02, ideal_seconds=(i + 1) * .02) for i in range(10)]
    assert pacing_metrics(events, transmitted_silence_frames=0)['valid']
    assert not pacing_metrics(events)['valid']
    events[5]['time_seconds'] += .05
    assert not pacing_metrics(events, transmitted_silence_frames=0)['valid']
    assert transmitted_silence_frames({'provider': 'inworld'}) == 50


@pytest.mark.parametrize('tail', [False, True, -1, 1, 49, 51, None, '0'])
def test_invalid_contract_rejected(tail):
    with pytest.raises(ValueError):
        validate(config(tail))


def test_other_providers_cannot_silently_omit_tail():
    c = json.loads(Path('config/models/smallest-pulse.json').read_text())
    with pytest.raises(ValueError, match='Only Inworld'):
        validate(dict(c, transmitted_silence_frames=0))


def test_probe_summary_uses_identical_valid_clips_and_counts():
    import importlib.util
    spec = importlib.util.spec_from_file_location('inworld_probe', 'scripts/inworld_stream_end_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    rows = []
    for variant in ('baseline', 'candidate'):
        for cid, valid in [('paired', True), ('failed', variant == 'baseline')]:
            rows.append(dict(clip_id=cid, variant=variant, valid=valid, sent_audio_seconds=1,
                             last_final_ms=100, word_errors=dict(substitutions=0, deletions=0,
                             insertions=1 if variant == 'baseline' else 0, reference_words=10)))
    summary = probe.summarize(rows)
    assert summary['paired_clips'] == 1 and summary['attempted_sessions'] == 4
    assert summary['variants']['baseline']['paired_wer']['wer'] == .1
    assert summary['variants']['candidate']['paired_wer']['wer'] == 0
    assert summary['variants']['baseline']['usable'] == 2
    assert summary['variants']['candidate']['usable'] == 1
