import io
import json
import urllib.error

import numpy as np
import pytest
import soundfile as sf

from stt_bench import batch
from stt_bench.data import FRAME_SAMPLES, RATE, SILENCE_FRAMES, sha256

CONFIG = batch.batch_config('nemotron-3.5-transcribe-30b-a4b')


def answer(text, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': text}}]}


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_payload_carries_the_clip_as_wav_and_the_configured_options():
    payload = batch.request_payload(CONFIG, b'RIFF-bytes')
    audio = payload['messages'][0]['content'][1]['audio_url']['url']
    assert audio == 'data:audio/wav;base64,UklGRi1ieXRlcw=='
    assert payload['model'] == 'nvidia/Nemotron-3.5-Transcribe-30B-A4B' and payload['temperature'] == 0
    assert payload['chat_template_kwargs'] == {'enable_thinking': False} and payload['stream'] is False


def test_only_a_complete_answer_is_a_transcript():
    assert batch.read_transcript(answer(' Hello there. ')) == 'Hello there.'
    with pytest.raises(RuntimeError, match='finish_reason'):
        batch.read_transcript(answer('Hello', finish='length'))
    with pytest.raises(RuntimeError, match='Endpoint error'):
        batch.read_transcript({'error': {'message': 'bad audio'}})


def test_a_transient_status_is_retried_and_a_refusal_is_not(monkeypatch):
    calls = []

    class Opener:
        def __init__(self, replies):
            self.replies = replies

        def open(self, request, timeout):
            calls.append(request.get_header('User-agent'))
            reply = self.replies.pop(0)
            if isinstance(reply, int):
                raise urllib.error.HTTPError(request.full_url, reply, 'status', {}, None)
            return Reply(json.dumps(reply).encode())

    monkeypatch.setattr(batch.urllib.request, 'build_opener', lambda *h: Opener([503, answer('Hi.')]))
    assert batch.transcribe_wav(CONFIG, b'wav', sleep=lambda s: None) == ('Hi.', 2)
    assert calls == ['brev-transcribe-client/1.0'] * 2
    monkeypatch.setattr(batch.urllib.request, 'build_opener', lambda *h: Opener([403]))
    with pytest.raises(RuntimeError, match='HTTP 403'):
        batch.transcribe_wav(CONFIG, b'wav', sleep=lambda s: None)


def frozen_dataset(tmp_path, references):
    audio_dir = tmp_path / 'audio'
    audio_dir.mkdir()
    clips = []
    for index, reference in enumerate(references):
        speech = 5
        pcm = np.zeros((speech + SILENCE_FRAMES) * FRAME_SAMPLES, dtype=np.int16)
        pcm[:speech * FRAME_SAMPLES] = np.random.default_rng(index).integers(-2000, 2000, speech * FRAME_SAMPLES)
        path = audio_dir / f'c{index}.wav'
        sf.write(path, pcm, RATE, subtype='PCM_16')
        clips.append(dict(clip_id=f'c{index}', audio=f'audio/c{index}.wav', audio_sha256=sha256(path), reference=reference,
                          condition='clean', speech_frames=speech, total_frames=speech + SILENCE_FRAMES))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=2, clips=clips)))
    return manifest


def test_a_rerun_sends_only_unsaved_clips_and_failures_are_scored_as_unusable(tmp_path):
    manifest = frozen_dataset(tmp_path, ['the cat sat', 'a dog ran', 'birds sing'])
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'transcripts.jsonl').write_text(json.dumps(dict(clip_id='c0', transcript='the cat sat', attempts=1)) + '\n')
    sent = []

    def transcribe(wav):
        sent.append(wav)
        if len(sent) == 2:
            raise RuntimeError('Incomplete transcript')
        return 'a dog ran', 1

    loaded = batch.transcribe_subset(manifest, out, transcribe, concurrency=1)
    assert len(sent) == 2 and all(wav.startswith(b'RIFF') for wav in sent)
    summary = batch.score_subset(loaded, out)
    assert (summary['planned'], summary['transcribed'], summary['usable']) == (3, 3, 2)
    assert summary['wer']['wer'] == 0.0
    failed = [c for c in json.loads((out / 'results.json').read_text())['clips'] if not c['usable']]
    assert len(failed) == 1 and failed[0]['error'] == 'Incomplete transcript'
