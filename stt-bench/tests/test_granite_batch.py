import io
import wave

import pytest

from stt_bench import granite_batch


def words(*spans):
    return [{'text': 'w', 'start': s, 'end': e} for s, e in spans]


def test_cuts_land_mid_pause_and_keep_every_piece_within_the_limit():
    # Speech in 10 s blocks with a 1 s pause after each: 0-10, 11-21, 22-32, ...
    spans = [(i * 11.0, i * 11.0 + 10.0) for i in range(10)]
    cuts = granite_batch.cut_points(words(*spans), 109.0, max_chunk=30, min_gap=0.6)
    assert cuts == [21.5, 43.5, 65.5, 87.5]
    bounds = [0.0, *cuts, 109.0]
    assert all(b - a <= 30 for a, b in zip(bounds, bounds[1:]))


def test_no_cut_when_the_recording_fits():
    assert granite_batch.cut_points(words((0, 5), (6, 10)), 12.0, max_chunk=110, min_gap=0.6) == []


def test_a_window_without_a_pause_is_refused_rather_than_cut_mid_word():
    with pytest.raises(ValueError):
        granite_batch.cut_points(words((0, 50), (50.1, 100)), 100.0, max_chunk=30, min_gap=0.6)


def test_pieces_join_back_to_the_whole_recording(tmp_path):
    rate = 16000
    pcm = bytes(range(256)) * (rate * 2 * 3 // 256)          # 3 s of 16-bit mono
    path = tmp_path / 'a.wav'
    with wave.open(str(path), 'wb') as sink:
        sink.setnchannels(1); sink.setsampwidth(2); sink.setframerate(rate); sink.writeframes(pcm)
    parts = granite_batch.pieces(str(path), [1.0, 2.0])
    frames = []
    for part in parts:
        with wave.open(io.BytesIO(part)) as source:
            assert source.getframerate() == rate
            frames.append(source.readframes(source.getnframes()))
    assert [len(f) for f in frames] == [rate * 2] * 3
    assert b''.join(frames) == pcm


def test_request_names_the_audio_part_and_prompt_and_scales_the_token_budget():
    config = {'model': 'm', 'temperature': 0, 'min_max_tokens': 256, 'max_tokens_per_audio_second': 8, 'prompt': 'p'}
    body = granite_batch.request_body(config, b'RIFF', 100.0)
    assert body['max_tokens'] == 800
    assert body['messages'][0]['content'][0]['type'] == 'audio_url'
    assert body['messages'][0]['content'][0]['audio_url']['url'].startswith('data:audio/wav;base64,')
    assert body['messages'][0]['content'][1] == {'type': 'text', 'text': 'p'}
    assert granite_batch.request_body(config, b'RIFF', 2.0)['max_tokens'] == 256


def test_the_shipped_config_names_no_endpoint():
    config = granite_batch.load_config(granite_batch.Path('config/batch/ibm-granite-speech-5-0-2b.json'))
    text = granite_batch.Path('config/batch/ibm-granite-speech-5-0-2b.json').read_text()
    assert 'http' not in text and 'endpoints.huggingface' not in text
    assert config['endpoint_env'] and config['token_env']
