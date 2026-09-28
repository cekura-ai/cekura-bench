"""The rebuilt private dataset has the frozen shape the full harness verifies."""
import importlib.util
import json
import math

import numpy as np
import pytest
import soundfile as sf

from stt_bench.data import sha256

spec = importlib.util.spec_from_file_location('prepare_private_longform', 'scripts/prepare_private_longform.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def dataset(root):
    rng = np.random.default_rng(7)
    for n in range(1, 5):
        folder = root/f'conversation-0{n}'
        folder.mkdir(parents=True)
        for label in 'AB':
            samples = 24000 + 331*n + (17 if label == 'B' else 0)
            sf.write(folder/f'{label}.flac', rng.uniform(-.3, .3, samples), 48000, subtype='PCM_24')
            segments = [dict(index=0, start=0, end=.4, text='one two', words=[
                dict(word='one', start=0.0, end=.2), dict(word='two', start=.3, end=.2)])]
            (folder/f'{label}.json').write_text(json.dumps(dict(
                audio_file=f'{label}.flac', conversation_id=f'conv-{n}', speaker_label=label,
                duration_seconds=samples/48000, transcript='One, two.', segments=segments)))


def test_rebuild_matches_the_frozen_shape_and_never_overwrites(tmp_path):
    dataset(tmp_path/'private-dataset')
    manifest = tmp_path/'workspace/manifest.json'
    builder.prepare(tmp_path/'private-dataset', tmp_path/'audio', manifest)
    m = json.loads(manifest.read_text())
    assert len(m['clips']) == 8 and len(m['source_inventory']) == 16
    for clip in m['clips']:
        speech_samples = math.ceil(sf.info(tmp_path/'private-dataset'/clip['source_audio']).frames/3)
        assert clip['speech_frames'] == math.ceil(speech_samples/320)
        assert clip['submitted_seconds'] == (clip['speech_frames']+50)/50
        assert clip['reference'] == 'One, two.'
        assert [w['timing_valid'] for w in clip['words']] == [True, False]
        for rate in (16000, 24000):
            audio = tmp_path/'audio'/clip['audio'][str(rate)]['path']
            assert sha256(audio) == clip['audio'][str(rate)]['sha256']
            pcm, sr = sf.read(audio, dtype='int16')
            assert sr == rate and len(pcm) == (clip['speech_frames']+50)*rate//50
            assert not np.any(pcm[-rate:]) and np.any(pcm[:rate//10])
    with pytest.raises(ValueError, match='never overwritten'):
        builder.prepare(tmp_path/'private-dataset', tmp_path/'audio', manifest)
