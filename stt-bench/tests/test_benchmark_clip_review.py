"""Review evidence checks: exact score agreement, lossless audio, portable export."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_clip_review import comparison, prepare_audio, share_zip
from build_benchmark_html import render
from stt_bench.score import word_errors


@pytest.mark.parametrize('reference,transcript', [
    ('the red cat sat', 'the blue cat sat today'),
    ('one two three', 'one three'),
    ('hello world', ''),
    ('', 'extra words'),
    ('I have twenty dollars.', 'i have $20'),
    ('one one two', 'one two two'),
])
def test_diff_reproduces_scored_words(reference, transcript):
    counts = word_errors(reference, transcript)
    diff = comparison(reference, transcript, counts)
    assert ' '.join(a[1] for a in diff if a[1]) == counts['reference_normalized']
    assert ' '.join(a[2] for a in diff if a[2]) == counts['hypothesis_normalized']
    s = sum(min(len(r.split()), len(h.split())) for kind, r, h in diff if kind == 'substitute')
    assert s == counts['substitutions']


def test_reject_different_text_and_saved_scores():
    counts = word_errors('hello world', 'hello there')
    with pytest.raises(ValueError, match='text differs'):
        comparison('hello world', 'changed transcript', counts)
    counts['substitutions'] = 0
    with pytest.raises(ValueError, match='counts differ'):
        comparison('hello world', 'hello there', counts)


def test_audio_hash_and_lossless_samples_checked_even_on_rebuild(tmp_path):
    source, target = tmp_path / 'source.wav', tmp_path / 'copy.flac'
    pcm = np.array([-32768, -100, 0, 100, 32767] * 400, dtype=np.int16)
    sf.write(source, pcm, 16000, subtype='PCM_16')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = prepare_audio(source, digest, target)
    assert result['seconds'] == .125
    assert np.array_equal(sf.read(target, dtype='int16')[0], pcm)
    sf.write(target, np.zeros(2000, dtype=np.int16), 16000, subtype='PCM_16')
    with pytest.raises(ValueError, match='not lossless'):
        prepare_audio(source, digest, target)
    with pytest.raises(ValueError, match='changed'):
        prepare_audio(source, 'wrong hash', target)


def test_share_zip_only_includes_current_assets(tmp_path):
    page = tmp_path / 'custom-name.html'
    page.write_text('review')
    (tmp_path / 'included.flac').write_bytes(b'audio')
    (tmp_path / 'old-private.flac').write_bytes(b'private')
    bundle = share_zip(page, dict(clips=[dict(audio='included.flac')], includes_private=False))
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {'index.html', 'README.txt', 'included.flac'}
        assert archive.read('index.html') == b'review'


def test_transcript_markup_stays_data():
    text = '</script><img src=x onerror=alert(1)>'
    page = render({'clip_review': {'transcript': text}}, '__CLIP_REVIEW_UI__<script>__BENCHMARK_DATA__</script>')
    assert text not in page
    assert '\\u003c/script\\u003e' in page
    assert 'id="verify-clips"' in page
    assert '__CLIP_REVIEW_UI__' not in page
