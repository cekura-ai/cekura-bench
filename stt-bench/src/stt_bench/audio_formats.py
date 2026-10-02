"""Hash-bound OpenAI derivatives. Frozen source manifests are never rewritten."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from .data import sha256, write_json

CONVERSION = {'algorithm': 'windowed-sinc-polyphase', 'up': 3, 'down': 2,
              'taps': 61, 'window': 'kaiser', 'beta': 5.0, 'rounding': 'nearest-even',
              'source_rate': 16000, 'sample_rate': 24000, 'channels': 1,
              'encoding': 'pcm_s16le', 'silence': 'resample speech; append exactly 24000 zeros',
              'version': 1}


def resample_24k(pcm, speech_frames):
    """Fixed symmetric FIR, zero-phase alignment, 3:2 rate conversion using numpy."""
    if pcm.ndim != 1 or len(pcm) != (speech_frames + 50) * 320 or np.any(pcm[-16000:]):
        raise ValueError('Expected frozen mono 16 kHz audio and exact silence tail')
    x = pcm[:speech_frames * 320].astype(np.float64)
    n = np.arange(-30, 31, dtype=np.float64)
    kernel = np.sinc(n / 3) * np.kaiser(61, 5.0)
    kernel *= 3 / kernel.sum()
    up = np.zeros(len(x) * 3, dtype=np.float64)
    up[::3] = x
    filtered = np.convolve(up, kernel, mode='full')[30:30 + len(up):2]
    speech = np.clip(np.rint(filtered), -32768, 32767).astype('<i2')
    return np.concatenate([speech, np.zeros(24000, dtype='<i2')])


def derivative(source, clip, cache):
    digest = sha256(source)
    identity = {**CONVERSION, 'source_sha256': digest, 'speech_frames': clip['speech_frames'],
                'numpy': np.__version__}
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    path, record = cache / f'{tag}.wav', cache / f'{tag}.json'
    if record.exists():
        saved = json.loads(record.read_text())
        if saved['conversion'] != identity or not path.exists() or saved['output_sha256'] != sha256(path):
            raise ValueError('Audio derivative changed')
        pcm, rate = sf.read(path, dtype='int16')
        if rate != 24000 or pcm.ndim != 1 or len(pcm) != (clip['speech_frames'] + 50) * 480 or np.any(pcm[-24000:]):
            raise ValueError('Audio derivative shape changed')
        return pcm.astype('<i2').tobytes(), saved
    if path.exists():
        raise ValueError('Uncheckpointed derivative exists; inspect it before retrying')
    pcm, rate = sf.read(source, dtype='int16')
    if rate != 16000:
        raise ValueError('Source must be 16 kHz')
    converted = resample_24k(pcm, clip['speech_frames'])
    cache.mkdir(parents=True, exist_ok=True)
    sf.write(path, converted, 24000, subtype='PCM_16')
    saved = {'conversion': identity, 'output_sha256': sha256(path),
             'pcm_sha256': hashlib.sha256(converted.tobytes()).hexdigest(),
             'samples': len(converted), 'duration_seconds': len(converted) / 24000}
    write_json(record, saved)
    return converted.tobytes(), saved


def prepare_derivatives(dataset):
    from .catalog import dataset_definition
    from .huggingface_data import verify_prepared
    root = verify_prepared(dataset_definition(dataset))
    results = {}
    for subset in ('smoke', 'full'):
        manifest = root / subset / 'manifest.json'
        clips = json.loads(manifest.read_text())['clips']
        outputs = []
        for clip in clips:
            _, record = derivative(manifest.parent / clip['audio'], clip, manifest.parent / 'derivatives/pcm24000')
            outputs.append({'clip_id': clip['clip_id'], **record})
        results[subset] = {'source_manifest_sha256': sha256(manifest), 'clips': outputs}
    return results
