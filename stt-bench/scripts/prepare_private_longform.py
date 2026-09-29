"""Rebuild the frozen eight-recording private dataset from `private-dataset/`.

Each speaker channel of each conversation is one recording. The reference is the
recording's supplied transcript; word times come from its segment annotations.
Audio uses the same 48 kHz -> 16 kHz FIR as the original freeze, padded to whole
20 ms frames plus one second of silence, so the output is sample-identical to the
frozen derivatives. The 24 kHz copy is the harness's own conversion of that file.
No provider is contacted and nothing is overwritten.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stt_bench.audio_formats import resample_24k  # noqa: E402
from stt_bench.data import sha256, write_json  # noqa: E402
from stt_bench.full_benchmark import PRIVATE, PRIVATE_MANIFEST  # noqa: E402

SOURCE = Path('private-dataset')
KERNEL_TAPS = np.arange(-60, 61, dtype=float)
KERNEL = np.sinc(KERNEL_TAPS / 3) * np.kaiser(121, 5)
KERNEL /= KERNEL.sum()


def downsample(source, target):
    """Write 16 kHz speech from a 48 kHz mono original; return speech samples."""
    info = sf.info(source)
    if info.samplerate != 48000 or info.channels != 1:
        raise ValueError('Source format changed')
    written = 0
    with sf.SoundFile(target, 'w', samplerate=16000, channels=1, subtype='PCM_16') as output:
        for start in range(0, info.frames, 480000):
            end = min(start + 480000, info.frames)
            left, right = max(0, start - 60), min(info.frames, end + 60)
            audio, _ = sf.read(source, start=left, stop=right, dtype='float64')
            filtered = np.convolve(audio, KERNEL, mode='full')[60:60 + len(audio)]
            y = np.clip(np.rint(filtered[start - left:end - left:3] * 32768), -32768, 32767).astype('<i2')
            output.write(y)
            written += len(y)
        speech_frames = math.ceil(written / 320)
        output.write(np.zeros(speech_frames * 320 - written + 16000, dtype='<i2'))
    return speech_frames


def words(annotation):
    duration = annotation['duration_seconds']
    out = []
    for segment in annotation['segments']:
        for w in segment['words']:
            start, end = w.get('start'), w.get('end')
            valid = (all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end))
                     and 0 <= start <= end <= duration)
            out.append(dict(text=w['word'], start=start, end=end, timing_valid=valid))
    return out


def prepare(source=SOURCE, audio_root=PRIVATE, manifest_path=PRIVATE_MANIFEST):
    source, audio_root, manifest_path = Path(source), Path(audio_root), Path(manifest_path)
    if manifest_path.exists():
        raise ValueError('Private manifest already exists; it is never overwritten')
    annotations = sorted(source.glob('conversation-*/*.json'))
    if len(annotations) != 8:
        raise ValueError('Expected eight recording annotations')
    audio_root.mkdir(parents=True, exist_ok=True)
    clips, inventory = [], {}
    for path in annotations:
        annotation = json.loads(path.read_text())
        clip_id = f"{path.parent.name}-{annotation['speaker_label']}"
        original = path.parent / annotation['audio_file']
        relative = str(original.relative_to(source))
        inventory[relative] = sha256(original)
        inventory[str(path.relative_to(source))] = sha256(path)
        audio16, audio24 = audio_root / f'{clip_id}.wav', audio_root / f'{clip_id}-24k.wav'
        if audio16.exists() or audio24.exists():
            raise ValueError('Private derivative already exists: ' + clip_id)
        speech_frames = downsample(original, audio16)
        pcm, _ = sf.read(audio16, dtype='int16')
        sf.write(audio24, resample_24k(pcm, speech_frames), 24000, subtype='PCM_16')
        clips.append(dict(clip_id=clip_id, conversation_id=annotation['conversation_id'],
                          speaker_label=annotation['speaker_label'], source_audio=relative,
                          reference=annotation['transcript'], words=words(annotation),
                          speech_frames=speech_frames, submitted_seconds=(speech_frames + 50) / 50,
                          audio={'16000': dict(path=audio16.name, sha256=sha256(audio16)),
                                 '24000': dict(path=audio24.name, sha256=sha256(audio24))}))
        print(clip_id + ' prepared', flush=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, dict(schema_version=1, dataset_id='private-longform',
                                   preparation='rebuilt from private-dataset/ by scripts/prepare_private_longform.py',
                                   source_inventory=inventory, counts=dict(recordings=len(clips)), clips=clips))
    print(json.dumps(dict(recordings=len(clips), manifest_sha256=sha256(manifest_path),
                          audio_seconds=sum(c['submitted_seconds'] for c in clips))))


if __name__ == '__main__':
    prepare()
