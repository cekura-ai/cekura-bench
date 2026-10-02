"""Import pinned public audio without optional Torch/FFmpeg audio decoders."""
import hashlib
import io
import json
import math
import re
import shutil
import importlib.metadata
from pathlib import Path

import soundfile as sf

from .catalog import dataset_definition, dataset_root
from .data import RATE, prepare_audio, sha256, write_json, load_manifest


def freeze_rows(rows, definition, out):
    """Validate every source row; publish manifests only after complete success."""
    out.mkdir(parents=True, exist_ok=False)
    full = out / 'full'
    (full / 'audio').mkdir(parents=True)
    clips, errors, seen = [], [], set()
    for index, row in enumerate(rows):
        sample_id = row.get('sample_id')
        try:
            if not isinstance(sample_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', sample_id):
                raise ValueError('Missing or unsafe sample_id')
            if sample_id in seen:
                raise ValueError('Duplicate sample_id')
            seen.add(sample_id)
            reference = row.get('transcription')
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError('Missing transcription')
            duration = row.get('duration_seconds')
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError('Invalid duration_seconds')
            encoded = row['audio']
            source_bytes = encoded.get('bytes')
            if source_bytes is None:
                source_bytes = Path(encoded['path']).read_bytes()
            source = io.BytesIO(source_bytes)
            info = sf.info(source)
            if info.samplerate != RATE or info.channels != 1 or info.frames <= 0:
                raise ValueError('Expected nonempty mono 16 kHz source audio')
            if abs(info.duration - duration) > 1 / RATE:
                raise ValueError('Source audio duration does not match declared duration')
            source.seek(0)
            pcm, details = prepare_audio(source)
            filename = sample_id + '.wav'
            target = full / 'audio' / filename
            sf.write(target, pcm, RATE, subtype='PCM_16')
            clips.append(dict(clip_id='pipecat-' + sample_id, source_id=sample_id,
                              sample_id=sample_id, filename=filename, reference=reference,
                              dataset_transcription=reference, source_duration_seconds=duration,
                              source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                              audio='audio/' + filename, audio_sha256=sha256(target),
                              condition='public_anchor', entities=None, **details))
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            errors.append(dict(row=index, sample_id=sample_id, error=str(exc)))
    if len(clips) + len(errors) != definition['expected_clips']:
        errors.append(dict(error='Unexpected source row count', expected=definition['expected_clips'],
                           actual=len(clips) + len(errors)))
    if errors:
        write_json(out / 'preparation-errors.json', dict(status='failed', errors=errors,
                                                        prepared_clips=len(clips)))
        raise ValueError(f'Dataset preparation failed; inspect {out / "preparation-errors.json"}')
    base = dict(schema_version=2, dataset_id=definition['id'], dataset=definition['repository'],
                source_revision=definition['revision'], split=definition['split'],
                language=definition['language'], license=None,
                source_url='https://huggingface.co/datasets/' + definition['repository'],
                available_clips=len(clips), source_features=['sample_id', 'audio', 'duration_seconds', 'transcription'],
                dependency_versions={p: importlib.metadata.version(p) for p in ('datasets', 'soundfile', 'numpy', 'webrtcvad-wheels')},
                preprocessing=dict(sample_rate=RATE, encoding='linear16', channels=1, frame_ms=20,
                    trailing_silence_ms=1000, boundary='last positive WebRTC VAD frame; automatic, needs listening review',
                    vad='webrtcvad-wheels==2.0.14', vad_mode=1,
                    transcript_provenance='Pipecat supplied reference text; no independent listening verification',
                    entity_provenance='No entity annotations supplied'))
    full_manifest = dict(**base, subset='full', selection={'algorithm': 'all source rows, original order', 'count': len(clips)}, clips=clips)
    smoke_clips = sorted(clips, key=lambda c: hashlib.sha256(f"{definition['seed']}:{c['sample_id']}".encode()).hexdigest())[:definition['smoke_count']]
    smoke = out / 'smoke'
    (smoke / 'audio').mkdir(parents=True)
    for clip in smoke_clips:
        shutil.copyfile(full / clip['audio'], smoke / clip['audio'])
    smoke_manifest = dict(**base, subset='smoke',
                         selection=dict(algorithm='sha256(seed:sample_id)', seed=definition['seed'], count=len(smoke_clips)),
                         clips=smoke_clips)
    write_json(smoke / 'manifest.json', smoke_manifest)
    write_json(full / 'manifest.json', full_manifest)
    for subset in ('full', 'smoke'):
        load_manifest(out / subset / 'manifest.json')
    write_json(out / 'prepared.json', dict(status='complete', definition=definition,
               manifests={s: sha256(out / s / 'manifest.json') for s in ('full', 'smoke')},
               source_seconds=sum(c['source_duration_seconds'] for c in clips),
               submitted_seconds=sum(c['submitted_seconds'] for c in clips)))
    return full / 'manifest.json'


def prepare_dataset(name):
    from datasets import Audio, load_dataset
    definition = dataset_definition(name)
    out = dataset_root(definition)
    if out.exists():
        verify_prepared(definition)
        return out / 'full' / 'manifest.json'
    # Keep the requested DatasetDict load; choose the benchmark split explicitly.
    ds = load_dataset(definition['repository'], revision=definition['revision'],
                      cache_dir='.cache/huggingface/datasets')
    rows = ds[definition['split']].cast_column('audio', Audio(decode=False))
    return freeze_rows(rows, definition, out)


def verify_prepared(definition):
    root = dataset_root(definition)
    saved = json.loads((root / 'prepared.json').read_text())
    if saved['definition'] != definition or saved['status'] != 'complete':
        raise ValueError('Prepared dataset definition changed or preparation incomplete')
    for subset in ('full', 'smoke'):
        path = root / subset / 'manifest.json'
        if sha256(path) != saved['manifests'][subset]:
            raise ValueError(f'Prepared {subset} manifest changed')
        load_manifest(path)
    return root
