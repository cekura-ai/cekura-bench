"""Durable sequential runs with a bounded, auditable second attempt."""
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil

import soundfile as sf

from .data import load_manifest, sha256, write_json
from .providers import validate, reduce_events, transcribe, require_credential, sample_rate
from .audio_formats import derivative
from .streaming import EventLog, pacing_metrics, read_events, stream_audio
from .diagnostics import validate_preflight, pacing_identity
from .catalog import dataset_identity
from .assemblyai_pacing import DESCRIPTION as ASSEMBLYAI_PACING_DESCRIPTION


def fingerprint(manifest_path, config, dry_run):
    return hashlib.sha256(json.dumps({'manifest': sha256(manifest_path), 'config': config,
                                     'mode': 'dry_run' if dry_run else 'live'}, sort_keys=True).encode()).hexdigest()


def assess(events, config, dry_run=False):
    reduced = reduce_events(events, config)
    pacing = pacing_metrics(events)
    reasons = []
    if not any(e['kind'] == 'clip_end' for e in events):
        reasons.append('interrupted_attempt')
    if 'interrupted_attempt' not in reasons and any(e['kind'] == 'error' and e.get('error_type') == 'Interrupted' for e in events):
        reasons.append('interrupted_attempt')
    if any(e['kind'] == 'error' and e.get('error_type') != 'Interrupted' for e in events):
        reasons.append('transport_or_provider_error')
    if not pacing['valid']:
        reasons.append('invalid_pacing')
    if not dry_run:
        if not reduced['transcript_complete']:
            reasons.append('incomplete_transcript')
        if not reduced['model_verified']:
            reasons.append('model_not_verified')
    return {**reduced, 'pacing': pacing, 'valid': not reasons, 'exclusion_reasons': reasons}


def recover_attempt(raw, config, dry_run, clip_id, number):
    events = read_events(raw, allow_truncated_final=True)
    summary = assess(events, config, dry_run)
    return {'clip_id': clip_id, 'attempt': number, 'raw_file': 'raw/' + raw.name,
            'raw_sha256': sha256(raw), **summary}


def select_attempt(attempts):
    ordered = sorted(attempts, key=lambda a: a['attempt'])
    return next((a for a in ordered if a['valid']), ordered[0])


def load_attempts(out, clip_id, config, dry_run):
    attempts = []
    for number in (1, 2):
        raw = out / 'raw' / f'{clip_id}--attempt-{number}.jsonl'
        meta = out / 'attempts' / f'{clip_id}--attempt-{number}.json'
        if not raw.exists():
            if meta.exists():
                raise ValueError(f'Raw evidence missing: {raw.name}')
            continue
        if meta.exists():
            item = json.loads(meta.read_text())
            if sha256(raw) != item['raw_sha256']:
                raise ValueError(f'Raw evidence changed: {raw.name}')
        else:
            item = recover_attempt(raw, config, dry_run, clip_id, number)
            write_json(meta, item)
        attempts.append(item)
    return attempts


async def run(manifest_path: Path, config_path: Path, out: Path, dry_run: bool, resume=False, pacing_check=None,
              *, max_attempts=2, authorized_private_manifest_sha256=None, stop_on_provider_failure=False):
    manifest = load_manifest(manifest_path)
    config = json.loads(config_path.read_text())
    validate(config)
    preflight = validate_preflight(pacing_check) if not dry_run else None
    key = require_credential(config) if not dry_run else ''
    if max_attempts not in (1, 2):
        raise ValueError('Expected one or two attempts')
    if not dry_run and any(c['condition'] not in {'public_anchor', 'public_entities'} for c in manifest['clips']):
        if (authorized_private_manifest_sha256 != sha256(manifest_path) or
                any(c['condition'] not in {'public_anchor', 'private_short'} for c in manifest['clips'])):
            raise ValueError('Private audio requires explicit authorization for this exact short-run manifest')
    expected = fingerprint(manifest_path, config, dry_run)
    runtime = pacing_identity()
    if resume:
        saved = json.loads((out / 'run.json').read_text())
        if saved.get('max_attempts', 2) != max_attempts:
            raise ValueError('Cannot change attempt budget on resume')
        if saved.get('schema_version') != 2 or saved.get('fingerprint') != expected:
            raise ValueError('Resume requires matching v2 dataset, configuration and run mode')
        if saved.get('measurement_version') != 4:
            raise ValueError('Cannot resume an older measurement version; use a new run directory')
        if saved.get('pacing_runtime') != runtime:
            raise ValueError('Cannot resume on a different timing runtime or host; use a new run')
        if sha256(out / 'manifest.json') != saved['manifest_sha256']:
            raise ValueError('Saved manifest changed')
        for name in saved['source_hashes']:
            if sha256(Path(__file__).parent / name) != saved['source_hashes'].get(name):
                raise ValueError(f'Cannot resume after changing {name}; use a new run')
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'raw').mkdir()
        (out / 'attempts').mkdir()
        shutil.copyfile(manifest_path, out / 'manifest.json')
        if manifest.get('annotations_sha256'):
            annotation_path = manifest_path.parent / 'annotations.json'
            if sha256(annotation_path) != manifest['annotations_sha256']:
                raise ValueError('Frozen annotation file changed')
            shutil.copyfile(annotation_path, out / 'annotations.json')
        write_json(out / 'run.json', dict(schema_version=2, measurement_version=4,
                   deadline_ms=[0, 250, 500, 1000], pacing_check=preflight, pacing_runtime=runtime,
                   mode='dry_run' if dry_run else 'live',
                   fingerprint=expected, started_at=datetime.now(timezone.utc).isoformat(),
                   manifest_sha256=sha256(manifest_path), config=config,
                   dataset_identity=dataset_identity(manifest),
                   model_identity={k: config.get(k) for k in ('model_id', 'provider', 'model', 'version', 'expected_model_uuid')},
                   python=platform.python_version(), system=platform.platform(),
                   dependency_versions={p: importlib.metadata.version(p) for p in
                                        ['jiwer', 'whisper-normalizer', 'websockets', 'webrtcvad-wheels', 'numpy', 'soundfile']},
                   source_hashes={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                   lock_sha256=sha256(Path('uv.lock')) if Path('uv.lock').exists() else None,
                   timing_clock='time.perf_counter; per-clip local origin',
                   pacing_policy=(ASSEMBLYAI_PACING_DESCRIPTION if config['provider']=='assemblyai' else '20 ms absolute deadlines; native asynchronous macOS timer with scoped latency activity and no spin, otherwise cooperative waits <=1 ms and final wait <=1 ms; bounded recovery with >=19 ms scheduled start spacing; 18–40 ms gaps and <=2% span drift; ordered frames and send duration <=40 ms'),
                   max_attempts=max_attempts, authorized_private_manifest_sha256=authorized_private_manifest_sha256,
                   retry_policy=('One attempt per clip; no automatic retries' if max_attempts == 1 else
                     'Original attempt for deadline metrics; first completed valid attempt for recovery accuracy; at most one retry')))
    lock = (out / '.runner.lock').open('a')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process is already running this output directory') from None
        histories = {c['clip_id']: load_attempts(out, c['clip_id'], config, dry_run) for c in manifest['clips']}

        def checkpoint():
            outcomes = [{**select_attempt(histories[c['clip_id']]), 'attempts': histories[c['clip_id']]}
                        for c in manifest['clips'] if histories[c['clip_id']]]
            write_json(out / 'outcomes.json', outcomes)
            return outcomes

        checkpoint()
        for number in range(1, max_attempts + 1):
            for index, clip in enumerate(manifest['clips'], 1):
                history = histories[clip['clip_id']]
                if any(a['valid'] for a in history) or any(a['attempt'] >= number for a in history):
                    continue
                print(f"[{index}/{len(manifest['clips'])}] attempt {number} {clip['clip_id']} ({clip['submitted_seconds']:.2f}s)", flush=True)
                path = out / 'raw' / f"{clip['clip_id']}--attempt-{number}.jsonl"
                log = EventLog(path)
                log.emit('clip_start', clip_id=clip['clip_id'], attempt=number,
                         mode='dry_run' if dry_run else 'live', pacing_check=preflight)
                try:
                    source = manifest_path.parent / clip['audio']
                    if sample_rate(config) == 24000:
                        payload, conversion = derivative(source, clip, manifest_path.parent / 'derivatives/pcm24000')
                        log.emit('audio_derivative', **conversion)
                    else:
                        pcm, _ = sf.read(source, dtype='int16')
                        payload = pcm.astype('<i2').tobytes()
                    if dry_run:
                        async def send_audio(frame):
                            pass
                        async def finalize(t0):
                            log.emit('dry_run_finalize', t0_seconds=t0)
                        await stream_audio(payload, clip['speech_frames'], send_audio, finalize, log, sample_rate=sample_rate(config))
                    else:
                        await transcribe(payload, clip['speech_frames'], config, key, log)
                except asyncio.CancelledError:
                    log.emit('error', error_type='Interrupted')
                    raise
                except Exception as exc:
                    log.emit('error', error_type=type(exc).__name__)
                    print(f'  Failed: {type(exc).__name__}; raw evidence saved', flush=True)
                finally:
                    log.emit('clip_end')
                    log.close()
                item = recover_attempt(path, config, dry_run, clip['clip_id'], number)
                write_json(out / 'attempts' / f"{clip['clip_id']}--attempt-{number}.json", item)
                history.append(item)
                checkpoint()
                print(f"  Valid: {item['valid']}; final ms: {item['finalize_latency_ms']}; reasons: {item['exclusion_reasons']}", flush=True)
                if stop_on_provider_failure and 'transport_or_provider_error' in item['exclusion_reasons']:
                    print('Provider error: stopping this model without retries or further clip submissions.', flush=True)
                    return checkpoint()
        return checkpoint()
    finally:
        lock.close()
