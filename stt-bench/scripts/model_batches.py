"""One model's smoke or full batches in an already provisioned Vercel session.

No transcription without --live. Never updates or resumes the historical Nova job.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

from stt_bench.catalog import dataset_definition, model_config
from stt_bench.data import sha256, write_json
from stt_bench.huggingface_data import verify_prepared
from stt_bench.model_jobs import job_identity, validate_smoke, rollup
from stt_bench.providers import require_credential, verify_model
from stt_bench.run import run
from stt_bench.score import score
from stt_bench.benchmark import smoke_passed
from stt_bench.diagnostics import local_probe, validate_preflight
from vercel_batches import partition, allowance


def export(root, run_root):
    archive = root / 'artifacts.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p != archive and p.name != 'artifacts.sha256' and not p.is_symlink():
                tar.add(p, arcname=str(p), recursive=False)
        if run_root.exists():
            tar.add(run_root, arcname=str(run_root))
    (root / 'artifacts.sha256').write_text(sha256(archive) + '  artifacts.tar.gz\n')


async def execute(args):
    if not args.live:
        raise ValueError('Transcription is disabled; --live is required at the later launch')
    model_config(args.model)
    for value in (args.run_id, args.dataset, args.session_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise ValueError('Invalid job identifier')
    if args.region != 'iad1' or not 300 <= args.budget_seconds <= 82800:
        raise ValueError('Expected iad1 and a bounded 300–82800 second budget')
    config_path = model_config(args.model)
    config = json.loads(config_path.read_text())
    require_credential(config)
    data = verify_prepared(dataset_definition(args.dataset))
    full_path, smoke_path = data / 'full/manifest.json', data / 'smoke/manifest.json'
    identity = job_identity(args.dataset, args.model, full_path, smoke_path, config_path)
    os.environ.update(STT_BENCH_COMPUTE_PROVIDER='vercel-sandbox', STT_BENCH_COMPUTE_REGION=args.region,
                      STT_BENCH_COMPUTE_INSTANCE=args.session_id)
    root = Path('reports') / args.dataset / args.model / args.run_id
    run_root = Path('runs') / args.dataset / args.model / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / 'batch-state.json'
        receipt_path = root / 'smoke-receipt.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(
            identity=identity, status='new', completed_batches=[], model_verification=verify_model(config))
        if state['identity'] != identity:
            raise ValueError('Dataset, model or source changed; use a new run')
        if state['status'] == 'complete' or (args.phase == 'smoke' and receipt_path.exists()):
            if receipt_path.exists():
                validate_smoke(json.loads(receipt_path.read_text()), identity)
            return
        if state['status'] not in ('new', 'smoke_passed', 'session_complete'):
            raise ValueError('Unfinished previous command: reconcile its saved ID, do not retranscribe')
        if args.phase == 'full':
            if not receipt_path.exists():
                raise ValueError('A model-specific smoke receipt is required')
            validate_smoke(json.loads(receipt_path.read_text()), identity)
        session = root / 'sessions' / f'{args.session_id}-{args.phase}'
        session.mkdir(parents=True, exist_ok=False)
        deadline = time.monotonic() + args.budget_seconds
        state.update(status='qualifying', active_session=args.session_id, phase=args.phase)
        write_json(state_path, state)
        try:
            with (session / 'qualification.log').open('x') as log:
                done = await asyncio.to_thread(subprocess.run,
                    [sys.executable, 'scripts/validate_harness.py', '--out', str(session / 'qualification')],
                    stdout=log, stderr=subprocess.STDOUT, timeout=240)
            if done.returncode:
                raise ValueError('Session timing qualification failed')
            await local_probe(session / 'pacing', seconds=10, repeats=3)
            preflight = session / 'pacing/pacing.json'
            validate_preflight(preflight)
            if args.phase == 'smoke':
                clips = json.loads(smoke_path.read_text())['clips']
                if time.monotonic() + allowance(clips) + 90 > deadline:
                    raise ValueError('Insufficient session budget for smoke and retry allowance')
                state['status'] = 'smoke_running'; write_json(state_path, state)
                await asyncio.wait_for(run(smoke_path, config_path, run_root / 'smoke', False, False, preflight), allowance(clips))
                report_dir = root / 'smoke'
                report = score(run_root / 'smoke', report_dir)
                passed = smoke_passed(report)
                write_json(receipt_path, dict(passed=passed, identity=identity,
                    report=str(report_dir / 'results.json'), report_sha256=sha256(report_dir / 'results.json')))
                if not passed:
                    raise ValueError('Smoke failed; full benchmark blocked')
                state['status'] = 'smoke_passed'
            else:
                full = json.loads(full_path.read_text())
                batches = partition(full['clips'])
                # Stable prefixes prevent omission, duplication or successful-batch replay.
                if [b['index'] for b in state['completed_batches']] != list(range(len(state['completed_batches']))):
                    raise ValueError('Invalid completed batch sequence')
                for index in range(len(state['completed_batches']), len(batches)):
                    clips = batches[index]
                    worst = allowance(clips)
                    if time.monotonic() + worst + 90 > deadline:
                        break
                    name = f'batch-{index:04d}'
                    # Put batch manifests next to originals so relative audio references stay valid.
                    manifest = full_path.parent / f'{args.model}-{args.run_id}-{name}.json'
                    if manifest.exists() or (run_root / name).exists():
                        raise ValueError('Uncheckpointed batch exists; inspect before continuing')
                    from vercel_batches import batch_manifest
                    write_json(manifest, batch_manifest(full, clips, sha256(full_path), index))
                    state.update(status='running', active_batch=index); write_json(state_path, state)
                    await asyncio.wait_for(run(manifest, config_path, run_root / name, False, False, preflight), worst)
                    report_dir = root / name
                    report = score(run_root / name, report_dir)
                    if not report['completeness']['run_complete'] or not any(r['accuracy_usable'] for r in report['clips']):
                        raise ValueError('Batch incomplete or entirely unusable')
                    path = report_dir / 'results.json'
                    state['completed_batches'].append(dict(index=index, clips=len(clips), session_id=args.session_id,
                        report=str(path), sha256=sha256(path)))
                    write_json(state_path, state)
                    rollup(state, root, full)
                state['status'] = 'complete' if len(state['completed_batches']) == len(batches) else 'session_complete'
                rollup(state, root, full)
            write_json(state_path, state)
        except BaseException as exc:
            state.update(status='failed', error_type=type(exc).__name__)
            write_json(state_path, state)
            raise
        finally:
            export(root, run_root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='pipecat-stt-benchmark')
    p.add_argument('--model', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--session-id', required=True)
    p.add_argument('--region', default='iad1')
    p.add_argument('--phase', choices=('smoke', 'full'), required=True)
    p.add_argument('--budget-seconds', type=int, default=82800)
    p.add_argument('--live', action='store_true')
    asyncio.run(execute(p.parse_args()))


if __name__ == '__main__':
    main()
