"""Sequential, independently recorded batches for bounded Vercel sessions.

Uses the existing streaming, retry, qualification and scoring implementations.
Never resumes a batch on another host. The controller rotates only between batches.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from stt_bench.benchmark import verify_model
from stt_bench.catalog import dataset_definition, model_config
from stt_bench.data import sha256, write_json
from stt_bench.diagnostics import local_probe, validate_preflight
from stt_bench.huggingface_data import verify_prepared
from stt_bench.measurement import summarize_deadlines
from stt_bench.run import run
from stt_bench.score import aggregate_wer, percentiles, score


def allowance(clips):
    # Reserve both attempts, 40 seconds of protocol/connection overhead per
    # attempt, and 20% audio scheduling headroom. Actual runs are usually faster.
    return math.ceil(sum(2 * (1.2 * c['submitted_seconds'] + 40) for c in clips))


def partition(clips, limit=900):
    batches, current = [], []
    for clip in clips:
        if allowance([clip]) > limit:
            raise ValueError('A clip exceeds the bounded batch allowance')
        if current and allowance(current + [clip]) > limit:
            batches.append(current)
            current = []
        current.append(clip)
    if current:
        batches.append(current)
    return batches


def batch_manifest(parent, clips, parent_hash, index):
    return {**parent, 'subset': f'full-batch-{index:04d}', 'clips': clips,
            'selection': {'algorithm': 'contiguous partition, original source order', 'count': len(clips)},
            'batch': {'index': index, 'parent_manifest_sha256': parent_hash,
                      'parent_count': len(parent['clips'])}}


def rollup(state, root, full):
    rows, sessions = [], {}
    cost = 0
    for batch in state['completed_batches']:
        path = Path(batch['report'])
        if sha256(path) != batch['sha256']:
            raise ValueError('Completed batch report changed')
        report = json.loads(path.read_text())
        rows.extend(report['clips'])
        sessions.setdefault(batch['session_id'], []).extend(report['clips'])
        cost += sum(g['estimated_cost_usd'] for g in report['results'])
    expected = [c['clip_id'] for c in full['clips']]
    actual = [c['clip_id'] for c in rows]
    if actual != expected[:len(actual)]:
        raise ValueError('Batch coverage is duplicated, missing, or out of order')
    good = [r for r in rows if r['accuracy_usable']]
    summary = dict(status=state['status'], planned_clips=len(expected),
        completed_clips=len(rows), scored_clips=len(good),
        full_coverage=actual == expected, dataset_revision=full['source_revision'],
        model='deepgram-nova-3', model_version=state['model_verification']['model']['version'],
        eventual_word_errors=aggregate_wer([r['word_errors'] for r in good]),
        deadlines=summarize_deadlines(rows) if rows else [],
        estimated_cost_usd=cost, smoke_cost_included=False,
        selected_exclusions=dict(Counter(reason for r in rows for reason in r['exclusion_reasons'])),
        finalize_acknowledgment_ms=percentiles([r['finalize_latency_ms'] for r in good if r['finalize_latency_ms'] is not None]),
        session_counts={key: len(value) for key, value in sessions.items()},
        notes=['Sequential streams across independently qualified Vercel sessions in iad1.',
               'Latency includes the Vercel-to-Deepgram network route; session identities remain in each batch report.',
               'Finalize acknowledgment is a protocol diagnostic, not complete-transcript availability.',
               'Supplied references and automatic speech boundaries have not been independently verified by listening.',
               'Partial results describe completed batches only; inspect full_coverage before treating this as the full dataset.'])
    write_json(root / 'summary.json', summary)
    write_json(root / 'clips.json', rows)
    w = summary['eventual_word_errors']['wer']
    (root / 'summary.md').write_text('\n'.join([
        '# Pipecat / Deepgram Nova-3 — Vercel sessions', '',
        f"Status: {state['status']}. Completed {len(rows)} / {len(expected)} clips; {len(good)} scored.", '',
        f"Eventual word error rate: {w:.2%}." if w is not None else 'Eventual word error rate: unavailable.',
        f'Estimated full-dataset cost so far: ${cost:.4f}; smoke is separate.', '',
        *summary['notes'], '', 'Detailed batch reports and raw events are retained.']))


async def execute(args):
    if not os.environ.get('DEEPGRAM_API_KEY'):
        raise ValueError('Deepgram key missing')
    os.environ.update(STT_BENCH_COMPUTE_PROVIDER='vercel-sandbox',
        STT_BENCH_COMPUTE_REGION='iad1', STT_BENCH_COMPUTE_INSTANCE=args.session_id)
    definition = dataset_definition('pipecat-stt-benchmark')
    data_root = verify_prepared(definition)
    full_path = data_root / 'full/manifest.json'
    full = json.loads(full_path.read_text())
    config_path = model_config('deepgram-nova-3')
    config = json.loads(config_path.read_text())
    root = Path('reports/pipecat-stt-benchmark/deepgram-nova-3') / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    batches = partition(full['clips'])
    identity = dict(manifest=sha256(full_path), config=sha256(config_path),
        wrapper=sha256(Path(__file__)), source={p.name: sha256(p) for p in sorted(Path('src/stt_bench').glob('*.py'))},
        batch_ids=[[c['clip_id'] for c in batch] for batch in batches])
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / 'batch-state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['identity'] != identity:
                raise ValueError('Code, configuration or data changed between sessions')
            if state['status'] not in ('session_complete', 'complete'):
                raise ValueError('Previous session did not finish cleanly; inspect evidence before continuing')
            if state['status'] == 'complete':
                return
        else:
            smoke = json.loads(Path(args.smoke_status).read_text())
            if not smoke.get('smoke_passed') or smoke.get('dataset') != 'pipecat-stt-benchmark':
                raise ValueError('A passing separate live smoke test is required')
            state = dict(identity=identity, completed_batches=[], status='starting',
                smoke_status=args.smoke_status, smoke_sha256=sha256(Path(args.smoke_status)),
                model_verification=verify_model(config), started_at=datetime.now(timezone.utc).isoformat())
        session_root = root / 'sessions' / args.session_id
        session_root.mkdir(parents=True, exist_ok=False)
        state.update(status='qualifying', active_session=args.session_id)
        write_json(state_path, state)
        deadline = time.monotonic() + args.budget_seconds
        try:
            with (session_root / 'qualification.log').open('x') as log:
                result = await asyncio.to_thread(subprocess.run, [sys.executable,
                    'scripts/validate_harness.py', '--out', str(session_root / 'qualification')],
                    stdout=log, stderr=subprocess.STDOUT, timeout=240)
            if result.returncode:
                raise ValueError('Session timing qualification failed')
            await local_probe(session_root / 'pacing', seconds=10, repeats=3)
            pacing = session_root / 'pacing/pacing.json'
            validate_preflight(pacing)
            for index in range(len(state['completed_batches']), len(batches)):
                clips = batches[index]
                worst = allowance(clips)
                if time.monotonic() + worst + 90 > deadline:
                    break
                name = f'batch-{index:04d}'
                manifest = full_path.parent / f'{args.run_id}-{name}.json'
                if manifest.exists():
                    raise ValueError('Uncheckpointed batch already exists; refusing to retranscribe')
                write_json(manifest, batch_manifest(full, clips, identity['manifest'], index))
                run_path = Path('runs/pipecat-stt-benchmark/deepgram-nova-3') / args.run_id / name
                state.update(status='running', active_batch=index)
                write_json(state_path, state)
                print(f'Full dataset: batch {index + 1}/{len(batches)}, {len(clips)} clips', flush=True)
                await asyncio.wait_for(run(manifest, config_path, run_path, False, False, pacing), timeout=worst)
                report_dir = root / name
                report = score(run_path, report_dir)
                if not report['completeness']['run_complete']:
                    raise ValueError('Batch incomplete')
                if not any(c['accuracy_usable'] for c in report['clips']):
                    raise ValueError('Entire batch is unusable; stopping for diagnosis')
                report_file = report_dir / 'results.json'
                state['completed_batches'].append(dict(index=index, session_id=args.session_id,
                    report=str(report_file), sha256=sha256(report_file), clips=len(clips)))
                write_json(state_path, state)
                rollup(state, root, full)
            state['status'] = 'complete' if len(state['completed_batches']) == len(batches) else 'session_complete'
            state.pop('active_batch', None)
            write_json(state_path, state)
            rollup(state, root, full)
            print(f"Session done: {state['status']}; batches {len(state['completed_batches'])}/{len(batches)}", flush=True)
        except BaseException as exc:
            state.update(status='failed', error_type=type(exc).__name__)
            write_json(state_path, state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--budget-seconds', type=int, required=True)
    parser.add_argument('--smoke-status', required=True)
    args = parser.parse_args()
    if not 1100 <= args.budget_seconds <= 2400:
        parser.error('Session budget must be between 1100 and 2400 seconds')
    asyncio.run(execute(args))
