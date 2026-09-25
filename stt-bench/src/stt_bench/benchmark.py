"""Smoke-before-full workflow with durable, explicitly named artifacts."""
from datetime import datetime, timezone
import asyncio
import fcntl
import json
import re
import urllib.request
from pathlib import Path

from .catalog import dataset_definition, model_config
from .data import write_json, sha256
from .huggingface_data import verify_prepared
from .diagnostics import local_probe, validate_preflight, pacing_identity
from .run import run
from .score import score


from .providers import verify_model, cost_total


def smoke_passed(report):
    return (report['completeness']['run_complete'] and bool(report['clips'])
            and all(c['valid'] and c['accuracy_usable'] for c in report['clips']))


async def benchmark(dataset, model, run_id=None, resume=False):
    definition = dataset_definition(dataset)
    config_path = model_config(model)
    config = json.loads(config_path.read_text())
    root = verify_prepared(definition)
    if resume and not run_id:
        raise ValueError('--resume requires --run-id')
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('run-id may contain only letters, digits, underscores and hyphens')
    runs = Path('runs') / dataset / model / run_id
    reports = Path('reports') / dataset / model / run_id
    if not resume:
        runs.mkdir(parents=True, exist_ok=False)
        reports.mkdir(parents=True, exist_ok=False)
    if not runs.is_dir() or not reports.is_dir():
        raise ValueError('Resume requires existing workflow directories')
    with (runs / '.workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('This benchmark workflow is already running') from None
        identity = dict(dataset=definition, model_id=model, config=config,
                        pacing_runtime=pacing_identity(),
                        source_hashes={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                        manifests={s: sha256(root / s / 'manifest.json') for s in ('smoke', 'full')})
        status_path = runs / 'benchmark.json'
        if resume:
            state = json.loads(status_path.read_text())
            if state['identity'] != identity:
                raise ValueError('Resume requires the same dataset, model and configuration')
            if state['status'] == 'complete':
                print(f'Already complete: {reports}', flush=True)
                return state
        else:
            state = dict(run_id=run_id, identity=identity, status='starting', reports={}, costs={})
            write_json(status_path, state)
        print(f'Benchmark {dataset} / {model} / {run_id}', flush=True)
        try:
            state['model_verification'] = await asyncio.to_thread(verify_model, config)
            state['pricing'] = config['pricing']
            state['pricing_note'] = 'Configured list-rate estimate; verification date recorded, not an invoice'
            for subset in ('smoke', 'full'):
                # A report is a completion checkpoint; never repeat a successful smoke run.
                if subset in state['reports']:
                    report_path = Path(state['reports'][subset]['path'])
                    if sha256(report_path) != state['reports'][subset]['sha256']:
                        raise ValueError('Saved workflow report changed')
                    report = json.loads(report_path.read_text())
                    if subset == 'smoke' and not smoke_passed(report):
                        raise ValueError('Smoke test failed; inspect its saved report before a new workflow')
                    continue
                state['status'] = f'{subset}_pacing_check'
                write_json(status_path, state)
                probe = reports / (subset + '-pacing-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
                await local_probe(probe, seconds=10, repeats=3)
                preflight = probe / 'pacing.json'
                validate_preflight(preflight)
                manifest = root / subset / 'manifest.json'
                state['status'] = f'{subset}_running'
                write_json(status_path, state)
                run_path = runs / subset
                await run(manifest, config_path, run_path, False, run_path.exists(), preflight)
                report_dir = reports / (subset + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
                report = score(run_path, report_dir)
                report_path = report_dir / 'results.json'
                state['reports'][subset] = dict(path=str(report_path), sha256=sha256(report_path))
                state['costs'][subset] = cost_total(g['estimated_cost_usd'] for g in report['results'])
                write_json(status_path, state)
                if subset == 'smoke' and not smoke_passed(report):
                    raise ValueError('Smoke test failed; full run was not started. Inspect ' + str(report_path))
            state.pop('error', None)
            state.pop('blocked_stage', None)
            state['status'] = 'complete'
            state['estimated_total_cost_usd'] = cost_total(state['costs'].values())
            write_json(status_path, state)
            write_json(reports / 'benchmark.json', state)
            print(f'Benchmark complete: {reports / "benchmark.json"}', flush=True)
            return state
        except Exception as exc:
            state['blocked_stage'] = state['status']
            state['status'] = 'blocked'
            state['error'] = str(exc)
            write_json(status_path, state)
            write_json(reports / 'benchmark.json', state)
            raise
