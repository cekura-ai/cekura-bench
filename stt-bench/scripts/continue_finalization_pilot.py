"""Continue only unattempted pilot sessions, with one capture process per provider.

Original captures are immutable. Interrupted sessions stay failed; no retries.
"""
import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

from finalization_pilot import MODELS, observation, summarize
from stt_bench.data import load_manifest, sha256, write_json
from stt_bench.diagnostics import local_probe, validate_preflight
from stt_bench.run import recover_attempt, run


def verify(root):
    plan = json.loads((root / 'plan.json').read_text())
    for p, expected in plan['hashes'].items():
        if sha256(Path(p)) != expected:
            raise ValueError('Original pilot code or input changed')
    return load_manifest(root / 'dataset/manifest.json')


def existing_row(root, model, variant, index, clip, config):
    original = root / 'captures' / model / variant / f'{index:02d}'
    files = list((original / 'raw').glob('*.jsonl'))
    if not files:
        return None
    if len(files) != 1:
        raise ValueError('More than one attempt in the no-retry pilot')
    # Preserve the old directory byte-for-byte, including empty interrupted checkpoints.
    reconciled = root / 'reconciled' / model / variant / f'{index:02d}'
    (reconciled / 'raw').mkdir(parents=True, exist_ok=False)
    raw = reconciled / 'raw' / files[0].name
    shutil.copyfile(files[0], raw)
    recovered = recover_attempt(raw, config, False, clip['clip_id'], 1)
    write_json(reconciled / 'outcomes.json', [recovered])
    return dict(model=model, capture_phase='original', **observation(reconciled, clip, config, variant))


async def worker(root, model):
    manifest = verify(root)
    pacing = root / 'isolated-pacing/pacing.json'
    validate_preflight(pacing)
    rows = []
    stopped = False
    for index, clip in enumerate(manifest['clips']):
        variants = ('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')
        for variant in variants:
            config_path = root / 'configs' / f'{model}-{variant}.json'
            config = json.loads(config_path.read_text())
            row = existing_row(root, model, variant, index, clip, config)
            if row is None and not stopped:
                capture = root / 'isolated-captures' / model / variant / f'{index:02d}'
                print(f'ISOLATED {model} {variant} clip {index+1}/20', flush=True)
                await run(root / 'dataset' / f'clip-{index:02d}.json', config_path, capture,
                          False, pacing_check=pacing, max_attempts=1, stop_on_provider_failure=True)
                row = dict(model=model, capture_phase='isolated', **observation(capture, clip, config, variant))
            if row is not None:
                rows.append(row)
                write_json(root / f'worker-{model}.json', rows)
                if 'transport_or_provider_error' in row['exclusion_reasons']:
                    stopped = True
    return rows


async def main(root):
    verify(root)
    with (root / 'isolated-start.json').open('x') as f:
        json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(),
                       script_sha256=sha256(Path(__file__)),
                       policy='Only unattempted sessions; interrupted sessions are failures, never retried'), f)
    await local_probe(root / 'isolated-pacing', seconds=10, repeats=3)
    validate_preflight(root / 'isolated-pacing/pacing.json')

    async def launch(models):
        for model in models:
            with (root / f'worker-{model}.log').open('w') as log:
                process = await asyncio.create_subprocess_exec(sys.executable, '-u', __file__,
                    '--out', str(root), '--model', model, stdout=log, stderr=log)
                code = await process.wait()
                if code:
                    raise RuntimeError(f'{model} worker exited {code}; inspect saved log')

    tasks = [asyncio.create_task(launch(models)) for models in ([MODELS[0]], [MODELS[1], MODELS[2]], [MODELS[3]])]
    def checkpoint():
        rows = []
        for model in MODELS:
            path = root / f'worker-{model}.json'
            if path.exists():
                rows.extend(json.loads(path.read_text()))
        keys = [(r['model'], r['variant'], r['clip_id']) for r in rows]
        if len(keys) != len(set(keys)) or len(keys) > 160:
            raise ValueError('Duplicate session or exceeded authorized session count')
        # Additional capture_phase metadata is retained for sensitivity analysis.
        write_json(root / 'results.json', summarize(rows))
        return rows
    while any(not task.done() for task in tasks):
        checkpoint()
        await asyncio.sleep(5)
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    rows = checkpoint()
    write_json(root / 'state.json', dict(finished_at=datetime.now(timezone.utc).isoformat(),
        sessions=len(rows), status='complete' if len(rows) == 160 else 'partial',
        worker_errors=[str(e) for e in outcomes if isinstance(e, Exception)],
        original_sessions=sum(r['capture_phase'] == 'original' for r in rows),
        isolated_sessions=sum(r['capture_phase'] == 'isolated' for r in rows)))
    print('Isolated continuation finished; no original sessions repeated', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    asyncio.run(worker(args.out, args.model) if args.model else main(args.out))
