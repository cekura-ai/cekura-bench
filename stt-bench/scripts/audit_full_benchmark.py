#!/usr/bin/env python3
"""Audit frozen identities, archives, merged scores and coverage without provider calls."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile
import soundfile as sf

from stt_bench.full_benchmark import combine, load_plan


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def audit(root):
    identity = json.loads((root / 'input.json').read_text())
    assert digest(root / 'plan.json') == identity['plan_sha256'], 'Frozen plan changed'
    assert digest(root / 'input.tar.gz') == identity['bundle_sha256'], 'Frozen bundle changed'
    original = json.loads((root / 'plan.json').read_text())
    plan_loader, reducer = load_plan, combine
    if original['models'] == ['speechmatics-linden-1']:
        from stt_bench.linden_full_benchmark import load_plan as plan_loader, combine as reducer
    audio_checked = 0
    with tarfile.open(root / 'input.tar.gz') as bundle:
        for name, expected_hash in original['code_hashes'].items():
            source = bundle.extractfile(name)
            assert source and hashlib.file_digest(source, 'sha256').hexdigest() == expected_hash, f'Bundled code changed: {name}'
        for clip in original['items']:
            for rate in ((16, 24) if clip['cohort'] == 'public' else (16,)):
                if clip['cohort'] == 'private':
                    content = bundle.extractfile(clip[f'audio{rate}']).read()
                    assert hashlib.sha256(content).hexdigest() == clip[f'audio{rate}_sha256']
                    audio = sf.info(io.BytesIO(content))
                else:
                    source = Path(clip[f'audio{rate}'])
                    assert digest(source) == clip[f'audio{rate}_sha256']
                    audio = sf.info(source)
                assert audio.samplerate == rate * 1000 and audio.channels == 1
                assert audio.frames == (clip['speech_frames'] + 50) * rate * 20
                audio_checked += 1
    runtime = root / 'runtime.json'
    plan = plan_loader(root / 'plan.json', identity['plan_sha256'],
                     str(runtime) if runtime.exists() else None,
                     digest(runtime) if runtime.exists() else None)
    runtime_source = root / 'runtime-source.json'
    if runtime.exists() and runtime_source.exists():
        source = json.loads(runtime_source.read_text())
        amendment = json.loads(runtime.read_text())
        assert source['runtime_sha256'] == digest(runtime)
        assert source['files'] == amendment['code_overrides']
        assert digest(root / source['bundle']) == source['bundle_sha256']
        with tarfile.open(root / source['bundle']) as bundle:
            for name, expected_hash in source['files'].items():
                archived = bundle.extractfile(name)
                assert archived and hashlib.file_digest(archived, 'sha256').hexdigest() == expected_hash
    control = json.loads((root / 'controller.json').read_text())
    report = json.loads((root / 'results.json').read_text())
    if control['status'] == 'complete':
        assert all(not model['active'] and not model['blocked'] and not model['privateBlocked'] for model in control['models'].values())
        assert all(batch['status'] in ('collected', 'skipped_blocked', 'failed_before_provider') for batch in control['batches'].values()), 'Unreconciled batch at completion'
        assert all(worker.get('computeStopped') for worker in control['workers'].values())
        assert control['preparation'].get('computeStopped')
    states = []
    raw_attempts = set()
    archives = []
    pre_provider_failures = 0
    for batch in control['batches'].values():
        if batch['status'] == 'failed_before_provider':
            receipt_path = root / batch['reconciliation_file']
            assert digest(receipt_path) == batch['reconciliation_sha256']
            receipt = json.loads(receipt_path.read_text())
            assert receipt['verifiedBeforeProvider'] and receipt['batchId'] == batch['id']
            assert not receipt['evidence']['output_exists']
            pre_provider_failures += 1
        if batch['status'] != 'collected':
            continue
        folder = root / 'batches' / batch['id']
        archive = folder / 'evidence.tar.gz'
        assert batch.get('verified'), f"Batch {batch['id']} not replayed"
        assert digest(archive) == batch['archiveHash'], f"Archive {batch['id']} changed"
        state = json.loads((folder / 'verified.json').read_text())
        assert state.get('verified') and state['assignment'] == batch['assignment']
        with tarfile.open(archive) as bundle:
            for row in state['rows']:
                key = (batch['model'], batch['assignment'].get('private_variant', 'single-session'), row['clip_id'], row['attempt'])
                assert key not in raw_attempts, f'Duplicate recorded attempt: {key}'
                raw_attempts.add(key)
                raw = bundle.extractfile(f"{row['clip_id']}--{row['attempt']}.jsonl")
                assert raw and hashlib.file_digest(raw, 'sha256').hexdigest() == row['raw_sha256']
        archives.append({'batch_id': batch['id'], 'sha256': batch['archiveHash']})
        states.append(state)
    merged = reducer(plan, states)
    assert merged['models'] == report['models'], 'Saved merged metrics do not reproduce'
    assert merged['previous_private_transport_attempts'] == report['previous_private_transport_attempts'], 'Prior attempts changed'
    # Exercise a different partitioning/order of the same saved observations.
    reversed_merge = reducer(plan, list(reversed(states)))
    assert reversed_merge['models'] == merged['models'], 'Metrics depend on worker partition order'
    expected = {(c['clip_id'], c['cohort']) for c in plan['items']}
    coverage = {}
    for model in plan['models']:
        items = report['models'][model]['items']
        assert len(items) == len(plan['items']) and {(i['clip_id'], i['cohort']) for i in items} == expected
        assert Counter(i['cohort'] for i in items) == Counter(i['cohort'] for i in plan['items'])
        coverage[model] = dict(Counter(i['status'] for i in items))
        for item in items:
            assert len(item['attempts']) <= 2
            assert [a['attempt'] for a in item['attempts']] == list(range(1, len(item['attempts']) + 1))
    return {'status': 'passed', 'run_status': control['status'], 'plan_sha256': identity['plan_sha256'],
            'local_audio_hashes_and_shapes_checked': audio_checked,
            'private_24khz_verification': ('Not used by the Linden run; its private 16 kHz input was checked above.'
                if original['models'] == ['speechmatics-linden-1'] else
                'Converted and hash-checked in the preparation sandbox and again before each Gradium dispatch.'),
            'archives': archives, 'raw_attempts_checked': len(raw_attempts),
            'reconciled_pre_provider_failures_checked': pre_provider_failures,
            'coverage': coverage, 'scores_reproduce': True, 'partition_order_invariant': True,
            'note': 'This local evidence audit does not itself establish stopped remote compute.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    (args.root / 'evidence-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'archives'}, indent=2))
