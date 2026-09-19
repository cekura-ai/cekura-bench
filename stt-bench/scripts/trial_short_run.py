"""Explicitly authorized 10+10 trial run. One pass, no automatic provider retries."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tarfile

from stt_bench.catalog import model_config
from stt_bench.data import load_manifest, sha256, write_json
from stt_bench.diagnostics import local_probe, validate_preflight
from stt_bench.run import run
from stt_bench.score import score

MODELS={'soniox-stt-rt-v5','smallest-pulse','sarvam-saaras-v3-realtime','inworld-stt-1'}


def validate_selection(manifest, expected_hash):
    if sha256(manifest)!=expected_hash:
        raise ValueError('Manifest differs from the specifically authorized 10+10 selection')
    m=load_manifest(manifest)
    if (Counter(c['condition'] for c in m['clips'])!={'public_anchor':10,'private_short':10}
            or any(not 1<c['submitted_seconds']<=20 for c in m['clips'])):
        raise ValueError('Expected exactly ten short private and ten short public clips')
    return m


async def main(args):
    if not args.live or args.model not in MODELS:
        raise ValueError('Explicit live authorization and one of the four trial models required')
    manifest=Path(args.manifest);validate_selection(manifest,args.manifest_sha256)
    config=model_config(args.model)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    os.environ.update(STT_BENCH_COMPUTE_PROVIDER='vercel-sandbox',STT_BENCH_COMPUTE_REGION='iad1',
                      STT_BENCH_COMPUTE_INSTANCE=args.session_id)
    state={'model':args.model,'manifest_sha256':args.manifest_sha256,'max_attempts':1,
           'planned_clips':20,'status':'qualifying','started_at':datetime.now(timezone.utc).isoformat()}
    write_json(root/'state.json',state)
    try:
        await local_probe(root/'pacing',seconds=10,repeats=3)
        pacing=root/'pacing/pacing.json';validate_preflight(pacing)
        state['status']='running';write_json(root/'state.json',state)
        await asyncio.wait_for(run(manifest,config,root/'capture',False,pacing_check=pacing,
            max_attempts=1,authorized_private_manifest_sha256=args.manifest_sha256,
            stop_on_provider_failure=True),timeout=900)
        report=score(root/'capture',root/'scored')
        state.update(status='complete' if report['completeness']['all_planned_clips_have_attempts'] else 'stopped_on_provider_error',
            attempted_clips=sum(bool(r['attempts']) for r in report['clips']),
            usable_clips=sum(r['accuracy_usable'] for r in report['clips']))
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        state['finished_at']=datetime.now(timezone.utc).isoformat();write_json(root/'state.json',state)
        with tarfile.open(root/'evidence.tar.gz','w:gz') as archive:
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.name not in ('evidence.tar.gz','evidence.sha256'):
                    archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
        (root/'evidence.sha256').write_text(sha256(root/'evidence.tar.gz')+'  evidence.tar.gz\n')
    print(json.dumps(state),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--live',action='store_true');p.add_argument('--model',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--session-id',required=True);p.add_argument('--out',required=True)
    asyncio.run(main(p.parse_args()))
