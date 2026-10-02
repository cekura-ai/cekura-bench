"""Run only the approved <=300 seconds per provider, once, with fresh pacing checks."""
import argparse
import asyncio
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

MODELS = {'gradium-default', 'reson8-realtime'}


def validate_plan(path, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError('Approved plan hash mismatch')
    plan = json.loads(path.read_text())
    if (plan.get('max_audio_seconds_per_provider') != 300 or plan.get('max_attempts_per_clip') != 1
            or {m['model'] for m in plan['models']} != MODELS or len(plan['models']) != 2
            or [g['id'] for g in plan['datasets']] != ['pipecat','fleurs-general','fleurs-entities']):
        raise ValueError('Unexpected model, cohort or credit limit')
    total_frames = 0; ids = set(); count = 0
    for group in plan['datasets']:
        manifest = (path.parent / group['manifest']).resolve()
        if not manifest.is_relative_to(path.parent.resolve()) or sha256(manifest) != group['manifest_sha256']:
            raise ValueError('Frozen selection changed or escaped the plan directory')
        m = load_manifest(manifest)
        frames = sum(c['total_frames'] for c in m['clips'])
        if frames != group['total_frames'] or len(m['clips']) != group['clips']:
            raise ValueError('Planned audio count differs from files')
        for c in m['clips']:
            if (c['condition'] not in {'public_anchor','public_entities'} or c['clip_id'] in ids
                    or not 50 < c['total_frames'] <= 1000
                    or abs(c['submitted_seconds']-c['total_frames']/50) > 1e-6):
                raise ValueError('Invalid, nonpublic or duplicate clip')
            ids.add(c['clip_id'])
        total_frames += frames; count += len(m['clips'])
    if total_frames > 15000 or count != plan['planned_clips'] or total_frames/50 != plan['planned_audio_seconds_per_provider']:
        raise ValueError('Five-minute budget exceeded or plan totals changed')
    return plan


async def main(args):
    if not args.live or args.model not in MODELS:
        raise ValueError('Explicit live flag and an approved model required')
    path = Path(args.plan); plan = validate_plan(path,args.plan_sha256)
    config = model_config(args.model)
    model = next(m for m in plan['models'] if m['model']==args.model)
    if sha256(config) != model['config_sha256']:
        raise ValueError('Provider configuration changed')
    root = Path(args.out); root.mkdir(parents=True,exist_ok=False)
    os.environ.update(STT_BENCH_COMPUTE_PROVIDER='vercel-sandbox',STT_BENCH_COMPUTE_REGION='iad1',
                      STT_BENCH_COMPUTE_INSTANCE=args.session_id)
    state = dict(status='qualifying',model=args.model,plan_sha256=args.plan_sha256,max_attempts=1,
                 planned_clips=plan['planned_clips'],max_audio_seconds=300,groups=[],
                 started_at=datetime.now(timezone.utc).isoformat())
    write_json(root/'state.json',state)
    try:
        await local_probe(root/'pacing',seconds=10,repeats=3)
        pacing = root/'pacing/pacing.json'; validate_preflight(pacing)
        async with asyncio.timeout(1200):
            for group in plan['datasets']:
                state.update(status='running',active_group=group['id']);write_json(root/'state.json',state)
                target=root/group['id']
                await run(path.parent/group['manifest'],config,target/'capture',False,pacing_check=pacing,
                          max_attempts=1,stop_on_provider_failure=True)
                report=score(target/'capture',target/'scored')
                result=report['results'][0]
                state['groups'].append(dict(id=group['id'],planned_clips=group['clips'],
                    attempted_clips=result['attempted_clips'],usable_clips=result['scored_clips'],
                    sent_audio_seconds=result['initial_sent_audio_seconds'],wer=result['wer']))
                write_json(root/'state.json',state)
                if result['reliability']['first_attempt_transport_failures']:
                    state['status']='stopped_on_provider_error';break
            else:
                state['status']='complete'
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


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');p.add_argument('--model',required=True)
    p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--session-id',required=True);p.add_argument('--out',required=True)
    asyncio.run(main(p.parse_args()))
