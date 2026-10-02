"""Full frozen Pipecat benchmark, with sequential streams and fresh batch pacing."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tarfile
from stt_bench.benchmark import smoke_passed
from stt_bench.catalog import dataset_definition, model_config
from stt_bench.data import sha256, write_json
from stt_bench.diagnostics import local_probe, validate_preflight
from stt_bench.huggingface_data import verify_prepared
from stt_bench.model_jobs import job_identity, rollup
from stt_bench.run import run
from stt_bench.score import score

MODEL='assemblyai-universal-3-5-pro'


def batches(full):
    if len(full['clips']) != 1000 or len({c['clip_id'] for c in full['clips']}) != 1000:
        raise ValueError('Expected exactly 1,000 distinct frozen Pipecat clips')
    return [full['clips'][i:i+50] for i in range(0,1000,50)]


async def main(args):
    if not args.live:raise ValueError('--live required')
    os.environ.update(STT_BENCH_COMPUTE_PROVIDER='vercel-sandbox',STT_BENCH_COMPUTE_REGION='iad1',
                      STT_BENCH_COMPUTE_INSTANCE=args.session_id)
    data=verify_prepared(dataset_definition('pipecat-stt-benchmark'))
    full_path=data/'full/manifest.json';smoke_path=data/'smoke/manifest.json'
    if sha256(full_path)!=args.manifest_sha256:raise ValueError('Frozen full manifest mismatch')
    full=json.loads(full_path.read_text());groups=batches(full)
    config=model_config(MODEL)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    state=dict(status='qualifying',started_at=datetime.now(timezone.utc).isoformat(),
               identity=job_identity('pipecat-stt-benchmark',MODEL,full_path,smoke_path,config),
               planned_clips=1000,completed_clips=0,completed_batches=[],
               max_attempts=2,stream_concurrency=1,smoke_passed=False)
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat();write_json(root/'state.json',state)
    async def qualify(label):
        probe=root/(label+'-pacing');await local_probe(probe,seconds=10,repeats=3)
        validate_preflight(probe/'pacing.json');return probe/'pacing.json'
    save()
    try:
        pacing=await qualify('smoke');state['status']='smoke_running';save()
        await asyncio.wait_for(run(smoke_path,config,root/'smoke/capture',False,pacing_check=pacing,
            max_attempts=2,stop_on_provider_failure=True),900)
        report=score(root/'smoke/capture',root/'smoke/scored')
        state['smoke_passed']=smoke_passed(report);save()
        if not state['smoke_passed']:
            state['status']='smoke_failed';return
        for index,clips in enumerate(groups):
            name=f'batch-{index:04d}'
            pacing=await qualify(name)
            manifest=full_path.parent/(args.run_id+'-'+name+'.json')
            if manifest.exists():raise ValueError('Existing batch requires reconciliation, not retransmission')
            write_json(manifest,{**full,'subset':name,'clips':clips,
                'batch':dict(index=index,parent_manifest_sha256=args.manifest_sha256,parent_count=1000)})
            state.update(status='running',active_batch=index);save()
            # At most one retry, matching the existing public benchmark contract.
            timeout=sum(2*(1.2*c['submitted_seconds']+40) for c in clips)
            await asyncio.wait_for(run(manifest,config,root/name/'capture',False,pacing_check=pacing,
                max_attempts=2,stop_on_provider_failure=True),timeout)
            report=score(root/name/'capture',root/name/'scored')
            path=root/name/'scored/results.json'
            if not report['completeness']['all_planned_clips_have_attempts']:
                state.update(status='stopped_on_provider_error',failed_batch=index);return
            state['completed_batches'].append(dict(index=index,clips=len(clips),session_id=args.session_id,
                report=str(path),sha256=sha256(path)))
            state['completed_clips']+=len(clips);save();rollup(state,root,full)
            print(json.dumps(dict(completed=state['completed_clips'],planned=1000)),flush=True)
            if not any(c['accuracy_usable'] for c in report['clips']):
                state['status']='stopped_on_unusable_batch';return
        state['status']='complete';state.pop('active_batch',None)
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__);raise
    finally:
        state['finished_at']=datetime.now(timezone.utc).isoformat();save()
        rollup(state,root,full)
        with tarfile.open(root/'evidence.tar.gz','w:gz',compresslevel=1) as archive:
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.name not in ('evidence.tar.gz','evidence.sha256'):
                    archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
        (root/'evidence.sha256').write_text(sha256(root/'evidence.tar.gz')+'  evidence.tar.gz\n')
        print(json.dumps(dict(status=state['status'],completed=state['completed_clips'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true')
    for name in ('manifest-sha256','session-id','run-id','out'):p.add_argument('--'+name,required=True)
    asyncio.run(main(p.parse_args()))
