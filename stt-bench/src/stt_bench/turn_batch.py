"""Frozen turn-only Vercel assignments and verified offline report assembly."""
import argparse
import asyncio
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import tarfile

from .data import sha256, write_json
from .diagnostics import local_probe, validate_preflight
from .turn_runner import run_turns, smoke_selection, validate_profile
from .turns import verify_turn_manifest, require
from .run import assess
from .streaming import read_events
from .full_benchmark import classify

MANIFEST='datasets/private-turns-v1/automatic-v1/manifest.json'


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    require(not (root/'plan.json').exists(),'Frozen plan already exists')
    m=verify_turn_manifest(Path(MANIFEST))
    paths=sorted(Path('config/profiles/private-turns-v1').glob('*.json'))
    configs={p.stem:validate_profile(json.loads(p.read_text())) for p in paths if '3.8' not in p.stem}
    require(len(configs)==23,'Expected the 23 non-Gemini-3.8 profiles')
    # Known account limits win over general defaults. Unknown ceilings ramp under
    # a bounded worker budget and permanently reduce after a rate-limit response.
    limits={'deepgram':150,'openai':50,'google':20,'speechmatics':2,'assemblyai':20,
            'gradium':3,'inworld':32,'reson8':32,'smallest':32,'soniox':32,'sarvam':16,
            'cartesia':16,'elevenlabs':16,'gemini':10}
    files=[Path('pyproject.toml'),Path('uv.lock'),*Path('src/stt_bench').glob('*.py'),
           *Path('src/stt_bench').glob('*.html'),*Path('tests').glob('*.py'),
           *[p for p in Path('config/models').glob('*.json') if '3.8' not in p.stem],*paths,
           *[p for p in Path(MANIFEST).parent.rglob('*') if p.is_file()]]
    # Offline tests reference small frozen FLEURS fixtures; include only these.
    files += [p for p in Path('datasets/fleurs-en-us-smoke-v1').rglob('*') if p.is_file()]
    files=sorted(set(files))
    plan=dict(schema_version=1,run_id=root.name,kind='private_turns',manifest=MANIFEST,
        manifest_sha256=sha256(Path(MANIFEST)),configs=configs,
        config_hashes={p.stem:sha256(p) for p in paths if p.stem in configs},
        models=list(configs),smoke_ids=smoke_selection(m),clip_ids=[c['clip_id'] for c in m['clips']],
        durations={c['clip_id']:c['submitted_seconds'] for c in m['clips']},
        provider_limits=limits,max_workers=128,max_workers_per_model=32,
        assemblyai_new_sessions_per_minute=5,attempts_per_turn=1,
        planned_sessions=len(configs)*len(m['clips']),audio_seconds_per_model=sum(c['submitted_seconds'] for c in m['clips']),
        authorization='User explicitly authorized all remaining models on the automatic turn dataset in Vercel; Gemini 3.8 excluded.',
        counts=m['counts'],hashes={str(p):sha256(p) for p in files})
    write_json(root/'plan.json',plan)
    with tarfile.open(root/'input.tar.gz','w:gz',compresslevel=1) as archive:
        for p in files:archive.add(p,arcname=str(p),recursive=False)
        archive.add(root/'plan.json',arcname='turn-input/plan.json')
    write_json(root/'input.json',dict(plan_sha256=sha256(root/'plan.json'),bundle_sha256=sha256(root/'input.tar.gz')))
    return {k:plan[k] for k in ('models','planned_sessions','audio_seconds_per_model','counts','provider_limits')}


def verify(plan_path,expected,*,replay_smoke=True):
    require(sha256(plan_path)==expected,'Frozen turn plan changed')
    plan=json.loads(plan_path.read_text())
    require(plan['kind']=='private_turns' and plan['manifest']==MANIFEST,'Not a turn benchmark')
    for name,value in plan['hashes'].items():require(sha256(Path(name))==value,'Frozen input changed: '+name)
    verify_turn_manifest(Path(plan['manifest']))
    for row in plan.get('replayed_smoke',[]) if replay_smoke else []:
        observed=assess(read_events(Path(row['raw_path'])),plan['configs'][row['model']])
        require(observed['valid']==row['valid'] and observed['turn_timing']==row['turn_timing'] and
                (row['clip_id'] not in plan['smoke_ids'] or row['valid']),
                'Corrected smoke replay differs from original event evidence')
    return plan


async def worker(plan_path,plan_hash,assignment_path,out):
    plan=verify(plan_path,plan_hash,replay_smoke=False)
    a=json.loads(assignment_path.read_text());model=a['model']
    require(a['plan_hash']==plan_hash and model in plan['models'],'Wrong turn assignment')
    require(not a.get('smoke_receipt') or a['smoke_receipt'].get('policy')==plan.get('smoke_policy'), 'Smoke policy differs from frozen plan')
    config=Path(f'config/profiles/private-turns-v1/{model}.json')
    require(sha256(config)==plan['config_hashes'][model],'Profile changed')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'assignment.json',a)
    result=dict(status='starting',assignment=a,rows=[])
    try:
        # A reusable worker has one live stream. Qualification is tied to its VM
        # session/code and refreshed before it becomes older than one hour.
        probe=Path('turn-pacing/pacing.json')
        try:validate_preflight(probe)
        except (FileNotFoundError,ValueError):
            if probe.parent.exists():shutil.move(probe.parent,Path(f'turn-pacing-previous-{out.name}'))
            await local_probe(probe.parent,seconds=10,repeats=3)
            validate_preflight(probe)
        rows=await run_turns(Path(plan['manifest']),config,out/'run',pacing_check=probe,
            authorized_private_manifest_sha256=plan['manifest_sha256'],selected_clip_ids=a['clip_ids'],
            smoke_receipt=a.get('smoke_receipt'))
        for row in rows:
            events=read_events(out/'run'/row['raw_file'])
            row['failure_class']=classify(events)
            error_text=' '.join(str(e.get('error_message','')).lower() for e in events if e['kind']=='error')
            if 'too many concurrent' in error_text:row['failure_class']='concurrency'
            elif 'exceeded your current quota' in error_text:row['failure_class']='quota'
        result.update(status='finished',rows=rows)
    except Exception as exc:
        # No credentials or audio in this diagnostic. Provider exceptions are
        # redacted in the shared runner's raw event log.
        result.update(status='worker_failed',error_type=type(exc).__name__,error=str(exc)[:300])
    finally:
        result['finished_at']=datetime.now(timezone.utc).isoformat()
        write_json(out/'state.json',result)
        with tarfile.open(out/'evidence.tar.gz','w:gz') as archive:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p.name not in ('evidence.tar.gz','evidence.sha256'):
                    archive.add(p,arcname=str(p.relative_to(out)),recursive=False)
        (out/'evidence.sha256').write_text(sha256(out/'evidence.tar.gz'))
    return {k:result[k] for k in ('status','error_type','error') if k in result}


def unpack(root,batch):
    folder=root/'batches'/batch
    require(sha256(folder/'evidence.tar.gz')==(folder/'evidence.sha256').read_text().strip(),'Archive hash mismatch')
    target=folder/'unpacked'
    if not target.exists():
        target.mkdir()
        # Every shard contains the same large source annotation files. Retain
        # those in the checksummed archive; only materialize replay inputs.
        with tarfile.open(folder/'evidence.tar.gz') as archive:
            for member in archive.getmembers():
                if (member.name in ('state.json','run/run.json') or
                        member.name.startswith(('run/raw/','run/attempts/'))):
                    archive.extract(member,target,filter='data')
    state=json.loads((target/'state.json').read_text())
    config=json.loads((root/'plan.json').read_text())['configs'][state['assignment']['model']]
    for row in state['rows']:
        raw=target/'run'/row['raw_file']
        require(sha256(raw)==row['raw_sha256'],'Raw evidence hash mismatch')
        observed=assess(read_events(raw),config)
        require(observed['valid']==row['valid'] and observed['turn_timing']==row['turn_timing'], 'Offline replay disagrees')
    write_json(folder/'verified.json',state)
    return state


def report(root):
    from .turn_report import build_turn_report
    root=Path(root);plan=json.loads((root/'plan.json').read_text());state=json.loads((root/'controller.json').read_text())
    report_root=root/('report-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    report_root.mkdir();models={}
    exclusions=json.loads((root/'user-exclusions.json').read_text()) if (root/'user-exclusions.json').exists() else {}
    excluded=set(exclusions.get('excluded_models',[]))
    for model in plan['models']:
        if model in excluded:continue
        combined=report_root/'evidence'/model;combined.mkdir(parents=True)
        (combined/'raw').mkdir();(combined/'attempts').mkdir()
        for name in ('manifest.json','review.json','draft.json'):
            os.link(Path(plan['manifest']).parent/name,combined/name)
        shards=[];runmeta=None
        for imported in plan.get('replayed_smoke',[]):
            if imported['model']!=model:continue
            source=Path(imported['source_run_path'])
            meta=json.loads((source/'run.json').read_text())
            expected=dict(meta['config'])
            reconstruction=plan['configs'][model].get('transcript_reconstruction')
            if reconstruction in ('elevenlabs-committed-segments-v2','speechmatics-empty-silence-ranges-v2'):
                expected['transcript_reconstruction']=reconstruction
            require(expected==plan['configs'][model] and meta['manifest_sha256']==plan['manifest_sha256'],
                    'Imported smoke changed more than the versioned transcript reconstruction')
            raw=source/'raw'/f"{imported['clip_id']}--attempt-1.jsonl"
            require(sha256(raw)==imported['raw_sha256'],'Imported original raw evidence changed')
            observed=assess(read_events(raw),plan['configs'][model])
            require(observed['valid']==imported['valid'] and observed['turn_timing']==imported['turn_timing'],'Imported smoke replay changed')
            for sub,name in [('raw',raw.name),('attempts',f"{imported['clip_id']}--attempt-1.json")]:
                os.link(source/sub/name,combined/sub/name)
            shards.append(dict(kind='original_first_attempt_replayed',run=meta,
                               raw_sha256=imported['raw_sha256'],source_run_path=str(source)))
        for b in state['batches'].values():
            if b['model']!=model or b['status']!='collected':continue
            verified=unpack(root,b['id']);source=root/'batches'/b['id']/'unpacked/run'
            if not (source/'run.json').exists():continue
            meta=json.loads((source/'run.json').read_text());runmeta=meta
            require(meta['manifest_sha256']==plan['manifest_sha256'] and meta['config']==plan['configs'][model],'Shard identity differs')
            shards.append(dict(batch_id=b['id'],archive_sha256=b['archiveHash'],run=meta))
            for row in verified['rows']:
                for sub,name in [('raw',Path(row['raw_file']).name),('attempts',f"{row['clip_id']}--attempt-1.json")]:
                    dest=combined/sub/name;require(not dest.exists(),'Duplicate original attempt across shards')
                    os.link(source/sub/name,dest)
        runmeta=runmeta or dict(measurement_version=5,mode='live',config=plan['configs'][model],manifest_sha256=plan['manifest_sha256'])
        runmeta={**runmeta,'execution_shards':shards,'source_manifest_path':str(Path(plan['manifest']).resolve()),
                 'composite_report_only':True}
        write_json(combined/'run.json',runmeta)
        result=build_turn_report(combined,report_root/model)
        result['execution']=state['models'][model]
        result['execution_shards']=shards
        result['reporting_code_sha256']=sha256(Path(__file__))
        result['user_exclusions']=exclusions
        result['cost']=dict(status='unavailable_account_rate_unverified',estimated_usd=None)
        rate=plan['configs'][model].get('pricing',{}).get('usd_per_minute')
        if rate is not None:
            seconds=sum(plan['durations'][r['clip_id']] for r in result['clips'] if r['attempted'])
            result['cost']=dict(status='configured_rate_estimate_not_invoice',estimated_usd=seconds/60*rate)
        write_json(report_root/model/'results.json',result);models[model]=result
    summaries={model:{k:r[k] for k in ('counts','accuracy','ttft','ttfs','exception_speech_end_to_final','deadlines','cost','execution')} for model,r in models.items()}
    write_json(report_root/'comparison.json',dict(plan=plan,models=summaries,user_exclusions=exclusions))
    def number(v):return '—' if v is None else f'{v:.1f}'
    rows=[]
    for model,r in models.items():
        count=r['counts'];wer=r['accuracy']['wer']
        rows.append(f'<tr><td><a href="{html.escape(model)}/index.html">{html.escape(model)}</a></td><td>{count["attempted"]}/{count["planned"]}</td><td>{count["valid"]}</td><td>{number(wer*100 if wer is not None else None)}</td><td>{number(r["ttft"]["p50_ms"])}</td><td>{number(r["ttfs"]["p50_ms"])}</td><td>{r["ttft"]["n"]}/{r["ttfs"]["n"]}</td><td>{html.escape(str(r["execution"].get("blocked") or ""))}</td></tr>')
    (report_root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Private turn benchmark</title><style>body{font:15px/1.5 system-ui;max-width:1400px;margin:40px auto;padding:20px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>Private conversational turns</h1><p>206 automatically prepared clips · 4 conversations · 8 speaker recordings · 94 excluded candidates. First attempts only. No Gemini 3.8. Blank metrics mean unavailable, never zero.</p><p>Open a model for p50/p90/p95, deadline accuracy, listening and raw transcript comparison. Full statistics and execution receipts: <a href="comparison.json">comparison.json</a>.</p><table><tr><th>Model</th><th>Attempted/planned</th><th>Valid</th><th>WER %</th><th>TTFT p50 ms</th><th>TTFS p50 ms</th><th>TTFT/TTFS n</th><th>Blocker</th></tr>'+''.join(rows)+'</table>')
    write_json(root/'latest-report.json',dict(path=str(report_root),models=summaries))
    return str(report_root)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','verify','worker','unpack','report'])
    p.add_argument('--root',type=Path);p.add_argument('--plan',type=Path,default=Path('turn-input/plan.json'))
    p.add_argument('--plan-hash');p.add_argument('--assignment',type=Path);p.add_argument('--out',type=Path);p.add_argument('--batch')
    a=p.parse_args()
    if a.mode=='prepare':result=prepare(a.root)
    elif a.mode=='verify':verify(a.plan,a.plan_hash);result={'verified':True}
    elif a.mode=='worker':result=asyncio.run(worker(a.plan,a.plan_hash,a.assignment,a.out))
    elif a.mode=='unpack':result=unpack(a.root,a.batch);result={k:result[k] for k in ('status','error_type','error') if k in result}
    else:result=report(a.root)
    print(json.dumps(result))


if __name__=='__main__':main()
