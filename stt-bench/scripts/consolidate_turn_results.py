"""Verify and consolidate chronological first attempts across turn continuations."""
import argparse
from datetime import datetime,timezone
import csv,html,json,os
from pathlib import Path
from stt_bench.data import sha256,write_json
from stt_bench.score import word_errors,aggregate_wer
from stt_bench.turn_batch import unpack
from stt_bench.turn_report import build_turn_report,percentiles,render


def recovery_summary(clips, attempts):
    originals={r['clip_id']:r for r in clips}
    valid=[r for r in attempts if r['valid']]
    recovered={r['clip_id']:r for r in valid if not originals[r['clip_id']]['valid']}
    errors={r['clip_id']:r['word_errors'] for r in clips if r['word_errors'] is not None}
    timings={r['clip_id']:r['turn_timing'] for r in clips if r['valid']}
    for cid,r in recovered.items():
        errors[cid]=word_errors(originals[cid]['reference'],r['transcript'])
        timings[cid]=r['turn_timing']
    return dict(attempted=len(attempts),valid=len(valid),failed=len(attempts)-len(valid),
        additional_turns_recovered=len(recovered),valid_after_recovery=sum(r['valid'] for r in clips)+len(recovered),
        recovery_subset_accuracy=aggregate_wer([word_errors(originals[r['clip_id']]['reference'],r['transcript']) for r in valid]),
        recovery_subset_ttft=percentiles([r['turn_timing'].get('ttft_ms') for r in valid]),
        recovery_subset_ttfs=percentiles([r['turn_timing'].get('ttfs_ms') for r in valid]),
        after_recovery_accuracy=aggregate_wer(list(errors.values())),
        after_recovery_ttft=percentiles([t.get('ttft_ms') for t in timings.values()]),
        after_recovery_ttfs=percentiles([t.get('ttfs_ms') for t in timings.values()]),
        interpretation='Supplementary recovery experiment. Primary metrics retain every original first attempt.')


def consolidate(roots,out):
    roots=[Path(r) for r in roots];out=Path(out);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((roots[0]/'plan.json').read_text());excluded={'deepgram-nova-2'}
    for root in roots:
        exclusion_file=root/'user-exclusions.json'
        if exclusion_file.exists():excluded.update(json.loads(exclusion_file.read_text()).get('excluded_models',[]))
    models=[m for m in plan['models'] if m not in excluded];first={m:{} for m in models};later=[];statuses=[];peaks={m:0 for m in models}
    for root in roots:
        state=json.loads((root/'controller.json').read_text());source_plan=json.loads((root/'plan.json').read_text())
        statuses.append(dict(root=str(root),status=state['status'],updated_at=state['updatedAt'],
            active_batches=sum(len(m['active']) for m in state['models'].values()),
            workers_stopped=sum(w.get('computeStopped',False) for w in state['workers'].values()),
            workers_total=len(state['workers']),preparation_stopped=state['preparation'].get('computeStopped',False)))
        for model,m in state['models'].items():
            if model in peaks:peaks[model]=max(peaks[model],m.get('peak',0))
        for b in state['batches'].values():
            model=b['model']
            if model not in first or b['status']!='collected':continue
            verified=unpack(root,b['id']);source=root/'batches'/b['id']/'unpacked/run'
            if not (source/'run.json').exists():continue
            meta=json.loads((source/'run.json').read_text())
            assert meta['manifest_sha256']==plan['manifest_sha256']
            for row in verified['rows']:
                item=dict(source=str(source.resolve()),batch_id=b['id'],root=str(root),row=row,run=meta,
                          archive_sha256=b['archiveHash'],raw_sha256=row['raw_sha256'])
                key=row['clip_id']
                if key in first[model]:
                    later.append(dict(model=model,clip_id=key,source=item['source'],original_source=first[model][key]['source'],
                        valid=row['valid'],transcript=row['transcript'],exclusion_reasons=row.get('exclusion_reasons',[]),
                        turn_timing=row['turn_timing'],raw_sha256=row['raw_sha256']))
                else:first[model][key]=item
    summaries={}
    for model in models:
        evidence=out/'evidence'/model;evidence.mkdir(parents=True);(evidence/'raw').mkdir();(evidence/'attempts').mkdir();(evidence/'metadata').mkdir()
        for name in ('manifest.json','draft.json','review.json'):os.link(Path(plan['manifest']).parent/name,evidence/name)
        config=dict(plan['configs'][model])
        if model=='elevenlabs-scribe-v2-realtime':config['transcript_reconstruction']='elevenlabs-committed-segments-v3'
        if model in ('speechmatics-enhanced','speechmatics-standard'):config['transcript_reconstruction']='speechmatics-empty-silence-ranges-v2'
        provenance=[];copied=set()
        for cid,item in first[model].items():
            source=Path(item['source']);raw=source/'raw'/f'{cid}--attempt-1.jsonl'
            assert sha256(raw)==item['raw_sha256']
            source_config=dict(item['run']['config']);target_config=dict(config)
            source_config.pop('transcript_reconstruction',None);target_config.pop('transcript_reconstruction',None)
            assert source_config==target_config,(model,'wire settings changed')
            for sub,name in [('raw',raw.name),('attempts',f'{cid}--attempt-1.json')]:os.link(source/sub/name,evidence/sub/name)
            metadata_name=Path(item['root']).name+'-'+item['batch_id']+'.json'
            if metadata_name not in copied:os.link(source/'run.json',evidence/'metadata'/metadata_name);copied.add(metadata_name)
            provenance.append(dict(clip_id=cid,source_root=item['root'],source_batch=item['batch_id'],raw_sha256=item['raw_sha256'],
                archive_sha256=item['archive_sha256'],execution_metadata='metadata/'+metadata_name,
                execution_metadata_sha256=sha256(source/'run.json')))
        write_json(evidence/'run.json',dict(schema_version=2,measurement_version=5,mode='live',config=config,
            manifest_sha256=plan['manifest_sha256'],source_manifest_path=str(Path(plan['manifest']).resolve()),
            composite_report_only=True,analysis_only=True,first_attempts_only=True,
            analysis_source_hashes={name:sha256(Path('src/stt_bench')/name) for name in ['turn_metrics.py','turn_report.py','provider_protocol.py','measurement.py']},
            execution_provenance=provenance))
        result=build_turn_report(evidence,out/model)
        result['execution_provenance']=provenance
        result['recovery_attempts']=[r for r in later if r['model']==model]
        result['recovery_summary']=recovery_summary(result['clips'],result['recovery_attempts'])
        result['policy']='First chronological attempt per turn across source runs. Later attempts remain separate. Collector corrections use original raw events; no timing timestamp is changed.'
        result['analysis_code_sha256']=sha256(Path(__file__))
        result['diagnostics']={
            'completion':percentiles([r.get('completion_latency_ms') for r in result['clips'] if r['valid']]),
            'finalization_ack':percentiles([r.get('finalize_latency_ms') for r in result['clips'] if r['valid'] and r.get('finalize_latency_status')=='observed']),
            'first_partial_after_speech_end':percentiles([(r.get('first_partial_after_t0') or {}).get('latency_ms') for r in result['clips'] if r['valid']]),
            'socket_close':{'status':'not_recorded_separately','n':0,'p50_ms':None,'p90_ms':None,'p95_ms':None},
            'peak_reserved_workers':peaks[model],
            'concurrency_note':'Peak allocated workers, including connection setup. This is not proof of the account maximum.'}
        count=result['counts'];count.update(failure_rate=count['failed']/count['planned'],not_run_rate=count['not_run']/count['planned'],missing_result_rate=(count['planned']-count['valid'])/count['planned'])
        rate=config.get('pricing',{}).get('usd_per_minute')
        result['cost']=dict(status='account_rate_unverified',estimated_usd=None)
        if rate is not None:
            seconds=0
            sample_rate=config.get('sample_rate',config.get('query',{}).get('sample_rate',16000))
            for item in first[model].values():
                raw=Path(item['source'])/item['row']['raw_file']
                with raw.open() as stream:
                    seconds+=sum(e.get('bytes',0)/(2*sample_rate) for e in map(json.loads,stream) if e['kind']=='audio_sent')
            result['cost']=dict(status='configured_list_rate_estimate_not_invoice',estimated_usd=seconds/60*rate,actual_sent_audio_seconds=seconds)
        write_json(out/model/'results.json',result)
        (out/model/'index.html').write_text(render(result))
        summaries[model]={k:result[k] for k in ['counts','accuracy','ttft','ttfs','exception_speech_end_to_final','deadlines','recovery_attempts','recovery_summary','diagnostics','cost']}
    complete=all(s['status'] in ('complete','partial_or_blocked') and not s['active_batches'] and s['workers_stopped']==s['workers_total'] and s['preparation_stopped'] for s in statuses)
    summary=dict(updated_at=datetime.now(timezone.utc).isoformat(),execution_complete=complete,
                 all_included_models_attempted_all_turns=all(m['counts']['attempted']==206 for m in summaries.values()),source_runs=statuses,
                 models=summaries,excluded_models=sorted(excluded|{'gemini-3.8'}),planned_turns_per_model=206,
                 planned_total=206*len(models),later_recovery_attempts=later)
    write_json(out/'comparison.json',summary)
    flat=[]
    for model,result in summaries.items():
        row={'model':model,**{key:result['counts'][key] for key in ('planned','attempted','valid','failed','not_run')},
             'wer_percent':None if result['accuracy']['wer'] is None else 100*result['accuracy']['wer']}
        for metric in ('ttft','ttfs','exception_speech_end_to_final'):
            row.update({metric+'_'+key:value for key,value in result[metric].items() if key in ('n','p50_ms','p90_ms','p95_ms')})
        for deadline in result['deadlines']:
            prefix='deadline_'+str(deadline['deadline_ms'])+'ms_'
            row[prefix+'wer_percent']=None if deadline['wer'] is None else deadline['wer']*100
            row[prefix+'measured_turns']=deadline['measured_clips']
        row['recovery_attempts']=len(result['recovery_attempts'])
        row['additional_turns_recovered']=result['recovery_summary']['additional_turns_recovered']
        row['valid_after_recovery']=result['recovery_summary']['valid_after_recovery']
        row['cost_status']=result['cost']['status']
        row['estimated_usd']=result['cost']['estimated_usd']
        flat.append(row)
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

    def n(x):return '—' if x is None else f'{x:,.1f}'
    rows=[]
    for model,m in summaries.items():
        c=m['counts'];rows.append(f'<tr><td><a href="{model}/index.html">{html.escape(model)}</a></td><td>{c["attempted"]}/{c["planned"]}</td><td>{c["valid"]}</td><td>{n(None if m["accuracy"]["wer"] is None else m["accuracy"]["wer"]*100)}</td><td>{n(m["ttft"]["p50_ms"])}</td><td>{n(m["ttfs"]["p50_ms"])}</td><td>{m["ttft"]["n"]}/{m["ttfs"]["n"]}</td></tr>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Turn benchmark results</title><style>body{font:15px/1.5 system-ui;max-width:1400px;margin:40px auto;padding:20px}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>Private turn benchmark</h1><p>'+('Final execution snapshot' if complete else 'Interim snapshot: provider work or collection is still running')+' · '+summary['updated_at']+'</p><p>206 automatically prepared clips · 4 conversations · 8 speaker recordings · 94 preparation exclusions. User-requested exclusions are listed in the full results.</p><p>First attempts only. Original raw events were replayed with versioned collector corrections. Later recovery attempts are listed separately. Missing latency is unavailable, not zero. Google Chirp has no controlled TTFS; observed final delay is separate.</p><p>Open a model for p50/p90/p95, deadline accuracy, uncertainty, reference text, provider output and listening. <a href="comparison.json">Full results and provenance</a> · <a href="metrics.csv">Download metrics CSV</a>.</p><table><tr><th>Model</th><th>Attempted / planned</th><th>Valid</th><th>WER %</th><th>TTFT p50 ms</th><th>TTFS p50 ms</th><th>TTFT / TTFS observations</th></tr>'+''.join(rows)+'</table>')
    return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args();s=consolidate(a.root,a.out);print(json.dumps({'path':a.out,'execution_complete':s['execution_complete'],'attempted':sum(m['counts']['attempted'] for m in s['models'].values())}))
