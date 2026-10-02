"""Separate offline reports for private conversational turns; no legacy pooling."""
from collections import Counter
import html
import json
from pathlib import Path

import numpy as np

from .data import sha256, write_json
from .measurement import deadline_observations, grouped_interval, summarize_deadlines
from .run import assess
from .score import aggregate_wer, word_errors
from .streaming import read_events
from .turn_metrics import MEASUREMENT
from .turns import require


def percentiles(values):
    values=[v for v in values if v is not None]
    return dict(n=len(values), **{f'p{p}_ms':float(np.percentile(values,p)) if values else None for p in (50,90,95)})


def latency_interval(rows, metric):
    groups={}
    for r in rows:
        value=r.get('turn_timing',{}).get(metric)
        if value is not None:groups.setdefault(r['dependency_group'],[]).append(value)
    if len(groups)<2:return dict(status='unavailable_insufficient_groups', groups=len(groups))
    values=list(groups.values());rng=np.random.default_rng(42)
    samples=[float(np.median([v for i in rng.integers(len(values),size=len(values)) for v in values[i]])) for _ in range(2000)]
    lo,hi=np.percentile(samples,[2.5,97.5])
    return dict(status='available',groups=len(groups),confidence=.95,lower_ms=float(lo),upper_ms=float(hi),
                iterations=2000,seed=42,method='conversation bootstrap of median turn latency')


def build_turn_report(root, out):
    root,out=Path(root),Path(out)
    run=json.loads((root/'run.json').read_text());m=json.loads((root/'manifest.json').read_text())
    require(run.get('measurement_version')==5 and run['config'].get('measurement_profile')==MEASUREMENT,'Expected turn measurement version 5')
    require(sha256(root/'manifest.json')==run['manifest_sha256'],'Saved manifest changed')
    for name in ('review','draft'):
        require(sha256(root/f'{name}.json')==m[f'{name}_sha256'],'Saved review evidence changed')
    require(not out.exists(),'Use a new report directory')
    rows=[]
    for c in m['clips']:
        raw=root/'raw'/f"{c['clip_id']}--attempt-1.jsonl"
        r=dict(clip_id=c['clip_id'],conversation_id=c['conversation_id'],speaker_id=c['speaker_id'],
               source_id=c['source_id'],dependency_group=c['conversation_id'],human_review_verified=m['listening_review_verified'],
               reference=c['reference'],transcript='',attempted=raw.exists(),valid=False,word_errors=None,
               turn_timing={},exclusion_reasons=[],audio=None)
        events=[];attempt=None
        if raw.exists():
            raw_hash=sha256(raw);meta=root/'attempts'/f"{c['clip_id']}--attempt-1.json"
            if meta.exists():require(json.loads(meta.read_text())['raw_sha256']==raw_hash,'Raw attempt changed')
            events=read_events(raw,allow_truncated_final=True)
            attempt=assess(events,run['config'],run['mode']=='dry_run')
            r.update(transcript=attempt['transcript'],valid=attempt['valid'],turn_timing=attempt['turn_timing'],
                     exclusion_reasons=attempt['exclusion_reasons'],raw_sha256=raw_hash,
                     first_partial_after_t0=attempt.get('first_partial_after_t0'),
                     completion_latency_ms=attempt['completion_latency_ms'],finalize_latency_ms=attempt['finalize_latency_ms'],
                     finalize_latency_status=attempt.get('finalize_latency_status'),
                     transcript_completion_basis=attempt.get('transcript_completion_basis'))
            if attempt['valid'] and run['mode']=='live':r['word_errors']=word_errors(c['reference'],attempt['transcript'])
        r['deadlines']=deadline_observations(events,c['reference'],None,attempt,run['mode'],run['config'])
        rows.append(r)
    good=[r for r in rows if r['word_errors'] is not None]
    attempted=sum(r['attempted'] for r in rows);valid=sum(r['valid'] for r in rows)
    result=dict(schema_version=1,measurement_version=5,measurement_profile=MEASUREMENT,
        title='Private conversational turns', mode=run['mode'],config=run['config'],
        preparation_mode=m.get('preparation_mode','manual-review-v1'),
        listening_review_verified=m['listening_review_verified'], boundary_detector=m.get('boundary_detector'),
        transcript_source=m.get('transcript_source','listening_reviewed'),
        manifest_sha256=run['manifest_sha256'],review_sha256=m['review_sha256'],
        counts={**m['counts'],'planned':len(rows),'attempted':attempted,'valid':valid,'failed':attempted-valid,'not_run':len(rows)-attempted},
        accuracy=dict(**aggregate_wer([r['word_errors'] for r in good]),measured_turns=len(good),
                      confidence_interval=grouped_interval(good,lambda r:r['word_errors'])),
        ttft={**percentiles([r['turn_timing'].get('ttft_ms') for r in rows]),
              'confidence_interval':latency_interval(rows,'ttft_ms'),
              'status_counts':dict(Counter(r['turn_timing'].get('ttft_status','not_run') for r in rows)),
              'first_text_kind_counts':dict(Counter(r['turn_timing'].get('first_text_kind') for r in rows if r['turn_timing'].get('ttft_ms') is not None))},
        ttfs={**percentiles([r['turn_timing'].get('ttfs_ms') for r in rows]),
              'confidence_interval':latency_interval(rows,'ttfs_ms'),
              'status_counts':dict(Counter(r['turn_timing'].get('ttfs_status','not_run') for r in rows)),
              'final_before_boundary':sum(bool(r['turn_timing'].get('final_text_before_boundary')) for r in rows)},
        exception_speech_end_to_final=percentiles([r['turn_timing'].get('observed_speech_end_to_final_ms') for r in rows
                                                  if r['turn_timing'].get('finalization_class')=='provider_exception']),
        deadlines=summarize_deadlines(rows),clips=rows,dataset_exclusions=m.get('exclusions', []),
        definitions=dict(ttft='First nonempty transcript receipt minus first actual audio packet send start; first text may be partial or final.',
            ttfs='Last change to completed final transcript minus actual delivery of the frozen speech-end frame; display clamped at zero, signed value retained.',
            grouping='Uncertainty is resampled by conversation, not independent turns. Four conversations provide limited population evidence.',
            scope='One original attempt per turn. Separate from full-recording measurements and historical combined ranking.'))
    out.mkdir(parents=True);(out/'audio').mkdir()
    parent=Path(run.get('source_manifest_path') or root/'manifest.json').parent
    for r,c in zip(rows,m['clips']):
        audio=parent/c['audio']
        if audio.exists():
            require(sha256(audio)==c['audio_sha256'],'Listening audio changed')
            link=out/'audio'/f"{c['clip_id']}.wav";link.symlink_to(audio.resolve());r['audio']='audio/'+link.name
    write_json(out/'results.json',result)
    (out/'index.html').write_text(render(result))
    return result


def render(result):
    def esc(x):return html.escape(str(x))
    def num(x):return 'Unavailable' if x is None else f'{x:.1f}'
    c=result['counts'];rows=[]
    for r in result['clips']:
        timing=r['turn_timing'];counts=r['word_errors']
        wer=(counts['substitutions']+counts['insertions']+counts['deletions'])/counts['reference_words'] if counts and counts['reference_words'] else None
        player=f'<audio controls preload="none" src="{esc(r["audio"])}"></audio>' if r['audio'] else '<small>Listening audio unavailable locally</small>'
        if timing.get('final_text_before_boundary'):
            player += '<p>Final text available before boundary; displayed TTFS is clamped to zero.</p>'
        rows.append(f'<details><summary>{esc(r["clip_id"])} · TTFT {num(timing.get("ttft_ms"))} ms · TTFS {num(timing.get("ttfs_ms"))} ms · WER {num(wer*100 if wer is not None else None)}%</summary>{player}<h3>Reference</h3><p>{esc(r["reference"])}</p><h3>Provider transcript</h3><p>{esc(r["transcript"])}</p><p>{esc(", ".join(r["exclusion_reasons"]))}</p><small>TTFT: {esc(timing.get("ttft_status","not_run"))}; first text: {esc(timing.get("first_text_kind"))}. TTFS: {esc(timing.get("ttfs_status","not_run"))}; signed delay: {num(timing.get("ttfs_signed_ms"))} ms.</small></details>')
    summaries=''.join(f'<tr><td>{name.upper()}</td><td>{result[name]["n"]}</td>'+''.join(f'<td>{num(result[name][f"p{p}_ms"])}</td>' for p in (50,90,95))+'</tr>' for name in ('ttft','ttfs'))
    deadlines=''.join(f'<tr><td>{d["deadline_ms"]} ms</td><td>{d["measured_clips"]} / {d["planned_clips"]}</td><td>{num(d["wer"]*100 if d["wer"] is not None else None)}%</td></tr>' for d in result['deadlines'])
    extra=''
    exception=result['exception_speech_end_to_final']
    if result['config'].get('turn_finalization_class')=='provider_exception':
        extra += '<h2>Observed final-text delay</h2><p>This provider has no supported controlled finalization. These values are excluded from controlled TTFS comparisons.</p>'
        extra += '<p>Measured turns: '+str(exception['n'])+' · '+ ' · '.join(f'p{p}: {num(exception[f"p{p}_ms"])} ms' for p in (50,90,95))+'</p>'
    extra += '<h2>Metric coverage and uncertainty</h2><p>Confidence intervals resample conversations. With only four conversations, these intervals do not establish a broad model ranking.</p>'
    for metric in ('ttft','ttfs'):
        stats=result[metric];ci=stats.get('confidence_interval',{})
        interval=(num(ci['lower_ms'])+'–'+num(ci['upper_ms'])+' ms') if ci.get('status')=='available' else 'Unavailable'
        extra += '<p>'+metric.upper()+' median 95% interval: '+interval+'. Statuses: '+esc(json.dumps(stats['status_counts']))+'</p>'
    extra += '<p>First text kind: '+esc(json.dumps(result['ttft']['first_text_kind_counts']))+'. Final text available before speech-end boundary: '+str(result['ttfs']['final_before_boundary'])+' turns.</p>'
    if result.get('diagnostics'):
        extra += '<h2>Separate completion diagnostics</h2><table><tr><th>Metric</th><th>Measured turns</th><th>p50 ms</th><th>p90 ms</th><th>p95 ms</th></tr>'
        for key,label in [('completion','Stream observation completion (includes tail)'),('finalization_ack','Finalization acknowledgment'),('first_partial_after_speech_end','First partial after speech end')]:
            stats=result['diagnostics'][key]
            extra += '<tr><td>'+label+'</td><td>'+str(stats['n'])+'</td>'+''.join('<td>'+num(stats[f'p{p}_ms'])+'</td>' for p in (50,90,95))+'</tr>'
        extra += '</table><p>These diagnostics use separate receipt events and are not TTFT or TTFS. Physical socket-close timing was not separately recorded.</p>'
    recovery=result.get('recovery_summary',{})
    if recovery.get('attempted'):
        extra += '<h2>Separate recovery experiment</h2><p>'+esc(recovery['interpretation'])+'</p>'
        extra += '<p>Recovery attempts: '+str(recovery['attempted'])+' · Valid: '+str(recovery['valid'])+' · Additional turns recovered: '+str(recovery['additional_turns_recovered'])+' · Valid turns after recovery: '+str(recovery['valid_after_recovery'])+' / '+str(c['planned'])+'</p>'
        extra += '<p>Pooled WER after recovery: '+num(recovery['after_recovery_accuracy']['wer']*100 if recovery['after_recovery_accuracy']['wer'] is not None else None)+'%. Primary WER above is unchanged.</p>'
        extra += '<table><tr><th>Recovery subset metric</th><th>Measured turns</th><th>p50 ms</th><th>p90 ms</th><th>p95 ms</th></tr>'
        for key,label in [('recovery_subset_ttft','TTFT'),('recovery_subset_ttfs','TTFS')]:
            stats=recovery[key];extra += '<tr><td>'+label+'</td><td>'+str(stats['n'])+'</td>'+''.join('<td>'+num(stats[f'p{p}_ms'])+'</td>' for p in (50,90,95))+'</tr>'
        extra += '</table>'
        for attempt in result['recovery_attempts']:
            extra += '<details><summary>Recovery: '+esc(attempt['clip_id'])+' · '+('Valid' if attempt['valid'] else 'Failed')+'</summary><p>'+esc(attempt['transcript'])+'</p><p>'+esc(', '.join(attempt.get('exclusion_reasons',[])))+'</p></details>'
    if result.get('cost'):
        extra += '<p>Cost: '+esc(result['cost']['status'])+'. Estimate USD: '+('Unavailable' if result['cost']['estimated_usd'] is None else f"{result['cost']['estimated_usd']:.4f}")+'.</p>'
    exclusions=''.join(f'<li>{esc(t["turn_id"])}: {esc(t.get("exclusion_reason") or t.get("reason") or t.get("notes", ""))}</li>' for t in result.get('dataset_exclusions', []))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Private conversational turns</title>
<style>body{{max-width:1050px;margin:auto;padding:48px 24px;background:#fbfbfa;color:#2f3437;font:16px/1.6 'Helvetica Neue',sans-serif}}h1{{font:44px Georgia,serif}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #eaeaea}}details{{background:white;padding:20px;margin:12px 0;border:1px solid #eaeaea;border-radius:8px}}summary{{cursor:pointer}}small{{color:#666}}audio{{margin-top:16px;max-width:100%}}</style>
<h1>Private conversational turns</h1><p>{esc(result['config'].get('model_id',result['config']['model']))} · {esc(result['mode'])}</p>
<p>Preparation: {esc(result['preparation_mode'])}. {'Boundaries and transcripts were listening-reviewed.' if result['listening_review_verified'] else 'Speech boundaries were automatically estimated; supplied transcripts were not independently listening-reviewed.'}</p>
<p>{c['conversations']} conversations · {c['speaker_recordings']} speaker recordings · {c['turns']} exported turns</p><p>Planned {c['planned']} · Attempted {c['attempted']} · Valid {c['valid']} · Failed {c['failed']} · Not run {c['not_run']}</p>
<h2>Pooled word error rate: {num(result['accuracy']['wer']*100 if result['accuracy']['wer'] is not None else None)}%</h2><p>Measured on {result['accuracy']['measured_turns']} turns. Latency uses first attempts and does not require correct words.</p>
<table><tr><th>Metric</th><th>Measured turns</th><th>p50 ms</th><th>p90 ms</th><th>p95 ms</th></tr>{summaries}</table>
<h2>Accuracy after speech end</h2><table><tr><th>Deadline</th><th>Measured / planned turns</th><th>Pooled word error rate</th></tr>{deadlines}</table>
{extra}<h2>Excluded during dataset preparation</h2><p>{len(result.get('dataset_exclusions', []))} entries excluded before benchmarking.</p><ul>{exclusions}</ul>
<p>{esc(result['definitions']['ttft'])}</p><p>{esc(result['definitions']['ttfs'])}</p><p>{esc(result['definitions']['grouping'])}</p><p>{esc(result['definitions']['scope'])}</p><p><a href="results.json">Full results, deadline accuracy, uncertainty, and exclusions</a></p><h2>Listen and inspect each turn</h2>{''.join(rows)}</html>'''
