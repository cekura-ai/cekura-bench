#!/usr/bin/env python3
"""Combine verified aggregate scores from runs on identical item identities."""
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def render(roots,out):
    out.mkdir(parents=True,exist_ok=True)
    models={}; sources=[]; expected=None; complete=True
    for root in roots:
        plan=json.loads((root/'plan.json').read_text());report=json.loads((root/'results.json').read_text())
        if expected is None:expected=plan['items']
        assert plan['items']==expected,'Comparison requires identical complete datasets'
        audit=json.loads((root/'evidence-audit.json').read_text()) if (root/'evidence-audit.json').exists() else None
        stopped=json.loads((root/'compute-stop-verification.json').read_text()) if (root/'compute-stop-verification.json').exists() else None
        concurrency=json.loads((root/'stream-concurrency.json').read_text()) if (root/'stream-concurrency.json').exists() else {}
        status=report['execution']['status']
        complete &= status=='complete' and report['execution']['all_compute_stopped'] and bool(audit and audit.get('status')=='passed' and audit.get('run_status')=='complete' and audit.get('plan_sha256')==sha(root/'plan.json')) and bool(stopped and stopped.get('allStopped'))
        sources.append({'root':str(root.resolve()),'run_id':plan['run_id'],'status':status,'plan_sha256':sha(root/'plan.json'),
                        'results_sha256':sha(root/'results.json'),'audit_present':bool(audit),'stop_check_present':bool(stopped)})
        for model,data in report['models'].items():
            assert model not in models,'Duplicate model configuration'
            models[model]={'config':plan['configs'][model], 'groups':data['groups'], 'concurrency':concurrency.get(model),
                           'execution':report['execution']['models'][model],'report':'../'+root.name+'/index.html',
                           'source_run':plan['run_id']}
    result={'status':'verified_complete' if complete else 'partial','generated_at':datetime.now(timezone.utc).isoformat(),
            'items_per_model':1008,'sources':sources,'models':models}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    headings=['Model / configuration','Usable items','Recovered','Combined WER','Public WER','Private WER','Public WER at +500 ms','Private word delay (median)','Accepted concurrency peak']
    rows=[]
    for model,d in models.items():
        g=d['groups'];deadline=next(x for x in g['public']['deadlines'] if x['deadline_ms']==500)
        def pct(v):return 'Unavailable' if v is None else f'{v:.2%}'
        delay=g['private']['word_finalization'].get('p50_ms')
        rows.append([model,f"{g['combined']['successful']}/1008",str(g['combined']['recovered']),
                     *[pct(g[c]['final_wer']['wer']) for c in ['combined','public','private']],pct(deadline['wer']),
                     'Unavailable' if delay is None else f'{delay:,.0f} ms',str((d['concurrency'] or {}).get('peak_accepted_session_overlap','Pending'))])
    notes=[f"Status: {result['status']}. Same 1,000 public clips and eight intact private recordings per model.",
      'WER is summed word errors divided by summed reference words. Lower is better. Final WER uses valid completed attempts and may include one recovery. First-attempt deadline accuracy and private word delays do not use recoveries.',
      'All results were collected with concurrent sandbox workers. AssemblyAI uses universal-3-5-pro with min_latency and language_codes=[en], 60 ms wire packets, and no silence overrides. Its confirmed account limit is five new streams per minute across all regions; existing streams may overlap. Other adapters use their existing 20 ms transport. This difference remains part of the comparison.',
      'Gradium private recordings use consecutive provider sessions of up to 270 seconds because the provider limits session length. Every original frame is retained; context resets and reconnect gaps are reported separately.',
      'Inworld private WER includes a genuine provider-output repetition in conversation-04-B (59.66% WER for that recording). The valid result was not dropped or rerun.',
      'Private word delays describe correctly aligned, finalized words, not every spoken word. See each report for measured-word coverage, exclusions, public deadline coverage, completion timing, failures, and recovery history.',
      'Earlier reports remain preserved. These rows are configuration-specific measurements, not a general model ranking.']
    md=['# Full concurrent benchmark comparison','',*sum(([n,''] for n in notes),[]),'| '+' | '.join(headings)+' |','| '+' | '.join(['---']*len(headings))+' |']
    md+=['| '+' | '.join(r)+' |' for r in rows]
    md+=['','Detailed reports:','']+[f"- [{m}]({d['report']})" for m,d in models.items()]
    (out/'RESULTS.md').write_text('\n'.join(md)+'\n')
    body='<h1>Full concurrent benchmark comparison</h1>'+''.join('<p>'+html.escape(n)+'</p>' for n in notes)
    body+='<div style="overflow:auto"><table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headings)+'</tr>'
    for row in rows:
        body+='<tr>'+''.join('<td>'+('<a href="'+models[row[0]]['report']+'">'+html.escape(v)+'</a>' if i==0 else html.escape(v))+'</td>' for i,v in enumerate(row))+'</tr>'
    body+='</table></div><p><a href="summary.json">Aggregate scores and source hashes</a> · <a href="RESULTS.md">Markdown report</a></p>'
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Concurrent STT comparison</title><style>body{font:16px system-ui;margin:40px auto;padding:0 24px;max-width:1400px;background:#fafaf7;color:#202522}p{line-height:1.6;max-width:100ch}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;text-align:left;border-bottom:1px solid #ddd}th{background:#eef1eb}a{color:#206244}</style><body>'+body+'</body></html>')
    print(json.dumps({'status':result['status'],'models':len(models),'report':str(out/'index.html')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();render(a.roots,a.out)
