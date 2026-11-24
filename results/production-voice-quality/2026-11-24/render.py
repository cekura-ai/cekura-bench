import html,json,pathlib
P=pathlib.Path(__file__).resolve().parent
def render(d):
    WEIGHTS=d["weights"]; DIMENSIONS=d["categories"]
    esc=html.escape
    def fmt(v,kind):return 'N/A' if v is None else (f'{v/1000:.2f} s' if kind=='ms' else f'{v:.1f}'+('%' if kind=='pass' else ''))
    rows=d['providers'];metrics=d['metrics'];logo=(P/'logo.svg').read_text()
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Production Voice Agent Quality Benchmark</title><style>body{margin:0;color:#182c40;background:#f8fafc;font:15px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}main{max-width:1280px;margin:auto;padding:32px}header{border-bottom:2px solid #3265ad;padding-bottom:18px;display:flex;justify-content:space-between}header svg{width:125px;height:auto}h1{font-size:40px;letter-spacing:-1.2px;margin:24px 0 8px}h2{margin:0 0 12px;font-size:24px}h3{margin:0;font-size:17px}p{color:#617184}section{overflow-x:auto;margin:28px 0;background:white;border:1px solid #dce5ee;border-radius:12px;padding:24px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid section{margin:0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:right;padding:12px 10px;border-bottom:1px solid #e4eaf0;white-space:nowrap}th{background:#eef4fa}th:first-child,td:first-child{text-align:left}small{font-size:11px;color:#617184}a{color:#3265ad}.bar{display:grid;grid-template-columns:110px 1fr 62px 74px;gap:8px;align-items:center;font-size:12px;margin:8px 0}.track{height:8px;background:#eef4fa;border-radius:5px}.fill{height:8px;background:#7395bd;border-radius:5px}.def{font-size:12px;min-height:70px}details{border-top:1px solid #e4eaf0;padding:14px 0}summary{cursor:pointer;font-weight:600}.stats{display:flex;flex-wrap:wrap;gap:25px;margin:22px 0}.stats b{font-size:24px}.stats span{display:block;font-size:12px;color:#617184}svg text{font-family:inherit}@media(max-width:750px){main{padding:16px}.grid{grid-template-columns:1fr}.grid section{min-width:0;padding:16px}.bar{grid-template-columns:90px 1fr 50px 65px;gap:5px}h1{font-size:30px}}@media print{section{break-inside:avoid}.scroll{overflow:visible}.grid{grid-template-columns:1fr}}.focus-row td{background:#e2f2f5;font-weight:700}.focus-bar{background:#e2f2f5;padding:6px;border-radius:6px;margin-left:-6px;margin-right:-6px}.focus-bar .fill{background:#007c91}.focus-panel{border:2px solid #007c91}.focus-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.focus-cards b{display:block;font-size:27px;color:#007c91}.focus-cards small{display:block}.concerns li{margin:10px 0}.focus-note{font-size:13px}@media(max-width:750px){.focus-cards{grid-template-columns:repeat(2,1fr)}} </style><main><header>'+logo+'<small>24 November 2026 · Cisco focus</small></header><h1>Production Voice Agent Quality Benchmark: Cisco</h1><p>Conversation quality of live production AI assistants, tested with the same generic support scenarios.</p><div class="stats">'+''.join(f'<div><b>{v:,}</b><span>{label}</span></div>' for v,label in [(d['provider_count'],'Providers'),(d['phone_number_count'],'Production phone numbers'),(d['selected_call_count'],'Selected calls'),(15,'Scenarios · three runs each')])+'</div>']
    parts.append(focus_summary(d))
    lat=[r['metrics']['latency']['value'] for r in rows if r['metrics']['latency']['value'] is not None]; xmax=max(lat)*1.18; ymin=min(r['composite'] for r in rows if r['composite'] is not None)-3; ymax=100
    svg=['<svg viewBox="0 0 1120 450" role="img" aria-label="Composite quality versus median turn latency">']
    for i in range(6):
        x=80+i*950/5;v=i*xmax/5;svg.append(f'<path d="M{x} 30V365" stroke="#e4eaf0"/><text x="{x}" y="390" text-anchor="middle" fill="#617184" font-size="12">{v/1000:.1f}s</text>')
        y=365-i*335/5;v=ymin+i*(ymax-ymin)/5;svg.append(f'<path d="M80 {y}H1030" stroke="#e4eaf0"/><text x="65" y="{y+4}" text-anchor="end" fill="#617184" font-size="12">{v:.0f}</text>')
    labels=[]
    points=[(80+950*r['metrics']['latency']['value']/xmax,365-335*(r['composite']-ymin)/(ymax-ymin)) for r in rows if r['composite'] is not None and r['metrics']['latency']['value'] is not None]
    for i,r in enumerate(rows):
        if r['composite'] is None or r['metrics']['latency']['value'] is None:continue
        x=80+950*r['metrics']['latency']['value']/xmax;y=365-335*(r['composite']-ymin)/(ymax-ymin);dy=-9 if i%2 else 16
        width=len(r['provider'])*6.2
        for dx,dy in [(8,-9),(8,18),(-width-8,-9),(-width-8,18),(8,-25),(8,32)]:
            box=(x+dx,y+dy-11,x+dx+width,y+dy+3)
            if not any(box[0]<b[2]+4 and box[2]>b[0]-4 and box[1]<b[3]+3 and box[3]>b[1]-3 for b in labels) and not any(box[0]-5<px<box[2]+5 and box[1]-5<py<box[3]+5 for px,py in points):break
        labels.append(box)
        svg.append(f'<g><title>{r["provider"]}: {r["composite"]:.2f}; latency {r["metrics"]["latency"]["value"]:.0f} ms</title><circle cx="{x}" cy="{y}" r="{8 if r["provider"]=="Cisco" else 5}" fill="{"#007c91" if r["provider"]=="Cisco" else "#a5b4c4"}"/><text x="{x+dx}" y="{y+dy}" font-size="11" fill="#182c40">{r["provider"]}</text></g>')
    svg+=['<text x="550" y="430" text-anchor="middle" font-size="13">Median turn latency · lower is better</text><text x="18" y="205" transform="rotate(-90 18 205)" text-anchor="middle" font-size="13">Composite quality · higher is better</text></svg>']
    parts+=['<section><h2>Quality and median turn latency</h2><p>Composite quality is scored out of 100. Median turn latency is measured separately.</p>'+''.join(svg)+'</section>']
    parts+=['<section><h2>Provider results</h2><p>Parentheses show the number of tested phone numbers. Each number receives equal weight in the quality score.</p><div class="scroll"><table><thead><tr><th>Provider</th><th>Composite /100</th>'+''.join('<th>'+esc(m['label'])+('<br>/5' if m['kind']=='score' else '<br>% passing' if m['kind']=='pass' else '<br>seconds')+'</th>' for m in metrics)+'</tr></thead><tbody>']
    for r in rows:parts+=['<tr class="'+('focus-row' if r['provider']=='Cisco' else '')+'"><td>'+r['provider']+f' ({r["phone_number_count"]})</td><td><b>'+fmt(r['composite'],'score')+'</b></td>'+''.join('<td>'+fmt(r['metrics'][m['key']]['value'],m['kind'])+'</td>' for m in metrics)+'</tr>']
    parts+=['</tbody></table></div></section><div class="grid">']
    for m in metrics:
        rr=sorted(rows,key=lambda r:(r['metrics'][m['key']]['value'] is None, (r['metrics'][m['key']]['value'] or 0)*(1 if m['kind']=='ms' else -1)));maxv=5 if m['kind']=='score' else 100 if m['kind']=='pass' else max(r['metrics'][m['key']]['value'] or 0 for r in rr)
        parts+=['<section><h3>'+m['label']+'</h3><small>'+('Separate latency measurement' if m['key']=='latency' else f'{WEIGHTS[m["key"]]*100:.0f}% of composite')+'</small><p class="def">'+esc(m['description'])+'</p>']
        for r in rr:
            a=r['metrics'][m['key']];v=a['value'];parts+=[f'<div class="bar {"focus-bar" if r["provider"]=="Cisco" else ""}"><span>{r["provider"]}</span><div class="track"><div class="fill" style="width:{0 if v is None else max(0,min(100,v/maxv*100))}%"></div></div><span>{fmt(v,m["kind"])}</span><small>{a["scored_calls"]}/{r["selected_calls"]} calls</small></div>']
        parts+=['</section>']
    parts+=['</div><section><h2>How the composite is calculated</h2><p>'+esc(d['aggregation'])+'</p><p>Graded scores are converted from 0–5 to 0–100. Binary checks use 100 for a pass and 0 for a failure. Stop-time quality is 100 at 300 ms or less, 0 at 3,000 ms or more, and linear between those anchors.</p><table><tr><th>Category</th><th>Weight</th><th>Metrics and weights</th></tr>']
    for cat,keys in DIMENSIONS.items():parts+=[f'<tr><td>{cat}</td><td>{sum(WEIGHTS[k] for k in keys)*100:.0f}%</td><td style="white-space:normal;text-align:left">'+', '.join(next(m['label'] for m in metrics if m['key']==k)+f' {WEIGHTS[k]*100:.0f}%' for k in keys)+'</td></tr>']
    parts+=['</table><p>Only the production AI interaction is assessed. Navigation, recorded menus, human handoffs and testing-agent idle check-ins are excluded. N/A means the call did not contain the evidence required for that metric; it does not count as a pass or failure. Coverage beside each chart shows scored calls / selected calls.</p></section><section><h2>Calls and coverage</h2><p>Phone numbers identify tested production access points. Multiple numbers may reach the same assistant; they are not a count of unique AI models.</p>']
    for r in sorted(rows,key=lambda r:r['provider']!='Cisco'):
        parts+=[f'<details {"open id=\"cisco-calls\"" if r["provider"]=="Cisco" else ""}><summary>{r["provider"]} ({r["phone_number_count"]} numbers) · {r["selected_calls"]} calls</summary>']
        for n in r['numbers']:
            parts+=[f'<h3>{n["phone_number"]} · 45 calls</h3>']
            for scenario in d['scenarios']:
                cs=[c for c in n['calls'] if c['scenario']==scenario];parts+=['<p><small>'+scenario+': </small>'+', '.join(f'<a href="{c.get("share_url",c["url"])}" target="_blank" rel="noreferrer">{c["run_id"]}</a>' for c in cs)+'</p>']
        parts+=['</details>']
    parts+=['</section></main></html>'];(P/'report.html').write_text(''.join(parts))

def focus_summary(d):
    cisco=next(p for p in d['providers'] if p['provider']=='Cisco')
    cs={c['run_id']:c for n in cisco['numbers'] for c in n['calls']}
    def link(rid,text):
        return '<a href="'+html.escape(cs[rid]['share_url'],quote=True)+'" target="_blank" rel="noreferrer">'+html.escape(text)+'</a>'
    rank=1+sum(p['composite']>cisco['composite'] for p in d['providers'])
    cards=[(f"{cisco['composite']:.1f}/100",'Cisco composite quality'),(f"{cisco['metrics']['latency']['value']/1000:.2f}s",'Median qualifying turn latency'),('45','Cisco selected calls'),(f'{rank} of 17','Position in this reference comparison')]
    out='<section class="focus-panel"><h2>Cisco: Mia / Webex assistant</h2><p>+1 984 314 9680 · 15 scenarios, three repetitions each · November 24, 2026</p><div class="focus-cards">'+''.join('<div><b>'+value+'</b><small>'+label+'</small></div>' for value,label in cards)+'</div><p class="focus-note">'+html.escape(d['comparison_note'])+'</p><h3>Remaining observed concerns</h3><ul class="concerns">'
    out+='<li><b>Explanation abandoned after “uh-huh” — 2/3 acknowledgment calls.</b> '+link(4780967,'01:02 · unfinished explanation never resumes')+'; '+link(4780979,'01:04 · stalled explanation, then a presence check')+'. Both fail cue handling; the latter also fails the silence check.</li>'
    out+='<li><b>Caller’s second question cut off — 2/3 pause calls.</b> '+link(4780972,'00:27 · premature AI response')+'; '+link(4780984,'00:26 · premature AI response')+'. Both interruption scores are 2.5/5.</li>'
    out+='<li><b>First distress cue missed — 1/3 difficult-day calls.</b> '+link(4780966,'00:23–00:53 · business overview without acknowledgment or appropriate escalation')+'.</li>'
    out+='<li><b>Wrong point clarified — 1/3 clarification calls.</b> '+link(4780994,'00:51–01:05 · request about Webex topic answered with question logging')+'. Response relevance scores the call 4/5.</li>'
    out+='</ul><p class="focus-note">Speech integrity passes all 45 selected Cisco calls; unnecessary repetition is 5/5. Backchannels and coughs are excluded from Stop Time, while abandoning an explanation is penalized by cue handling. Expected Outcome remains a separate diagnostic: it still falsely passes the wrong-point clarification, but is excluded from the composite.</p><p><a href="#cisco-calls">Open all 45 Cisco calls and scenario coverage ↓</a></p></section>'
    return out

if __name__=='__main__':render(json.loads((P/'results.json').read_text()))
