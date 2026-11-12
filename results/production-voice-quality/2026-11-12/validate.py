"""Recompute the published metric averages and composite from selected run scores."""
import collections, hashlib, json, math, pathlib, statistics
P=pathlib.Path(__file__).resolve().parent
mean=lambda xs:statistics.mean(xs) if xs else None
def balanced(calls,key):
    groups=collections.defaultdict(list)
    for c in calls:
        v=c['metrics'][key]
        if v is not None:groups[c['scenario']].append(v)
    return mean([mean(v) for v in groups.values()])
def equal_numbers(numbers,key):
    return mean([v for n in numbers if (v:=balanced(n['calls'],key)) is not None])
def close(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,abs_tol=1e-9)),(a,b)
def validate(path=P/'results.json'):
    d=json.loads(path.read_text());ids=[];counts=collections.Counter();calls=0
    close(sum(d['weights'].values()),1)
    assert len(d['providers'])==16 and d['phone_number_count']==46 and d['selected_call_count']==2070
    assert hashlib.sha256((path.parent/'report.html').read_bytes()).hexdigest()==d['report_sha256']
    for p in d['providers']:
        assert p['provider']!='Cognigy'
        assert len(p['numbers'])==p['phone_number_count']
        for n in p['numbers']:
            cs=n['calls'];assert len(cs)==45
            assert set(collections.Counter(c['scenario'] for c in cs))==set(d['scenarios'])
            assert set(collections.Counter(c['scenario'] for c in cs).values())=={3}
            ids.extend(c['run_id'] for c in cs);calls+=len(cs);counts[p['provider']]+=1
            for c in cs:
                assert c['run_id']>1_000_000 and c['result_id']>1_000_000
                assert c['url']==f"https://dashboard.cekura.ai/16745/results/{c['result_id']}?call_id={c['run_id']}"
                assert c['status']=='completed'
                for m in d['metrics']:
                    v=c['metrics'][m['key']]
                    assert v is None or (not isinstance(v,bool) and math.isfinite(v) and v>=0)
                    if m['kind']=='score':assert v is None or v<=5
                    if m['kind']=='pass':assert v in (None,0,100)
        for m in d['metrics']:
            key=m['key']; a=p['metrics'][key]
            value=equal_numbers(p['numbers'],key)
            if m['kind']=='ms': value=mean([v for n in p['numbers'] for c in n['calls'] for v in c['events_ms'][key]])
            close(value,a['value'])
            norm=None if key=='latency' else value*(20 if m['kind']=='score' else 1)
            if key=='stop':
                number_values=[]
                for n in p['numbers']:
                    groups=collections.defaultdict(list)
                    for c in n['calls']:
                        ev=c['events_ms']['stop']
                        if ev:groups[c['scenario']].append(mean([max(0,min(100,100*(3000-v)/2700)) for v in ev]))
                    v=mean([mean(x) for x in groups.values()])
                    if v is not None:number_values.append(v)
                norm=mean(number_values)
            close(norm,a['normalized'])
            assert a['scored_calls']==sum(c['metrics'][key] is not None for n in p['numbers'] for c in n['calls'])
        score=sum(d['weights'][k]*p['metrics'][k]['normalized'] for k in d['weights'] if d['weights'][k]>0)
        close(score,p['composite'])
    assert calls==2070 and len(set(ids))==2070 and sum(counts.values())==46
    print(f'Validated {len(d["providers"])} providers, {sum(counts.values())} numbers, {calls} selected calls, metric averages and composites.')
if __name__=='__main__':validate()
