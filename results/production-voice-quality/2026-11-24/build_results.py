"""Append Cisco's selected calls to the unchanged November 12 reference panel."""
import collections,copy,hashlib,json,pathlib,statistics
from render import render
P=pathlib.Path(__file__).resolve().parent
mean=lambda xs:statistics.mean(xs) if xs else None

def aggregate(calls,metrics):
    result={}
    for metric in metrics:
        key=metric['key'];kind=metric['kind']
        groups=collections.defaultdict(list);quality=collections.defaultdict(list);events=[]
        for call in calls:
            value=call['metrics'][key]
            if value is None:continue
            groups[call['scenario']].append(value)
            normalized=mean([max(0,min(100,100*(3000-v)/2700)) for v in call['events_ms']['stop']]) if key=='stop' else value*(20 if kind=='score' else 1)
            quality[call['scenario']].append(normalized)
            if kind=='ms':events.extend(call['events_ms'][key])
        value=(statistics.median(events) if events else None) if key=='latency' else mean(events) if kind=='ms' else mean([mean(v) for v in groups.values()])
        result[key]={'value':value,'normalized':None if key=='latency' else mean([mean(v) for v in quality.values()]),'scored_calls':sum(c['metrics'][key] is not None for c in calls),'eligible_scenarios':len(groups),'events':len(events)}
    return result

def main():
    reference=json.loads((P.parent/'2026-11-12/results.json').read_text())
    calls=json.loads((P/'cisco-calls.json').read_text())
    number_metrics=aggregate(calls,reference['metrics'])
    provider_metrics={k:{'value':v['value'],'normalized':v['normalized'],'scored_calls':v['scored_calls'],'eligible_numbers':int(v['value'] is not None),'events':v['events']} for k,v in number_metrics.items()}
    composite=sum(reference['weights'][k]*provider_metrics[k]['normalized'] for k in reference['weights'] if reference['weights'][k]>0)
    cisco={'provider':'Cisco','phone_number_count':1,'selected_calls':45,'composite':composite,'metrics':provider_metrics,'numbers':[{'phone_number':'+19843149680','calls':calls,'metrics':number_metrics}]}
    data=copy.deepcopy(reference)
    data.update(snapshot_date='2026-11-24',provider_count=17,phone_number_count=32,selected_call_count=1440,focus_provider='Cisco',reference_snapshot='2026-11-12')
    data['provider_snapshot_dates']={p['provider']:'2026-11-12' for p in reference['providers']}
    data['provider_snapshot_dates']['Cisco']='2026-11-24'
    data['comparison_note']='Cisco was tested on November 24; the other 16 providers retain the November 12 reference scores. Cisco uses repaired judges with the same metric meanings, weights and scoped export. Comparisons are directional, not a simultaneous retest of every provider.'
    data['cisco_execution_lineage']={'executions':47,'selected_calls':45,'replacements':[{'original_run_id':4780955,'selected_run_id':4782023,'reason':'Unscripted caller rescue after the acknowledgment contaminated the original test.'},{'original_run_id':4780959,'selected_run_id':4782488,'reason':'Original ended without an assessable reply; replacement exercises the intended direction-change interruption.'}]}
    data['providers'].append(cisco)
    data['providers'].sort(key=lambda p:-p['composite'])
    render(data)
    data['report_sha256']=hashlib.sha256((P/'report.html').read_bytes()).hexdigest()
    (P/'results.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(f'Cisco {composite:.2f}/100; latency {provider_metrics["latency"]["value"]:.0f}ms; 17 providers, 32 numbers, 1,440 selected calls.')

if __name__=='__main__':main()
