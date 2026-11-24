import importlib.util,json,pathlib,sys

P=pathlib.Path(__file__).resolve().parents[1]/'results/production-voice-quality/2026-11-24'
spec=importlib.util.spec_from_file_location('cisco_quality_validator',P/'validate.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)

def data():return json.loads((P/'results.json').read_text())
def cisco():return next(p for p in data()['providers'] if p['provider']=='Cisco')

def test_cisco_snapshot_reconciles_with_unchanged_reference():
    validator.validate()

def test_replacements_preserve_full_scenario_parity():
    calls=cisco()['numbers'][0]['calls'];ids={c['run_id'] for c in calls}
    assert 4782023 in ids and 4782488 in ids
    assert 4780955 not in ids and 4780959 not in ids
    assert data()['cisco_execution_lineage']['executions']==47

def test_backchannel_exclusion_does_not_hide_abandoned_explanations():
    calls={c['run_id']:c for c in cisco()['numbers'][0]['calls']}
    for rid in [4780967,4780979]:
        assert calls[rid]['metrics']['stop'] is None
        assert calls[rid]['events_ms']['stop']==[]
        assert calls[rid]['metrics']['cues']==0
        assert calls[rid]['metrics']['gibberish']==100
    assert calls[4782023]['metrics']['cues']==100
    assert calls[4780972]['metrics']['interruption']==2.5
    assert calls[4780984]['metrics']['interruption']==2.5

def test_latency_uses_published_scope_and_focus_is_present():
    assert cisco()['metrics']['latency']['events']==77
    assert cisco()['metrics']['latency']['value']==3760
    page=(P/'report.html').read_text()
    assert page.count('class="bar focus-bar"')==13
    assert 'id="cisco-calls"' in page and 'Cisco: Mia / Webex assistant' in page
    assert 'METRIC_FIX' not in page and 'X-Amz' not in page
    assert all('/share/result/' in c['share_url'] for c in cisco()['numbers'][0]['calls'])

def test_latency_pools_turns_instead_of_averaging_call_medians():
    sys.path.insert(0,str(P))
    spec=importlib.util.spec_from_file_location('cisco_quality_builder',P/'build_results.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    calls=[
        {'scenario':'A','metrics':{'latency':100},'events_ms':{'latency':[100]}},
        {'scenario':'B','metrics':{'latency':1000},'events_ms':{'latency':[900,1000,1100]}},
    ]
    result=builder.aggregate(calls,[{'key':'latency','kind':'ms'}])['latency']
    # Four turn events have median 950; the two call medians would yield 550.
    assert result['value']==950
    assert result['events']==4
    assert result['normalized'] is None
