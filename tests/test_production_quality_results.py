import importlib.util,pathlib,json
P=pathlib.Path(__file__).resolve().parents[1]/'results/production-voice-quality/2026-11-12'
spec=importlib.util.spec_from_file_location('production_quality_validator',P/'validate.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def call(case,score):return {'scenario':case,'metrics':{'quality':score}}
def test_equal_number_and_scenario_weighting():
    # Repeated calls and extra scored scenarios cannot overweight a number.
    a={'calls':[call('A',5),call('A',5),call('B',1)]}
    b={'calls':[call('A',1)]}
    assert v.balanced(a['calls'],'quality')==3
    assert v.equal_numbers([a,b],'quality')==2
def test_na_is_not_a_pass_or_zero():
    a={'calls':[call('A',None),call('B',4)]}
    b={'calls':[call('A',None)]}
    assert v.equal_numbers([a,b],'quality')==4
    assert v.equal_numbers([b],'quality') is None
def test_published_snapshot_reconciles():v.validate()
def test_regal_stop_ends_before_resumed_answer():
    d=json.loads((P/'results.json').read_text())
    regal=next(p for p in d['providers'] if p['provider']=='Regal')
    call=next(c for n in regal['numbers'] for c in n['calls'] if c['run_id']==4508159)
    # The pause at 27.56s separates the interrupted utterance from the resumed answer.
    # A 1.2s segment merge previously included that answer in the stop duration.
    assert call['events_ms']['stop']==[4750.0]
    assert abs(regal['metrics']['stop']['value']-4663.333333333333)<1e-9
