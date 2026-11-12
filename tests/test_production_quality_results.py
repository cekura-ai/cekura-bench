import importlib.util,pathlib
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
