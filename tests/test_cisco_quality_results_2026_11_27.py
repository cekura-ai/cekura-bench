import importlib.util, json, pathlib

P = pathlib.Path(__file__).resolve().parents[1] / 'results/production-voice-quality/2026-11-27'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, P / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data():
    return json.loads((P / 'results.json').read_text())


def test_snapshot_reconciles_and_keeps_reference_providers():
    load('cisco_quality_validator_2026_11_27', 'validate.py').validate()


def test_committed_results_match_a_fresh_build():
    assert load('cisco_quality_builder_2026_11_27', 'build_results.py').build() == data()


def test_voice_tone_is_marked_provisional_until_nisqa_scores_replace_it():
    d = data()
    assert d['cisco_audio_source'] == 'platform_voice_tone_provisional'
    assert d['provider_snapshot_dates']['Cisco'] == '2026-11-27'
    assert d['cisco_execution_lineage']['replacements'] == []


def test_cisco_timing_matches_the_rerun():
    cisco = next(p for p in data()['providers'] if p['provider'] == 'Cisco')
    assert cisco['metrics']['latency']['value'] == 1920 and cisco['metrics']['latency']['events'] == 82
    assert cisco['metrics']['stop']['events'] == 6
    assert round(cisco['metrics']['stop']['value'], 2) == 793.33
