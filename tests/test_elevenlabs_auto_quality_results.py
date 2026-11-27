import importlib.util
import json
import pathlib

P = pathlib.Path(__file__).resolve().parents[1] / 'results/production-voice-quality/2026-11-27-elevenlabs'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, P / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_calls_and_provider_aggregation_reconcile():
    load('elevenlabs_quality_validator', 'validate.py').validate()


def test_snapshot_rebuild_is_reproducible():
    assert load('elevenlabs_quality_builder', 'build_results.py').build() == json.loads((P / 'results.json').read_text())


def test_caller_replacements_and_repetition_scores_are_preserved():
    numbers = json.loads((P / 'elevenlabs-calls.json').read_text())
    calls = {c['run_id']: c for n in numbers for c in n['calls']}
    for rid in (4790076, 4790077, 4790122, 4790212, 4790081):
        assert calls[rid]['metrics']['repetition'] == 3.33
    assert calls[4790080]['metrics']['repetition'] == 5
    assert calls[4790080]['metrics']['cues'] is None


def test_audio_is_complete_and_private_recordings_are_not_published():
    proof = json.loads((P / 'elevenlabs-nisqa-validation.json').read_text())
    assert len(proof['scores']) == 135
    assert len({s['run_id'] for s in proof['scores']}) == 135
    assert min(s['unique_source_seconds'] for s in proof['scores']) >= 5
    text = json.dumps(proof)
    assert 'voice_recording_url' not in text and 'X-Amz-' not in text and 'data:audio' not in text
