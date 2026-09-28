"""A turn plan can freeze a subset of profiles without changing the default."""
import json

import pytest

from stt_bench import turn_batch


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    manifest = tmp_path/'dataset/manifest.json'
    manifest.parent.mkdir()
    manifest.write_text('{}')
    clips = [dict(clip_id=f'c{i}', submitted_seconds=2.0) for i in range(3)]
    monkeypatch.setattr(turn_batch, 'MANIFEST', str(manifest))
    monkeypatch.setattr(turn_batch, 'verify_turn_manifest', lambda path: dict(clips=clips, counts={'clips': 3}))
    monkeypatch.setattr(turn_batch, 'smoke_selection', lambda m: ['c0'])
    return tmp_path


def test_selected_models_are_the_only_planned_models(frozen):
    turn_batch.prepare(frozen/'plan-a', ['soniox-stt-rt-v5', 'inworld-stt-1'])
    plan = json.loads((frozen/'plan-a/plan.json').read_text())
    assert plan['models'] == ['soniox-stt-rt-v5', 'inworld-stt-1']
    assert set(plan['configs']) == set(plan['config_hashes']) == set(plan['models'])
    assert plan['planned_sessions'] == 6 and 'soniox-stt-rt-v5, inworld-stt-1' in plan['authorization']


def test_default_plan_keeps_all_profiles_and_unknown_models_fail(frozen):
    turn_batch.prepare(frozen/'plan-b')
    assert len(json.loads((frozen/'plan-b/plan.json').read_text())['models']) == 23
    with pytest.raises(ValueError, match='Unknown or repeated'):
        turn_batch.prepare(frozen/'plan-c', ['not-a-model'])
    with pytest.raises(ValueError, match='Unknown or repeated'):
        turn_batch.prepare(frozen/'plan-d', ['inworld-stt-1', 'inworld-stt-1'])
