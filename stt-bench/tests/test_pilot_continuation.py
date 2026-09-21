import asyncio
import importlib
import json
from pathlib import Path


def test_started_sessions_are_preserved_and_never_resent(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    module = importlib.import_module('continue_finalization_pilot')
    model = module.MODELS[0]
    (tmp_path / 'configs').mkdir()
    for variant in ('baseline', 'candidate'):
        (tmp_path / 'configs' / f'{model}-{variant}.json').write_text('{}')
    clip = {'clip_id': 'public-test'}
    monkeypatch.setattr(module, 'verify', lambda root: {'clips': [clip]})
    monkeypatch.setattr(module, 'validate_preflight', lambda path: None)
    saved = dict(model=model, variant='baseline', clip_id='public-test', valid=False,
                 exclusion_reasons=['interrupted_attempt'], capture_phase='original')
    monkeypatch.setattr(module, 'existing_row', lambda root, model, variant, *args:
                        saved if variant == 'baseline' else None)
    calls = []
    async def fake_run(manifest, config, capture, *args, **kwargs):
        calls.append((capture, kwargs))
    monkeypatch.setattr(module, 'run', fake_run)
    monkeypatch.setattr(module, 'observation', lambda *args: dict(variant='candidate',
                        clip_id='public-test', valid=True, exclusion_reasons=[]))
    rows = asyncio.run(module.worker(tmp_path, model))
    assert len(calls) == 1
    assert '/candidate/' in str(calls[0][0])
    assert calls[0][1]['max_attempts'] == 1
    assert rows[0] == saved
    assert rows[1]['capture_phase'] == 'isolated'


def test_reconciliation_does_not_change_original_files(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    module = importlib.import_module('continue_finalization_pilot')
    model = module.MODELS[0]
    raw_dir = tmp_path / 'captures' / model / 'baseline' / '00' / 'raw'
    raw_dir.mkdir(parents=True)
    raw = raw_dir / 'public-test--attempt-1.jsonl'
    raw.write_text('{"kind":"error","time_seconds":0,"error_type":"Interrupted"}\n')
    before = raw.read_bytes()
    monkeypatch.setattr(module, 'recover_attempt', lambda *args: {'valid': False})
    monkeypatch.setattr(module, 'observation', lambda *args: {'valid': False})
    row = module.existing_row(tmp_path, model, 'baseline', 0, {'clip_id': 'public-test'}, {})
    assert row['capture_phase'] == 'original' and not row['valid']
    assert raw.read_bytes() == before
    assert not (raw_dir.parent / 'outcomes.json').exists()
    assert (tmp_path / 'reconciled' / model / 'baseline' / '00' / 'raw' / raw.name).read_bytes() == before
