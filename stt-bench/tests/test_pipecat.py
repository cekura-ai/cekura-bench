import asyncio
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stt_bench.catalog import dataset_definition, dataset_identity, model_config
from stt_bench.data import load_manifest, sha256, write_json
from stt_bench.huggingface_data import freeze_rows, verify_prepared
from stt_bench.benchmark import benchmark, smoke_passed
from stt_bench.streaming import wait_until


def source_row(sample_id='clip-1', rate=16000):
    audio = io.BytesIO()
    # Only an importer fixture, never provider performance evidence.
    sf.write(audio, .3 * np.sin(2 * np.pi * 180 * np.arange(rate) / rate), rate,
             format='WAV', subtype='PCM_16')
    return dict(sample_id=sample_id, duration_seconds=1., transcription='  Hello,  world! ',
                audio={'bytes': audio.getvalue(), 'path': None})


def definition(count=1):
    d = dataset_definition('pipecat-stt-benchmark')
    return {**d, 'expected_clips': count, 'smoke_count': min(10, count)}


def test_freezes_all_rows_and_keeps_duplicate_text(tmp_path):
    rows = [source_row(f'clip-{i}') for i in range(12)]
    d = definition(12)
    a, b = tmp_path / 'a', tmp_path / 'b'
    freeze_rows(rows, d, a)
    freeze_rows(rows, d, b)
    full = load_manifest(a / 'full/manifest.json')
    smoke = load_manifest(a / 'smoke/manifest.json')
    assert len(full['clips']) == 12 and len(smoke['clips']) == 10
    assert all(c['reference'] == '  Hello,  world! ' and c['entities'] is None for c in full['clips'])
    assert [c['sample_id'] for c in full['clips']] == [r['sample_id'] for r in rows]
    assert sha256(a / 'smoke/manifest.json') == sha256(b / 'smoke/manifest.json')
    assert dataset_identity(full)['source_revision'] == d['revision']
    assert full['license'] is None
    audio = a / 'full' / full['clips'][0]['audio']
    audio.write_bytes(audio.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Frozen audio changed'):
        load_manifest(a / 'full/manifest.json')


@pytest.mark.parametrize('kind', ['duplicate', 'reference', 'format', 'duration', 'missing', 'count', 'unsafe', 'silence'])
def test_import_errors_are_recorded_without_publishing_partial_manifest(tmp_path, kind):
    rows = [source_row()]
    d = definition()
    if kind == 'duplicate':
        rows *= 2
        d = definition(2)
    elif kind == 'reference':
        rows[0]['transcription'] = ''
    elif kind == 'format':
        rows = [source_row(rate=8000)]
    elif kind == 'duration':
        rows[0]['duration_seconds'] = 2
    elif kind == 'missing':
        rows[0]['audio']['bytes'] = b'not audio'
    elif kind == 'count':
        d = definition(2)
    elif kind == 'unsafe':
        rows[0]['sample_id'] = '../escape'
    else:
        audio = io.BytesIO()
        sf.write(audio, np.zeros(16000), 16000, format='WAV')
        rows[0]['audio']['bytes'] = audio.getvalue()
    out = tmp_path / 'data'
    with pytest.raises(ValueError, match='preparation failed'):
        freeze_rows(rows, d, out)
    assert json.loads((out / 'preparation-errors.json').read_text())['errors']
    assert not (out / 'full/manifest.json').exists()
    assert not (out / 'prepared.json').exists()


def test_verification_rejects_changed_definition_or_manifest(tmp_path, monkeypatch):
    import stt_bench.huggingface_data as module
    d = definition()
    out = tmp_path / 'prepared'
    freeze_rows([source_row()], d, out)
    monkeypatch.setattr(module, 'dataset_root', lambda _: out)
    assert verify_prepared(d) == out
    with pytest.raises(ValueError, match='definition changed'):
        verify_prepared({**d, 'seed': 43})
    p = out / 'full/manifest.json'
    p.write_text(p.read_text() + ' ')
    with pytest.raises(ValueError, match='manifest changed'):
        verify_prepared(d)


def test_unknown_selectors_fail_before_network():
    with pytest.raises(ValueError, match='Unknown dataset'):
        dataset_definition('other')
    with pytest.raises(ValueError, match='Unknown model'):
        model_config('other')


def test_smoke_requires_every_clip_valid():
    report = dict(completeness={'run_complete': True}, clips=[{'valid': True, 'accuracy_usable': True}])
    assert smoke_passed(report)
    report['clips'].append({'valid': False, 'accuracy_usable': False})
    assert not smoke_passed(report)


def test_wait_yields_coarsely_and_late_deadlines_do_not_sleep(monkeypatch):
    import stt_bench.timing as module
    clock = [0.]
    sleeps = []
    def now():
        clock[0] += .00001
        return clock[0]
    async def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(module.asyncio, 'sleep', sleep)
    monkeypatch.setattr(module.time, 'perf_counter', now)
    asyncio.run(wait_until(.02, now))
    assert len(sleeps) > 1 and all(0 < seconds <= .001 for seconds in sleeps)
    assert clock[0] >= .02
    count = len(sleeps)
    asyncio.run(wait_until(0, now))
    assert len(sleeps) == count


@pytest.mark.parametrize('passes', [True, False])
def test_workflow_gates_full_and_resume_preserves_success(tmp_path, monkeypatch, passes):
    import stt_bench.benchmark as module
    d = definition()
    prepared = tmp_path / 'data'
    freeze_rows([source_row()], d, prepared)
    config = json.loads(model_config('deepgram-nova-3').read_text())
    config_path = tmp_path / 'model.json'
    write_json(config_path, config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, 'dataset_definition', lambda _: d)
    monkeypatch.setattr(module, 'model_config', lambda _: config_path)
    monkeypatch.setattr(module, 'verify_prepared', lambda _: prepared)
    monkeypatch.setattr(module, 'verify_model', lambda _: {'verified': True})
    async def probe(out, **kwargs):
        out.mkdir(parents=True)
        write_json(out / 'pacing.json', {})
    monkeypatch.setattr(module, 'local_probe', probe)
    monkeypatch.setattr(module, 'validate_preflight', lambda _: {})
    called = []
    async def run(manifest, config, out, *args):
        called.append(out.name)
        out.mkdir()
    def score(run_dir, out):
        out.mkdir()
        report = dict(completeness={'run_complete': True},
                      clips=[dict(valid=passes, accuracy_usable=passes)],
                      results=[{'estimated_cost_usd': .1}])
        write_json(out / 'results.json', report)
        return report
    monkeypatch.setattr(module, 'run', run)
    monkeypatch.setattr(module, 'score', score)
    args = ('pipecat-stt-benchmark', 'deepgram-nova-3', 'trial')
    if passes:
        result = asyncio.run(benchmark(*args))
        assert result['estimated_total_cost_usd'] == .2
        assert called == ['smoke', 'full']
        asyncio.run(benchmark(*args, resume=True))
        assert called == ['smoke', 'full']
    else:
        with pytest.raises(ValueError, match='Smoke test failed'):
            asyncio.run(benchmark(*args))
        assert called == ['smoke']
        with pytest.raises(ValueError, match='Smoke test failed'):
            asyncio.run(benchmark(*args, resume=True))
        assert called == ['smoke']
    assert (tmp_path / 'reports/pipecat-stt-benchmark/deepgram-nova-3/trial/benchmark.json').exists()


def test_pipecat_report_labels_and_missing_annotations(tmp_path):
    from stt_bench.score import score
    root = tmp_path / 'run'
    root.mkdir()
    manifest = dict(dataset_id='pipecat-stt-benchmark', dataset='pipecat-ai/stt-benchmark-data',
                    source_revision='frozen-revision', split='train', subset='smoke', language='en-US',
                    preprocessing={'entity_provenance': 'No entity annotations supplied'},
                    clips=[dict(clip_id='clip', condition='public_anchor', reference='hello',
                                submitted_seconds=1.02, entities=None)])
    write_json(root / 'manifest.json', manifest)
    config = json.loads(model_config('deepgram-nova-3').read_text())
    write_json(root / 'run.json', dict(schema_version=2, measurement_version=3, mode='dry_run',
               config=config, manifest_sha256=sha256(root / 'manifest.json')))
    out = tmp_path / 'report'
    result = score(root, out)
    assert result['dataset_identity']['subset'] == 'smoke'
    assert result['run_identity']['model_id'] == 'deepgram-nova-3'
    assert result['results'][0]['entity_error_rate'] is None
    assert result['results'][0]['wer'] is None
    assert result['review']['verified_clips'] == 0
    for name in ('results.md', 'per-clip.md', 'entity-errors.md'):
        text = (out / name).read_text()
        assert 'pipecat-ai/stt-benchmark-data' in text and 'frozen-revision' in text
        assert 'FLEURS' not in text
    assert not any('reviewed by Codex' in note for note in result['notes'])
