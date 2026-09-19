import importlib.util
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from stt_bench.diagnostics import pacing_identity


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_region_and_instance_are_part_of_resume_identity(monkeypatch):
    for field in ('COMPUTE_PROVIDER', 'COMPUTE_REGION', 'COMPUTE_INSTANCE'):
        monkeypatch.delenv('STT_BENCH_' + field, raising=False)
    original = pacing_identity()
    monkeypatch.setenv('STT_BENCH_COMPUTE_PROVIDER', 'vercel-sandbox')
    with pytest.raises(ValueError, match='requires provider, region and instance'):
        pacing_identity()
    monkeypatch.setenv('STT_BENCH_COMPUTE_REGION', 'iad1')
    monkeypatch.setenv('STT_BENCH_COMPUTE_INSTANCE', 'test-vm')
    remote = pacing_identity()
    assert original != remote
    monkeypatch.setenv('STT_BENCH_COMPUTE_REGION', 'sfo1')
    assert remote != pacing_identity()


@pytest.mark.parametrize('live,fail_at,expected', [(False, None, 2), (True, None, 3),
                                                (True, 0, 1), (True, 1, 2), (True, 2, 3)])
def test_remote_pipeline_gates_provider_and_exports_failure(tmp_path, monkeypatch, live, fail_at, expected):
    module = script('remote_job')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(module, 'dataset_definition', lambda _: {})
    monkeypatch.setattr(module, 'model_config', lambda _: None)
    monkeypatch.setattr(module, 'verify_prepared', lambda _: None)
    monkeypatch.setenv('DEEPGRAM_API_KEY', 'fixture-secret-not-real')
    Path('uv.lock').write_text('fixture')
    Path('.env').write_text('PRIVATE=fixture-secret-not-real')
    Path('reports/unrelated').mkdir(parents=True)
    Path('reports/unrelated/private.txt').write_text('do not export')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['STT_BENCH_COMPUTE_REGION'] == 'iad1'
        return SimpleNamespace(returncode=1 if len(calls) - 1 == fail_at else 0)
    monkeypatch.setattr(module.subprocess, 'run', run)
    args = SimpleNamespace(dataset='pipecat-stt-benchmark', model='deepgram-nova-3',
                           provider='vercel-sandbox', region='iad1', instance='vm-test',
                           run_id='test-job', live=live)
    assert module.run_job(args) == (0 if fail_at is None else 1)
    assert len(calls) == expected
    out = Path('reports/remote-jobs/test-job')
    state = json.loads((out / 'job.json').read_text())
    assert state['status'] == ('failed' if fail_at is not None else
                              'benchmark_complete' if live else 'qualified_no_transcription_calls')
    with tarfile.open(out / 'artifacts.tar.gz') as tar:
        names = tar.getnames()
        assert str(out / 'job.json') in names
        assert not any('unrelated' in name or '.env' in name for name in names)
        assert all(b'fixture-secret' not in tar.extractfile(name).read() for name in names)
    with pytest.raises(FileExistsError):
        module.run_job(args)
    assert len(calls) == expected


def test_bundle_selection_excludes_credentials_and_previous_results(tmp_path, monkeypatch):
    module = script('bundle_remote')
    monkeypatch.chdir(tmp_path)
    root = Path('datasets/frozen')
    root.mkdir(parents=True)
    monkeypatch.setattr(module, 'dataset_definition', lambda _: {})
    monkeypatch.setattr(module, 'verify_prepared', lambda _: root)
    monkeypatch.setattr(module, 'regression_files', lambda: [])
    for name in ('prepared.json', '.env'):
        (root / name).write_text('{}')
    for name in ('pyproject.toml', 'uv.lock', 'README.md', '.env'):
        Path(name).write_text('fixture')
    for subset in ('full', 'smoke'):
        p = root / subset
        p.mkdir()
        (p / 'manifest.json').write_text(json.dumps({'clips': [{'audio': 'clip.wav'}]}))
        (p / 'clip.wav').write_bytes(b'fixture audio')
    files = module.bundle_files()
    assert len(files) == 8
    assert not any(p.name == '.env' for p in files)
    # Reject links even when a referenced file is inside the workspace.
    (root / 'full/clip.wav').unlink()
    (root / 'full/clip.wav').symlink_to((root / 'smoke/clip.wav').resolve())
    with pytest.raises(ValueError, match='links'):
        module.bundle_files()


def test_bundle_refuses_disk_pressure_and_existing_output(tmp_path, monkeypatch):
    module = script('bundle_remote')
    monkeypatch.chdir(tmp_path)
    Path('source.py').write_text('print(1)')
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match='Insufficient free space'):
        module.build_bundle(Path('bundle.tar.gz'), [Path('source.py')])
    assert not Path('bundle.tar.gz').exists()
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1024**3))
    module.build_bundle(Path('bundle.tar.gz'), [Path('source.py')])
    with tarfile.open('bundle.tar.gz') as tar:
        assert tar.getnames() == ['source.py']
    with pytest.raises(ValueError, match='already exists'):
        module.build_bundle(Path('bundle.tar.gz'), [Path('source.py')])
