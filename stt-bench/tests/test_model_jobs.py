import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from stt_bench.data import sha256, write_json
from stt_bench.model_jobs import validate_smoke, rollup, compare_summaries
from stt_bench.preparation import check
from stt_bench.providers import reduce_events, transcript_at
from stt_bench import deepgram


def test_smoke_is_bound_to_model_code_data_config_and_report(tmp_path):
    p = tmp_path / 'smoke.json'
    report = {'manifest_sha256': 'smoke-hash', 'completeness': {'run_complete': True},
              'clips': [{'valid': True, 'accuracy_usable': True}]}
    write_json(p, report)
    identity = {'model': 'a', 'config_sha256': 'config', 'sources': {'run.py': 'hash'},
                'smoke_manifest_sha256': 'smoke-hash', 'full_manifest_sha256': 'full-hash'}
    receipt = {'passed': True, 'identity': identity, 'report': str(p), 'report_sha256': sha256(p)}
    validate_smoke(receipt, identity)
    for field in identity:
        changed = {**identity, field: 'other'}
        with pytest.raises(ValueError, match='exact'):
            validate_smoke(receipt, changed)
    p.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        validate_smoke(receipt, identity)


def test_rollup_rejects_duplicate_missing_or_reordered_clips(tmp_path):
    p = tmp_path / 'batch.json'
    write_json(p, {'clips': [{'clip_id': 'b'}]})
    state = {'completed_batches': [{'report': str(p), 'sha256': sha256(p)}]}
    with pytest.raises(ValueError, match='coverage'):
        rollup(state, tmp_path, {'clips': [{'clip_id': 'a'}, {'clip_id': 'b'}]})


def test_readiness_never_claims_live_access_or_accepts_stale_validation(tmp_path, monkeypatch):
    import stt_bench.preparation as module
    for subset, count in [('smoke', 10), ('full', 1000)]:
        p = tmp_path / subset / 'manifest.json'; p.parent.mkdir()
        write_json(p, {'clips': [{'submitted_seconds': 1}]*count})
    monkeypatch.setattr(module, 'dataset_definition', lambda _: {})
    monkeypatch.setattr(module, 'verify_prepared', lambda _: tmp_path)
    monkeypatch.setattr(module, 'VALIDATION', tmp_path / 'validation.json')
    monkeypatch.setattr(module, 'source_identity', lambda: {'source': 'now'})
    monkeypatch.setattr(module, 'models', lambda: [{'model_id': 'test', 'provider': 'test',
        'missing_prerequisites': [], 'live_verification': 'pending'}])
    write_json(module.VALIDATION, {'passed': True, 'source_identity': {'source': 'before'}})
    report = check()
    assert report['local_validation'] == 'missing_or_stale'
    assert not report['transcription_calls_made'] and not report['launch_authorized']


def test_v3_nova_replay_matches_existing_protocol_exactly():
    # Freeze a v3-style raw transcript fixture, including duplicate and cleanup finals.
    config = json.loads(Path('config/models/deepgram-nova-3.json').read_text())
    def result(text, at, start, ack=False):
        return {'kind': 'provider_message', 'time_seconds': at, 'message': {
            'type': 'Results', 'is_final': True, 'from_finalize': ack, 'start': start, 'duration': .1,
            'channel': {'alternatives': [{'transcript': text}]},
            'metadata': {'model_uuid': config['expected_model_uuid'], 'model_info': {'version': config['version']}}}}
    first = result('hello', .5, 0)
    events = [first, first, {'kind': 'speech_end', 'time_seconds': 1},
              {'kind': 'finalize_requested', 'time_seconds': 1.01}, result('world', 1.2, .1, True),
              {'kind': 'audio_complete', 'time_seconds': 2},
              {'kind': 'close_stream_requested', 'time_seconds': 2.01}, result('again', 2.1, .2),
              {'kind': 'provider_message', 'time_seconds': 2.2, 'message': {'type': 'Metadata'}}]
    assert reduce_events(events, config) == deepgram.reduce_events(events, config)
    for cutoff in (1, 1.25, 1.5, 2):
        assert transcript_at(events, cutoff, config) == deepgram.transcript_at(events, cutoff)
    from stt_bench.report import include_complete_text
    scored = include_complete_text(reduce_events(events, config), events, config)
    assert scored['transcript'] == 'hello world again'
    assert scored['transcript_at_finalize'] == 'hello world'


def test_new_remote_entry_requires_explicit_live_flag(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    from model_batches import execute
    with pytest.raises(ValueError, match='disabled'):
        asyncio.run(execute(SimpleNamespace(live=False)))


def test_model_sessions_require_smoke_then_continue_without_repeating_batches(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    import model_batches as job
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'data'
    for subset, ids in [('smoke', ['s']), ('full', ['a', 'b'])]:
        (data / subset).mkdir(parents=True)
        write_json(data / subset / 'manifest.json', {'clips': [{'clip_id': x} for x in ids]})
    config = tmp_path / 'model.json'; write_json(config, {'provider': 'fixture'})
    monkeypatch.setattr(job, 'model_config', lambda _: config)
    monkeypatch.setattr(job, 'require_credential', lambda _: 'fixture')
    monkeypatch.setattr(job, 'dataset_definition', lambda _: {})
    monkeypatch.setattr(job, 'verify_prepared', lambda _: data)
    monkeypatch.setattr(job, 'job_identity', lambda *a: {'model': 'fixture', 'sources': 'fixed'})
    monkeypatch.setattr(job, 'verify_model', lambda _: {})
    monkeypatch.setattr(job, 'validate_smoke', lambda receipt, identity: None if receipt['passed'] and receipt['identity'] == identity else pytest.fail('invalid smoke'))
    monkeypatch.setattr(job.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    clock = [0]
    monkeypatch.setattr(job.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(job, 'allowance', lambda _: 100)
    monkeypatch.setattr(job, 'partition', lambda clips: [[c] for c in clips])
    monkeypatch.setattr(job, 'validate_preflight', lambda _: None)
    qualifications, captured = [], []
    async def probe(path, **kw): qualifications.append(str(path))
    async def capture(manifest, config, out, *a):
        out.mkdir(parents=True)
        captured.extend(c['clip_id'] for c in json.loads(manifest.read_text())['clips'])
        clock[0] += 150
    def scored(run, out):
        out.mkdir(parents=True)
        result = {'completeness': {'run_complete': True}, 'clips': [{'accuracy_usable': True}]}
        write_json(out / 'results.json', result)
        return result
    monkeypatch.setattr(job, 'local_probe', probe)
    monkeypatch.setattr(job, 'run', capture)
    monkeypatch.setattr(job, 'score', scored)
    monkeypatch.setattr(job, 'smoke_passed', lambda _: True)
    monkeypatch.setattr(job, 'rollup', lambda *a: None)
    args = SimpleNamespace(live=True, model='fixture', dataset='dataset', run_id='run',
                           session_id='session1', region='iad1', phase='full', budget_seconds=300)
    with pytest.raises(ValueError, match='smoke receipt'):
        asyncio.run(job.execute(args))
    args.phase = 'smoke'; asyncio.run(job.execute(args))
    args.phase = 'full'; args.session_id = 'session2'; asyncio.run(job.execute(args))
    args.session_id = 'session3'; asyncio.run(job.execute(args))
    asyncio.run(job.execute(args))  # Completed jobs return without more dispatch.
    assert captured == ['s', 'a', 'b']
    assert len(qualifications) == 3
    state = json.loads(Path('reports/dataset/fixture/run/batch-state.json').read_text())
    assert state['status'] == 'complete'
    assert [b['session_id'] for b in state['completed_batches']] == ['session2', 'session3']


def test_v4_report_scores_24k_duration_deadlines_and_unknown_price(tmp_path):
    from stt_bench.report import build_report
    from stt_bench.catalog import model_config
    c = json.loads(model_config('openai-gpt-4o-transcribe').read_text())
    root = tmp_path / 'run'; (root / 'raw').mkdir(parents=True)
    manifest = {'clips': [{'clip_id': 'a', 'reference': 'hello', 'condition': 'public_anchor',
                          'submitted_seconds': 1.2}], 'schema_version': 2}
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'run.json', {'schema_version': 2, 'measurement_version': 4, 'mode': 'live',
        'manifest_sha256': sha256(root / 'manifest.json'), 'config': c, 'source_hashes': {}})
    events = [{'kind': 'audio_sent', 'index': i, 'time_seconds': i*.02, 'ideal_seconds': i*.02,
               'bytes': 960, 'sample_rate': 24000, 'phase': 'speech' if i < 10 else 'silence'} for i in range(60)]
    def e(kind, at, **kw): return {'kind': kind, 'time_seconds': at, **kw}
    events += [e('clip_start', 0), e('model_accepted', 0, model=c['model']), e('speech_end', .2),
               e('finalize_requested', .2),
               e('provider_message', .21, message={'type': 'input_audio_buffer.committed', 'item_id': 'a'}),
               e('provider_message', .3, message={'type': 'conversation.item.input_audio_transcription.completed', 'item_id': 'a', 'transcript': 'hello'}),
               e('audio_complete', 1.2), e('close_stream_requested', 1.2), e('provider_terminal', 1.2), e('clip_end', 1.21)]
    raw = root / 'raw/a--attempt-1.jsonl'
    raw.write_text(''.join(json.dumps(x)+'\n' for x in sorted(events, key=lambda e:e['time_seconds'])))
    report = build_report(root, tmp_path / 'report')
    assert report['measurement_version'] == 4
    assert report['results'][0]['wer'] == 0
    assert report['results'][0]['initial_sent_audio_seconds'] == 1.2
    assert report['results'][0]['estimated_cost_usd'] is None
    assert report['clips'][0]['deadlines'][0]['status'] == 'no_text_yet'
    assert report['clips'][0]['deadlines'][1]['text'] == 'hello'
