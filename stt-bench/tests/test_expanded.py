import asyncio
import copy
import json
from pathlib import Path

import pytest

from stt_bench.data import load_manifest, write_json
from stt_bench.expanded import boundary_review, prepare_expanded
from stt_bench.deepgram import reduce_events
from stt_bench.run import run, select_attempt
from stt_bench.report import include_complete_text
from stt_bench.score import entity_errors, score
from stt_bench.streaming import read_events

CONFIG = json.loads(Path('config/deepgram.json').read_text())


@pytest.mark.parametrize('kind,reference,correct,incorrect', [
    ('number', '29¾', '29¾', '29½'),
    ('date', 'January 21', 'January 21', 'January 22'),
    ('phone', '+1 202-555-0100', '+1 202-555-0100', '+1 202-555-0101'),
    ('email', 'Sam@example.com', 'Sam@example.com', 'sam@example.com'),
    ('amount', '$20', '$20', 'twenty dollars'),
    ('spelled_sequence', 'A B C', 'A B C', 'A X C'),
])
def test_strict_entity_categories(kind, reference, correct, incorrect):
    label = [{'type': kind, 'text': reference, 'start': 0, 'end': len(reference)}]
    assert entity_errors(reference, correct, label)['errors'] == 0
    assert entity_errors(reference, incorrect, label)['errors'] == 1
    assert entity_errors(reference, '', label)['errors'] == 1


@pytest.mark.parametrize('kind,ref,hyp', [
    ('number', '20', '120'), ('number', '20', '200'), ('amount', '$20', '$120'),
    ('spelled_sequence', 'A B C', 'X A B C'), ('spelled_sequence', 'A B C', 'A B C X'),
    ('email', 'x@example.com', 'xx@example.com'),
])
def test_entity_insertions_at_internal_and_external_edges(kind, ref, hyp):
    labels = [{'type': kind, 'text': ref, 'start': 0, 'end': len(ref)}]
    assert entity_errors(ref, hyp, labels)['errors'] == 1


def test_entity_whitespace_only_and_case_and_numeric_unit_spans():
    ref = 'Pay $20 now'
    label = [{'type': 'amount', 'text': '$20', 'start': 4, 'end': 7}]
    assert entity_errors(ref, 'Pay  $ 20   now', label)['errors'] == 0
    assert entity_errors(ref, 'Pay $20 later', label)['errors'] == 0
    assert entity_errors(ref, 'Pay $20, now', label)['errors'] == 0
    assert entity_errors(ref, 'Pay $20.00 now', label)['errors'] == 1
    assert entity_errors('35mm', '36mm', [{'type': 'number', 'text': '35', 'start': 0, 'end': 2}])['errors'] == 1
    with pytest.raises(ValueError):
        entity_errors(ref, ref, label + label)


def response(text='hello', at=.12, final=True, ack=True):
    return {'kind': 'provider_message', 'time_seconds': at, 'message': {
        'type': 'Results', 'is_final': final, 'from_finalize': ack, 'start': 0, 'duration': .02,
        'channel': {'alternatives': [{'transcript': text}]}, 'channel_index': [0, 1],
        'metadata': {'model_uuid': CONFIG['expected_model_uuid'], 'model_info': {'version': CONFIG['version']}}}}


def test_partials_after_final_or_cleanup_are_not_measured():
    base = [{'kind': 'speech_end', 'time_seconds': .02}, {'kind': 'finalize_requested', 'time_seconds': .021}]
    for end in [response(), {'kind': 'close_stream_requested', 'time_seconds': .1}]:
        events = base + [end, response('late partial', .2, False, False)]
        assert reduce_events(events, CONFIG)['first_partial_after_t0'] is None


def test_cleanup_final_words_count_for_accuracy_but_not_finalize_latency():
    events = [{'kind': 'speech_end', 'time_seconds': .02},
              {'kind': 'finalize_requested', 'time_seconds': .021}, response('hello'),
              {'kind': 'close_stream_requested', 'time_seconds': 1.02},
              response('world', at=1.2, ack=False)]
    events[-1]['message']['start'] = 1
    original = reduce_events(events, CONFIG)
    scored = include_complete_text(original, events + [events[-1]])
    assert scored['transcript_at_finalize'] == 'hello'
    assert scored['transcript'] == 'hello world'
    assert scored['finalize_latency_ms'] == original['finalize_latency_ms'] == pytest.approx(100)
    assert scored['final_transcript_received_seconds'] == 1.2
    assert len(scored['additional_final_segments_after_ack']) == 1


def test_interrupted_log_only_allows_truncation_of_last_record(tmp_path):
    p = tmp_path / 'raw.jsonl'
    p.write_text('{"kind":"clip_start"}\n{"kin')
    assert len(read_events(p, allow_truncated_final=True)) == 1
    with pytest.raises(ValueError):
        read_events(p)
    p.write_text('{"kin\n{"kind":"clip_start"}\n')
    with pytest.raises(ValueError):
        read_events(p, allow_truncated_final=True)


def test_first_valid_attempt_never_fastest_or_best_text():
    first = {'attempt': 1, 'valid': True, 'finalize_latency_ms': 500, 'transcript': 'wrong'}
    second = {'attempt': 2, 'valid': True, 'finalize_latency_ms': 100, 'transcript': 'correct'}
    assert select_attempt([second, first]) == first
    first['valid'] = False
    assert select_attempt([second, first]) == second


def small_manifest(tmp_path):
    import shutil
    src = Path('datasets/fleurs-en-us-smoke-v1')
    target = tmp_path / 'data'
    shutil.copytree(src, target)
    manifest = json.loads((target / 'manifest.json').read_text())
    manifest['clips'] = manifest['clips'][:1]
    write_json(target / 'manifest.json', manifest)
    return target / 'manifest.json'


def fake_success(log):
    for i in range(51):
        log.emit('audio_sent', at=(i + 1) * .02, ideal_seconds=(i + 1) * .02, index=i, bytes=640,
                 phase='speech' if i == 0 else 'silence')
    log.emit('speech_end', at=.02)
    log.emit('finalize_requested', at=.021)
    message = response()['message']
    log.emit('provider_message', at=.12, message=message)
    log.emit('audio_complete', at=1.02)
    log.emit('close_stream_requested', at=1.03)
    log.emit('provider_message', at=1.1, message={'type': 'Metadata'})


def test_bounded_retry_resume_and_offline_attempt_costs(tmp_path, monkeypatch):
    monkeypatch.setattr('stt_bench.run.validate_preflight', lambda path: {'test_fixture': True})
    manifest = small_manifest(tmp_path)
    out = tmp_path / 'run'
    calls = []
    async def fake(pcm, speech_frames, config, key, log):
        calls.append(1)
        if len(calls) == 1:
            # Send evidence must contribute to retry-inclusive costs even when request fails.
            log.emit('audio_sent', at=.02, ideal_seconds=.02, index=0, bytes=640, phase='speech')
            raise RuntimeError('fixture failure')
        fake_success(log)
    monkeypatch.setattr('stt_bench.run.transcribe', fake)
    monkeypatch.setenv('DEEPGRAM_API_KEY', 'unit-test-key')
    outcomes = asyncio.run(run(manifest, Path('config/deepgram.json'), out, False))
    assert len(calls) == 2
    assert outcomes[0]['attempt'] == 2
    assert len(outcomes[0]['attempts']) == 2
    asyncio.run(run(manifest, Path('config/deepgram.json'), out, False, resume=True))
    assert len(calls) == 2
    report = score(out, tmp_path / 'report')
    g = report['results'][0]
    assert g['retry_count'] == 1 and g['scored_clips'] == 1
    assert g['initial_sent_audio_seconds'] == pytest.approx(.02)
    assert g['retry_sent_audio_seconds'] == pytest.approx(1.02)
    assert report == score(out, tmp_path / 'report2')
    config = copy.deepcopy(CONFIG)
    config['pricing']['usd_per_minute'] *= 2
    write_json(tmp_path / 'changed.json', config)
    with pytest.raises(ValueError, match='matching'):
        asyncio.run(run(manifest, tmp_path / 'changed.json', out, False, resume=True))


def test_cancelled_attempt_is_recovered_without_repeating_successes(tmp_path, monkeypatch):
    monkeypatch.setattr('stt_bench.run.validate_preflight', lambda path: {'test_fixture': True})
    manifest = small_manifest(tmp_path)
    out = tmp_path / 'run'
    calls = []
    async def cancelled(pcm, speech_frames, config, key, log):
        calls.append(1)
        raise asyncio.CancelledError()
    monkeypatch.setattr('stt_bench.run.transcribe', cancelled)
    monkeypatch.setenv('DEEPGRAM_API_KEY', 'unit-test-key')
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run(manifest, Path('config/deepgram.json'), out, False))
    async def successful(pcm, speech_frames, config, key, log):
        calls.append(1)
        fake_success(log)
    monkeypatch.setattr('stt_bench.run.transcribe', successful)
    outcomes = asyncio.run(run(manifest, Path('config/deepgram.json'), out, False, resume=True))
    assert len(calls) == 2 and outcomes[0]['attempt'] == 2
    assert 'interrupted_attempt' in outcomes[0]['attempts'][0]['exclusion_reasons']


def test_two_invalid_attempts_are_not_retried_forever(tmp_path, monkeypatch):
    monkeypatch.setattr('stt_bench.run.validate_preflight', lambda path: {'test_fixture': True})
    manifest = small_manifest(tmp_path)
    calls = []
    async def failed(*args):
        calls.append(1)
        raise RuntimeError('fixture')
    monkeypatch.setattr('stt_bench.run.transcribe', failed)
    monkeypatch.setenv('DEEPGRAM_API_KEY', 'unit-test-key')
    for resume in (False, True):
        outcomes = asyncio.run(run(manifest, Path('config/deepgram.json'), tmp_path / 'run', False, resume=resume))
    assert len(calls) == 2 and not outcomes[0]['valid']
    report = score(tmp_path / 'run', tmp_path / 'report')
    group = report['results'][0]
    assert report['completeness']['run_complete']
    assert group['wer'] is None and group['finalize_latency']['p50_ms'] is None
    assert group['excluded_clips'] == 1 and group['failed_or_invalid_attempts'] == 2
    assert 'unavailable' in (tmp_path / 'report' / 'per-clip.md').read_text()
    assert 'transport_or_provider_error' in (tmp_path / 'report' / 'results.md').read_text()


def test_expanded_dataset_is_reproducible_reviewed_and_disjoint(tmp_path):
    frozen = Path('datasets/fleurs-en-us-deepgram-v2/manifest.json')
    manifest = load_manifest(frozen)
    groups = {condition: [c for c in manifest['clips'] if c['condition'] == condition]
              for condition in ('public_anchor', 'public_entities')}
    assert len(groups['public_anchor']) == 100
    assert len(groups['public_entities']) > 0
    assert len({c['source_id'] for c in manifest['clips']}) == len(manifest['clips'])
    assert all(c['entities'] is not None and c['boundary_review']['status'] == 'passed_signal_checks' for c in manifest['clips'])
    assert all(c['entities'] for c in groups['public_entities'])
    assert manifest['exclusions']
    reproduced = prepare_expanded(Path('test.tsv'), Path('test'), Path('annotations/fleurs-en-us-entities-v1.json'), tmp_path / 'reproduced')
    assert reproduced.read_bytes() == frozen.read_bytes()
