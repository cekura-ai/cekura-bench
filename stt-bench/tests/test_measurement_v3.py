import copy
import json
from pathlib import Path

import pytest

from stt_bench.deepgram import transcript_at, reduce_events
from stt_bench.measurement import deadline_observations, grouped_interval, summarize_deadlines
from stt_bench.entities import canonical, extract, value_errors
from stt_bench.run import assess
from stt_bench.review import export_review, load_review
from stt_bench.compare import paired_difference, compare_reports
from stt_bench.score import word_errors, score
from stt_bench.data import write_json, sha256
from stt_bench.streaming import EventLog, read_events

CONFIG = json.loads(Path('config/deepgram.json').read_text())


def msg(text, at, *, final=False, start=0, duration=1, words=None, ack=False):
    alternative = {'transcript': text}
    if words is not None:
        alternative['words'] = words
    return {'kind': 'provider_message', 'time_seconds': at, 'message': {
        'type': 'Results', 'is_final': final, 'from_finalize': ack,
        'start': start, 'duration': duration, 'channel_index': [0, 1],
        'channel': {'alternatives': [alternative]},
        'metadata': {'model_info': {'version': CONFIG['version']}, 'model_uuid': CONFIG['expected_model_uuid']}}}


def base():
    return [dict(kind='audio_sent', time_seconds=(i+1)*.02, ideal_seconds=(i+1)*.02,
                 send_completed_seconds=(i+1)*.02+.0001, phase='speech' if i == 0 else 'silence',
                 index=i, bytes=640) for i in range(51)] + [
        dict(kind='speech_end', time_seconds=.02), dict(kind='finalize_requested', time_seconds=.021),
        dict(kind='audio_complete', time_seconds=1.02), dict(kind='close_stream_requested', time_seconds=1.03),
        dict(kind='provider_message', time_seconds=1.1, message={'type': 'Metadata'}),
        dict(kind='clip_end', time_seconds=1.2)]


def annotation(kind, reference, value=None):
    text = value or reference
    start = reference.index(text)
    return dict(type=kind, text=text, start=start, end=start+len(text))


def test_partials_replace_and_deadline_never_uses_later_text():
    events = [msg('transfer', .01), msg('transfer twenty', .2),
              msg('transfer twenty dollars', .6, final=True)]
    assert transcript_at(events, .25)['text'] == 'transfer twenty'
    assert transcript_at(events, .5)['provisional']
    assert transcript_at(events, .6)['text'] == 'transfer twenty dollars'
    assert not transcript_at(events, .6)['provisional']


def test_finals_append_and_same_words_at_different_times_survive():
    a = msg('yes', .1, final=True, start=0, duration=.5)
    events = [a, copy.deepcopy(a), msg('yes', .2, final=True, start=.5, duration=.5),
              msg('please', .3, start=1)]
    assert transcript_at(events, .5)['text'] == 'yes yes please'


def test_partial_suffix_survives_split_final_with_word_timing():
    events = [msg('they they are', .1, duration=2, words=[
        {'word': 'they', 'start': 0}, {'word': 'they', 'start': 1}, {'word': 'are', 'start': 1.5}]),
        msg('they', .2, final=True, duration=1)]
    assert transcript_at(events, .25)['text'] == 'they they are'
    assert transcript_at(events, .25)['reconstruction_status'] == 'supported'
    events[0]['message']['channel']['alternatives'][0].pop('words')
    assert transcript_at(events, .25)['reconstruction_status'] == 'unsupported_overlap'
    events[0]['message']['channel']['alternatives'][0]['words'] = []
    assert transcript_at(events, .25)['reconstruction_status'] == 'unsupported_overlap'
    events += [msg('they are', .3, start=1)]
    assert transcript_at(events, .35)['text'] == 'they they are'
    assert transcript_at(events, .35)['reconstruction_status'] == 'supported'


def test_overlapping_finals_are_not_silently_concatenated():
    events = [msg('one', .1, final=True), msg('one two', .2, final=True)]
    assert transcript_at(events, .3)['reconstruction_status'] == 'unsupported_overlap'


def test_empty_partial_clears_previous_hypothesis():
    assert transcript_at([msg('ghost', .1), msg('', .2)], .3)['text'] == ''


def test_ack_is_not_stream_completion_and_missing_ack_does_not_invalidate_completion():
    events = base() + [msg('hello', .12, final=True)]
    status = assess(events, CONFIG)
    assert status['valid'] and status['transcript_complete']
    assert status['finalize_latency_ms'] is None
    ack_only = [e for e in events if e.get('message', {}).get('type') != 'Metadata']
    ack_only += [msg('', .2, final=True, ack=True, start=1)]
    assert not assess(ack_only, CONFIG)['transcript_complete']
    assert assess(ack_only, CONFIG)['finalize_latency_ms'] is not None


def test_deadline_retains_empty_failure_and_pre_speech_partial():
    events = base() + [msg('hello', .01)]
    observations = deadline_observations(events, 'hello', [], assess(events, CONFIG), 'live')
    assert observations[0]['text'] == 'hello'
    assert observations[0]['provisional']
    failure = [{'kind': 'error', 'time_seconds': .1}, {'kind': 'clip_end', 'time_seconds': .1}]
    observations = deadline_observations(failure, 'hello there', [], assess(failure, CONFIG), 'live')
    assert all(d['word_errors']['deletions'] == 2 for d in observations)
    assert all(d['status'] == 'failed_before_speech_end' for d in observations)


def test_future_error_does_not_change_deadline_status():
    events = base() + [msg('hello', .12), {'kind': 'error', 'time_seconds': .8}]
    observations = deadline_observations(events, 'hello', [], assess(events, CONFIG), 'live')
    assert observations[1]['status'] == 'text_available'
    assert observations[3]['status'] == 'request_failed'


def test_truncated_observation_is_unavailable_not_backfilled():
    events = [dict(kind='speech_end', time_seconds=.02), msg('hello', .12)]
    observations = deadline_observations(events, 'hello there', [], assess(events, CONFIG), 'live')
    assert observations[0]['word_errors']['deletions'] == 2
    assert observations[1]['status'] == 'observation_window_incomplete'
    assert observations[1]['word_errors'] is None


@pytest.mark.parametrize('kind,ref,hyp', [
    ('amount', '$20', 'twenty dollars'), ('amount', '£20.00', 'twenty pounds'),
    ('number', '20', 'twenty'), ('date', 'January 21', 'January twenty first'),
    ('phone', '+1 202-555-0100', '+12025550100'),
    ('email', 'Sam@EXAMPLE.com', 'Sam@example.com'), ('spelled_sequence', 'A B C', 'A-B-C'),
    ('spelled_sequence', 'A B C', 'a b c')])
def test_typed_equivalence(kind, ref, hyp):
    result = value_errors(ref, hyp, [annotation(kind, ref)])['by_type'][kind]
    assert result['status'] == 'measured'
    assert result['counts']['correct'] == 1
    assert result['counts']['spurious'] == 0


@pytest.mark.parametrize('kind,ref,hyp,entity', [
    ('amount', '$20', '$30', '$20'), ('date', 'January 21', 'January 22', 'January 21'),
    ('identifier', 'Account ID 0051', 'Account ID 51', '0051'),
    ('email', 'Sam@example.com', 'sam@example.com', 'Sam@example.com'),
    ('number', '0051', '51', '0051')])
def test_typed_wrong_values(kind, ref, hyp, entity):
    c = value_errors(ref, hyp, [annotation(kind, ref, entity)])['by_type'][kind]['counts']
    assert c['wrong'] == 1


def test_entity_occurrences_missing_wrong_spurious_and_no_reference_amount():
    ref = '20 then 20'
    labels = [dict(type='number', text='20', start=0, end=2), dict(type='number', text='20', start=8, end=10)]
    c = value_errors(ref, 'twenty', labels)['by_type']['number']['counts']
    assert c['correct'] == 1 and c['missing'] == 1
    c = value_errors(ref, '20 then 20 then 30', labels)['by_type']['number']['counts']
    assert c['correct'] == 2 and c['spurious'] == 1
    assert value_errors('thank you', 'thank you $20', [])['by_type']['amount']['counts']['spurious'] == 1


def test_unsupported_or_incomplete_reference_inventory_not_scored_as_hallucination():
    ref = 'F1 and 20'
    result = value_errors(ref, ref, [annotation('number', ref, 'F1')])
    assert result['unsupported']
    assert result['by_type']['number']['counts'] is None
    assert canonical('date', 'third century BCE') is None
    assert value_errors('20', '20', [])['by_type']['number']['status'] == 'unsupported_reference_inventory'


def test_number_at_sentence_end_and_no_ordinary_prose_spelling_false_positive():
    assert extract('pay 20.')[0]['value'] == canonical('number', '20')
    assert not extract('I am a person')
    assert canonical('number', 'twenty point five') == canonical('number', '20.5')
    assert canonical('number', 'a thousand') == canonical('number', '1000')


def test_grouped_bootstrap_and_paired_differences():
    rows = [dict(dependency_group='a', word_errors=word_errors('one two', 'one')),
            dict(dependency_group='a', word_errors=word_errors('three', 'three')),
            dict(dependency_group='b', word_errors=word_errors('four', 'four'))]
    ci = grouped_interval(rows, lambda r:r['word_errors'])
    assert ci['groups'] == 2 and ci['lower'] <= .25 <= ci['upper']
    assert ci == grouped_interval(rows, lambda r:r['word_errors'])
    assert grouped_interval([{'word_errors': rows[0]['word_errors']}], lambda r:r['word_errors'])['status'].startswith('unavailable')
    left = [dict(**r, deadlines=[dict(word_errors=r['word_errors'])]) for r in rows]
    right = copy.deepcopy(left)
    paired = paired_difference(left, right, 0)
    assert paired['lower'] == paired['upper'] == paired['left_minus_right_wer'] == 0
    assert not paired['ordering_resolved']


def test_review_hash_binding_and_no_automatic_verification(tmp_path):
    manifest = Path('datasets/fleurs-en-us-smoke-v1/manifest.json')
    review = export_review(manifest, tmp_path / 'review')
    statuses, summary = load_review(manifest, review)
    assert summary['verified_clips'] == 0
    assert not any(r['human_review_verified'] for r in statuses.values())
    contents = json.loads(review.read_text())
    contents['clips'][0]['reference'] = 'changed'
    write_json(review, contents)
    with pytest.raises(ValueError, match='changed'):
        load_review(manifest, review)


def test_writer_preserves_order_and_drains_on_close(tmp_path):
    path = tmp_path / 'events.jsonl'
    log = EventLog(path)
    for i in range(1000):
        log.emit('fixture', index=i)
    log.close()
    assert [e['index'] for e in read_events(path)] == list(range(1000))


def test_retry_never_replaces_first_request_in_report(tmp_path):
    root = tmp_path / 'run'
    (root / 'raw').mkdir(parents=True)
    (root / 'attempts').mkdir()
    write_json(root / 'manifest.json', {'clips': [dict(clip_id='fixture', condition='public_anchor',
               reference='hello', entities=[], submitted_seconds=1.02)]})
    write_json(root / 'run.json', dict(schema_version=2, mode='live', config=CONFIG,
                                     manifest_sha256=sha256(root / 'manifest.json')))
    events1 = [dict(kind='error', time_seconds=.1), dict(kind='clip_end', time_seconds=.2)]
    events2 = base() + [msg('hello', .1, final=True)]
    for number, events in enumerate((events1, events2), 1):
        (root / 'raw' / f'fixture--attempt-{number}.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    report = score(root, tmp_path / 'report')
    group = report['results'][0]
    assert group['wer'] == 0
    assert all(d['wer'] == 1 for d in group['deadlines'])
    assert group['reliability']['recovered_on_retry'] == 1
    assert not any(d['headline_eligible'] for d in group['deadlines'])
    with pytest.raises(ValueError, match='identical'):
        other = copy.deepcopy(report)
        other['manifest_sha256'] = 'different'
        write_json(tmp_path / 'other.json', other)
        compare_reports(tmp_path / 'report' / 'results.json', tmp_path / 'other.json', tmp_path / 'comparison')


def test_adjacent_provider_timestamps_tolerate_float_rounding():
    events = [msg('first', .1, final=True, start=3.615, duration=4.575),
              msg('second', .2, final=True, start=8.19, duration=4.29)]
    result = transcript_at(events, .3)
    assert result['text'] == 'first second'
    assert result['reconstruction_status'] == 'supported'


def test_interruption_is_not_a_provider_failure():
    events = [dict(kind='error', time_seconds=.1, error_type='Interrupted')]
    a = assess(events, CONFIG)
    assert 'transport_or_provider_error' not in a['exclusion_reasons']
    result = deadline_observations(events, 'hello', [], a, 'live')
    assert all(d['status'] == 'locally_interrupted' and d['word_errors'] is None for d in result)


def test_preflight_rejects_failed_stale_or_other_host_results(tmp_path):
    from datetime import datetime, timezone, timedelta
    from stt_bench.diagnostics import validate_preflight, pacing_identity
    path = tmp_path / 'pacing.json'
    report = dict(mode='local_websocket_probe', identity=pacing_identity(), seconds=10, repeats=3,
                  probe_version=4, receiver_process='independent',
                  trials=[{'valid': True, 'duplex_valid': True, 'send_pacing_valid': True,
                           'client_delivery_valid': True, 'fixture_valid': True} for _ in range(3)],
                  receiver_delay_ms=0, injected_send_stall_ms=0, receiver_stall_ms=0, client_stall_ms=0,
                  completed_at=datetime.now(timezone.utc).isoformat())
    write_json(path, report)
    assert validate_preflight(path)['sha256'] == sha256(path)
    for change in ({'trials': [{'valid': False, 'duplex_valid': True}] * 3},
                   {'trials': [{'valid': True, 'duplex_valid': False}] * 3},
                   {'trials': [{'valid': True}] * 3}, {'identity': {}},
                   {'receiver_stall_ms': 80}, {'client_stall_ms': 80},
                   {'completed_at': (datetime.now(timezone.utc)-timedelta(hours=2)).isoformat()}):
        write_json(path, {**report, **change})
        with pytest.raises(ValueError):
            validate_preflight(path)
    for change in ({'probe_version': 1}, {'probe_version': 3}, {'receiver_process': 'same_event_loop'}):
        write_json(path, {**report, **change})
        with pytest.raises(ValueError, match='Legacy pacing check'):
            validate_preflight(path)
    legacy = {key: value for key, value in report.items()
              if key not in {'probe_version', 'receiver_process'}}
    write_json(path, legacy)
    with pytest.raises(ValueError, match='Legacy pacing check'):
        validate_preflight(path)
    with pytest.raises(ValueError, match='require'):
        validate_preflight(None)


def test_live_run_requires_preflight_before_requests_or_output_creation(tmp_path):
    import asyncio
    from stt_bench.run import run
    with pytest.raises(ValueError, match='pacing-check'):
        asyncio.run(run(Path('datasets/fleurs-en-us-smoke-v1/manifest.json'),
                        Path('config/deepgram.json'), tmp_path / 'live', False))
    assert not (tmp_path / 'live').exists()


def test_human_review_requires_all_attestations_and_policy(tmp_path):
    manifest = Path('datasets/fleurs-en-us-smoke-v1/manifest.json')
    path = export_review(manifest, tmp_path / 'review')
    review = json.loads(path.read_text())
    review['transcription_policy'] = 'Verbatim; fixtures only, not actual listening evidence'
    review['grouping_policy'] = 'Synthetic fixture group IDs'
    for row in review['clips']:
        row.update(reviewed_by='fixture', reviewed_at='2026-09-12', reference_listened_verified=True,
                   boundary_listened_verified=True, entities_listened_verified=True, dependency_group='shared')
    write_json(path, review)
    statuses, summary = load_review(manifest, path)
    assert summary['status'] == 'verified'
    assert {r['dependency_group'] for r in statuses.values()} == {'shared'}
    review['clips'][0]['boundary_listened_verified'] = False
    write_json(path, review)
    assert load_review(manifest, path)[1]['status'] == 'review_incomplete'


def test_compare_preserves_coverage_and_rejects_different_capture_protocol(tmp_path):
    config = dict(measurement_version=3, mode='live', manifest_sha256='same', normalization={},
                  scoring_source_hashes={'scorer': 'same'}, review={},
                  capture_protocol={'measurement_version':3, 'source_hashes': {'streaming.py': 'same', 'deepgram.py': 'same'}},
                  run_identity={'provider': 'fixture'}, run_configuration={}, run_started_at='fixture')
    clips = []
    for i in range(3):
        clips.append(dict(clip_id=str(i), condition='public_anchor', dependency_group=str(i),
                          deadlines=[dict(pacing_valid=True, word_errors=word_errors('hello', 'hello')) for _ in range(4)]))
    left = {**config, 'clips': clips}
    right = copy.deepcopy(left)
    right['clips'][2]['deadlines'][0]['pacing_valid'] = False
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    write_json(a, left)
    write_json(b, right)
    comparison = compare_reports(a, b, tmp_path / 'comparison')['comparisons'][0]
    assert comparison['planned_clips'] == 3 and comparison['common_clips'] == 2
    assert comparison['left_eligible'] == 3 and comparison['right_eligible'] == 2
    assert comparison['paired']['left_minus_right_wer'] == 0
    right['capture_protocol']['measurement_version'] = 2
    write_json(b, right)
    with pytest.raises(ValueError, match='capture_protocol'):
        compare_reports(a, b, tmp_path / 'rejected')
