"""Independent aggregation checks using synthetic records; no provider calls."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('unified', Path(__file__).resolve().parents[1] / 'scripts/unified_benchmark.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def counts(errors=0, words=10):
    return dict(substitutions=errors, insertions=0, deletions=0, reference_words=words)


def record(cid, cohort='pipecat', errors=0, words=10, delay=100, valid=True):
    first = dict(attempt=1, valid=valid, t0_seconds=2,
                 final_transcript_received_seconds=2 + delay / 1000,
                 first_partial_after_t0={'latency_ms': delay / 2},
                 completion_latency_ms=delay + 1000)
    return dict(id=cid, cohort=cohort, counts=counts(errors, words) if valid else None,
                first=first, attempts=[first], words=None)


def reduce(rows, planned=None):
    return report.reduce_model('deepgram-nova-3', rows, planned or dict(pipecat=1, fleurs=1, private=1), True, [])


def test_combined_wer_uses_words_not_average_of_dataset_percentages():
    rows = [record('p', errors=1, words=1), record('f', 'fleurs', words=99), record('r', 'private', words=100)]
    result = reduce(rows)
    assert result['combined']['wer'] == .005
    assert result['combined']['reference_words'] == 200
    assert result['rankable']


def test_public_percentiles_exclude_fleurs_and_private_timing():
    rows = [record('p', delay=0), record('f', 'fleurs', delay=100), record('r', 'private', delay=100000)]
    rows[-1]['words'] = dict(reference_words=10, words=[{'delay_ms': v} for v in (1000, 2000, 3000)])
    result = reduce(rows)
    assert result['timings']['final'] == pytest.approx(dict(n=1, p50_ms=0, p90_ms=0, p95_ms=0, p99_ms=0))
    assert result['timings']['word']['p99_ms'] == 2980
    assert result['timings']['interim']['n'] == 1


def test_missing_values_and_invalid_first_attempts_never_use_recovery_timing():
    row = record('p', valid=False)
    row['counts'] = counts(1, 100)
    row['attempts'].append(dict(attempt=2, valid=True, completion_latency_ms=1))
    result = reduce([row])
    assert result['combined']['wer'] == .01
    assert result['timings']['final'] == dict(n=0, p50_ms=None, p90_ms=None, p95_ms=None, p99_ms=None)
    assert result['timings']['interim']['n'] == 0
    assert (result['failed_attempts'], result['retries']) == (1, 1)
    assert not result['rankable']
    assert result['cohorts']['fleurs']['wer'] is None


def test_duplicate_clips_fail_instead_of_double_counting():
    with pytest.raises(ValueError, match='Duplicate'):
        reduce([record('p'), record('p')])


def test_percentiles_tail_interpolation_and_empty_values():
    assert report.percentiles([0, None, 100]) == dict(n=2, p50_ms=50, p90_ms=90, p95_ms=95, p99_ms=99)
    assert report.percentiles([])['p99_ms'] is None
    with pytest.raises(ValueError, match='timing'):
        report.percentiles([float('nan')])


def test_attempt_one_deadlines_retained_from_selected_clip_summary():
    first = dict(attempt=1, valid=False)
    row = dict(clip_id='x', attempts=[first, dict(attempt=2, valid=True)],
               accuracy_usable=True, word_errors=counts(), deadlines=[{'deadline_ms': 500}])
    result = report.clip_record(row, 'fleurs')
    assert result['first']['attempt'] == 1
    assert result['first']['deadlines'] == row['deadlines']
    assert 'deadlines' not in first


def test_export_record_has_no_transcripts_or_private_words():
    row = record('p')
    row['first']['transcript'] = 'PRIVATE TRANSCRIPT SENTINEL'
    result = reduce([row])
    assert 'PRIVATE TRANSCRIPT SENTINEL' not in str(result)
    assert 'counts' not in result  # Only aggregate corpus counts are exported.


def test_failure_rates_use_fixed_planned_count_and_distinguish_missing_and_recovery():
    failed = record('failed', valid=False)
    recovered = record('recovered', valid=False)
    recovered['counts'] = counts()
    recovered['attempts'].append(dict(attempt=2, valid=True))
    result = reduce([failed, recovered], planned=dict(pipecat=4, fleurs=0, private=0))
    assert result['reliability'] == dict(planned=4, attempted=2, usable=1, failed=1,
                                       not_run=2, failure_rate=.25, not_run_rate=.5,
                                       missing_result_rate=.75)
    assert result['failed_attempts'] == 2  # A recovered attempt is not a final item failure.
    assert result['combined']['wer'] == 0


def test_no_attempts_is_unavailable_failure_rate_not_zero():
    result = reduce([], planned=dict(pipecat=4, fleurs=0, private=0))
    assert result['reliability']['failure_rate'] is None
    assert result['reliability']['not_run_rate'] == 1


def test_equal_attempted_sets_rank_only_shared_successes_and_keep_all_available():
    # Both models attempted every clip. One failed clip must be excluded from
    # both ranking scores, even though the other model transcribed it correctly.
    rows = {
        'deepgram-nova-3': [record('shared', errors=1), record('failed-elsewhere', words=100),
                            record('private', 'private')],
        'cartesia-ink-2': [record('shared'), record('failed-elsewhere', valid=False),
                           record('private', 'private')],
    }
    models = [report.reduce_model(model, records, dict(pipecat=2, fleurs=0, private=1), True, [])
              for model, records in rows.items()]
    original = {m['id']: dict(m['cohorts']['pipecat']) for m in models}
    common = report.rank_common_public(models, rows)
    assert common['clip_ids'] == ['shared']
    assert common['reference_words'] == 10
    assert all(m['headline']['n'] == 1 and m['headline']['reference_words'] == 10 for m in models)
    assert [(m['rank'], m['headline']['wer']) for m in models] == [(2, .1), (1, 0)]
    assert all(m['cohorts']['pipecat'] == original[m['id']] for m in models)
    assert models[0]['cohorts']['pipecat']['wer'] == 1 / 110
    assert models[1]['reliability']['failed'] == 1
