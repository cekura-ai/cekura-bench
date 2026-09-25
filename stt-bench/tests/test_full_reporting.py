"""Cross-worker private aggregation and recovery invariants."""
from stt_bench.full_benchmark import MODELS, combine
from stt_bench.score import word_errors


def attempt(cid, number, delays, valid=True):
    return dict(clip_id=cid, cohort='private', attempt=number, valid=valid,
                word_errors=word_errors('one two three four', 'one two three four'),
                failure_class=None if valid else 'transient', pacing={'valid': valid},
                sent_audio_seconds=10, completion_latency_ms=1200,
                private_latency=None if number == 2 else {
                    'reference_words': 4,
                    'words': [{'delay_ms': d} for d in delays],
                    'exclusions': {'invalid_first_attempt' if not valid else 'ambiguous_alignment': 4-len(delays)}})


def saved(rows):
    return {'verified': True, 'assignment': {'model': MODELS[0]}, 'rows': rows}


def test_private_percentiles_merge_individual_words_and_keep_failed_first_attempts():
    plan = {'run_id': 'fixture', 'items': [{'clip_id': c, 'cohort': 'private'} for c in ('a', 'b', 'c')],
            'configs': dict.fromkeys(MODELS, {})}
    rows = [attempt('a', 1, [10, 20, 30, 40]), attempt('b', 1, [1000]),
            attempt('c', 1, [], False), attempt('c', 2, [])]
    complete = combine(plan, [saved(rows)])['models']
    partitioned = combine(plan, [saved(rows[:1]), saved(rows[1:3]), saved(rows[3:])])['models']
    assert complete == partitioned
    groups = complete[MODELS[0]]['groups']
    timing = groups['private']['word_finalization']
    assert timing['n'] == 5 and timing['p50_ms'] == 30
    assert timing['reference_words'] == 12
    assert timing['exclusions'] == {'ambiguous_alignment': 3, 'invalid_first_attempt': 4}
    assert groups['private']['recovered'] == 1
    assert groups['private']['completion_first_attempt']['n'] == 2


def test_previous_private_variants_remain_separate_and_audio_is_accounted():
    plan = {'run_id': 'fixture', 'items': [{'clip_id': 'a', 'cohort': 'private'}],
            'configs': dict.fromkeys(MODELS, {}), 'private_variants': {MODELS[0]: 'approved-session-mode'}}
    old = saved([attempt('a', 1, [], False), attempt('a', 2, [], False)])
    current = saved([attempt('a', 1, [100, 200])])
    current['assignment']['private_variant'] = 'approved-session-mode'
    report = combine(plan, [old, current])
    model = report['models'][MODELS[0]]
    assert model['groups']['private']['attempts'] == 1
    assert model['groups']['private']['word_finalization']['n'] == 2
    assert model['previous_private_transport_submitted_seconds'] == 20
    assert model['all_variants_submitted_seconds'] == 30
    assert len(report['previous_private_transport_attempts']) == 2
