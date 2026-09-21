"""Fixed membership and independently calculated pooled scores; no provider calls."""
import copy
import json
from pathlib import Path

import pytest

from test_unified_benchmark import record, report


def fixture():
    rows = {
        'deepgram-nova-3': [record('p', errors=1, words=10), record('r', 'private', words=30),
                            record('extra', words=100), record('f', 'fleurs', words=1000)],
        'cartesia-ink-2': [record('p', words=10), record('r', 'private', errors=4, words=30),
                           record('extra', words=100)],
    }
    models = [report.reduce_model(k, v, dict(pipecat=2, private=1, fleurs=1), True, []) for k, v in rows.items()]
    manifest = dict(version='test-v1', basis='pooled', datasets={'pipecat': {'p': 10}, 'private': {'r': 30}})
    return rows, models, manifest


def test_pool_integer_counts_exclude_extra_public_and_fleurs():
    rows, models, manifest = fixture()
    before = copy.deepcopy(models)
    ranking = report.rank_frozen_combined(models, rows, manifest)
    assert ranking['reference_words'] == 40
    assert [m['rank'] for m in models] == [1, 2]
    assert [m['public_rank'] for m in models] == [2, 1]
    assert [m['ranking_score']['wer'] for m in models] == [.025, .1]
    for m, old in zip(models, before):
        assert m['ranking_score']['n'] == 2
        for key in ('cohorts', 'reliability', 'timings', 'deadlines', 'combined'):
            assert m[key] == old[key]
    report.rank_frozen_combined(models, rows, manifest)
    assert models[0]['ranking_score']['wer'] == .025


@pytest.mark.parametrize('cohort,cid', [('pipecat', 'p'), ('private', 'r')])
@pytest.mark.parametrize('missing', [True, False])
def test_missing_or_failed_required_item_never_shrinks_set(cohort, cid, missing):
    rows, models, manifest = fixture()
    if missing:
        rows[models[0]['id']] = [r for r in rows[models[0]['id']] if r['id'] != cid]
    else:
        next(r for r in rows[models[0]['id']] if r['id'] == cid)['counts'] = None
    ranking = report.rank_frozen_combined(models, rows, manifest)
    assert ranking['reference_words'] == 40
    assert models[0]['rank'] is None and not models[0]['rankable']
    assert models[0]['ranking_score']['wer'] is None
    assert models[1]['ranking_score']['wer'] == .1
    assert models[1]['ranking_score']['reference_words'] == 40
    assert models[1]['rank'] == 1
    assert ranking['datasets']['private']['clip_ids'] == ['r']


def test_same_count_different_private_ids_is_incomplete():
    rows, models, manifest = fixture()
    rows[models[0]['id']][1]['id'] = 'other-private'
    report.rank_frozen_combined(models, rows, manifest)
    assert models[0]['rank'] is None


def test_reject_changed_reference_words_and_duplicates():
    rows, models, manifest = fixture()
    rows[models[0]['id']][0]['counts']['reference_words'] = 11
    with pytest.raises(ValueError, match='Frozen reference words differ'):
        report.rank_frozen_combined(models, rows, manifest)
    rows, models, manifest = fixture()
    rows[models[0]['id']].append(rows[models[0]['id']][0])
    with pytest.raises(ValueError, match='Duplicate'):
        report.rank_frozen_combined(models, rows, manifest)


def test_empty_inputs_no_rank_and_order_independent_ties():
    rows, models, manifest = fixture()
    assert report.rank_frozen_combined([], {}, manifest)['models'] == []
    for rs in rows.values():
        for r in rs:
            r['counts']['substitutions'] = 0
    report.rank_frozen_combined(models, rows, manifest)
    expected = {m['id']: m['rank'] for m in models}
    report.rank_frozen_combined(list(reversed(models)), rows, manifest)
    assert {m['id']: m['rank'] for m in models} == expected
    assert models[1]['rank'] == 1  # Stable model ID tie break.


def test_pinned_manifest_exact_denominators():
    m = json.loads(Path('config/rankings/combined-public-private-v1.json').read_text())
    assert len(m['datasets']['pipecat']) == 864
    assert len(m['datasets']['private']) == 8
    assert sum(m['datasets']['pipecat'].values()) == 20865
    assert sum(m['datasets']['private'].values()) == 12555
