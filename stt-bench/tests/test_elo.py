from stt_bench import elo


def test_outcome_margin_and_failures():
    assert elo.outcome(90.0, 87.0, 5.0) == 0.5
    assert elo.outcome(95.0, 80.0, 5.0) == 1.0
    assert elo.outcome(None, 80.0, 5.0) == 0.0
    assert elo.outcome(None, None, 5.0) is None


def test_bradley_terry_orders_by_wins():
    models = ['a', 'b', 'c']
    matches = [('a', 'b', 1.0, 1.0)] * 8 + [('b', 'c', 1.0, 1.0)] * 8 + [('a', 'c', 1.0, 1.0)] * 8
    r = elo.bt(matches, models)
    assert r['a'] > r['b'] > r['c']
    assert abs(sum(r.values()) / 3 - 1000) < 5


def test_pipecat_counting_vendored():
    assert elo.count_reference_words("I'm here at 7:30, it's fine.") == 8  # I am, here, at, 7:30, it is, fine
    assert elo.error_weight({'type': 'deletion', 'reference': 'the big dog', 'hypothesis': None}) == 3


def test_bootstrap_resamples_whole_conversations_across_measures():
    import random
    assert elo.unit('c003-s1-seg0008') == 'c003'
    assert elo.unit('pipecat-0042') == 'pipecat-0042'
    turns = {f'c00{i}-s1-seg000{j}': {'a': 90.0, 'b': 80.0} for i in (1, 2) for j in (1, 2)}
    M = {'ocular': (turns, {}), 'phone': (turns, {}), 'pipecat': ({'p1': {}, 'p2': {}}, {})}
    s = elo.bootstrap_sample(M, random.Random(1))
    assert s['ocular'] == s['phone']  # one conversation draw shared by every measure built on it
    assert len(s['ocular']) == 4 and len({c.split('-')[0] for c in s['ocular']}) <= 2
    assert all(c.split('-')[0] in {x.split('-')[0] for x in s['ocular']} for c in s['ocular'])
    assert len(s['pipecat']) == 2


def test_every_clip_with_reference_words_is_scored():
    we = lambda n, e: dict(reference_words=n, substitutions=e, insertions=0, deletions=0)
    runs = {'a': {'short': ('valid', we(2, 0), 'yes please'), 'empty': ('valid', we(0, 0), ''), 'long': ('valid', we(6, 1), '')},
            'b': {'short': ('valid', we(2, 1), 'yes please'), 'empty': ('valid', we(0, 1), ''), 'long': ('failed', None, '')}}
    rows, wts = elo.measure_rows('wer', runs)
    assert set(rows) == {'short', 'long'}  # a 2-word reply counts; a clip with no reference words does not
    assert rows['short'] == {'a': 100.0, 'b': 50.0} and wts['short'] == 2
    assert rows['long']['b'] is None and wts['long'] == 6


def test_table_wer_is_plain_wer_with_no_floor():
    we = lambda n, e: dict(reference_words=n, substitutions=0, insertions=e, deletions=0)
    runs = {'a': {'x': ('valid', we(1, 9), ''), 'y': ('valid', we(9, 1), ''), 'z': ('failed', None, '')}}
    assert elo.wer_totals(runs) == {'a': (10, 10)}  # 9 inserted words on a 1-word reply all count; the failed clip does not
