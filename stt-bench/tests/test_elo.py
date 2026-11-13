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
