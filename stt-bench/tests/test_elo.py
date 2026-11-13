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
