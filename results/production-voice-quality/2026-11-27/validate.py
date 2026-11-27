"""Recompute Cisco's November 27 aggregates and composite from its selected calls, and check every other provider is unchanged."""
import collections, json, math, pathlib, statistics

P = pathlib.Path(__file__).resolve().parent
mean = lambda xs: statistics.mean(xs) if xs else None


def close(a, b):
    assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, abs_tol=1e-9)), (a, b)


def validate(path=P / 'results.json'):
    d = json.loads(path.read_text())
    previous = json.loads((P.parent / '2026-11-24/results.json').read_text())
    assert d['weights'] == previous['weights'] and d['metrics'] == previous['metrics']
    assert {p['provider']: p for p in d['providers'] if p['provider'] != 'Cisco'} == \
           {p['provider']: p for p in previous['providers'] if p['provider'] != 'Cisco'}
    assert len(d['providers']) == 17 and d['selected_call_count'] == 1440
    assert d['cisco_audio_source'] in ('platform_voice_tone_provisional', 'nisqa')

    cisco = next(p for p in d['providers'] if p['provider'] == 'Cisco')
    calls = cisco['numbers'][0]['calls']
    assert calls == json.loads((P / 'cisco-calls.json').read_text())
    assert len(calls) == 45 and len({c['run_id'] for c in calls}) == 45
    per_scenario = collections.Counter(c['scenario'] for c in calls)
    assert set(per_scenario) == set(d['scenarios']) and set(per_scenario.values()) == {3}
    for c in calls:
        assert c['result_id'] == 1194670 and c['status'] == 'completed'
        assert c['url'] == f"https://dashboard.cekura.ai/8793/results/1194670?call_id={c['run_id']}"
        for m in d['metrics']:
            v = c['metrics'][m['key']]
            assert v is None or (not isinstance(v, bool) and math.isfinite(v) and v >= 0)
            if m['kind'] == 'score':
                assert v is None or v <= 5
            if m['kind'] == 'pass':
                assert v in (None, 0, 100)

    for m in d['metrics']:
        key, kind, agg = m['key'], m['kind'], cisco['metrics'][m['key']]
        groups = collections.defaultdict(list)
        for c in calls:
            if c['metrics'][key] is not None:
                groups[c['scenario']].append(c['metrics'][key])
        value = mean([mean(v) for v in groups.values()])
        norm = None if key == 'latency' or value is None else value * (20 if kind == 'score' else 1)
        if kind == 'ms':
            events = [e for c in calls for e in c['events_ms'][key]]
            value = (statistics.median(events) if events else None) if key == 'latency' else mean(events)
        if key == 'stop':
            stop_groups = collections.defaultdict(list)
            for c in calls:
                if c['events_ms']['stop']:
                    stop_groups[c['scenario']].append(mean([max(0, min(100, 100 * (3000 - v) / 2700)) for v in c['events_ms']['stop']]))
            norm = mean([mean(v) for v in stop_groups.values()])
        close(value, agg['value'])
        close(norm, agg['normalized'])
        assert agg['scored_calls'] == sum(c['metrics'][key] is not None for c in calls)
    close(sum(d['weights'][k] * cisco['metrics'][k]['normalized'] for k in d['weights'] if d['weights'][k] > 0), cisco['composite'])
    assert [p['composite'] for p in d['providers']] == sorted((p['composite'] for p in d['providers']), reverse=True)
    print(f"Validated Cisco November 27: {len(calls)} calls, composite {cisco['composite']:.4f}; 16 reference providers unchanged.")


if __name__ == '__main__':
    validate()
