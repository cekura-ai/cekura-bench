"""Independently reconcile the selected calls, model evidence and provider aggregates."""
import collections
import json
import math
import pathlib
import statistics

P = pathlib.Path(__file__).resolve().parent
mean = lambda values: statistics.mean(values) if values else None


def close(actual, expected):
    assert (actual is None and expected is None) or (actual is not None and expected is not None and math.isclose(actual, expected, abs_tol=1e-9)), (actual, expected)


def validate(path=P / 'results.json'):
    data = json.loads(path.read_text())
    previous = json.loads((P.parent / '2026-11-27' / 'results.json').read_text())
    for key in ('weights', 'metrics', 'scenarios', 'repetitions', 'stop_anchors_ms', 'aggregation', 'cisco_audio_source', 'cisco_execution_lineage'):
        assert data[key] == previous[key], key
    assert {p['provider']: p for p in data['providers'] if p['provider'] != 'ElevenLabs'} == {
        p['provider']: p for p in previous['providers'] if p['provider'] != 'ElevenLabs'}
    assert data['provider_count'] == len(data['providers']) == 17
    assert data['phone_number_count'] == sum(p['phone_number_count'] for p in data['providers']) == 34
    assert data['selected_call_count'] == sum(p['selected_calls'] for p in data['providers']) == 1530
    assert data['audio_model']['scored_calls'] == sum(p['metrics']['audio']['scored_calls'] for p in data['providers'])
    assert data['elevenlabs_audio_source'] == 'nisqa' and data['weights']['latency'] == 0
    assert all(m['metric_id'] != 276247 for m in data['metrics'])
    row = next(p for p in data['providers'] if p['provider'] == 'ElevenLabs')
    assert row['phone_number_count'] == 3 and row['selected_calls'] == 135
    calls = [c for number in row['numbers'] for c in number['calls']]
    selected = {c['run_id'] for c in calls}
    assert len(calls) == len(selected) == 135
    lineage = data['elevenlabs_execution_lineage']
    assert lineage == json.loads((P / 'elevenlabs-selection.json').read_text())
    assert lineage['executions'] == 146 and len(lineage['replacements']) == 6 and len(lineage['excluded_attempt_ids']) == 5
    assert selected == {c['run_id'] for c in lineage['calls']}
    assert not selected.intersection(lineage['excluded_attempt_ids'])
    assert all(r['new_run_id'] in selected and r['old_run_id'] not in selected for r in lineage['replacements'])
    inputs = json.loads((P / 'elevenlabs-calls.json').read_text())
    for number, original in zip(row['numbers'], inputs, strict=True):
        assert number['calls'] == original['calls']
        assert len(number['calls']) == 45
        coverage = collections.Counter(c['scenario'] for c in number['calls'])
        assert set(coverage) == set(data['scenarios']) and set(coverage.values()) == {3}
        for c in number['calls']:
            assert c['status'] == 'completed'
            assert c['url'] == f"https://dashboard.cekura.ai/16745/results/{c['result_id']}?call_id={c['run_id']}"
            assert set(c['metrics']) == set(data['weights'])
            for definition in data['metrics']:
                v = c['metrics'][definition['key']]
                assert v is None or (not isinstance(v, bool) and math.isfinite(v) and v >= 0)
                if definition['kind'] == 'score': assert v is None or v <= 5
                if definition['kind'] == 'pass': assert v in (None, 0, 100)
        for definition in data['metrics']:
            key, kind = definition['key'], definition['kind']
            groups = collections.defaultdict(list)
            for c in number['calls']:
                v = c['metrics'][key]
                if v is not None: groups[c['scenario']].append(v)
            value = mean([mean(values) for values in groups.values()])
            normalized = None if key == 'latency' or value is None else value * (20 if kind == 'score' else 1)
            if kind == 'ms':
                events = [e for c in number['calls'] for e in c['events_ms'][key]]
                value = statistics.median(events) if key == 'latency' and events else mean(events)
                assert number['metrics'][key]['events'] == len(events)
            if key == 'stop':
                qualities = collections.defaultdict(list)
                for c in number['calls']:
                    if c['events_ms']['stop']:
                        qualities[c['scenario']].append(mean([max(0, min(100, 100 * (3000 - v) / 2700)) for v in c['events_ms']['stop']]))
                normalized = mean([mean(values) for values in qualities.values()])
            close(number['metrics'][key]['value'], value)
            close(number['metrics'][key]['normalized'], normalized)
            assert number['metrics'][key]['scored_calls'] == sum(c['metrics'][key] is not None for c in number['calls'])
    for definition in data['metrics']:
        key = definition['key']
        eligible = [number['metrics'][key] for number in row['numbers'] if number['metrics'][key]['value'] is not None]
        close(row['metrics'][key]['value'], mean([m['value'] for m in eligible]))
        close(row['metrics'][key]['normalized'], None if key == 'latency' else mean([m['normalized'] for m in eligible]))
        assert row['metrics'][key]['eligible_numbers'] == len(eligible)
        assert row['metrics'][key]['scored_calls'] == sum(m['scored_calls'] for m in eligible)
    close(row['composite'], sum(weight * row['metrics'][key]['normalized'] for key, weight in data['weights'].items() if weight))
    proof = json.loads((P / 'elevenlabs-nisqa-validation.json').read_text())
    assert proof['checkpoint_sha256'] == data['audio_model']['checkpoint_sha256'] == previous['audio_model']['checkpoint_sha256']
    assert proof['scored_calls'] == 135
    assert {r['run_id']: r['score'] for r in proof['scores']} == {c['run_id']: c['metrics']['audio'] for c in calls}
    assert all(r['unique_source_seconds'] >= 5 and 30 <= r['model_seconds'] <= 60 for r in proof['scores'])
    assert all(r['input_samples_regenerated_byte_identically'] and r['prediction_csv_matches'] for r in proof['scores'])
    assert [p['composite'] for p in data['providers']] == sorted((p['composite'] for p in data['providers']), reverse=True)
    print(f"Validated ElevenLabs:135 selected calls,3 numbers,{row['composite']:.5f}/100;16 provider entries unchanged.")


if __name__ == '__main__':
    validate()
