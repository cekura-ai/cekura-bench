"""Replace Cisco's November 24 calls with its November 27 rerun; every other provider is unchanged.

Voice tone uses isolated-AI NISQA scores with the same November 24 checkpoint and preparation.
The tested phone number is omitted from this snapshot at Luis's request.
"""
import copy, importlib.util, json, pathlib, sys

P = pathlib.Path(__file__).resolve().parent
PREVIOUS = P.parent / '2026-11-24'
AUDIO_SOURCE = 'nisqa'

# Reuse the November 24 aggregation unchanged (its module imports render.py from its own folder).
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('cisco_quality_builder_2026_11_24', PREVIOUS / 'build_results.py')
_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_builder)
aggregate = _builder.aggregate


def build():
    previous = json.loads((PREVIOUS / 'results.json').read_text())
    calls = json.loads((P / 'cisco-calls.json').read_text())
    number_metrics = aggregate(calls, previous['metrics'])
    provider_metrics = {k: {'value': v['value'], 'normalized': v['normalized'], 'scored_calls': v['scored_calls'],
                            'eligible_numbers': int(v['value'] is not None), 'events': v['events']}
                        for k, v in number_metrics.items()}
    weights = previous['weights']
    composite = sum(weights[k] * provider_metrics[k]['normalized'] for k in weights if weights[k] > 0)
    cisco = {'provider': 'Cisco', 'phone_number_count': 1, 'selected_calls': len(calls), 'composite': composite,
             'metrics': provider_metrics,
             'numbers': [{'calls': calls, 'metrics': number_metrics}]}
    data = copy.deepcopy(previous)
    for key in ('report_sha256', 'cisco_execution_lineage'):
        data.pop(key, None)
    data['snapshot_date'] = '2026-11-27'
    data['provider_snapshot_dates']['Cisco'] = '2026-11-27'
    data['comparison_note'] = ('Cisco was retested on November 27, after Cisco changed the number and tenant of its agent; '
                               'the other 16 providers retain the November 12 reference scores. Comparisons are directional, '
                               'not a simultaneous retest of every provider.')
    data['cisco_audio_source'] = AUDIO_SOURCE
    data['cisco_execution_lineage'] = {'project_id': 8793, 'result_id': 1194670, 'executions': len(calls),
                                       'selected_calls': len(calls), 'replacements': []}
    data['providers'] = [p for p in data['providers'] if p['provider'] != 'Cisco'] + [cisco]
    data['providers'].sort(key=lambda p: -p['composite'])
    data['audio_model']['scored_calls'] = sum(p['metrics']['audio']['scored_calls'] for p in data['providers'])
    return data


def main():
    data = build()
    (P / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    cisco = next(p for p in data['providers'] if p['provider'] == 'Cisco')
    rank = 1 + [p['provider'] for p in data['providers']].index('Cisco')
    print(f"Cisco {cisco['composite']:.2f}/100 ({rank} of {len(data['providers'])}); "
          f"latency {cisco['metrics']['latency']['value']:.0f} ms; voice tone source: {data['cisco_audio_source']}.")


if __name__ == '__main__':
    main()
