"""Refresh ElevenLabs with three auto-shop deployments using the standard benchmark aggregation."""
import copy
import importlib.util
import json
import pathlib
import sys

P = pathlib.Path(__file__).resolve().parent
PREVIOUS = P.parent / '2026-11-27'
AUDIO_SOURCE = 'nisqa'
BUILDER = P.parent / '2026-11-24'
sys.path.insert(0, str(BUILDER))
spec = importlib.util.spec_from_file_location('published_quality_aggregate', BUILDER / 'build_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
aggregate = module.aggregate


def build():
    data = copy.deepcopy(json.loads((PREVIOUS / 'results.json').read_text()))
    numbers = json.loads((P / 'elevenlabs-calls.json').read_text())
    for number in numbers:
        number['metrics'] = aggregate(number['calls'], data['metrics'])
    metrics = {}
    for definition in data['metrics']:
        key = definition['key']
        eligible = [number['metrics'][key] for number in numbers if number['metrics'][key]['value'] is not None]
        metrics[key] = {
            'value': sum(metric['value'] for metric in eligible) / len(eligible) if eligible else None,
            'normalized': sum(metric['normalized'] for metric in eligible) / len(eligible) if eligible and key != 'latency' else None,
            'scored_calls': sum(metric['scored_calls'] for metric in eligible),
            'eligible_numbers': len(eligible),
            'events': sum(metric['events'] for metric in eligible),
        }
    composite = sum(weight * metrics[key]['normalized'] for key, weight in data['weights'].items() if weight)
    provider = {'provider': 'ElevenLabs', 'phone_number_count': len(numbers), 'selected_calls': sum(len(n['calls']) for n in numbers),
                'composite': composite, 'metrics': metrics, 'numbers': numbers}
    data['providers'] = [p for p in data['providers'] if p['provider'] != 'ElevenLabs'] + [provider]
    data['providers'].sort(key=lambda p: -p['composite'])
    data['snapshot_date'] = '2026-11-27'
    data['focus_provider'] = 'ElevenLabs'
    data['provider_snapshot_dates']['ElevenLabs'] = '2026-11-25'
    data['comparison_note'] = ('ElevenLabs auto-shop agents were tested November 24–25, and Cisco was tested November 27. '
                               'The other 15 providers retain their November 12 snapshots. These are live agents from '
                               'different businesses; the comparison is directional, not a simultaneous retest of every provider.')
    data['elevenlabs_audio_source'] = AUDIO_SOURCE
    data['elevenlabs_execution_lineage'] = json.loads((P / 'elevenlabs-selection.json').read_text())
    data['provider_count'] = len(data['providers'])
    data['phone_number_count'] = sum(p['phone_number_count'] for p in data['providers'])
    data['selected_call_count'] = sum(p['selected_calls'] for p in data['providers'])
    data['audio_model']['scored_calls'] = sum(p['metrics']['audio']['scored_calls'] for p in data['providers'])
    return data


if __name__ == '__main__':
    data = build()
    (P / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    row = next(p for p in data['providers'] if p['provider'] == 'ElevenLabs')
    print(f"ElevenLabs {row['composite']:.5f}/100; {row['selected_calls']} calls, {row['phone_number_count']} numbers.")
