"""Export aggregate scores/coverage only; no call or phone identifiers."""
import argparse
import copy
import json
import pathlib

P = pathlib.Path(__file__).resolve().parent


def export(previous, commit, pull_request):
    source = json.loads((P / 'results.json').read_text())
    data = copy.deepcopy(previous)
    for key in ('snapshot_date', 'provider_count', 'phone_number_count', 'selected_call_count', 'audio_model', 'comparison_note', 'provider_snapshot_dates', 'elevenlabs_audio_source'):
        data[key] = source[key]
    provider = next(p for p in source['providers'] if p['provider'] == 'ElevenLabs')
    row = {k: copy.deepcopy(v) for k, v in provider.items() if k != 'numbers'}
    row['numbers'] = [{'selected_calls': len(n['calls']), 'metrics': n['metrics']} for n in provider['numbers']]
    data['providers'] = [p for p in data['providers'] if p['provider'] != 'ElevenLabs'] + [row]
    data['providers'].sort(key=lambda p: -p['composite'])
    data['source'] = {'repository': 'cekura-ai/cekura-bench', 'path': 'results/production-voice-quality/2026-11-27-elevenlabs/results.json',
                      'pull_request': pull_request, 'commit': commit}
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--pull-request', type=int, required=True)
    args = parser.parse_args()
    data = export(json.loads(args.previous.read_text()), args.commit, args.pull_request)
    args.output.write_text(json.dumps(data, indent=1, allow_nan=False) + '\n')
