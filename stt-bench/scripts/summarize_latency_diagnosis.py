"""Check the saved causal controls and create a compact evidence inventory."""
import json
from pathlib import Path

from stt_bench.data import sha256, write_json


def main():
    root = Path('reports')
    checks, sources = [], []
    def read(path):
        sources.append(dict(path=str(path), sha256=sha256(path)))
        return json.loads(path.read_text())
    for name, expected_state in [('loop-cpu', 1), ('gil', 3)]:
        data = read(root / f'latency-control-{name}/native-wait-evidence.json')
        injected = data['injections'][0]
        callback = max((c for c in data['callbacks'] if c['start'] <= injected['start'] and c['end'] >= injected['end']),
                       key=lambda c: c['end']-c['start'])
        samples = [s for s in data['thread_samples'] if injected['start'] <= s['before'] <= injected['end']]
        state_fraction = sum(s['state'] == expected_state for s in samples) / len(samples)
        cpu_ms = callback['cpu_seconds'] * 1000
        passed = state_fraction >= .8 and (cpu_ms > 60 if name == 'loop-cpu' else cpu_ms < 10)
        checks.append(dict(control=name, passed=passed, callback_cpu_ms=cpu_ms,
                           callback_wall_ms=1000*(callback['end']-callback['start']),
                           samples=len(samples), expected_state_fraction=state_fraction))
        assert passed, f'Known {name} fault was not distinguished by the measurement'
    data = read(root / 'latency-sparse-stock-disk/native-wait-evidence.json')
    timer = max(data['timers'], key=lambda t: t['resumed']-t['deadline'])
    callback = next(c for c in data['callbacks'] if c['callback'] == 'MacOSDeadlineTimer._ready'
                    and c['start'] <= timer['observed'] <= c['end'])
    assert timer['observed']-timer['deadline'] < .001
    assert callback['cpu_seconds'] < .001 and callback['end']-callback['start'] > .050
    anchor = dict(timer=timer, callback=callback,
                  timer_observation_late_ms=1000*(timer['observed']-timer['deadline']),
                  resumption_after_observation_ms=1000*(timer['resumed']-timer['observed']),
                  callback_cpu_ms=1000*callback['cpu_seconds'],
                  callback_wall_ms=1000*(callback['end']-callback['start']))
    memory = read(root / 'latency-memory-no-gc-60s/native-wait-evidence.json')
    report = read(root / 'latency-memory-no-gc-60s/pacing.json')
    assert not memory['gc']
    assert 'send_gap_above_40ms' in report['trials'][0]['gate_reasons']
    matrix = read(root / 'latency-controls-20260912/results.json')
    native = read(root / 'latency-native-no-python-60s.json')
    native_summary = [{k: v for k, v in trial.items() if k != 'observations'} for trial in native['trials']]
    result = dict(controls=checks, representative_delay=anchor,
                  logging_and_gc_disabled=report['trials'][0],
                  matrix=[dict(mode=r['mode'], repeat=r['repeat'], valid=r['trial']['valid'],
                               max_send_gap_ms=r['trial']['interval_ms_max'], policy=r['policy']) for r in matrix],
                  native_reference=native_summary, evidence=sources,
                  interpretation='Delayed process execution is supported; no claim that CPU capacity, a specific app, disk pressure, or a specific kernel policy caused it. Sampled thread states are not continuous scheduler traces.')
    write_json(root / 'latency-diagnosis-verified.json', result)
    print(json.dumps(dict(controls=checks, representative_delay=anchor, native_reference=native_summary), indent=2))


if __name__ == '__main__':
    main()
