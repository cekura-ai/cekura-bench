"""Reproducible local qualification; every trial is retained, no provider calls."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from stt_bench.data import write_json
from stt_bench.diagnostics import local_probe, validate_preflight


async def main(out):
    out.mkdir(parents=True, exist_ok=False)
    summary = {'provider_calls': 0, 'checks': []}
    cases = [
        ('preflight', dict(seconds=10, repeats=3), True, None),
        ('continuous', dict(seconds=60, repeats=1), True, None),
        ('send-stall', dict(seconds=2, repeats=1, stall_ms=80), False, 'send_gap_above_40ms'),
        ('client-stall', dict(seconds=2, repeats=1, client_stall_ms=80), False, 'message_receipt_delay_above_40ms'),
        ('receiver-stall', dict(seconds=2, repeats=1, receiver_stall_ms=80), False, 'heartbeat_emission_coverage_gap_above_40ms'),
    ]
    for name, options, expected_valid, expected_reason in cases:
        cpu, started = time.process_time(), time.perf_counter()
        report = await local_probe(out / name, **options)
        wall = time.perf_counter() - started
        row = dict(name=name, expected_valid=expected_valid, expected_reason=expected_reason,
                   actual_valid=all(t['valid'] for t in report['trials']),
                   report=str(out / name / 'pacing.json'),
                   parent_cpu_percent_of_one_core=100*(time.process_time()-cpu)/wall,
                   trials=[{k: t.get(k) for k in (
                       'valid', 'send_pacing_valid', 'client_delivery_valid', 'fixture_valid',
                       'interval_ms_max', 'actual_over_ideal', 'message_receipt_delay_ms_max',
                       'heartbeat_send_lateness_ms_max', 'heartbeat_emission_coverage_gap_ms_max', 'gate_reasons')}
                       for t in report['trials']])
        row['passed'] = row['actual_valid'] == expected_valid and (expected_reason is None or
            all(expected_reason in t['gate_reasons'] for t in report['trials']))
        summary['checks'].append(row)
        write_json(out / 'summary.json', summary)
        print(json.dumps(row), flush=True)
    try:
        validate_preflight(out / 'preflight' / 'pacing.json')
        summary['preflight_accepted'] = True
    except ValueError as exc:
        summary.update(preflight_accepted=False, preflight_error=str(exc))
    summary['qualified'] = summary['preflight_accepted'] and all(c['passed'] for c in summary['checks'])
    write_json(out / 'summary.json', summary)
    return summary['qualified']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(main(args.out)) else 1)
