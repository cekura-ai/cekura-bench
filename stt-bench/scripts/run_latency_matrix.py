"""Balanced, diagnostic-only controls; all outcomes retained, no provider calls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    modes = [('disk-observed', ['--observe']), ('memory-observed', ['--observe', '--memory-log']),
             ('disk-interactive', ['--observe', '--interactive-qos']), ('disk-unobserved', [])]
    plan = [(repeat, mode, flags) for repeat in range(3)
            for mode, flags in (modes if repeat % 2 == 0 else list(reversed(modes)))]
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=2))
    rows = []
    for repeat, mode, flags in plan:
        out = args.out / f'{repeat}-{mode}'
        command = [sys.executable, 'scripts/diagnose_wait_boundary.py', '--library', str(args.library),
                   '--out', str(out), '--seconds', '10', '--repeats', '1', '--sparse', '--stock-selector', *flags]
        started = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        row = dict(repeat=repeat, mode=mode, started_at_unix=started, exit_code=result.returncode,
                   stdout=result.stdout, stderr=result.stderr)
        if result.returncode == 0:
            evidence = json.loads((out / 'native-wait-evidence.json').read_text())
            report = json.loads((out / 'pacing.json').read_text())
            row.update(trial=report['trials'][0], policy=evidence['policy_after'])
        rows.append(row)
        (args.out / 'results.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(dict(repeat=repeat, mode=mode, exit_code=result.returncode,
                             send_gap_ms=row.get('trial', {}).get('interval_ms_max'),
                             reasons=row.get('trial', {}).get('gate_reasons'), policy=row.get('policy'))), flush=True)
    return 0 if all(row['exit_code'] == 0 for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
