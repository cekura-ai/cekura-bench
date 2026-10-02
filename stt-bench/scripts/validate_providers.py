"""Run local regressions and record a source-bound preparation receipt. No API calls."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from stt_bench.data import write_json
from stt_bench.preparation import source_identity


def main():
    root = Path('reports/provider-preparation')
    root.mkdir(parents=True, exist_ok=True)
    before = source_identity()
    commands = [([sys.executable, '-m', 'pytest', '-q'], 'pytest.log'),
                (['node', '--test', 'tests/vercel_models.test.mjs', 'tests/vercel_command_wait.test.mjs'], 'node-tests.log'),
                (['node', '--check', 'scripts/vercel_models.mjs'], 'node-syntax.log'),
                (['git', 'diff', '--check'], 'diff-check.log')]
    results = []
    for command, name in commands:
        with (root / name).open('w') as log:
            p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        results.append({'command': command, 'exit_code': p.returncode, 'log': str(root / name)})
    after = source_identity()
    passed = all(r['exit_code'] == 0 for r in results) and before == after
    write_json(root / 'validation.json', {'passed': passed, 'source_identity': after,
        'source_unchanged_during_tests': before == after, 'completed_at': datetime.now(timezone.utc).isoformat(),
        'tests': results, 'provider_calls': 0, 'validation_kind': 'local fixtures and loopback WebSocket servers'})
    print('Local preparation validation passed' if passed else 'Validation failed; inspect reports/provider-preparation')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
