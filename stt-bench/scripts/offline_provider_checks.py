"""Run provider regression tests with Python network access limited to loopback."""
import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sys

TESTS = ['tests/test_reson8.py', 'tests/test_gradium.py', 'tests/test_trial_providers.py', 'tests/test_providers.py', 'tests/test_chirp.py',
         'tests/test_pacing_integrity.py', 'tests/test_measurement_v3.py',
         'tests/test_model_jobs.py', 'tests/test_pipecat.py', 'tests/test_benchmark.py']


def network_guard(event, args):
    if event == 'socket.connect':
        address = args[1]
        # Unix sockets stay local. TCP/UDP clients may only reach loopback.
        if isinstance(address, tuple):
            try:
                permitted = ipaddress.ip_address(address[0]).is_loopback
            except ValueError:
                permitted = address[0] == 'localhost'
            if not permitted:
                raise RuntimeError('Offline checks prohibit external socket connections')
    elif event == 'socket.getaddrinfo' and args[0] not in (None, 'localhost', '127.0.0.1', '::1'):
        raise RuntimeError('Offline checks prohibit external DNS lookups')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from stt_bench.credentials import NAMES
    from stt_bench.preparation import source_identity
    from stt_bench.catalog import dataset_definition
    from stt_bench.huggingface_data import verify_prepared
    payload = json.loads(args.payload.read_text()) if args.payload else None
    if payload:
        assert not Path('.env').exists() and not Path('.secrets').exists()
        assert not any(os.environ.get(k) for names in NAMES.values() for k in names)
        for name, expected in payload['hashes'].items():
            path = Path(name)
            assert not path.is_symlink() and path.resolve().is_relative_to(Path.cwd())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        with Path('input.tar.gz').open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == payload['baselineBundleHash']
    before = source_identity()
    sys.addaudithook(network_guard)
    verify_prepared(dataset_definition('pipecat-stt-benchmark'))
    import pytest
    exit_code = pytest.main(['-q', '--tb=short', *TESTS])
    after = source_identity()
    passed = exit_code == 0 and before == after
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(passed=passed, source_identity=after,
        source_unchanged=before == after, validation_kind='offline fixtures and loopback only',
        provider_calls=0, live_access='untested', dataset_verified=True,
        completed_at=datetime.now(timezone.utc).isoformat(), test_exit_code=int(exit_code)), indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
