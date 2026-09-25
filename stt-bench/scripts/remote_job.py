"""Run the existing qualification and benchmark on Linux; never provision compute.

Run from the extracted repository root after `uv sync --locked`.
Default mode makes no transcription calls. --live enables smoke then full.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile

from stt_bench.catalog import dataset_definition, model_config
from stt_bench.credentials import command_environment
from stt_bench.data import sha256, write_json
from stt_bench.huggingface_data import verify_prepared


def run_job(args):
    if platform.system() != 'Linux':
        raise ValueError('This entry point requires Linux remote compute')
    for name in ('run_id', 'provider', 'region', 'instance'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', getattr(args, name)):
            raise ValueError(f'{name} must contain only letters, digits, underscores or hyphens')
    definition = dataset_definition(args.dataset)
    model_config(args.model)
    verify_prepared(definition)
    config_path = model_config(args.model)
    provider = json.loads(config_path.read_text())['provider'] if config_path else 'deepgram'
    selected_env = command_environment(provider) if args.live else {}
    for base in ('runs', 'reports'):
        if (Path(base) / args.dataset / args.model / args.run_id).exists():
            raise ValueError('Use a new run ID; this remote entry point never overwrites or restarts a workflow')
    out = Path('reports/remote-jobs') / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, STT_BENCH_COMPUTE_PROVIDER=args.provider,
               STT_BENCH_COMPUTE_REGION=args.region, STT_BENCH_COMPUTE_INSTANCE=args.instance)
    env.update(selected_env)
    state = dict(run_id=args.run_id, dataset=args.dataset, model=args.model,
                 compute=dict(provider=args.provider, region=args.region, instance=args.instance,
                              provenance='Supplied by launcher; compare with cloud control-plane record'),
                 started_at=datetime.now(timezone.utc).isoformat(), live=args.live,
                 lock_sha256=sha256(Path('uv.lock')), status='starting', stages=[])
    commands = [
        ('regression', [sys.executable, '-m', 'pytest', '-q']),
        ('qualification', [sys.executable, 'scripts/validate_harness.py', '--out', str(out / 'qualification')]),
    ]
    if args.live:
        commands.append(('benchmark', [sys.executable, '-m', 'stt_bench.cli', 'benchmark',
                         '--dataset', args.dataset, '--model', args.model, '--run-id', args.run_id]))
    write_json(out / 'job.json', state)
    try:
        for name, command in commands:
            state['status'] = name
            write_json(out / 'job.json', state)
            print(f'{name}: output in {out / (name + ".log")}', flush=True)
            with (out / (name + '.log')).open('x') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['stages'].append(dict(name=name, exit_code=result.returncode))
            if result.returncode:
                state.update(status='failed', failed_stage=name)
                return result.returncode
        state['status'] = 'benchmark_complete' if args.live else 'qualified_no_transcription_calls'
        return 0
    except BaseException as exc:
        state.update(status='interrupted_or_error', error=type(exc).__name__)
        raise
    finally:
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(out / 'job.json', state)
        # Export only this job. Credentials, other runs and historical data are excluded.
        # The archive is local to the remote VM: download it before deleting compute.
        archive = out / 'artifacts.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p != archive and not p.is_symlink():
                    tar.add(p, arcname=str(p), recursive=False)
            for base in ('runs', 'reports'):
                root = Path(base) / args.dataset / args.model / args.run_id
                if root.exists():
                    tar.add(root, arcname=str(root))
        (out / 'artifacts.sha256').write_text(sha256(archive) + '  artifacts.tar.gz\n')
        print(f"{state['status']}: {out / 'job.json'}; download {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--instance', required=True, help='Actual sandbox name or VM ID')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--dataset', default='pipecat-stt-benchmark')
    parser.add_argument('--model', default='deepgram-nova-3')
    parser.add_argument('--live', action='store_true')
    return run_job(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
