"""Local-only model inventory, dataset checks and evidence-bound readiness."""
import json
from pathlib import Path
from .catalog import MODELS, BLOCKED_MODELS, model_config, dataset_definition
from .credentials import credential, NAMES
from .data import sha256, write_json
from .huggingface_data import verify_prepared
from .providers import validate, sample_rate

VALIDATION = Path('reports/provider-preparation/validation.json')


def source_identity():
    paths = [Path('pyproject.toml'), Path('uv.lock')]
    for directory, patterns in [('src', ('*.py',)), ('scripts', ('*.py', '*.mjs')),
                                ('tests', ('*.py', '*.mjs')), ('config', ('*.json',))]:
        for pattern in patterns:
            paths.extend(Path(directory).rglob(pattern))
    return {str(p): sha256(p) for p in sorted(set(paths))}


def models():
    rows = []
    for name in MODELS:
        config = json.loads(model_config(name).read_text())
        validate(config)
        key, source = credential(config['provider'])
        rows.append(dict(model_id=name, provider=config['provider'], api_model=config['model'],
            version=config['version'], credential_present=bool(key), credential_variable=source,
            canonical_variable=NAMES[config['provider']][0], sample_rate=sample_rate(config),
            identity_policy=config.get('identity_policy', 'exact_version_and_uuid'),
            finalization=config.get('finalization', 'manual_at_speech_end'),
            completion_basis=config.get('completion_basis', 'close_stream_metadata'),
            supported_measurements={'deadline_accuracy': True, 'eventual_accuracy': True,
                'completion_diagnostic': True, 'finalize_ack_diagnostic': config.get('finalization') != 'stream_end_after_tail'
                    and config.get('finalize_ack_supported', True)},
            pricing=config['pricing'], live_verification='pending',
            missing_prerequisites=[] if key else [NAMES[config['provider']][0]]))
    rows.extend(dict(model_id=name, missing_prerequisites=[reason], live_verification='blocked')
                for name, reason in BLOCKED_MODELS.items())
    return rows


def check(dataset='pipecat-stt-benchmark', out=None):
    definition = dataset_definition(dataset)
    root = verify_prepared(definition)
    inputs = {}
    for subset in ('smoke', 'full'):
        p = root / subset / 'manifest.json'
        m = json.loads(p.read_text())
        inputs[subset] = dict(manifest_sha256=sha256(p), clips=len(m['clips']),
                              submitted_seconds=sum(c['submitted_seconds'] for c in m['clips']))
    validation = json.loads(VALIDATION.read_text()) if VALIDATION.exists() else {}
    verified = validation.get('source_identity') == source_identity() and validation.get('passed') is True
    rows = models()
    for row in rows:
        row['local_validation'] = 'passed' if verified and row.get('provider') else 'pending' if row.get('provider') else 'not_implemented'
        row['status'] = 'missing_prerequisites' if row['missing_prerequisites'] else 'locally_validated' if verified else 'local_validation_pending'
    result = dict(dataset=definition, inputs=inputs, models=rows,
                  local_validation='passed' if verified else 'missing_or_stale',
                  validation_path=str(VALIDATION), transcription_calls_made=False,
                  launch_authorized=False, note='Credential presence is not live model access proof.')
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'readiness.json', result)
        lines = ['# STT model readiness', '', 'Local tests only. No provider transcription or deployment performed.', '',
                 '| Model | Local validation | Credentials / prerequisites | Live verification |', '|---|---|---|---|']
        for r in rows:
            lines.append(f"| {r['model_id']} | {r['local_validation']} | {', '.join(r['missing_prerequisites']) or 'configured'} | {r['live_verification']} |")
        lines += ['', f"Dataset: {inputs['full']['clips']} full clips; {inputs['smoke']['clips']} separate smoke clips.",
                  'Unknown account prices remain null. No zero-cost claim is inferred.', '']
        (out / 'readiness.md').write_text('\n'.join(lines))
    return result
