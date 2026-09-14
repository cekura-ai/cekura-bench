"""Import historical Nova-3 evidence by re-scoring and checking every saved outcome.

This is offline only. It cannot resume a provider request or alter old artifacts.
"""
import json
from pathlib import Path
from .catalog import dataset_definition, model_config
from .data import sha256, write_json
from .huggingface_data import verify_prepared
from .model_jobs import job_identity, rollup
from .report import build_report

KNOWN_CAPTURE = {
    'deepgram.py': 'd285fbf9775574d312c5174a570a093f338decfddbf1672306312eaee097774a',
    'streaming.py': 'b6bc347cffa8b0a24aac45a529871c1afa09e85c86c82400201b8415c2dd4231',
    'timing.py': '9bf24aa19623fe5c461fab9c76443cc84744bdac1c0427a55d5f152729f563dc',
}


def safe_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Evidence path leaves the supplied root')
    return path


def import_baseline(evidence_root, batch_state, out):
    root, out = Path(evidence_root), Path(out)
    old = json.loads(Path(batch_state).read_text())
    dataset, model = 'pipecat-stt-benchmark', 'deepgram-nova-3'
    data = verify_prepared(dataset_definition(dataset))
    full_path, smoke_path = data / 'full/manifest.json', data / 'smoke/manifest.json'
    config = model_config(model)
    if old['identity']['manifest'] != sha256(full_path) or old['identity']['config'] != sha256(config):
        raise ValueError('Baseline dataset or model configuration mismatch')
    if any(old['identity']['source'].get(k) != v for k, v in KNOWN_CAPTURE.items()):
        raise ValueError('Unreviewed historical capture implementation')
    out.mkdir(parents=True, exist_ok=False)
    state = {'identity': job_identity(dataset, model, full_path, smoke_path, config),
             'status': old['status'], 'completed_batches': []}
    compare_fields = ('clip_id', 'reference', 'transcript', 'accuracy_usable', 'word_errors', 'deadlines',
                      'selected_attempt', 'finalize_latency_ms', 'completion_latency_ms')
    for b in old['completed_batches']:
        report_path = safe_path(root, b['report'])
        if sha256(report_path) != b['sha256']:
            raise ValueError('Original baseline batch report changed')
        original = json.loads(report_path.read_text())
        if original.get('measurement_version') != 3 or any(
                original['capture_protocol']['source_hashes'].get(k) != v for k, v in KNOWN_CAPTURE.items()):
            raise ValueError('Baseline batch capture mismatch')
        rel = Path(b['report'])
        if rel.parts[0] != 'reports' or rel.parts[-1] != 'results.json':
            raise ValueError('Unexpected baseline report location')
        run_root = safe_path(root, Path('runs', *rel.parts[1:-1]))
        dest = out / f"batch-{b['index']:04d}"
        fresh = build_report(run_root, dest)
        if [{k: r.get(k) for k in compare_fields} for r in original['clips']] != [
                {k: r.get(k) for k in compare_fields} for r in fresh['clips']]:
            raise ValueError('Rescoring changed historical outcomes; baseline comparison blocked')
        state['completed_batches'].append({**b, 'report': str(dest / 'results.json'),
                                           'sha256': sha256(dest / 'results.json')})
    write_json(out / 'batch-state.json', state)
    result = rollup(state, out, json.loads(full_path.read_text()))
    result.update(capture_measurement_version=3, compatibility='verified_by_exact_offline_outcome_comparison',
                  original_batch_state_sha256=sha256(Path(batch_state)), original_capture=KNOWN_CAPTURE)
    write_json(out / 'summary.json', result)
    return result
