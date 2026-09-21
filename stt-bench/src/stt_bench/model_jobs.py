"""Model-specific smoke receipts, portable batch identities and full-set rollups."""
import json
from pathlib import Path
from .data import sha256, write_json
from .preparation import source_identity
from .providers import cost_total
from .score import aggregate_wer, percentiles
from .measurement import summarize_deadlines


def job_identity(dataset, model, full, smoke, config):
    return dict(dataset=dataset, model=model, full_manifest_sha256=sha256(full),
                smoke_manifest_sha256=sha256(smoke), config_sha256=sha256(config),
                sources=source_identity(), measurement_version=4)


def validate_smoke(receipt, identity):
    if not receipt.get('passed') or receipt.get('identity') != identity:
        raise ValueError('Smoke must pass for this exact model, configuration, dataset and code')
    report = Path(receipt['report'])
    if not report.exists() or sha256(report) != receipt['report_sha256']:
        raise ValueError('Smoke report is missing or changed')
    r = json.loads(report.read_text())
    from .benchmark import smoke_passed
    if not smoke_passed(r) or r['manifest_sha256'] != identity['smoke_manifest_sha256']:
        raise ValueError('Smoke receipt does not match its report')


def rollup(state, root, full):
    reports = []
    for batch in state['completed_batches']:
        path = Path(batch['report'])
        if sha256(path) != batch['sha256']:
            raise ValueError('Completed batch report changed')
        reports.append(json.loads(path.read_text()))
    rows = [r for report in reports for r in report['clips']]
    ids = [r['clip_id'] for r in rows]
    expected = [c['clip_id'] for c in full['clips']]
    if ids != expected[:len(ids)]:
        raise ValueError('Batch coverage is duplicated, missing or out of order')
    good = [r for r in rows if r['accuracy_usable']]
    attempts = [a for r in rows for a in r['attempts']]
    summary = dict(schema_version=4, measurement_version=4, status=state['status'],
        model=state['identity']['model'], identity=state['identity'], planned_clips=len(expected),
        completed_clips=len(rows), scored_clips=len(good), full_coverage=ids == expected,
        eventual_word_errors=aggregate_wer([r['word_errors'] for r in good]),
        deadlines=summarize_deadlines(rows) if rows else [],
        completion_latency=percentiles([r['completion_latency_ms'] for r in good if r['completion_latency_ms'] is not None]),
        finalize_latency=percentiles([r['finalize_latency_ms'] for r in good if r['finalize_latency_ms'] is not None]),
        failures=sum(not a['valid'] for a in attempts), retries=sum(a['attempt'] > 1 for a in attempts),
        estimated_cost_usd=cost_total(g['estimated_cost_usd'] for report in reports for g in report['results']),
        smoke_cost_included=False, provider_contract=reports[0]['provider_contract'] if reports else None,
        normalization=reports[0]['normalization'] if reports else None,
        shared_measurement_contract=reports[0]['shared_measurement_contract'] if reports else None,
        sessions=[b['session_id'] for b in state['completed_batches']], clips=rows)
    write_json(root / 'summary.json', summary)
    (root / 'summary.md').write_text(f"# {summary['model']}\n\n{len(rows)}/{len(expected)} clips completed; {len(good)} scored.\n\n"
        f"Word errors: {json.dumps(summary['eventual_word_errors'])}\n\n"
        'Completion and finalize timings have provider-specific meanings. See summary.json.\n')
    return summary


def compare_summaries(paths, out):
    summaries = [json.loads(Path(p).read_text()) for p in paths]
    if not summaries:
        raise ValueError('Supply at least one model summary')
    first = summaries[0]
    models = set()
    for s in summaries:
        if s['model'] in models:
            raise ValueError('Duplicate model summary')
        models.add(s['model'])
        if s.get('measurement_version') != 4 or s['identity']['full_manifest_sha256'] != first['identity']['full_manifest_sha256']:
            raise ValueError('Comparison requires the same frozen input and v4 contract')
        if s['normalization'] != first['normalization'] or s['shared_measurement_contract'] != first['shared_measurement_contract']:
            raise ValueError('Shared measurement or normalization mismatch')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    rows = [{k: v for k, v in s.items() if k != 'clips'} for s in summaries]
    write_json(out / 'comparison.json', {'models': rows, 'note': 'Provider-specific completion diagnostics are not interchangeable. Partial coverage remains explicit; no statistical ranking claimed.'})
    def fmt(v):
        return 'unavailable' if v is None else f'{v:.4f}'
    lines = ['# Model comparison', '', '| Model | Completed / planned | Scored | WER | Failures | Retries | Estimated USD |',
             '|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['model']} | {s['completed_clips']}/{s['planned_clips']} | {s['scored_clips']} | {fmt(s['eventual_word_errors']['wer'])} | {s['failures']} | {s['retries']} | {fmt(s['estimated_cost_usd'])} |")
    lines += ['', 'Deadline accuracy, timing sample counts and provider completion rules are in comparison.json.',
              'Incomplete datasets and unknown costs are explicitly retained. Smoke costs are separate.', '']
    (out / 'comparison.md').write_text('\n'.join(lines))
