"""Offline reports from immutable attempts, including failed attempts and coverage."""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path

from .data import sha256, write_json
from .providers import reduce_events, is_nova, sample_rate
from .run import assess, select_attempt
from .score import ENTITY_TYPES, NORMALIZATION, aggregate_wer, entity_errors, percentiles, plot, word_errors
from .streaming import pacing_metrics, read_events
from .entities import aggregate_values, value_errors
from .measurement import deadline_observations, grouped_interval, summarize_deadlines
from .review import load_review
from .catalog import dataset_identity


def include_complete_text(attempt, events, config=None):
    """Score every final text segment, while preserving the Finalize snapshot.

    CloseStream sometimes returns additional words even after a Finalize ack.
    Those words belong to accuracy, but must not replace the ack latency.
    """
    if config is not None and not is_nova(config):
        return attempt
    segments, seen, late = [], set(), []
    last_at = None
    ack = attempt['finalize_ack_received_seconds']
    for event in events:
        message = event.get('message', {})
        if message.get('type') != 'Results' or not message.get('is_final'):
            continue
        alternatives = message.get('channel', {}).get('alternatives', [])
        text = alternatives[0].get('transcript', '') if alternatives else ''
        key = (tuple(message.get('channel_index', [])), message.get('start'), message.get('duration'), text)
        if key in seen or not text:
            continue
        seen.add(key)
        segments.append((message.get('start', 0), text))
        last_at = event['time_seconds']
        if ack is not None and last_at > ack:
            late.append({'text': text, 'received_seconds': last_at})
    return {**attempt, 'transcript_at_finalize': attempt['transcript'] if ack is not None else None,
            'transcript': ' '.join(text for _, text in sorted(segments, key=lambda s: s[0])),
            'final_transcript_received_seconds': last_at, 'additional_final_segments_after_ack': late}


def attempts_for_clip(root, clip, run, legacy):
    cid, config = clip['clip_id'], run['config']
    attempts = []
    if run.get('schema_version', 1) == 2:
        for number in (1, 2):
            raw = root / 'raw' / f'{cid}--attempt-{number}.jsonl'
            meta = root / 'attempts' / f'{cid}--attempt-{number}.json'
            if not raw.exists():
                if meta.exists():
                    raise ValueError(f'Missing saved raw evidence: {raw.name}')
                continue
            digest = sha256(raw)
            if meta.exists() and json.loads(meta.read_text())['raw_sha256'] != digest:
                raise ValueError(f'Raw evidence changed: {raw.name}')
            events = read_events(raw, allow_truncated_final=True)
            attempts.append({'attempt': number, 'raw_sha256': digest, 'raw_file': 'raw/' + raw.name,
                             **assess(events, config, run['mode'] == 'dry_run'),
                             'sent_audio_seconds': sum(e.get('bytes', 0) for e in events if e['kind'] == 'audio_sent') / (sample_rate(config) * 2)})
    else:
        raw = root / 'raw' / f'{cid}.jsonl'
        if cid in legacy and (not raw.exists() or sha256(raw) != legacy[cid]['raw_sha256']):
            raise ValueError(f'Raw evidence changed or missing: {raw.name}')
        if raw.exists():
            events = read_events(raw)
            reduced, pacing = reduce_events(events, config), pacing_metrics(events)
            reasons = []
            if not reduced['transcript_complete']:
                reasons.append('incomplete_transcript')
            if not reduced['model_verified']:
                reasons.append('model_not_verified')
            if not pacing['valid']:
                reasons.append('invalid_pacing')
            attempts.append({'attempt': 1, 'raw_sha256': sha256(raw), 'raw_file': 'raw/' + raw.name,
                             **reduced, 'pacing': pacing, 'valid': not reasons, 'exclusion_reasons': reasons,
                             'sent_audio_seconds': sum(e.get('bytes', 0) for e in events if e['kind'] == 'audio_sent') / (sample_rate(config) * 2)})
    return [include_complete_text(a, read_events(root / a['raw_file'], allow_truncated_final=True), config) for a in attempts]


def build_report(root: Path, out: Path, review_path: Path | None = None):
    run = json.loads((root / 'run.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    review_rows, review_status = load_review(root / 'manifest.json', review_path)
    if sha256(root / 'manifest.json') != run['manifest_sha256']:
        raise ValueError('Run manifest changed')
    for package, expected in {'jiwer': '4.0.0', 'whisper-normalizer': '0.1.12'}.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'Expected {package}=={expected}; run uv sync --locked')
    if manifest.get('annotations_sha256') and sha256(root / 'annotations.json') != manifest['annotations_sha256']:
        raise ValueError('Run annotations changed')
    outcome_path = root / 'outcomes.json'
    legacy = {r['clip_id']: r for r in json.loads(outcome_path.read_text())} if outcome_path.exists() else {}
    config = run['config']
    rows = []
    for clip in manifest['clips']:
        attempts = attempts_for_clip(root, clip, run, legacy)
        selected = select_attempt(attempts) if attempts else {
            **reduce_events([], config), 'attempt': None, 'pacing': {'valid': False},
            'exclusion_reasons': ['not_attempted'], 'valid': False, 'sent_audio_seconds': 0}
        usable = (run['mode'] == 'live' and selected['transcript_complete'] and selected['model_verified']
                  and selected['pacing']['valid'] and 'interrupted_attempt' not in selected['exclusion_reasons'])
        first = next((a for a in attempts if a['attempt'] == 1), None)
        first_events = read_events(root / first['raw_file'], allow_truncated_final=True) if first else []
        deadlines = deadline_observations(first_events, clip['reference'], clip.get('entities'), first, run['mode'], config)
        rows.append({'clip_id': clip['clip_id'], 'condition': clip['condition'],
                     'dependency_group': clip.get('dependency_group'), 'human_review_verified': False,
                     **review_rows.get(clip['clip_id'], {}),
                     'deadlines': deadlines,
                     'reference': clip['reference'], 'submitted_seconds': clip['submitted_seconds'],
                     'planned_entities': clip.get('entities'), 'boundary_review': clip.get('boundary_review'),
                     **selected, 'selected_attempt': selected['attempt'], 'attempts': attempts,
                     'accuracy_usable': usable,
                     'word_errors': word_errors(clip['reference'], selected['transcript']) if usable else None,
                     'entity_values': value_errors(clip['reference'], selected['transcript'], clip.get('entities')) if usable else None,
                     'entities': entity_errors(clip['reference'], selected['transcript'], clip.get('entities')) if usable else None})
    groups = []
    for condition in sorted({c['condition'] for c in rows}):
        all_rows = [r for r in rows if r['condition'] == condition]
        good = [r for r in all_rows if r['accuracy_usable']]
        timed = [r for r in good if r['finalize_latency_ms'] is not None]
        partials = [r for r in good if r['first_partial_after_t0'] is not None]
        per_type = {}
        for kind in sorted(ENTITY_TYPES):
            available = sum(sum(e['type'] == kind for e in (r['planned_entities'] or [])) for r in all_rows)
            annotations = [e for r in good for e in r['entities']['details'] if e['type'] == kind]
            errors = sum(e['incorrect'] for e in annotations)
            per_type[kind] = {'dataset_entities': available, 'reference_entities': len(annotations), 'errors': errors,
                              'error_rate': errors / len(annotations) if annotations else None,
                              'status': 'measured' if annotations else 'no_scored_examples' if available else 'no_examples_available'}
        entity_n = sum(v['reference_entities'] for v in per_type.values())
        entity_wrong = sum(v['errors'] for v in per_type.values())
        attempts = [a for r in all_rows for a in r['attempts']]
        initial_seconds = sum(a['sent_audio_seconds'] for a in attempts if a['attempt'] == 1)
        retry_seconds = sum(a['sent_audio_seconds'] for a in attempts if a['attempt'] > 1)
        price = config['pricing']['usd_per_minute'] if run['mode'] == 'live' else 0
        groups.append({'provider': config['provider'], 'model': config['model'], 'version': config['version'],
                       'deadlines': summarize_deadlines(all_rows),
                       'eventual_wer_interval': grouped_interval(good, lambda r: r['word_errors']),
                       'entity_values': aggregate_values([r['entity_values'] for r in good]),
                       'completion_latency': percentiles([r['completion_latency_ms'] for r in good if r.get('completion_latency_ms') is not None]),
                       'reliability': {'planned_clips': len(all_rows),
                           'first_attempt_success': sum(bool(r['attempts'] and r['attempts'][0]['valid']) for r in all_rows),
                           'first_attempt_transport_failures': sum(bool(r['attempts'] and 'transport_or_provider_error' in r['attempts'][0]['exclusion_reasons']) for r in all_rows),
                           'first_attempt_pacing_failures': sum(bool(r['attempts'] and not r['attempts'][0]['pacing']['valid']) for r in all_rows),
                           'first_attempt_missing_ack': sum(bool(r['attempts'] and r['attempts'][0]['finalize_latency_ms'] is None
                               and r['attempts'][0].get('finalize_latency_status') != 'unsupported') for r in all_rows),
                           'first_attempt_completion_timeouts': sum(bool(r['attempts'] and r['attempts'][0].get('completion_timed_out')) for r in all_rows),
                           'first_attempt_interruptions': sum(bool(r['attempts'] and 'interrupted_attempt' in r['attempts'][0]['exclusion_reasons']) for r in all_rows),
                           'recovered_on_retry': sum(r['selected_attempt'] == 2 and r['valid'] for r in all_rows)},
                       'condition': condition, 'planned_clips': len(all_rows),
                       'attempted_clips': sum(bool(r['attempts']) for r in all_rows),
                       'scored_clips': len(good), 'excluded_clips': len(all_rows) - len(good),
                       'pacing_valid_clips': sum(r['pacing']['valid'] for r in all_rows),
                       'attempt_count': len(attempts), 'retry_count': sum(a['attempt'] > 1 for a in attempts),
                       'failed_or_invalid_attempts': sum(not a['valid'] for a in attempts),
                       'scored_clips_with_additional_final_text_after_ack': sum(bool(r.get('additional_final_segments_after_ack')) for r in good),
                       'exclusion_reason_counts': dict(Counter(reason for r in all_rows for reason in r['exclusion_reasons'])),
                       'all_attempt_failure_reasons': dict(Counter(reason for a in attempts for reason in a['exclusion_reasons'])),
                       **aggregate_wer([r['word_errors'] for r in good]),
                       'entity_metric_name': 'strict reference-entity error rate',
                       'entity_error_rate': entity_wrong / entity_n if entity_n else None,
                       'entity_errors': entity_wrong, 'reference_entities': entity_n,
                       'entity_annotated_clips': sum(r['planned_entities'] is not None for r in all_rows),
                       'entity_reviewed_without_entities_clips': sum(r['planned_entities'] == [] for r in all_rows),
                       'entity_unreviewed_clips': sum(r['planned_entities'] is None for r in all_rows),
                       'entity_scored_clips': sum(r['entities']['status'] == 'annotated' for r in good),
                       'entity_by_type': per_type,
                       'finalize_latency': {**percentiles([r['finalize_latency_ms'] for r in timed]),
                                            'missing_among_scored': len(good) - len(timed),
                                            'unavailable_clips': len(all_rows) - len(timed),
                                            'sample_size_sufficient': len(timed) >= 30,
                                            'headline_eligible': False,
                                            'interpretation': 'Acknowledgment diagnostic; not transcript availability latency',
                                            'full_coverage': len(timed) == len(all_rows)},
                       'first_partial_after_t0': {**percentiles([r['first_partial_after_t0']['latency_ms'] for r in partials]),
                                                 'absent_among_scored': len(good) - len(partials),
                                                 'unscored_clips': len(all_rows) - len(good)},
                       'usd_per_minute': config['pricing']['usd_per_minute'],
                       'initial_sent_audio_seconds': initial_seconds, 'retry_sent_audio_seconds': retry_seconds,
                       'estimated_initial_cost_usd': initial_seconds / 60 * price if price is not None else None,
                       'estimated_retry_cost_usd': retry_seconds / 60 * price if price is not None else None,
                       'estimated_cost_usd': (initial_seconds + retry_seconds) / 60 * price if price is not None else None})
    missing_categories = [kind for kind in sorted(ENTITY_TYPES)
                          if not any(g['entity_by_type'][kind]['dataset_entities'] for g in groups)]
    anchor = next((g for g in groups if g['condition'] == 'public_anchor'), None)
    gaps = []
    private_conditions = (('private_short',) if any(g['condition'] == 'private_short' for g in groups)
                          else ('private_mic', 'private_telephony'))
    for condition in private_conditions:
        private = next((g for g in groups if g['condition'] == condition), None)
        available = bool(anchor and private and anchor['wer'] is not None and private['wer'] is not None)
        gaps.append({'provider': config['provider'], 'condition': condition,
                     'private_minus_public_wer': private['wer'] - anchor['wer'] if available else None,
                     'percentage_points': (private['wer'] - anchor['wer']) * 100 if available else None,
                     'status': 'available' if available else 'private_data_pending'})
    run_complete = all(r['attempts'] and (any(a['valid'] for a in r['attempts']) or len(r['attempts']) >= run.get('max_attempts', 2))
                       for r in rows) if run.get('schema_version', 1) == 2 else all(r['attempts'] for r in rows)
    report = {'schema_version': 4 if run.get('measurement_version') == 4 else 3, 'measurement_version': 4 if run.get('measurement_version') == 4 else 3, 'mode': run['mode'], 'manifest_sha256': run['manifest_sha256'],
              'run_identity': {k: config.get(k) for k in ('model_id', 'provider', 'model', 'endpoint', 'version')},
              'dataset_identity': dataset_identity(manifest),
              'run_started_at': run.get('started_at'), 'run_configuration': config, 'review': review_status,
              'capture_protocol': {'measurement_version': run.get('measurement_version', run.get('schema_version', 1)),
                                   'pacing_runtime': run.get('pacing_runtime'),
                                   'pacing_policy': run.get('pacing_policy'),
                                   'source_hashes': {name: run.get('source_hashes', {}).get(name)
                                                     for name in (tuple(run.get('source_hashes', {})) if run.get('measurement_version') == 4 else ('deepgram.py', 'streaming.py', 'timing.py', 'macos_timer.py', 'macos_activity.py'))}},
              'scoring_source_hashes': {name: sha256(Path(__file__).parent / name)
                                       for name in ('score.py', 'report.py', 'deepgram.py', 'streaming.py', 'run.py', 'measurement.py', 'entities.py', 'review.py', 'providers.py', 'provider_protocol.py')},
              'normalization': dict(NORMALIZATION),
              'entity_comparison': 'Case-sensitive raw characters, whitespace ignored; annotated reference spans; no semantic equivalence. Unrelated hallucinated entities outside reference spans are not measured.',
              'shared_measurement_contract': {'version': 4, 'frame_ms': config.get('frame_ms', 20), 'tail_ms': 1000, 'deadlines_ms': [0, 250, 500, 1000], 'attempt_policy': 'first-request-deadlines-first-valid-recovery', 'reference_manifest': manifest.get('batch', {}).get('parent_manifest_sha256', run['manifest_sha256']), 'scoring_hashes': {name: sha256(Path(__file__).parent / name) for name in ('score.py', 'report.py', 'measurement.py', 'streaming.py', 'timing.py')}},
              'provider_contract': {'finalization': config.get('finalization', 'manual_at_speech_end'), 'completion_basis': config.get('completion_basis', 'close_stream_metadata'), 'sample_rate': sample_rate(config)},
              'pricing': config['pricing'], 'results': groups, 'public_private_gap': gaps,
              'completeness': {'all_requested_metrics_available': bool(run_complete and anchor
                                      and anchor['finalize_latency']['sample_size_sufficient']
                                      and all(g['status'] == 'available' for g in gaps) and not missing_categories
                                      and all(g['wer'] is not None for g in groups)),
                               'run_complete': bool(run_complete),
                               'public_anchor_latency_minimum_met': bool(anchor and anchor['finalize_latency']['sample_size_sufficient']),
                               'private_data_status': 'private data pending' if any(g['status'] != 'available' for g in gaps) else 'available',
                               'unrepresented_entity_categories': missing_categories,
                               'all_planned_clips_have_attempts': all(r['attempts'] for r in rows),
                               'deadline_headline_eligible': all(d['headline_eligible'] for g in groups for d in g['deadlines'])},
              'notes': ['Deadline metrics always use attempt one; successful retries never replace its outcome. Every attempt contributes to estimated cost.',
                        'Deadline totals retain pacing-invalid attempts as diagnostics; the controlled subset is secondary. Neither supports a headline when coverage or listening review is incomplete.',
                        'Requests failing before speech end are empty hypotheses in the first-request availability score; this is a failure penalty, not an observed speech-end timestamp.',
                        'Incomplete observation windows and unsupported transcript overlaps remain unavailable, not silently empty.',
                        'Eventual WER uses the first completed valid recovery attempt; completion timing follows the recorded provider contract and includes the deliberate observation window.',
                        'Entity values use a bounded English grammar; unsupported reference inventories are excluded explicitly. Spurious counts are extractor candidates.',
                        'First valid attempt is selected; no selection by lowest WER or fastest latency.',
                        'Accuracy includes every final text segment through stream closure. Additional text after the Finalize acknowledgment is reported separately; acknowledgment latency is not a guarantee that all final text has arrived.',
                        'Unavailable observations are null, never zero. A zero reference-entity count means no measured denominator.',
                        'Pacing exclusions reduce the evaluated sample. Results describe retained clips; meeting n >= 30 does not establish full frozen-dataset coverage.',
                        manifest.get('preprocessing', {}).get('entity_provenance', 'Entity annotations were reviewed by Codex from supplied reference text, not independently human-verified against audio.'),
                        'Boundaries passed signal checks, not human listening. Source transcripts may contain errors.',
                        'Public/private differences do not prove training-data contamination.'],
              'clips': rows}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'results.json', report)
    write_tables(report, out)
    plot(groups, run['mode'], out / 'deadline-wer.png')
    return report


def number(value, percent=False):
    if value is None:
        return 'unavailable'
    return f'{value * 100:.2f}%' if percent else f'{value:.2f}'


def write_tables(report, out):
    dataset = report.get('dataset_identity', {})
    environment = (report.get('capture_protocol', {}).get('pacing_runtime') or {}).get('runtime', {}).get('execution_environment')
    location = ('Compute: ' + ', '.join(f'{k}={v}' for k, v in environment.items())
                if environment else 'Compute location: not supplied; see recorded host and runtime in results.json.')
    lines = ['# Streaming speech-to-text benchmark', '',
             f"Dataset: {dataset.get('dataset')} | revision `{dataset.get('source_revision')}` | split `{dataset.get('split')}` | subset `{dataset.get('subset') or 'legacy selection'}`.", '',
             'Run complete.' if report['completeness']['run_complete'] else 'RUN IN PROGRESS: these are partial results.', '',
             ('Private excerpts included; references and boundaries are not independently listening-verified.'
              if any(c['condition'].startswith('private') for c in report['clips']) else
              'Private data pending. Missing categories and observations remain unavailable.'), '',
             f"Model: {report['results'][0]['model']} version `{report['results'][0]['version']}`.", '',
             location, '',
             '| Metric | ' + ' | '.join(g['condition'] for g in report['results']) + ' |',
             '|---|' + '---:|' * len(report['results'])]
    metrics = [
        ('Scored / planned clips', lambda g: f"{g['scored_clips']} / {g['planned_clips']}"),
        ('Excluded clips', lambda g: str(g['excluded_clips'])),
        ('Eventual corpus WER (recovery allowed)', lambda g: number(g['wer'], True)),
        ('S / I / D', lambda g: f"{g['substitutions']} / {g['insertions']} / {g['deletions']}"),
        ('Reference words', lambda g: str(g['reference_words'])),
        ('Strict reference-entity error rate', lambda g: number(g['entity_error_rate'], True)),
        ('Incorrect / evaluated entities', lambda g: f"{g['entity_errors']} / {g['reference_entities']}"),
        ('Reviewed clips with no entities / unreviewed clips', lambda g: f"{g['entity_reviewed_without_entities_clips']} / {g['entity_unreviewed_clips']}"),
        ('Acknowledgment p50 / p90 (ms; diagnostic)', lambda g: f"{number(g['finalize_latency']['p50_ms'])} / {number(g['finalize_latency']['p90_ms'])}"),
        ('Stream completion p50 / p90 (ms; includes observation window)', lambda g: f"{number(g['completion_latency']['p50_ms'])} / {number(g['completion_latency']['p90_ms'])}"),
        ('Finalize n / unavailable clips', lambda g: f"{g['finalize_latency']['n']} / {g['finalize_latency']['unavailable_clips']}"),
        ('Finalize n >= 30', lambda g: str(g['finalize_latency']['sample_size_sufficient'])),
        ('Finalize full coverage', lambda g: str(g['finalize_latency']['full_coverage'])),
        ('Scored clips with extra final text after Finalize ack', lambda g: str(g['scored_clips_with_additional_final_text_after_ack'])),
        ('Partial after t=0 p50 / p90 (ms)', lambda g: f"{number(g['first_partial_after_t0']['p50_ms'])} / {number(g['first_partial_after_t0']['p90_ms'])}"),
        ('Partial observed / absent among scored', lambda g: f"{g['first_partial_after_t0']['n']} / {g['first_partial_after_t0']['absent_among_scored']}"),
        ('Partial unscored clips', lambda g: str(g['first_partial_after_t0']['unscored_clips'])),
        ('Streaming list rate ($/min)', lambda g: number(g['usd_per_minute'])),
        ('Initial / retry audio (seconds)', lambda g: f"{g['initial_sent_audio_seconds']:.2f} / {g['retry_sent_audio_seconds']:.2f}"),
        ('Estimated initial / retry cost ($)', lambda g: f"{number(g['estimated_initial_cost_usd'])} / {number(g['estimated_retry_cost_usd'])}"),
        ('Estimated total cost ($)', lambda g: number(g['estimated_cost_usd'])),
        ('Attempts / retries', lambda g: f"{g['attempt_count']} / {g['retry_count']}"),
        ('Failed or invalid attempts', lambda g: str(g['failed_or_invalid_attempts'])),
        ('Private mic / telephony WER gaps', lambda g: 'unavailable / unavailable'),
    ]
    for label, fn in metrics:
        lines.append('| ' + label + ' | ' + ' | '.join(fn(g) for g in report['results']) + ' |')
    lines += ['', '## Original-request accuracy at fixed deadlines', '',
              'All measurable first attempts are included, including pacing-invalid attempts. These are diagnostic results until pacing, coverage and human review pass. Controlled-subset WER is secondary; retries never replace original requests.', '',
              '| Group | Deadline ms | WER | Measured / planned | Pacing invalid | Provisional | Controlled subset WER / n | Headline eligible |',
              '|---|---:|---:|---:|---:|---:|---|---|']
    for g in report['results']:
        for d in g['deadlines']:
            c = d['controlled_subset']
            lines.append(f"| {g['condition']} | {d['deadline_ms']} | {number(d['wer'], True)} | {d['measured_clips']} / {d['planned_clips']} | {d['pacing_invalid_clips']} | {d['provisional_clips']} | {number(c['wer'], True)} / {c['clips']} | {d['headline_eligible']} |")
    lines += ['', '## Reliability on original requests', '',
              '| Group | Planned | Successful | Transport/provider failures | Pacing failures | Missing ack | Recovered on retry |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in report['results']:
        r = g['reliability']
        lines.append(f"| {g['condition']} | {r['planned_clips']} | {r['first_attempt_success']} | {r['first_attempt_transport_failures']} | {r['first_attempt_pacing_failures']} | {r['first_attempt_missing_ack']} | {r['recovered_on_retry']} |")
    lines += ['', '## Entity-value accuracy (bounded English grammar)', '',
              'These eventual-text scores preserve values rather than exact formatting. Unsupported reference inventories are excluded. Spurious counts are unmatched extracted candidates; extraction is not exhaustive. Deadline entity scores are in JSON.', '',
              '| Group | Type | Reference occurrences | Wrong | Missing | Spurious candidates | Unsupported clips | Error rate |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in report['results']:
        for kind, c in g['entity_values']['by_type'].items():
            lines.append(f"| {g['condition']} | {kind} | {c['reference_entities']} | {c['wrong']} | {c['missing']} | {c['spurious']} | {c['unsupported_clips']} | {number(c['error_rate'], True)} |")
    lines += ['', '## Reference review and uncertainty', '',
              f"Review status: {report['review']['status']}; verified {report['review']['verified_clips']} / {report['review']['planned_clips']} clips.", '',
              'Intervals require explicit dependency groups (for example, connected conversations/speakers). Missing groups are not treated as independent clips. Full deadline intervals and status counts are in JSON.', '']
    for g in report['results']:
        lines.append(f"- {g['condition']} eventual WER interval: {json.dumps(g['eventual_wer_interval'], sort_keys=True)}")
    lines += ['', '## Exclusions', '',
              'The table above uses eligible observations only. Raw timings from excluded attempts remain in JSON for auditing.', '']
    for g in report['results']:
        lines += [f"- {g['condition']}: selected-attempt exclusions {json.dumps(g['exclusion_reason_counts'], sort_keys=True)}; all-attempt failures {json.dumps(g['all_attempt_failure_reasons'], sort_keys=True)}."]
    lines += ['', '## Recorded price evidence', '', '```json', json.dumps(report['pricing'], indent=2), '```']
    lines += ['', '## Entity coverage', '', '| Group | Category | Dataset entities | Evaluated | Errors | Rate | Status |', '|---|---|---:|---:|---:|---:|---|']
    for g in report['results']:
        for kind, counts in g['entity_by_type'].items():
            lines.append(f"| {g['condition']} | {kind} | {counts['dataset_entities']} | {counts['reference_entities']} | {counts['errors']} | {number(counts['error_rate'], True)} | {counts['status']} |")
    lines += ['', '## Interpretation', ''] + ['- ' + n for n in report['notes']]
    (out / 'results.md').write_text('\n'.join(lines) + '\n')
    identity_line = f"Dataset: {dataset.get('dataset')} | revision `{dataset.get('source_revision')}` | split `{dataset.get('split')}` | subset `{dataset.get('subset') or 'legacy selection'}` | model `{report['run_identity'].get('model')}` version `{report['run_identity'].get('version')}`."
    clips = ['# Per-clip results', '', identity_line, '',
             'Excluded measurements are unavailable here; their raw values and all attempts remain in results.json.', '',
             '| Clip | Group | Selected attempt | WER | S/I/D | Reference words | Entity errors / evaluated | Final ms | Partial ms | Status |',
             '|---|---|---:|---:|---|---:|---|---:|---:|---|']
    examples = ['# Strict entity errors', '', identity_line, '', 'These compare raw reference text. Capitalization, punctuation and written-number differences count as errors.', '']
    for r in report['clips']:
        w = r['word_errors']
        wer = (w['substitutions'] + w['insertions'] + w['deletions']) / w['reference_words'] if w and w['reference_words'] else None
        sid = f"{w['substitutions']}/{w['insertions']}/{w['deletions']}" if w else 'unavailable'
        partial = r['first_partial_after_t0']
        entity_counts = f"{r['entities']['errors']} / {r['entities']['reference_entities']}" if r['entities'] and r['entities']['status'] == 'annotated' else 'unavailable'
        final_ms = r['finalize_latency_ms'] if r['accuracy_usable'] else None
        partial_ms = partial['latency_ms'] if partial and r['accuracy_usable'] else None
        clips.append(f"| {r['clip_id']} | {r['condition']} | {r['selected_attempt']} | {number(wer, True)} | {sid} | {w['reference_words'] if w else 'unavailable'} | {entity_counts} | {number(final_ms)} | {number(partial_ms)} | {', '.join(r['exclusion_reasons']) or 'valid'} |")
        if r['entities']:
            for e in r['entities']['details']:
                if e['incorrect']:
                    examples += [f"## {r['clip_id']} — {e['type']}", '',
                                 f"Reference entity: {json.dumps(e['text'], ensure_ascii=False)}", '',
                                 f"Aligned output: {json.dumps(e['observed_aligned_text'], ensure_ascii=False)}", '',
                                 f"Full reference: {r['reference']}", '', f"Full output: {r['transcript']}", '']
    (out / 'per-clip.md').write_text('\n'.join(clips) + '\n')
    (out / 'entity-errors.md').write_text('\n'.join(examples) + '\n')
