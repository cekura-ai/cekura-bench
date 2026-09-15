"""Offline, explicit-source reduction for the unified English STT dashboard.

Keep observations internal. Only counts, percentiles and provenance enter HTML.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

COUNTS = ('substitutions', 'insertions', 'deletions', 'reference_words')
COHORTS = ('pipecat', 'fleurs', 'private')
REMOVED = {'deepgram-nova-2', 'deepgram-flux-multilingual',
           'openai-gpt-realtime-whisper', 'openai-gpt-4o-mini-transcribe', 'google-chirp-2'}
LABELS = {
    'deepgram-nova-3': 'Deepgram Nova-3', 'deepgram-flux-en': 'Deepgram Flux English',
    'openai-gpt-4o-transcribe': 'GPT-4o Transcribe',
    'gemini-3.5-transcribe-live': 'Gemini 3.5', 'google-chirp-3': 'Google Chirp 3',
    'elevenlabs-scribe-v2-realtime': 'ElevenLabs Scribe v2', 'cartesia-ink-2': 'Cartesia Ink 2',
    'speechmatics-standard': 'Speechmatics Standard', 'speechmatics-enhanced': 'Speechmatics Enhanced',
    'smallest-pulse': 'Smallest Pulse', 'gradium-default': 'Gradium',
    'reson8-realtime': 'Reson8', 'inworld-stt-1': 'Inworld STT-1',
    'assemblyai-universal-3-5-pro-min-latency': 'AssemblyAI Universal 3.5 Pro',
    'sarvam-saaras-v3-realtime': 'Sarvam Saaras v3', 'soniox-stt-rt-v5': 'Soniox STT-RT v5',
}
PUBLIC_RUNS = {
    'deepgram-nova-3': 'vocera-deepgram-nova-3-20260912',
    'deepgram-flux-en': 'vocera-deepgram-flux-en-20260912',
    'openai-gpt-4o-transcribe': 'vocera-openai-gpt-4o-transcribe-20260912',
    'gemini-3.5-transcribe-live': 'vocera-gemini-3-5-transcribe-live-20260912',
    'elevenlabs-scribe-v2-realtime': 'vocera-elevenlabs-scribe-v2-realtime-20260912',
    'cartesia-ink-2': 'vocera-cartesia-ink-2-20260912',
    'speechmatics-standard': 'vocera-speechmatics-standard-20260912',
    'speechmatics-enhanced': 'vocera-speechmatics-enhanced-20260912-serial',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def aggregate(rows):
    totals = dict.fromkeys(COUNTS, 0)
    for row in rows:
        require(all(isinstance(row.get(k), int) and row[k] >= 0 for k in COUNTS),
                'Invalid word-error counts')
        for key in COUNTS:
            totals[key] += row[key]
    totals['wer'] = sum(totals[k] for k in COUNTS[:3]) / totals['reference_words'] if totals['reference_words'] else None
    return totals


def check_counts(actual, saved):
    require(all(actual[k] == saved[k] for k in COUNTS), 'Recomputed word counts differ from saved evidence')
    if 'wer' in saved:
        require(actual['wer'] == saved['wer'] or (
            actual['wer'] is not None and saved['wer'] is not None and
            math.isclose(actual['wer'], saved['wer'], abs_tol=1e-10)), 'Saved WER differs from counts')


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values), 'Invalid timing observation')
    out = {'n': len(values)}
    for p in (50, 90, 95, 99):
        i = (len(values) - 1) * p / 100
        out[f'p{p}_ms'] = (values[math.floor(i)] + (values[math.ceil(i)] - values[math.floor(i)]) * (i % 1)) if values else None
    return out


def clip_record(row, cohort):
    attempts = row['attempts']
    first = next((a for a in attempts if a['attempt'] == 1), None)
    if first and 'deadlines' in row:
        first = {**first, 'deadlines': row['deadlines']}
    return dict(id=row['clip_id'], cohort=cohort,
                selected_attempt=row.get('selected_attempt'),
                counts=row['word_errors'] if row['accuracy_usable'] else None,
                first=first, attempts=attempts, words=None)


def full_record(row):
    attempts = row['attempts']
    chosen = next((a for a in attempts if a['attempt'] == row['selected_attempt']), None)
    require(chosen is None or chosen['valid'], 'Selected full-run attempt is invalid')
    first = next((a for a in attempts if a['attempt'] == 1), None)
    return dict(id=row['clip_id'], cohort='pipecat' if row['cohort'] == 'public' else 'private',
                selected_attempt=row['selected_attempt'],
                counts=chosen['word_errors'] if chosen else None, first=first,
                attempts=attempts, words=first.get('private_latency') if first else None)


def reduce_model(model, records, planned, terminal, sources):
    require(len({(r['cohort'], r['id']) for r in records}) == len(records), 'Duplicate dataset item')
    groups = {}
    for cohort in COHORTS:
        rows = [r for r in records if r['cohort'] == cohort]
        good = [r for r in rows if r['counts'] is not None]
        groups[cohort] = dict(usable=len(good), attempted=sum(bool(r['attempts']) for r in rows),
                              planned=planned[cohort], **aggregate([r['counts'] for r in good]))
    public_first = [r['first'] for r in records if r['cohort'] == 'pipecat' and r['first'] and r['first']['valid']]
    final = [(a['final_transcript_received_seconds'] - a['t0_seconds']) * 1000
             for a in public_first if a.get('final_transcript_received_seconds') is not None and a.get('t0_seconds') is not None]
    interim = [(a.get('first_partial_after_t0') or {}).get('latency_ms') for a in public_first]
    word_records = [r['words'] for r in records if r['cohort'] == 'private' and r['words']]
    words = [w['delay_ms'] for r in word_records for w in r['words']]
    deadlines = []
    for ms in (0, 250, 500, 1000):
        observations = [d for r in records if r['cohort'] == 'pipecat' and r['first']
                        for d in r['first'].get('deadlines', [])
                        if d['deadline_ms'] == ms and d.get('word_errors') is not None and d.get('pacing_valid')]
        deadlines.append(dict(deadline_ms=ms, n=len(observations), **aggregate([d['word_errors'] for d in observations])))
    attempts = [a for r in records for a in r['attempts']]
    planned_count = sum(planned.values())
    attempted_count = sum(g['attempted'] for g in groups.values())
    usable_count = sum(g['usable'] for g in groups.values())
    require(0 <= usable_count <= attempted_count <= planned_count, 'Invalid reliability coverage')
    # Each planned recording counts once; a valid recovery resolves its failure.
    # Missing runs remain distinct from attempted items without a usable result.
    reliability = dict(planned=planned_count, attempted=attempted_count, usable=usable_count,
                       failed=attempted_count - usable_count, not_run=planned_count - attempted_count,
                       failure_rate=(attempted_count - usable_count) / planned_count if attempted_count and planned_count else None,
                       not_run_rate=(planned_count - attempted_count) / planned_count if planned_count else None,
                       missing_result_rate=(planned_count - usable_count) / planned_count if planned_count else None)
    # Only completed full public/private runs qualify for an ordinal rank.
    rankable = terminal and groups['pipecat']['attempted'] == planned['pipecat'] and groups['private']['attempted'] == planned['private'] and all(groups[c]['usable'] for c in ('pipecat', 'private'))
    # Saved run usage and list-rate estimates do not establish actual charges.
    # Keep this unavailable until billing evidence is reconciled to the model's runs.
    return dict(id=model, label=LABELS[model], cohorts=groups, reliability=reliability,
                actual_cost_usd=None, actual_cost_status='unverified',
                combined=aggregate([r['counts'] for r in records if r['counts'] is not None]),
                rankable=rankable, terminal=terminal, sources=sources,
                timings={'final': percentiles(final), 'interim': percentiles(interim),
                         'completion': percentiles([a.get('completion_latency_ms') for a in public_first]),
                         'word': {**percentiles(words), 'reference_words': sum(r['reference_words'] for r in word_records)}},
                timing_eligible=len(public_first), deadlines=deadlines,
                failed_attempts=sum(not a['valid'] for a in attempts),
                retries=sum(max(0, len(r['attempts']) - 1) for r in records))


def rank_common_public(models, records):
    """Compute once over the full ranked set, independently of UI filtering."""
    ranked = [m for m in models if m['rankable']]
    sets = [{r['id'] for r in records[m['id']] if r['cohort'] == 'pipecat' and r['counts'] is not None}
            for m in ranked]
    common = set.intersection(*sets) if sets else set()
    denominators = set()
    for m in models:
        rows = [r for r in records[m['id']] if r['cohort'] == 'pipecat' and r['id'] in common]
        m['headline'] = aggregate([r['counts'] for r in rows]) if m['rankable'] else aggregate([])
        m['headline']['n'] = len(common) if m['rankable'] else 0
        if m['rankable']:
            denominators.add(m['headline']['reference_words'])
        public, private = (m['cohorts'][c]['wer'] for c in ('pipecat', 'private'))
        m['private_minus_public_pp'] = (private - public) * 100 if public is not None and private is not None else None
        m['rank'] = None
    require(len(denominators) <= 1, 'Common public reference-word denominators differ')
    ordered = sorted(ranked, key=lambda m: (m['headline']['wer'] if m['headline']['wer'] is not None else math.inf, m['id']))
    for i, m in enumerate(ordered, 1):
        if common:
            m['rank'] = i
    return dict(clip_ids=sorted(common), clips=len(common), reference_words=next(iter(denominators), 0),
                models=sorted(m['id'] for m in ranked), basis='Usable public clips shared by every ranked model')


def finalization_contract(model):
    if model == 'google-chirp-3':
        return dict(group='unavailable', label='No full public results')
    if model.startswith('speechmatics'):
        return dict(group='stream_end', label='EndOfStream after silence tail')
    if model.startswith('assemblyai'):
        return dict(group='stream_end', label='Native endpointing; Terminate after silence tail')
    signals = {'deepgram-nova-3': 'Finalize', 'deepgram-flux-en': 'ForceEndTurn',
               'openai-gpt-4o-transcribe': 'commit', 'gemini-3.5-transcribe-live': 'activityEnd',
               'elevenlabs-scribe-v2-realtime': 'commit', 'cartesia-ink-2': 'finalize',
               'smallest-pulse': 'finalize', 'gradium-default': 'flush', 'reson8-realtime': 'flush_request',
               'inworld-stt-1': 'endTurn', 'sarvam-saaras-v3-realtime': 'speech_end', 'soniox-stt-rt-v5': 'finalize'}
    return dict(group='signal_at_speech_end', label=signals[model] + ' at speech end')


def build(reports):
    from offline_rescore import rescore_records, score_projection
    from stt_bench.score import NORMALIZATION
    sources = {}
    def read(name):
        raw = (reports / name).read_bytes()
        sources[name] = {'path': 'reports/' + name, 'sha256': hashlib.sha256(raw).hexdigest()}
        return json.loads(raw)

    records = {m: [] for m in LABELS}
    model_sources = {m: [] for m in LABELS}
    terminal = dict.fromkeys(LABELS, True)
    fleurs_path = 'deepgram-public-v3-final/results.json'
    fleurs = read(fleurs_path)
    fleurs_ids = {r['clip_id'] for r in fleurs['clips']}
    require(len(fleurs_ids) == len(fleurs['clips']) == 180, 'FLEURS frozen coverage changed')
    planned = dict(pipecat=1000, fleurs=len(fleurs_ids), private=8)
    public_ids = None
    for model, run in PUBLIC_RUNS.items():
        name = f'vercel-models/{run}/hourly/summary.json'
        saved = read(name)
        require(saved['model'] == model and saved['status'] == 'complete', 'Public run identity or status differs')
        ids = [r['clip_id'] for r in saved['clips']]
        require(len(ids) == len(set(ids)) == 1000, 'Public coverage differs')
        identity = saved['identity']['full_manifest_sha256']
        if public_ids is None:
            public_ids, public_hash = set(ids), identity
            normalization = saved['normalization']
        require(public_ids == set(ids) and identity == public_hash, 'Public frozen dataset differs')
        require(saved['normalization'] == normalization, 'Text normalization differs')
        rows = [clip_record(r, 'pipecat') for r in saved['clips']]
        check_counts(aggregate([r['counts'] for r in rows if r['counts']]), saved['eventual_word_errors'])
        require(sum(r['counts'] is not None for r in rows) == saved['scored_clips'], 'Public usable count differs')
        records[model] += rows
        model_sources[model].append(name)

    private_name = 'private-longform-recovery-v2/comparison/comparison.json'
    private = read(private_name)
    require(private['complete'] and private['all_results_valid'], 'Private recovery is incomplete')
    private_ids = {r['clip_id'] for r in private['recordings']}
    require(len(private_ids) == 8, 'Private frozen coverage differs')
    for m in private['models']:
        model = m['model']
        if model in REMOVED:
            continue
        require(m['archive_verified'], 'Private archive verification missing')
        rows = []
        for r in private['recordings']:
            if r['model'] != model:
                continue
            e = r['evidence']
            rows.append(dict(id=r['clip_id'], cohort='private',
                             selected_attempt=e['selected']['attempt'] if e['selected'] else None,
                             counts=e['selected']['word_errors'] if e['selected'] else None,
                             first=e['attempts'][0] if e['attempts'] else None,
                             attempts=e['attempts'], words=e['first_attempt_latency']))
        check_counts(aggregate([r['counts'] for r in rows if r['counts']]), m['word_errors'])
        require(len(rows) == m['scored'] == 8, 'Private coverage differs')
        records[model] += rows
        model_sources[model].append(private_name)

    for folder, allow_partial in [('full-parallel-20260915', False), ('assemblyai-min-latency-full-20260915', True)]:
        name = folder + '/results.json'
        saved = read(name)
        complete = saved['execution']['status'] == 'complete'
        require(complete or allow_partial, 'Full run is incomplete')
        if complete:
            audit = read(folder + '/evidence-audit.json')
            require(audit['status'] == 'passed' and audit['run_status'] == 'complete', 'Full evidence audit missing')
            require(read(folder + '/compute-stop-verification.json')['allStopped'], 'Full stop verification missing')
        for model, data in saved['models'].items():
            require(model in records, 'Unselected full-run model')
            rows = [full_record(r) for r in data['items']]
            require({r['id'] for r in rows if r['cohort'] == 'pipecat'} == public_ids, 'Full public IDs differ')
            require({r['id'] for r in rows if r['cohort'] == 'private'} == private_ids, 'Full private IDs differ')
            check_counts(aggregate([r['counts'] for r in rows if r['counts']]), data['groups']['combined']['final_wer'])
            records[model] = rows
            model_sources[model] = [name]
            terminal[model] = complete

    require(fleurs['normalization'] == normalization, 'FLEURS normalization differs')
    records['deepgram-nova-3'] += [clip_record(r, 'fleurs') for r in fleurs['clips']]
    model_sources['deepgram-nova-3'].append(fleurs_path)
    for model in ('gradium-default', 'reson8-realtime'):
        for cohort in ('fleurs-general', 'fleurs-entities'):
            name = f'limited-gradium-reson8-20260914/{model}/evidence/{cohort}/scored/results.json'
            saved = read(name)
            require(saved['normalization'] == normalization, 'FLEURS normalization differs')
            require(all(r['clip_id'] in fleurs_ids for r in saved['clips']), 'Unknown FLEURS item')
            records[model] += [clip_record(r, 'fleurs') for r in saved['clips']]
            model_sources[model].append(name)
    # Only trial models without full runs remain; overlapping short private
    # snippets and Pipecat trial clips must not be counted again.
    for model in ('sarvam-saaras-v3-realtime', 'soniox-stt-rt-v5'):
        name = f'trial-short-20260914/{model}/evidence/scored/results.json'
        saved = read(name)
        records[model] = [clip_record(r, 'pipecat') for r in saved['clips'] if r['clip_id'] in public_ids]
        model_sources[model].append(name)
        terminal[model] = False

    before = {m: reduce_model(m, records[m], planned, terminal[m], model_sources[m]) for m in LABELS}
    records, audit = rescore_records(records)
    models = [reduce_model(m, records[m], planned, terminal[m], model_sources[m]) for m in LABELS]
    common = rank_common_public(models, records)
    audit['models'] = {m['id']: dict(before=before[m['id']]['cohorts'], after=m['cohorts'],
                                    headline=m['headline'], reliability=m['reliability']) for m in models}
    audit['corrected_records'] = score_projection(records)
    audit['sources'] = list(sources.values())
    audit['common_public'] = common
    for m in models:
        require(m['reliability'] == before[m['id']]['reliability'], 'Re-scoring changed reliability')
    for m in models:
        m.setdefault('rank', None)
        m['note'] = ''
        m['finalization_contract'] = finalization_contract(m['id'])
        if m['id'] == 'google-chirp-3':
            m['note'] = 'Private results verified; full Pipecat scores unavailable locally.'
        elif m['id'].startswith('assemblyai'):
            m['note'] = ('Verified complete public and private run; ' if m['terminal'] else 'Partial scored snapshot; ') + 'Universal 3.5 Pro in min_latency mode, 60 ms packets.'
        elif not m['terminal']:
            m['note'] = 'Small trial only; no completed full run.'
        elif m['id'] == 'gradium-default':
            m['note'] = 'Private audio uses consecutive sessions of up to 270 seconds.'
        elif m['id'] == 'gemini-3.5-transcribe-live':
            m['note'] = 'Private audio uses nine-minute session handoffs.'
        elif m['id'] == 'inworld-stt-1':
            m['note'] = 'Later text after stream closure extends last-final-text latency. Repetition also occurred before speech end; see evidence notes.'
    return dict(schema_version=2, title='English STT benchmark — corrected offline scores',
                normalization=dict(NORMALIZATION), common_public=common, _rescore_audit=audit,
                generated_at=datetime.now(timezone.utc).isoformat(), planned=planned,
                models=sorted(models, key=lambda m: m['rank'] or 999),
                sources=list(sources.values()),
                removed_models=sorted(REMOVED),
                assemblyai_scores_as_of=json.loads((reports / 'assemblyai-min-latency-full-20260915/results.json').read_text())['generated_at'])
