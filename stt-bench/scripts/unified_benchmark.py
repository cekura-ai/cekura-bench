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
REMOVED = {'deepgram-nova-2'}
LABELS = {
    'deepgram-flux-multilingual': 'Deepgram Flux Multilingual',
    'openai-gpt-realtime-whisper': 'GPT Realtime Whisper',
    'openai-gpt-4o-mini-transcribe': 'GPT-4o Mini Transcribe',
    'google-chirp-2': 'Google Chirp 2',
    'deepgram-nova-3': 'Deepgram Nova-3', 'deepgram-flux-en': 'Deepgram Flux English',
    'openai-gpt-4o-transcribe': 'GPT-4o Transcribe',
    'gemini-3.5-transcribe-live': 'Gemini 3.5', 'google-chirp-3': 'Google Chirp 3',
    'elevenlabs-scribe-v2-realtime': 'ElevenLabs Scribe v2', 'cartesia-ink-2': 'Cartesia Ink 2',
    'speechmatics-standard': 'Speechmatics Standard', 'speechmatics-enhanced': 'Speechmatics Enhanced',
    'speechmatics-linden-1': 'Speechmatics Linden',
    'smallest-pulse': 'Smallest Pulse', 'gradium-default': 'Gradium',
    'reson8-realtime': 'Reson8', 'inworld-stt-1': 'Inworld STT-1',
    'assemblyai-universal-3-5-pro-min-latency': 'AssemblyAI Universal 3.5 Pro · Min latency',
    'assemblyai-universal-3-5-pro': 'AssemblyAI Universal 3.5 Pro',
    'sarvam-saaras-v3-realtime': 'Sarvam Saaras v3', 'soniox-stt-rt-v5': 'Soniox STT-RT v5',
}
PUBLIC_RUNS = {
    'google-chirp-2': 'vocera-google-chirp-2-20260913-genlang',
    'google-chirp-3': 'vocera-google-chirp-3-20260913-genlang',
    'deepgram-flux-multilingual': 'vocera-deepgram-flux-multilingual-20260912',
    'openai-gpt-realtime-whisper': 'vocera-openai-gpt-realtime-whisper-20260912',
    'openai-gpt-4o-mini-transcribe': 'vocera-openai-gpt-4o-mini-transcribe-20260912',
    'deepgram-nova-3': 'vocera-deepgram-nova-3-20260912',
    'deepgram-flux-en': 'vocera-deepgram-flux-en-20260912',
    'openai-gpt-4o-transcribe': 'vocera-openai-gpt-4o-transcribe-20260912',
    'gemini-3.5-transcribe-live': 'vocera-gemini-3-5-transcribe-live-20260912',
    'elevenlabs-scribe-v2-realtime': 'vocera-elevenlabs-scribe-v2-realtime-20260912',
    'cartesia-ink-2': 'vocera-cartesia-ink-2-20260912',
    'speechmatics-standard': 'vocera-speechmatics-standard-20260912',
    'speechmatics-enhanced': 'vocera-speechmatics-enhanced-20260912-serial',
}

# Ordered replacements on the same frozen public/private datasets. A later
# turn run or small diagnostic cannot replace a full-recording comparison.
FULL_RUNS = (
    ('full-parallel-20260915', False),
    ('assemblyai-min-latency-full-20260915', True),
    ('inworld-full-stream-end-v2-20260915', False),
    ('linden-full-20260915', False),
)


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


def apply_comparison_selection(models, records):
    """User-selected accuracy view, separate from immutable full-set evidence/ranks."""
    for model in models:
        private = dict(model['cohorts']['private'])
        exclusions = []
        if model['id'] == 'inworld-stt-1':
            cid = 'conversation-04-B'
            matches = [r for r in records[model['id']] if r['cohort'] == 'private' and r['id'] == cid]
            require(len(matches) == 1 and matches[0]['counts'] is not None, 'Missing Inworld exclusion evidence')
            removed = matches[0]['counts']
            totals = {k: private[k] - removed[k] for k in COUNTS}
            private.update(aggregate([totals]), usable=private['usable'] - 1)
            exclusions = [dict(clip_id=cid, reason='Excluded at user request because of excessive repeated provider output.',
                               word_errors=removed)]
        private['excluded'] = len(exclusions)
        public = model['cohorts']['pipecat']
        combined = aggregate([public, private] if public['usable'] and private['usable'] else [])
        combined.update(usable=public['usable'] + private['usable'], planned=public['planned'] + private['planned'],
                        attempted=public['attempted'] + private['attempted'],
                        failed=public['attempted'] - public['usable'] + model['cohorts']['private']['attempted'] - model['cohorts']['private']['usable'],
                        excluded=len(exclusions))
        model['comparison_private'] = private
        model['comparison_combined'] = combined
        model['comparison_exclusions'] = exclusions


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
    if model in ('google-chirp-2', 'google-chirp-3'):
        return dict(group='stream_end', label='Input stream closure after silence tail')
    if model == 'speechmatics-linden-1':
        return dict(group='signal_at_speech_end', label='ForceEndOfUtterance at speech end; EndOfTranscript after tail')
    if model.startswith('speechmatics'):
        return dict(group='stream_end', label='EndOfStream after silence tail')
    if model.startswith('assemblyai'):
        return dict(group='stream_end', label='Native endpointing; Terminate after silence tail')
    signals = {'deepgram-nova-3': 'Finalize', 'deepgram-flux-en': 'ForceEndTurn',
               'openai-gpt-4o-transcribe': 'commit', 'openai-gpt-4o-mini-transcribe': 'commit',
               'openai-gpt-realtime-whisper': 'commit', 'deepgram-flux-multilingual': 'ForceEndTurn',
               'gemini-3.5-transcribe-live': 'activityEnd',
               'elevenlabs-scribe-v2-realtime': 'commit', 'cartesia-ink-2': 'finalize',
               'smallest-pulse': 'finalize', 'gradium-default': 'flush', 'reson8-realtime': 'flush_request',
               'inworld-stt-1': 'endTurn', 'sarvam-saaras-v3-realtime': 'speech_end', 'soniox-stt-rt-v5': 'finalize'}
    return dict(group='signal_at_speech_end', label=signals[model] + ' at speech end')


def rank_frozen_combined(models, records, manifest):
    """Pool integer counts on pinned IDs; missing results never shrink the set."""
    datasets = manifest['datasets']
    require(set(datasets) == {'pipecat', 'private'} and all(datasets.values()),
            'Ranking requires nonempty public and private datasets')
    dataset_meta = {c: dict(clip_ids=sorted(ids), clips=len(ids),
                           reference_words=sum(ids.values())) for c, ids in datasets.items()}
    for m in models:
        index = {(r['cohort'], r['id']): r for r in records[m['id']]}
        require(len(index) == len(records[m['id']]), 'Duplicate dataset item')
        scores = {}
        for cohort, ids in datasets.items():
            counts = []
            for cid, words in ids.items():
                row = index.get((cohort, cid))
                if row is not None and row['counts'] is not None:
                    require(row['counts']['reference_words'] == words,
                            f'Frozen reference words differ: {m["id"]} {cohort} {cid}')
                    counts.append(row['counts'])
            scores[cohort] = dict(usable=len(counts), complete=len(counts) == len(ids),
                                  **aggregate(counts))
        public_complete = m['terminal'] and scores['pipecat']['complete']
        m['headline'] = aggregate([scores['pipecat']]) if public_complete else aggregate([])
        m['headline']['n'] = len(datasets['pipecat']) if public_complete else 0
        m['public_rank'] = None
        m['rankable'] = bool(m['rankable'] and all(s['complete'] for s in scores.values()))
        m['ranking_score'] = dict(**aggregate(list(scores.values()) if m['rankable'] else []),
                                  n=sum(len(ids) for ids in datasets.values()) if m['rankable'] else 0,
                                  datasets=scores, version=manifest['version'])
        m['rank'] = None
        public, private = (m['cohorts'][c]['wer'] for c in ('pipecat', 'private'))
        m['private_minus_public_pp'] = (private - public) * 100 if public is not None and private is not None else None
    for score, rank in [('headline', 'public_rank'), ('ranking_score', 'rank')]:
        ordered = sorted((m for m in models if m[score]['wer'] is not None),
                         key=lambda m: (m[score]['wer'], m['id']))
        for i, m in enumerate(ordered, 1):
            m[rank] = i
    return dict(version=manifest['version'], basis=manifest['basis'], datasets=dataset_meta,
                reference_words=sum(d['reference_words'] for d in dataset_meta.values()),
                models=sorted(m['id'] for m in models if m['rankable']))


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
        folder = 'vercel-chirp' if model.startswith('google-chirp-') else 'vercel-models'
        name = f'{folder}/{run}/hourly/summary.json'
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

    for folder, allow_partial in FULL_RUNS:
        name = folder + '/results.json'
        saved = read(name)
        complete = saved['execution']['status'] == 'complete'
        require(complete or allow_partial, 'Full run is incomplete')
        if complete:
            audit = read(folder + '/evidence-audit.json')
            require(audit['status'] == 'passed' and audit['run_status'] == 'complete', 'Full evidence audit missing')
            if folder == 'linden-full-20260915':
                control = read(folder + '/controller.json')
                require(control['status'] == 'complete' and control['preparation'].get('computeStopped')
                        and bool(control['workers']) and all(w.get('computeStopped') and w.get('remoteStatus') == 'stopped'
                        for w in control['workers'].values()), 'Linden saved stop receipts incomplete')
            else:
                require(read(folder + '/compute-stop-verification.json')['allStopped'], 'Full stop verification missing')
        for model, data in saved['models'].items():
            if folder == 'inworld-full-stream-end-v2-20260915':
                require(model == 'inworld-stt-1', 'Unexpected model in Inworld replacement run')
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

    from assemblyai_standard_evidence import load as load_standard, SOURCE as normal_assembly
    read(normal_assembly)
    normal_saved = load_standard(reports)
    normal_items = normal_saved['models']['assemblyai-universal-3-5-pro']['items']
    require({r['clip_id'] for r in normal_items if r['cohort'] == 'private'} == private_ids,
            'AssemblyAI standard private dataset differs')
    require({r['clip_id'] for r in normal_items if r['cohort'] == 'public'} <= public_ids,
            'AssemblyAI standard public dataset differs')
    records['assemblyai-universal-3-5-pro'] = [full_record(r) for r in normal_items]
    for archive in normal_saved['source_archives']:
        sources[archive['path'].removeprefix('reports/')] = archive
    model_sources['assemblyai-universal-3-5-pro'] = [normal_assembly]
    terminal['assemblyai-universal-3-5-pro'] = False
    before = {m: reduce_model(m, records[m], planned, terminal[m], model_sources[m]) for m in LABELS}
    from elevenlabs_private_correction import load as load_eleven, MODEL as eleven, SOURCE as eleven_source
    corrected_private = load_eleven(reports)
    read(eleven_source)
    records[eleven] = [r for r in records[eleven] if r['cohort'] != 'private']
    for row in corrected_private['recordings']:
        e = row['evidence']
        records[eleven].append(dict(id=row['clip_id'], cohort='private',
            selected_attempt=e['selected']['attempt'], counts=e['selected']['word_errors'],
            first=e['attempts'][0], attempts=e['attempts'], words=e['first_attempt_latency']))
    model_sources[eleven] = [s for s in model_sources[eleven] if s != private_name] + [eleven_source]
    records, audit = rescore_records(records)
    audit['elevenlabs_private_correction'] = {k:corrected_private[k] for k in
        ('version','before','after','sources','code_hashes','scope','timing','provider_calls')}
    models = [reduce_model(m, records[m], planned, terminal[m], model_sources[m]) for m in LABELS]
    ranking_path = Path(__file__).resolve().parents[1] / 'config/rankings/combined-public-private-v1.json'
    ranking_raw = ranking_path.read_bytes()
    manifest = json.loads(ranking_raw)
    require(manifest['normalization'] == NORMALIZATION['version'], 'Frozen ranking normalization differs')
    ranking = rank_frozen_combined(models, records, manifest)
    ranking['manifest_sha256'] = hashlib.sha256(ranking_raw).hexdigest()
    ranking['source_results_sha256'] = manifest['source_results_sha256']
    common = dict(**ranking['datasets']['pipecat'], models=manifest['original_public_models'],
                  basis='Frozen shared public clips from the original public-only report')
    audit['models'] = {m['id']: dict(before=before[m['id']]['cohorts'], after=m['cohorts'],
                                    headline=m['headline'], ranking_score=m['ranking_score'],
                                    rank=m['rank'], reliability=m['reliability']) for m in models}
    audit['corrected_records'] = score_projection(records)
    audit['sources'] = list(sources.values())
    audit['common_public'] = common
    audit['ranking'] = ranking
    for m in models:
        require(m['reliability'] == before[m['id']]['reliability'], 'Re-scoring changed reliability')
    for m in models:
        m.setdefault('rank', None)
        m['note'] = ''
        m['finalization_contract'] = finalization_contract(m['id'])
        if m['id'] in ('google-chirp-2', 'google-chirp-3'):
            m['note'] = 'Completed 1,000-clip public run collected from its saved sandbox output; private full recordings verified separately.'
        elif m['id'] == 'assemblyai-universal-3-5-pro':
            m['note'] = normal_saved['note']
        elif m['id'].startswith('assemblyai'):
            m['note'] = ('Verified complete public and private run; ' if m['terminal'] else 'Partial scored snapshot; ') + 'Universal 3.5 Pro in min_latency mode, 60 ms packets.'
        elif not m['terminal']:
            m['note'] = 'Small trial only; no completed full run.'
        elif m['id'] == 'gradium-default':
            m['note'] = 'Private audio uses consecutive sessions of up to 270 seconds.'
        elif m['id'] == eleven:
            m['note'] = 'Private full-recording transcripts corrected offline: paired plain/timestamp finals counted once. All eight recordings and original attempts retained; first-attempt word delay replayed. Public coverage and turn results unchanged.'
        elif m['id'] == 'gemini-3.5-transcribe-live':
            m['note'] = 'Private audio uses nine-minute session handoffs.'
        elif m['id'] == 'inworld-stt-1':
            m['note'] = 'Updated from the verified full stream-ending rerun: 1,000 public clips and eight private recordings. No artificial silence is sent after endTurn; all final text remains scored.'
            m['finalization_contract'] = dict(group='signal_at_speech_end', label='endTurn then closeStream at speech end; no artificial silence tail')
        elif m['id'] == 'speechmatics-linden-1':
            m['note'] = 'Completed full run; archived receipts and merged scores verified. Two public clips remain invalid, including two frozen ranking items, so no combined rank is assigned.'
    from benchmark_uncertainty import add_uncertainty
    add_uncertainty({'models': models, 'ranking': ranking}, records)
    apply_comparison_selection(models, records)
    return dict(schema_version=3, title='English STT benchmark — combined public and private ranking',
                normalization=dict(NORMALIZATION), common_public=common, ranking=ranking, _rescore_audit=audit,
                generated_at=datetime.now(timezone.utc).isoformat(), planned=planned,
                models=sorted(models, key=lambda m: m['rank'] or 999),
                sources=list(sources.values()),
                removed_models=sorted(REMOVED),
                assemblyai_scores_as_of=json.loads((reports / 'assemblyai-min-latency-full-20260915/results.json').read_text())['generated_at'])
