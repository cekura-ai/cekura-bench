"""Public output is an allowlisted derivative, never a raw private evidence bundle."""
from copy import deepcopy
import json
from pathlib import Path
import re
import zipfile

LOCAL_PATH = re.compile(r'(?:/Users/|/home/|/private/|/vercel/sandbox/|[A-Za-z]:\\\\)[^\s"<>]*')
FORBIDDEN = re.compile(r'/Users/|/home/|/private/|/vercel/sandbox/|private-dataset/|hifi-audio-|"speaker_id"|"source_inventory"')
PRIVATE_KEYS = {'source_inventory', 'source_audio', 'source_metadata', 'speaker_id', 'original_source',
                'execution_provenance', 'source_runs', 'later_recovery_attempts'}


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in PRIVATE_KEYS and not LOCAL_PATH.search(k) and not (k == 'recovery_attempts' and isinstance(v, list))}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return LOCAL_PATH.sub('[local path omitted]', value)
    return value


def public_data(data):
    top = {'schema_version', 'title', 'normalization', 'common_public', 'ranking', 'generated_at',
           'planned', 'models', 'sources', 'removed_models', 'assemblyai_scores_as_of', 'clip_review', 'turns'}
    result = {k: deepcopy(v) for k, v in data.items() if k in top}
    model_keys = {'id', 'label', 'cohorts', 'reliability', 'actual_cost_usd', 'actual_cost_status', 'combined',
        'rankable', 'terminal', 'sources', 'timings', 'timing_eligible', 'deadlines', 'failed_attempts', 'retries',
        'headline', 'rank', 'public_rank', 'ranking_score', 'private_minus_public_pp', 'note', 'finalization_contract',
        'comparison_combined', 'comparison_private', 'comparison_exclusions', 'ranking_uncertainty', 'timing_policy'}
    result['models'] = [{k: v for k, v in m.items() if k in model_keys} for m in result.get('models', [])]
    # Private utterances, speaker metadata and raw receipts never enter this derivative.
    review = result.get('clip_review')
    if review:
        review['clips'] = [c for c in review['clips'] if c['cohort'] in ('pipecat', 'fleurs')]
        review['includes_private'] = False
        review['manifests'] = []
    if result.get('turns'):
        turn_keys = {'models', 'generated_at', 'counts', 'first_attempts', 'valid', 'failed', 'recovery_attempts',
                     'recovered', 'excluded_models', 'sources', 'manifest_sha256', 'preparation',
                     'listening_review_verified', 'compute_stopped', 'refresh'}
        result['turns'] = {k: v for k, v in result['turns'].items() if k in turn_keys}
        summary_keys = {'id', 'label', 'counts', 'accuracy', 'ttft', 'ttfs', 'exception_speech_end_to_final',
            'deadlines', 'recovery_summary', 'diagnostics', 'cost', 'overall', 'overall_rank',
            'overall_unavailable_reason', 'profile', 'manifest_sha256', 'result_sha256', 'timing_correction',
            'historical_stream_end_ttfs', 'timing_note', 'private_full', 'overall_fixed', 'overall_coverage'}
        result['turns']['models'] = [{k: v for k, v in m.items() if k in summary_keys} for m in result['turns']['models']]
        for model in result['turns']['models']:
            if 'profile' in model:
                model['profile'] = {k: v for k, v in model['profile'].items() if k in {
                    'provider', 'model', 'model_id', 'version', 'finalization', 'completion_basis',
                    'transcript_reconstruction', 'transmitted_silence_frames', 'turn_finalization_class',
                    'measurement_profile', 'sample_rate', 'channels', 'frame_ms', 'delay_in_frames',
                    'language', 'mode', 'force_endpoint', 'voice_profile'}}
            model['proof'] = None
            model['public_summary_only'] = True
            model.pop('recovery_attempts', None)
        result['turns']['public_summary_only'] = True
    result = scrub(result)
    result['publication'] = dict(visibility='public', private_evidence_included=False,
        notice='Private results are aggregate statistics only. Private audio, transcripts and receipt logs are not distributed.')
    assert_public_text(json.dumps(result))
    return result


def assert_public_text(text, forbidden_identifiers=()):
    if FORBIDDEN.search(text) or any(v and v in text for v in forbidden_identifiers):
        raise ValueError('Public export contains private identifiers or local paths')


def verify_public_directory(root, forbidden_identifiers=()):
    root = Path(root)
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlinks are not public assets: ' + str(path.relative_to(root)))
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if path.suffix not in ('.html', '.json', '.txt', '.flac', '.zip'):
            raise ValueError('Unexpected public asset: ' + rel)
        if rel.startswith('turn-proof/') or 'conversation-' in rel or 'private' in rel.lower():
            raise ValueError('Private asset in public directory: ' + rel)
        if path.suffix in ('.html', '.json', '.txt'):
            assert_public_text(path.read_text(), forbidden_identifiers)
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name not in ('index.html', 'README.txt', 'score-evidence.json', 'correction-audit.json') and not re.fullmatch(r'audio/(?:pipecat|fleurs)/(?:pipecat|fleurs)[A-Za-z0-9_.-]*\.flac', name):
                        raise ValueError('Unexpected public ZIP entry: ' + name)
                    if name.endswith(('.html', '.txt', '.json')):
                        assert_public_text(z.read(name).decode(), forbidden_identifiers)
    return dict(status='verified', private_identifiers_checked=len(forbidden_identifiers))


def timing_contracts(data):
    local = {'openai-gpt-realtime-whisper', 'openai-gpt-4o-transcribe', 'openai-gpt-4o-mini-transcribe',
             'gemini-3.5-transcribe-live', 'elevenlabs-scribe-v2-realtime', 'deepgram-flux-en', 'deepgram-flux-multilingual'}
    for m in data.get('models', []):
        if 'finalization_contract' not in m or 'timings' not in m:
            continue
        contract = m['finalization_contract']
        if m['id'] == 'inworld-stt-1':
            contract['group'] = 'signal_without_tail'
        elif m['id'] == 'gradium-default':
            contract['group'] = 'gradium_legacy_finality'
            contract['label'] = 'Historical text finality waits for stream end after the one-second tail'
        m['timing_policy'] = dict(transmitted_silence_ms=0 if m['id'] == 'inworld-stt-1' else 1000,
            completion_kind='local_observation_end' if m['id'] in local else 'server_terminal_then_local_marker',
            note='Completion includes the harness observation window; compare only matching protocols.')
        if m['id'] in local:
            # Preserve historical numbers, but do not offer a local marker as provider latency.
            old = m['timings']['completion']
            if 'harness_completion' not in m['timings']:
                m['timings']['harness_completion'] = deepcopy(old)
            m['timings']['completion'] = dict(n=0, **{f'p{p}_ms': None for p in (50, 90, 95, 99)},
                status='local_observation_end_not_provider_completion')


def turn_timing_contracts(data):
    for m in (data.get('turns') or {}).get('models', []):
        if m['id'] == 'gradium-default' and m['profile'].get('transcript_reconstruction') != 'gradium-append-only-text-v2':
            m.setdefault('historical_stream_end_ttfs', deepcopy(m['ttfs']))
            unavailable = dict(n=0, p50_ms=None, p90_ms=None, p95_ms=None,
                confidence_interval={'status': 'unavailable_legacy_stream_end_finality'},
                status_counts={'legacy_stream_end_finality': m['counts']['planned']}, final_before_boundary=0)
            m['ttfs'] = unavailable
            m['recovery_summary']['after_recovery_ttfs'] = deepcopy(unavailable)
            m['timing_note'] = 'Historical Gradium finality includes stream closure. Corrected text timing requires replay of saved receipts.'
