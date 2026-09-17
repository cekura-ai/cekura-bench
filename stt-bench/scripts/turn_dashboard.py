"""Package verified turn results beside the historical dashboard, without pooling them."""
from pathlib import Path
import gzip
import hashlib
import html
import json
import math
import re
import shutil

LABELS = {
    'assemblyai-universal-3-5-pro': 'AssemblyAI Universal 3.5 Pro',
    'assemblyai-universal-3-5-pro-min-latency': 'AssemblyAI Universal 3.5 Pro · Min latency',
    'deepgram-flux-en': 'Deepgram Flux English', 'deepgram-flux-multilingual': 'Deepgram Flux Multilingual',
    'deepgram-nova-3': 'Deepgram Nova-3', 'cartesia-ink-2': 'Cartesia Ink 2',
    'elevenlabs-scribe-v2-realtime': 'ElevenLabs Scribe v2', 'gemini-3.5-transcribe-live': 'Gemini 3.5',
    'google-chirp-2': 'Google Chirp 2', 'google-chirp-3': 'Google Chirp 3',
    'gradium-default': 'Gradium', 'inworld-stt-1': 'Inworld STT-1',
    'openai-gpt-4o-mini-transcribe': 'GPT-4o Mini Transcribe', 'openai-gpt-4o-transcribe': 'GPT-4o Transcribe',
    'openai-gpt-realtime-whisper': 'GPT Realtime Whisper', 'reson8-realtime': 'Reson8',
    'sarvam-saaras-v3-realtime': 'Sarvam Saaras v3', 'smallest-pulse': 'Smallest Pulse',
    'speechmatics-linden-1': 'Speechmatics Linden',
}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_model(model, saved):
    rows = model['clips']; c = model['counts']
    require(len(rows) == c['planned'] == 206 and len({r['clip_id'] for r in rows}) == 206, 'Turn clip coverage differs')
    require(sum(r['attempted'] for r in rows) == c['attempted'] == 206, 'Turn attempts incomplete')
    require(sum(r['valid'] for r in rows) == c['valid'] and c['failed'] + c['valid'] == 206, 'Turn validity counts differ')
    require(c['not_run'] == 0 and saved['counts'] == c, 'Turn summary counts differ')
    errors = [r['word_errors'] for r in rows if r['word_errors'] is not None]
    for key in ('substitutions', 'insertions', 'deletions', 'reference_words'):
        require(sum(e[key] for e in errors) == model['accuracy'][key], 'Turn word counts differ')
    a = model['accuracy']; expected = sum(a[k] for k in ('substitutions', 'insertions', 'deletions')) / a['reference_words']
    require(math.isclose(expected, a['wer']), 'Turn WER arithmetic differs')
    for key in ('accuracy', 'ttft', 'ttfs', 'deadlines', 'recovery_summary'):
        require(model[key] == saved[key], 'Turn model and comparison disagree: ' + key)
    for metric in ('ttft', 'ttfs'):
        values = sorted(r['turn_timing'][metric + '_ms'] for r in rows if r['turn_timing'].get(metric + '_ms') is not None)
        stats = model[metric]
        require(len(values) == stats['n'] and sum(stats['status_counts'].values()) == 206, 'Turn timing counts differ')
        for q in (50, 90, 95):
            ix = (len(values) - 1) * q / 100 if values else 0
            expected = values[math.floor(ix)] + (values[math.ceil(ix)] - values[math.floor(ix)]) * (ix % 1) if values else None
            actual = stats[f'p{q}_ms']
            require((actual is None and expected is None) or actual is not None and expected is not None and math.isclose(actual, expected, abs_tol=1e-7), 'Turn percentile differs')


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or digest(source) != digest(destination):
        shutil.copyfile(source, destination)


def pack_raw(source, destination, expected):
    require(digest(source) == expected, 'Original turn receipt hash differs')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as f, destination.open('wb') as out:
        with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as z:
            shutil.copyfileobj(f, z)
    return {'path': destination.parent.name + '/' + destination.name, 'raw_sha256': expected, 'gzip_sha256': digest(destination)}


PROOF_STYLE = """body{max-width:1050px;margin:auto;padding:32px 24px;background:#f7f8fa;color:#20242b;font:15px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}h1{font-size:30px;letter-spacing:-.03em}h2{font-size:21px;margin-top:32px}a{color:#225b53;text-underline-offset:3px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px;border-bottom:1px solid #e1e4e8}details{background:white;padding:18px;margin:12px 0;border:1px solid #e1e4e8;border-radius:10px}summary{cursor:pointer;font-weight:550}small{color:#59616c}audio{display:block;margin:16px 0;max-width:100%}.proof-links{display:flex;flex-wrap:wrap;gap:16px;font-size:13px}.proof-back{display:block;margin-bottom:28px}code{overflow-wrap:anywhere}a:focus-visible,summary:focus-visible{outline:3px solid #287d6c;outline-offset:4px}@media(max-width:600px){body{padding:24px 16px}table{display:block;overflow:auto}h1{font-size:26px}details{padding:14px}}"""


def attach_turns(data, reports_root, output_dir):
    source = Path(reports_root) / 'private-turns-consolidated-20260915/final'
    if not source.exists():
        return None
    summary = json.loads((source / 'comparison.json').read_text())
    validation = json.loads((source / 'validation.json').read_text())
    compute = json.loads((source / 'remote-compute-verification.json').read_text())
    require(summary['execution_complete'] and summary['all_included_models_attempted_all_turns'] and validation['status'] == 'verified' and compute['all_stopped'], 'Turn benchmark verification is incomplete')
    proof = Path(output_dir) / 'turn-proof'; proof.mkdir(parents=True, exist_ok=True)
    hashes = []
    for name in ('comparison.json', 'metrics.csv', 'validation.json', 'remote-compute-verification.json'):
        copy(source / name, proof / name); hashes.append({'path': 'turn-proof/' + name, 'sha256': digest(source / name)})
    historical = {m['id']: m for m in data['models']}
    models = []
    for mid, saved in summary['models'].items():
        result = json.loads((source / mid / 'results.json').read_text()); validate_model(result, saved)
        evidence = source / 'evidence' / mid
        if not models:
            for name in ('manifest.json', 'draft.json', 'review.json'):
                copy(evidence / name, proof / name)
        require(digest(proof / 'manifest.json') == result['manifest_sha256'], 'Turn manifest hash differs')
        model_dir = proof / mid; model_dir.mkdir(exist_ok=True)
        copy(source / mid / 'results.json', model_dir / 'results.json')
        for path in (evidence / 'metadata').glob('*.json'):
            copy(path, model_dir / 'metadata' / path.name)
        page = (source / mid / 'index.html').read_text()
        page = re.sub(r'<style>.*?</style>', '<style>' + PROOF_STYLE + '</style>', page, flags=re.S)
        page = page.replace('<h1>', '<a class="proof-back" href="../../index.html#latest">← Back to all models</a><h1>', 1)
        page = page.replace('src="audio/', 'src="../audio/')
        page = page.replace('<h2>Listen and inspect each turn</h2>', '<h2 id="turn-clips">Listen and inspect each turn</h2>')
        page = re.sub(r'<h2>Excluded during dataset preparation</h2>(.*?<ul>.*?</ul>)', r'<details><summary>Excluded candidates and reasons</summary>\1</details>', page, flags=re.S)
        provenance = {p['clip_id']: p for p in result['execution_provenance']}
        raw_index = []
        for row in result['clips']:
            cid = row['clip_id']; raw = evidence / 'raw' / f'{cid}--attempt-1.jsonl'
            record = pack_raw(raw, model_dir / 'raw' / (raw.name + '.gz'), row['raw_sha256']); record['clip_id'] = cid
            raw_index.append(record)
            audio = source / mid / row['audio']; copy(audio, proof / 'audio' / f'{cid}.wav')
            meta = provenance[cid]['execution_metadata']
            links = f'<p class="proof-links"><a download href="raw/{raw.name}.gz">Original receipt log (.jsonl.gz)</a><a href="{html.escape(meta)}">Session and model settings</a></p>'
            pattern = r'(<summary>' + re.escape(html.escape(cid)) + r' ·.*?</summary>)'
            page, count = re.subn(pattern, lambda m: m.group(1) + links, page, count=1)
            require(count == 1, 'Missing clip in listening proof page')
        for index, attempt in enumerate(result['recovery_attempts'], 1):
            cid = attempt['clip_id']; raw = Path(attempt['source']) / 'raw' / f'{cid}--attempt-1.jsonl'
            name = f'{cid}--recovery-{index}.jsonl.gz'
            record = pack_raw(raw, model_dir / 'recovery' / name, attempt['raw_sha256']); record.update(clip_id=cid, recovery=True); raw_index.append(record)
            page = re.sub(r'(<summary>Recovery: ' + re.escape(html.escape(cid)) + r' ·.*?</summary>)', lambda m: m.group(1) + f'<p><a download href="recovery/{name}">Recovery receipt log (.jsonl.gz)</a></p>', page, count=1)
        (model_dir / 'receipt-index.json').write_text(json.dumps(raw_index, indent=2) + '\n')
        (model_dir / 'index.html').write_text(page)
        old = historical.get(mid)
        models.append({'id': mid, 'label': LABELS.get(mid, mid), **saved,
            'overall': old['comparison_combined'] if old and old['comparison_combined']['wer'] is not None else None,
            'overall_unavailable_reason': ('No scored full run' if mid == 'assemblyai-universal-3-5-pro' else
                                           'Public pilot only; no full private run' if mid == 'sarvam-saaras-v3-realtime' else
                                           'No verified combined results'),
            'private_full': old['cohorts']['private'] if old else None,
            'overall_fixed': old['ranking_score'] if old else None,
            'overall_coverage': dict(public_attempted=old['cohorts']['pipecat']['attempted'],public_planned=1000,public_usable=old['cohorts']['pipecat']['usable'],private_usable=old['comparison_private']['usable']) if old else None,
            'overall_rank': old['rank'] if old and old['rankable'] else None,
            'profile': result['config'], 'manifest_sha256': result['manifest_sha256'],
            'result_sha256': digest(source / mid / 'results.json'), 'proof': 'turn-proof/' + mid + '/index.html'})
    manifest = json.loads((proof / 'manifest.json').read_text())
    (proof / 'exclusions.json').write_text(json.dumps(manifest['exclusions'], indent=2) + '\n')
    (proof / 'proof-index.json').write_text(json.dumps({'sources': hashes, 'manifest_sha256': digest(proof / 'manifest.json'), 'models': [{'id': m['id'], 'result_sha256': m['result_sha256']} for m in models]}, indent=2) + '\n')
    (proof / 'overall-source-selection.json').write_text(json.dumps({
        'scope': 'Full public clips and whole private recordings; turn observations are separate.',
        'ranking_version': data['ranking']['version'],
        'ranking_reference_words': data['ranking']['reference_words'],
        'models': [{'id': m['id'], 'sources': m['sources'], 'cohorts': m['cohorts'],
                    'rank': m['rank'], 'ranking_score': m['ranking_score'],
                    'comparison_private': m['comparison_private'], 'comparison_combined': m['comparison_combined'],
                    'comparison_exclusions': m['comparison_exclusions'],
                    'timings': m['timings'], 'note': m['note']} for m in data['models']],
        'source_hashes': data['sources'],
    }, indent=2) + '\n')
    return {'models': models, 'generated_at': summary['updated_at'], 'counts': manifest['counts'],
        'first_attempts': validation['attempted'], 'valid': validation['valid'], 'failed': validation['failures'],
        'recovery_attempts': len(summary['later_recovery_attempts']), 'recovered': sum(m['recovery_summary']['additional_turns_recovered'] for m in models),
        'excluded_models': summary['excluded_models'], 'sources': hashes, 'manifest_sha256': digest(proof / 'manifest.json'),
        'preparation': manifest.get('preparation_mode'), 'listening_review_verified': manifest['listening_review_verified'],
        'compute_stopped': compute['verified']}


def attach_turn_summary(data, reports_root):
    """Read aggregate turn statistics without copying any private evidence assets."""
    source = Path(reports_root) / 'private-turns-consolidated-20260915/final'
    if not source.exists():
        return None
    summary = json.loads((source / 'comparison.json').read_text())
    validation = json.loads((source / 'validation.json').read_text())
    compute = json.loads((source / 'remote-compute-verification.json').read_text())
    require(summary['execution_complete'] and validation['status'] == 'verified' and compute['all_stopped'],
            'Turn benchmark verification is incomplete')
    historical = {m['id']: m for m in data['models']}
    models = []
    for mid, saved in summary['models'].items():
        result = json.loads((source / mid / 'results.json').read_text())
        validate_model(result, saved)
        old = historical.get(mid)
        models.append({'id': mid, 'label': LABELS.get(mid, mid), **{k: saved[k] for k in
            ('counts', 'accuracy', 'ttft', 'ttfs', 'exception_speech_end_to_final', 'deadlines', 'recovery_summary')},
            'overall': old['comparison_combined'] if old else None,
            'private_full': old['cohorts']['private'] if old else None,
            'overall_fixed': old['ranking_score'] if old else None,
            'overall_coverage': dict(public_attempted=old['cohorts']['pipecat']['attempted'],public_planned=1000,public_usable=old['cohorts']['pipecat']['usable'],private_usable=old['comparison_private']['usable']) if old else None,
            'overall_rank': old['rank'] if old else None, 'profile': result['config'],
            'manifest_sha256': result['manifest_sha256'], 'result_sha256': digest(source / mid / 'results.json'),
            'proof': None, 'public_summary_only': True})
    return {'models': models, 'generated_at': summary['updated_at'], 'counts': {},
        'first_attempts': validation['attempted'], 'valid': validation['valid'], 'failed': validation['failures'],
        'recovery_attempts': len(summary['later_recovery_attempts']),
        'recovered': sum(m['recovery_summary']['additional_turns_recovered'] for m in models),
        'excluded_models': summary['excluded_models'], 'sources': [],
        'manifest_sha256': models[0]['manifest_sha256'], 'compute_stopped': compute['verified'],
        'public_summary_only': True}
