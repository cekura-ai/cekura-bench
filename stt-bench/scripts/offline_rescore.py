"""Verified, offline scoring migration. Source evidence is never written."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import importlib.metadata
import tarfile

from stt_bench.score import NORMALIZATION, rescore_saved_word_errors

COUNTS = ('substitutions', 'insertions', 'deletions', 'reference_words')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def evidence_without_scores(value):
    """Includes transcript, attempt selection, gates, timestamps and word timing."""
    if isinstance(value, dict):
        return {k: evidence_without_scores(v) for k, v in value.items()
                if k not in ('counts', 'word_errors')}
    if isinstance(value, list):
        return [evidence_without_scores(v) for v in value]
    return value


def rescore_records(records):
    for package, expected in [('jiwer', '4.0.0'), ('whisper-normalizer', '0.1.12')]:
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'Offline scoring requires {package}=={expected}')
    corrected = deepcopy(records)
    visited = set()
    validated = 0

    def walk(value):
        nonlocal validated
        if isinstance(value, dict):
            if id(value) in visited:
                return
            visited.add(id(value))
            if all(k in value for k in (*COUNTS, 'reference_normalized', 'hypothesis_normalized')):
                value.update(rescore_saved_word_errors(value))
                validated += 1
            else:
                for child in value.values():
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(corrected)
    before = digest(evidence_without_scores(records))
    after = digest(evidence_without_scores(corrected))
    if before != after:
        raise ValueError('Re-scoring changed attempt selection, transcript, eligibility or timing evidence')
    root = Path(__file__).resolve().parents[1]
    code_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ('src/stt_bench/score.py', 'scripts/offline_rescore.py',
                                'scripts/unified_benchmark.py', 'scripts/benchmark_clip_review.py')}
    return corrected, dict(normalization=NORMALIZATION, scoring_source_hashes=code_hashes,
                           validated_alignments=validated,
                           original_evidence_sha256=before, corrected_evidence_sha256=after)


def score_projection(records):
    """Persist per-item corrected text/counts and deadline scores for independent review."""
    return {model: [dict(clip_id=r['id'], cohort=r['cohort'],
                        selected_attempt=r.get('selected_attempt'), word_errors=r['counts'],
                        attempts=[dict(attempt=a['attempt'], valid=a['valid'],
                                       raw_sha256=a.get('raw_sha256'), word_errors=a.get('word_errors'))
                                  for a in r['attempts']],
                        deadlines=(r['first'] or {}).get('deadlines')) for r in rows]
            for model, rows in records.items()}


def inworld_evidence(reports):
    """Reproduce the published note from selected, checksum-matched raw receipts."""
    folder = reports / 'full-parallel-20260915'
    saved = json.loads((folder / 'results.json').read_text())
    selected = {r['clip_id']: next(a for a in r['attempts'] if a['attempt'] == r['selected_attempt'])
                for r in saved['models']['inworld-stt-1']['items'] if r['cohort'] == 'public'}
    verified, late, tap = set(), 0, None
    archives = []
    for path in sorted((folder / 'batches').glob('*/verified.json')):
        batch = json.loads(path.read_text())
        if batch['assignment']['model'] != 'inworld-stt-1':
            continue
        targets = {r['clip_id'] for r in batch['rows'] if r['clip_id'] in selected
                   and r['raw_sha256'] == selected[r['clip_id']]['raw_sha256']}
        if not targets:
            continue
        archive = path.parent / 'evidence.tar.gz'
        archives.append(dict(path=str(archive.relative_to(reports)), sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
        with tarfile.open(archive) as tar:
            for member in tar:
                cid = Path(member.name).name.split('--')[0]
                if not member.name.endswith('.jsonl') or cid not in targets:
                    continue
                raw = tar.extractfile(member).read()
                if hashlib.sha256(raw).hexdigest() != selected[cid]['raw_sha256'] or cid in verified:
                    raise ValueError('Inworld raw evidence mismatch or duplicate')
                events = [json.loads(line) for line in raw.splitlines()]
                t0 = next(e['time_seconds'] for e in events if e['kind'] == 'speech_end')
                close = next(e['time_seconds'] for e in events if e['kind'] == 'close_stream_requested')
                transcripts = [(e['time_seconds'], e['message']['result']['transcription']) for e in events
                               if e.get('message', {}).get('result', {}).get('transcription')]
                finals = [(at, t['transcript'].strip()) for at, t in transcripts
                          if t.get('isFinal') and t.get('transcript', '').strip()]
                if finals and finals[-1][0] > close and finals[-1][1] in ('Oh.', 'I.', "I'm not sure.", 'Yeah.'):
                    late += 1
                if cid == 'pipecat-b980f45a-7289-f63f-0923-2fe102deb8c2':
                    first = next(at for at, t in transcripts if 'tap tap tap' in t.get('transcript', '').lower())
                    tap = dict(clip_id=cid, raw_sha256=selected[cid]['raw_sha256'],
                               archive=str(archive.relative_to(reports)), member=member.name,
                               first_repetition_received_after_speech_end_ms=(first - t0) * 1000)
                verified.add(cid)
    if verified != set(selected) or tap is None or tap['first_repetition_received_after_speech_end_ms'] >= 0 or not late:
        raise ValueError('Saved Inworld evidence does not support the diagnostic note')
    return dict(verified_selected_public_clips=len(verified),
                clips_with_one_of_four_named_final_segments_after_close=late,
                interpretation='Receipt timing and text only; not a verified hallucination rate or causal test.',
                repetition_example=tap, archives=archives)


def write_correction(reports, data, audit):
    # Re-read every source before publishing the derived report.
    for source in data['sources']:
        path = reports / source['path'].removeprefix('reports/')
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f'Source changed during re-scoring: {source["path"]}')
    output = reports / ('offline-rescore-combined-v1' if 'ranking' in data else 'offline-rescore-v2')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('audit.json', audit), ('results.json', data)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
