"""Verified offline correction of private full-recording ElevenLabs transcripts."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

from stt_bench.elevenlabs_segments import VERSION, replay
from stt_bench.full_private_metrics import latency
from stt_bench.score import aggregate_wer, normalize_words, word_errors

MODEL = 'elevenlabs-scribe-v2-realtime'
SOURCE = 'elevenlabs-private-correction-v1/results.json'
ORIGINAL = 'private-longform-recovery-v2/comparison/comparison.json'
ROOT = Path(__file__).resolve().parents[1]
CODE = ('src/stt_bench/elevenlabs_segments.py', 'src/stt_bench/full_private_metrics.py',
        'src/stt_bench/score.py', 'scripts/elevenlabs_private_correction.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def generate(reports):
    destination = reports / SOURCE
    require(not destination.exists(), 'Correction is immutable; verify existing output instead')
    manifest_path = ROOT / 'workspaces/private-longform-recovery-v2/dataset/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    clips = {c['clip_id']: c for c in manifest['clips']}
    original = json.loads((reports / ORIGINAL).read_text())
    rows = deepcopy([r for r in original['recordings'] if r['model'] == MODEL])
    require(len(rows) == 8, 'Expected eight private recordings')
    original_counts, corrected_counts, proof = [], [], []
    archives = {}
    for row in rows:
        archive = Path(row['raw_archive'])
        archives[str(archive)] = digest(archive)
        evidence = row['evidence']
        selected = evidence['selected']['attempt']
        with tarfile.open(archive) as tar:
            for attempt in evidence['attempts']:
                names = [n for n in tar.getnames() if n.endswith(attempt['raw'].removeprefix('output/'))]
                require(len(names) == 1, 'Missing or ambiguous archived receipt')
                raw = tar.extractfile(names[0]).read()
                require(hashlib.sha256(raw).hexdigest() == attempt['raw_sha256'], 'Raw receipt hash changed')
                events = [json.loads(line) for line in raw.splitlines()]
                result = replay(events)
                require(normalize_words(result['original']) == normalize_words(attempt['transcript']),
                        'Historical transcript does not replay')
                require(not result['conflicts'], 'Word-changing annotation requires separate review')
                before = word_errors(clips[row['clip_id']]['reference'], attempt['transcript'])
                attempt['transcript'] = result['transcript']
                if attempt['word_errors'] is not None:
                    attempt['word_errors'] = word_errors(clips[row['clip_id']]['reference'], result['transcript'])
                if attempt['attempt'] == 1:
                    timing = latency(clips[row['clip_id']], result['snapshots'], result['frames'], attempt['valid'])
                    attempt['first_attempt_latency'] = timing
                    evidence['first_attempt_latency'] = timing
                entry = dict(clip_id=row['clip_id'], attempt=attempt['attempt'], selected=attempt['attempt']==selected,
                    valid=attempt['valid'], raw_sha256=attempt['raw_sha256'], archive=str(archive), member=names[0],
                    duplicate_pairs=result['duplicates'], before=before, after=attempt['word_errors'])
                proof.append(entry)
                if attempt['attempt'] == selected:
                    evidence['selected'] = deepcopy(attempt)
                    original_counts.append(before)
                    corrected_counts.append(attempt['word_errors'])
    result = dict(version=VERSION, provider_calls=0, recordings=rows,
        before=aggregate_wer(original_counts), after=aggregate_wer(corrected_counts), attempts=proof,
        scope='Eight full private recordings only. Original attempt selection and validity retained. Public and turn results unchanged.',
        timing='First-attempt finalized-word delay recomputed from original receipt timestamps and corrected snapshots; not new latency measurements.',
        sources=[dict(path=str(reports / ORIGINAL), sha256=digest(reports / ORIGINAL)),
                 dict(path=str(manifest_path), sha256=digest(manifest_path)),
                 *[dict(path=p, sha256=h) for p,h in archives.items()]],
        code_hashes={p:digest(ROOT/p) for p in CODE})
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    return result


def load(reports):
    result = json.loads((reports / SOURCE).read_text())
    require(result['version'] == VERSION, 'Unexpected ElevenLabs correction version')
    for item in result['sources']:
        require(digest(item['path']) == item['sha256'], 'Original ElevenLabs evidence changed')
    for name, expected in result['code_hashes'].items():
        require(digest(ROOT/name) == expected, 'ElevenLabs correction code changed; create a new correction version')
    require(aggregate_wer([r['evidence']['selected']['word_errors'] for r in result['recordings']]) == result['after'],
            'Corrected ElevenLabs totals differ')
    return result


def restore_review(data, reports):
    from benchmark_clip_review import comparison, COUNTS
    saved = load(reports)
    rows = {r['clip_id']: r for r in saved['recordings']}
    for clip in data['clip_review']['clips']:
        if clip['cohort'] != 'private':
            continue
        attempt = rows[clip['id']]['evidence']['selected']
        counts = attempt['word_errors']
        clip['results'][MODEL].update(transcript=attempt['transcript'],
            counts={k:counts[k] for k in COUNTS}, source=SOURCE,
            diff=comparison(clip['reference'], attempt['transcript'], counts))


if __name__ == '__main__':
    reports = ROOT / 'reports'
    result = load(reports) if (reports / SOURCE).exists() else generate(reports)
    print(json.dumps({k:result[k] for k in ('version','provider_calls','before','after')},indent=2))
