"""Read the overlooked standard-profile archive without counting smoke twice."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

MODEL = 'assemblyai-universal-3-5-pro'
SOURCE = 'assemblyai-full-20260914/private/wire60/state.json'


def load(reports):
    root = Path(reports) / 'assemblyai-full-20260914'
    sources = []
    def verified_archive(variant):
        path = root / variant / 'evidence.tar.gz'
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = (path.parent / 'evidence.sha256').read_text().split()[0]
        if actual != expected: raise ValueError('AssemblyAI archive checksum changed')
        sources.append(dict(path='reports/' + str(path.relative_to(reports)), sha256=actual))
        return tarfile.open(path)
    items = []
    with verified_archive('private/wire60') as archive:
        raw = (root / 'private/wire60/state.json').read_bytes()
        state = json.loads(raw)
        if json.load(archive.extractfile('state.json')) != state:
            raise ValueError('AssemblyAI private state differs from its archive')
        if state['status'] != 'complete' or state['usable_recordings'] != 8 or state['model']['model_id'] != MODEL:
            raise ValueError('AssemblyAI private completion or identity differs')
        for row in state['recordings']:
            wire = archive.extractfile(row['clip_id'] + '.jsonl').read()
            if hashlib.sha256(wire).hexdigest() != row['raw_sha256']:
                raise ValueError('AssemblyAI private raw hash differs')
            attempt = dict(deepcopy(row['protocol']), attempt=1, valid=row['valid'],
                transcript=row['transcript'], word_errors=deepcopy(row['word_errors']),
                raw_sha256=row['raw_sha256'], pacing=row['pacing'], exclusion_reasons=row['errors'])
            items.append(dict(clip_id=row['clip_id'],cohort='private', selected_attempt=1 if row['valid'] else None, attempts=[attempt]))
        totals = {key:sum(r['word_errors'][key] for r in state['recordings']) for key in ('substitutions','insertions','deletions','reference_words')}
        if any(totals[k] != state['word_errors'][k] for k in totals):raise ValueError('AssemblyAI private counts differ')
    with verified_archive('pipecat/wire60') as archive:
        saved = json.load(archive.extractfile('batch-0000/scored/results.json'))
        if saved['run_configuration'] != state['model']:
            raise ValueError('AssemblyAI public/private profiles differ')
        for row in saved['clips']:
            if not row['attempts']: continue
            for attempt in row['attempts']:
                member='batch-0000/capture/raw/' + row['clip_id'] + '--attempt-' + str(attempt['attempt']) + '.jsonl'
                wire=archive.extractfile(member).read()
                if hashlib.sha256(wire).hexdigest() != attempt['raw_sha256']:
                    raise ValueError('AssemblyAI public raw hash differs')
                attempt['word_errors']=deepcopy(row['word_errors']) if attempt['valid'] else None
                attempt['deadlines']=row['deadlines']
            items.append(dict(clip_id=row['clip_id'],cohort='public',reference=row['reference'],
                selected_attempt=row['selected_attempt'] if row['accuracy_usable'] else None,attempts=row['attempts']))
    return dict(models={MODEL:dict(items=items)}, source_archives=sources,
        note='Standard profile: 8/8 private recordings plus 9 usable of 10 attempted main-run public clips. Smoke clips are excluded. Public coverage is incomplete; no fixed-set rank.')


def restore_review(data, reports):
    """Attach the recovered transcripts using the same checked display scorer."""
    from benchmark_clip_review import review_counts, comparison
    saved=load(reports)
    clips={(c['cohort'],c['id']):c for c in data['clip_review']['clips']}
    for item in saved['models'][MODEL]['items']:
        cohort='pipecat' if item['cohort']=='public' else 'private'
        clip=clips[(cohort,item['clip_id'])]
        attempt=next((a for a in item['attempts'] if a['attempt']==item['selected_attempt']),None)
        shown=attempt or item['attempts'][-1]
        counts=review_counts(clip['reference'],shown['transcript'],attempt['word_errors']) if attempt else None
        clip['results'][MODEL]=dict(status='scored' if counts else 'excluded',transcript=shown['transcript'],
            attempt=shown['attempt'],attempts=len(item['attempts']),failed_attempts=sum(not a['valid'] for a in item['attempts']),
            counts={k:counts[k] for k in ('substitutions','insertions','deletions','reference_words')} if counts else None,
            diff=comparison(clip['reference'],shown['transcript'],counts) if counts else None,
            source=SOURCE,raw_sha256=shown['raw_sha256'])
    return saved
