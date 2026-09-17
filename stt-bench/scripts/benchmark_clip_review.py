"""Portable listening review, derived only from the dashboard's selected evidence."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

from unified_benchmark import COUNTS, aggregate, check_counts, require


def comparison(reference, transcript, counts):
    """Use the scorer's normalizer and jiwer alignment; reject different scores."""
    import jiwer
    from stt_bench.score import normalize_words
    ref, hyp = normalize_words(reference), normalize_words(transcript)
    require(ref == counts['reference_normalized'] and hyp == counts['hypothesis_normalized'],
            'Review text differs from saved scoring text')
    alignment = jiwer.process_words(ref, hyp)
    actual = dict(substitutions=alignment.substitutions, insertions=alignment.insertions,
                  deletions=alignment.deletions, reference_words=len(ref.split()))
    check_counts(actual, counts)
    rw, hw = ref.split(), hyp.split()
    return [[a.type, ' '.join(rw[a.ref_start_idx:a.ref_end_idx]),
             ' '.join(hw[a.hyp_start_idx:a.hyp_end_idx])]
            for a in alignment.alignments[0]]


def review_counts(reference, transcript, counts):
    """Accept either saved scoring version, then verify and apply current scoring."""
    from stt_bench.score import NORMALIZER, normalize_words, rescore_saved_word_errors
    require(any(normalize(reference) == counts['reference_normalized'] and
                normalize(transcript) == counts['hypothesis_normalized']
                for normalize in (NORMALIZER, normalize_words)),
            'Original review text differs from saved scoring text')
    return rescore_saved_word_errors(counts)


def prepare_audio(source, expected_hash, destination):
    """Lossless listening copy of frozen 16 kHz input, including its silence tail."""
    import numpy as np
    import soundfile as sf
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    require(raw_hash == expected_hash, f'Frozen review audio changed: {source.name}')
    samples, rate = sf.read(source, dtype='int16', always_2d=True)
    require(rate == 16000 and samples.shape[1] == 1, 'Expected mono 16 kHz review audio')
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, samples, rate, format='FLAC', subtype='PCM_16')
    restored, restored_rate = sf.read(destination, dtype='int16', always_2d=True)
    require(restored_rate == rate and np.array_equal(samples, restored),
            f'Listening audio is not lossless: {destination.name}')
    return dict(sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                input_sha256=raw_hash, seconds=len(samples) / rate)


def build_review(data, reports, root, output, include_private=False):
    manifests = [
        ('pipecat', root / 'datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/full/manifest.json', None),
        ('fleurs', root / 'datasets/fleurs-en-us-deepgram-v2/manifest.json', None),
    ]
    if include_private:
        manifests.append(('private', root / 'workspaces/private-longform-recovery-v2/dataset/manifest.json',
                          reports / 'assemblyai-private-20260914/dataset'))
    clips, manifest_sources = {}, []
    for cohort, manifest, audio_root in manifests:
        raw = manifest.read_bytes()
        manifest_sources.append(dict(path=str(manifest.relative_to(root)),
                                     sha256=hashlib.sha256(raw).hexdigest()))
        for row in json.loads(raw)['clips']:
            cid = row['clip_id']
            require(re.fullmatch(r'[A-Za-z0-9_-]+', cid), 'Unsafe clip ID')
            key = (cohort, cid)
            require(key not in clips, 'Duplicate review clip')
            if cohort == 'private':
                source = audio_root / row['audio']['16000']['path']
                expected = row['audio']['16000']['sha256']
            else:
                source, expected = manifest.parent / row['audio'], row['audio_sha256']
            relative = f'audio/{cohort}/{cid}.flac'
            audio = prepare_audio(source, expected, output / relative)
            clips[key] = dict(id=cid, cohort=cohort, reference=row['reference'],
                              original_reference=row.get('reference_original', row['reference']),
                              audio=relative, **audio, results={})

    cache = {}
    known_hashes = {s['path'].removeprefix('reports/'): s['sha256'] for s in data['sources']}
    for model in data['models']:
        for name in model['sources']:
            if name not in cache:
                raw = (reports / name).read_bytes()
                require(hashlib.sha256(raw).hexdigest() == known_hashes[name], 'Review source changed during build')
                cache[name] = json.loads(raw)
            saved = cache[name]
            if name == 'assemblyai-full-20260914/private/wire60/state.json':
                from assemblyai_standard_evidence import load
                saved = load(reports)
            if 'clips' in saved:
                rows = [('fleurs' if r['clip_id'].startswith('fleurs-') else 'pipecat', r, None)
                        for r in saved['clips']
                        if not name.startswith('trial-short-20260914/')
                        or ('pipecat', r['clip_id']) in clips]
            elif 'recordings' in saved:
                rows = [('private', r, r['evidence']) for r in saved['recordings'] if r['model'] == model['id']]
            else:
                rows = [('pipecat' if r['cohort'] == 'public' else 'private', r, r)
                        for r in saved['models'][model['id']]['items']]
            for cohort, row, evidence in rows:
                key = (cohort, row['clip_id'])
                if cohort == 'private' and not include_private:
                    continue
                require(key in clips, f'Unknown review clip: {key}')
                clip = clips[key]
                require(model['id'] not in clip['results'], 'Duplicate model/clip review result')
                attempts = (evidence or row)['attempts']
                if evidence is None:
                    chosen = next((a for a in attempts if a['attempt'] == row.get('selected_attempt')), None)
                    counts = row['word_errors'] if row['accuracy_usable'] else None
                elif 'selected' in evidence:
                    chosen = evidence['selected']
                    counts = chosen['word_errors'] if chosen else None
                else:
                    chosen = next((a for a in attempts if a['attempt'] == evidence['selected_attempt']), None)
                    counts = chosen['word_errors'] if chosen else None
                shown = chosen if chosen is not None else (attempts[-1] if attempts else None)
                transcript = shown.get('transcript') if shown else None
                if counts is not None:
                    require(chosen is not None and chosen['valid'] and isinstance(transcript, str),
                            'Scored review has no valid selected transcript')
                    if 'reference' in row:
                        require(row['reference'] == clip['reference'], 'Reference differs from frozen manifest')
                    counts = review_counts(clip['reference'], transcript, counts)
                    diff = comparison(clip['reference'], transcript, counts)
                else:
                    diff = None
                clip['results'][model['id']] = dict(
                    status='scored' if counts is not None else ('excluded' if attempts else 'not_run'),
                    transcript=transcript, attempt=shown['attempt'] if shown else None,
                    attempts=len(attempts), failed_attempts=sum(not a['valid'] for a in attempts),
                    counts={k: counts[k] for k in COUNTS} if counts is not None else None,
                    diff=diff, source=name, raw_sha256=shown.get('raw_sha256') if shown else None)
        for cohort, _, _ in manifests:
            good = [c['results'][model['id']]['counts'] for c in clips.values()
                    if c['cohort'] == cohort and model['id'] in c['results']
                    and c['results'][model['id']]['counts'] is not None]
            check_counts(aggregate(good), model['cohorts'][cohort])
            require(len(good) == model['cohorts'][cohort]['usable'], 'Review usable coverage differs')
    return dict(clips=list(clips.values()), manifests=manifest_sources, includes_private=include_private,
                audio_format='Lossless FLAC of the frozen mono 16 kHz input; terminal silence retained.')


def share_zip(page, review, proof_root=None, *, visibility='public'):
    """Package exactly this build's assets, excluding stale files and screenshots."""
    if visibility not in ('public', 'private-review'):
        raise ValueError('Unknown export visibility')
    if visibility == 'public' and (review.get('includes_private') or proof_root is not None):
        raise ValueError('Private evidence requires explicit private-review export')
    if visibility == 'public':
        for clip in review['clips']:
            require(clip['cohort'] in ('pipecat', 'fleurs'), 'Public ZIP cannot include private recordings')
            asset = page.parent / clip['audio']
            require(asset.resolve().is_relative_to((page.parent / 'audio' / clip['cohort']).resolve()),
                    'Public audio must stay inside its dataset directory')
    destination = page.parent / 'benchmark-review.zip'
    readme = ('Extract this entire ZIP, then open index.html in Chrome, Edge, Firefox, or Safari.\n'
              'Keep the audio folder beside index.html. No server or internet is needed.\n'
              'Use Verify clips to listen, choose a model, and inspect transcript differences.\n'
              f"Includes {len(review['clips'])} recordings. Private recordings included: {review['includes_private']}.\n"
              'Audio is losslessly compressed; transcript scores come from the saved benchmark.\n'
              'Listening can verify transcript errors, but cannot independently verify network latency.\n')
    if proof_root is not None:
        readme += 'Keep turn-proof beside index.html. It contains turn audio, model proof pages, compressed provider receipt logs, and source hashes.\n'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr('README.txt', readme)
        archive.write(page, 'index.html')
        for clip in review['clips']:
            archive.write(page.parent / clip['audio'], clip['audio'], compress_type=zipfile.ZIP_STORED)
        if proof_root is not None:
            require(proof_root.parent.resolve() == page.parent.resolve(), 'Proof must be beside the dashboard')
            for asset in sorted(proof_root.rglob('*')):
                if asset.is_file():
                    archive.write(asset, str(asset.relative_to(page.parent)))
    return destination
