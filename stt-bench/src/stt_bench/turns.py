"""Local conversational turn datasets with manual or automatic provenance."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from .audio_formats import resample_24k
from .data import load_manifest, sha256, write_json

DATASET = 'private-turns-v1'
REVIEW_VERSION = 1


def clean(text):
    return ' '.join(re.sub(r'<[^>]*>', ' ', text).split())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def inside(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Source path escapes input directory')
    require(path.is_file(), f'Missing input: {name}')
    return path


def draft_file(path):
    path = Path(path)
    return path / 'draft.json' if path.is_dir() else path


def read_sources(source):
    """Existing metadata.json, or explicit stereo/paired input.json (see TURN_BENCHMARK.md)."""
    source = Path(source).resolve()
    index = source / ('input.json' if (source / 'input.json').exists() else 'metadata.json')
    spec = json.loads(index.read_text())
    sources, inventory = [], {str(index): sha256(index)}
    for ci, conv in enumerate(spec['conversations']):
        require(len(conv['speakers']) == 2, 'Each conversation needs exactly two speakers')
        for si, speaker in enumerate(conv['speakers']):
            audio_path = inside(source, speaker.get('audio_file', conv.get('audio_file', '')))
            meta_path = inside(source, speaker['metadata_file'])
            detail = json.loads(meta_path.read_text())
            info = sf.info(audio_path)
            channel = speaker.get('channel')
            if info.channels == 2:
                require(type(channel) is int and channel in (0, 1), 'Stereo requires explicit zero-based channel mapping')
            else:
                require(info.channels == 1 and channel in (None, 0), 'Expected mono or explicitly mapped stereo')
                channel = 0
            offset = speaker.get('starts_at_seconds', detail.get('starts_at_seconds', 0))
            require(finite(offset) and offset >= 0, 'Invalid recording synchronization offset')
            label = speaker.get('label', detail.get('speaker_label'))
            require(isinstance(label, str) and label.strip(), 'Speaker label required')
            if 'speaker_label' in detail:
                require(detail['speaker_label'] == label, 'Speaker label conflicts with metadata')
            require(info.subtype in ('PCM_16', 'PCM_24', 'PCM_32'), 'V1 requires lossless integer PCM WAV/FLAC sources')
            record = dict(source_id=f'c{ci+1:03d}-s{si+1}', conversation_id=conv['conversation_id'],
                          speaker_id=detail.get('speaker_id', label), speaker_label=label,
                          audio_path=str(audio_path), metadata_path=str(meta_path), channel=channel,
                          offset_seconds=offset, sample_rate=info.samplerate, frames=info.frames,
                          seconds=info.duration, subtype=info.subtype, segments=detail['segments'])
            record['transcript'] = detail.get('transcript', ' '.join(s['text'] for s in detail['segments']))
            sources.append(record)
            for path in (audio_path, meta_path):
                if str(path) not in inventory:
                    inventory[str(path)] = sha256(path)
        a, b = sources[-2:]
        require(a['speaker_label'] != b['speaker_label'], 'Duplicate speaker labels')
        if a['audio_path'] == b['audio_path']:
            require(a['channel'] != b['channel'] and a['offset_seconds'] == b['offset_seconds'], 'Stereo speakers must use distinct channels on one timeline')
        else:
            for own, other in ((a, b), (b, a)):
                detail = json.loads(Path(own['metadata_path']).read_text())
                partner = detail.get('other_speaker_audio_file')
                if partner:
                    require(Path(partner).name == Path(other['audio_path']).name, 'Partner recording mismatch')
    require(sources and len({s['source_id'] for s in sources}) == len(sources), 'Empty or duplicate sources')
    require(len({c['conversation_id'] for c in spec['conversations']}) == len(spec['conversations']), 'Duplicate conversation IDs')
    return sources, inventory


def prepare_turns(source, out):
    sources, inventory = read_sources(source)
    out = Path(out)
    require(not out.exists(), 'Use a new draft output directory')
    turns, units = [], {}
    for s in sources:
        sid = s['source_id']
        for i, seg in enumerate(s['segments']):
            segment_id = f'{sid}-seg{i+1:04d}'
            unit_ids, flags = [], []
            words = seg.get('words') or []
            # Segments without word timings remain represented and reviewable as one unit.
            for j, word in enumerate(words or [dict(word=seg['text'], start=seg['start'], end=seg['end'])]):
                uid = f'{segment_id}-w{j+1:04d}'
                units[uid] = dict(source_id=sid, segment_id=segment_id, text=word['word'],
                                  start=word['start'], end=word['end'], order=len(units),
                                  word_timing=bool(words))
                unit_ids.append(uid)
                if not (finite(word['start']) and finite(word['end']) and 0 <= word['start'] < word['end'] <= s['seconds']):
                    flags.append('invalid_word_timing')
                elif finite(seg['start']) and finite(seg['end']) and not seg['start'] <= word['start'] < word['end'] <= seg['end']:
                    flags.append('word_outside_segment')
            if not (finite(seg['start']) and finite(seg['end']) and 0 <= seg['start'] < seg['end'] <= s['seconds']):
                flags.append('invalid_segment_timing')
            if words and clean(seg['text']) != clean(' '.join(w['word'] for w in words)):
                flags.append('segment_word_text_disagreement')
            if any(finite(a['end']) and finite(b['start']) and b['start']-a['end'] >= 1 for a, b in zip(words, words[1:])):
                flags.append('internal_pause_at_least_1s')
            if any(finite(a['end']) and finite(b['start']) and b['start'] < a['end'] for a,b in zip(words,words[1:])):
                flags.append('overlapping_word_timings')
            if clean(s['transcript']) != clean(' '.join(x['text'] for x in s['segments'])):
                flags.append('full_segment_text_disagreement')
            turns.append(dict(turn_id=segment_id, source_id=sid, unit_ids=unit_ids,
                              start=seg['start'], end=seg['end'], reference=clean(seg['text']),
                              boundary_approved=False, transcript_approved=False,
                              status='include', reason='', notes='', flags=sorted(set(flags)),
                              word_corrections={}))
    source_map = {s['source_id']: s for s in sources}
    for i, a in enumerate(turns):
        sa = source_map[a['source_id']]
        for b in turns[i+1:]:
            sb = source_map[b['source_id']]
            if sa['conversation_id'] != sb['conversation_id'] or not all(finite(x) for x in (a['start'], a['end'], b['start'], b['end'])):
                continue
            if max(a['start']+sa['offset_seconds'], b['start']+sb['offset_seconds']) < min(a['end']+sa['offset_seconds'], b['end']+sb['offset_seconds']):
                flag = 'same_speaker_overlap' if sa['source_id'] == sb['source_id'] else 'cross_speaker_overlap'
                a['flags'] = sorted(set(a['flags']+[flag])); b['flags'] = sorted(set(b['flags']+[flag]))
    out.mkdir(parents=True)
    (out/'media').mkdir()
    # Link originals locally: this review needs no duplicate private recordings.
    for s in sources:
        link = out/'media'/f"{s['source_id']}{Path(s['audio_path']).suffix}"
        link.symlink_to(Path(s['audio_path']))
        s['review_audio'] = 'media/'+link.name
    draft = dict(schema_version=1, dataset_id=DATASET, source_inventory=inventory,
                 sources=sources, units=units, turns=turns,
                 status='draft_not_listening_reviewed')
    write_json(out/'draft.json', draft)
    review = dict(schema_version=REVIEW_VERSION, draft_sha256=sha256(out/'draft.json'),
                  source_inventory=inventory, reviewed_by='', reviewed_at='',
                  transcription_policy='Supplied text, corrected against audio; inline non-speech tags excluded from WER.',
                  turns=deepcopy(turns))
    write_json(out/'review.json', review)
    template = Path(__file__).with_name('turn_review.html').read_text()
    payload = json.dumps(dict(draft=draft, review=review), ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (out/'index.html').write_text(template.replace('__TURN_DATA__', payload))
    return out/'draft.json'


def load_draft(path):
    path = draft_file(path)
    d = json.loads(path.read_text())
    require(d.get('schema_version') == 1 and d.get('dataset_id') == DATASET, 'Unsupported turn draft')
    for name, expected in d['source_inventory'].items():
        require(Path(name).is_file() and sha256(Path(name)) == expected, f'Source changed: {name}')
    return path, d


def validate_review(draft_path, review_path):
    path, draft = load_draft(draft_path)
    r = json.loads(Path(review_path).read_text())
    return validate_review_data(draft, r, sha256(path))


def validate_review_data(draft, r, draft_hash):
    automatic = r.get('preparation_mode') == 'automatic-silero-v1'
    require(r.get('schema_version') == (2 if automatic else REVIEW_VERSION) and r.get('draft_sha256') == draft_hash, 'Review belongs to a different draft')
    require(r.get('source_inventory') == draft['source_inventory'], 'Review source hashes differ')
    if automatic:
        from .turn_auto import validate_automatic
        validate_automatic(draft,r)
    else:
        require(all(isinstance(r.get(k), str) and r[k].strip() for k in ('reviewed_by', 'reviewed_at', 'transcription_policy')), 'Reviewer, review date and transcription policy required')
    rows = r.get('turns', [])
    require(rows and len({t['turn_id'] for t in rows}) == len(rows), 'Missing or duplicate turns')
    require(all(re.fullmatch(r'[A-Za-z0-9_-]+', t['turn_id']) for t in rows), 'Unsafe turn ID')
    counts = Counter(u for t in rows for u in t['unit_ids'])
    require(set(counts) == set(draft['units']) and all(n == 1 for n in counts.values()), 'Every source word/segment unit must occur exactly once, including exclusions')
    sources = {s['source_id']: s for s in draft['sources']}
    all_corrections = {uid:w for t in rows for uid,w in t.get('word_corrections',{}).items()}
    for t in rows:
        require(t['source_id'] in sources and t['unit_ids'], 'Unknown or empty turn source')
        units = [draft['units'][u] for u in t['unit_ids']]
        require(all(u['source_id'] == t['source_id'] for u in units), 'Cannot merge different speakers')
        require([u['order'] for u in units] == sorted(u['order'] for u in units), 'Source word order changed')
        require(t.get('status') in ('include', 'exclude'), 'Invalid turn disposition')
        if t['status'] == 'exclude':
            require(isinstance(t.get('reason'), str) and t['reason'].strip(), 'Excluded turns require a reason')
            continue
        if not automatic:
            require(t.get('boundary_approved') is True and t.get('transcript_approved') is True, f"Turn is not listening reviewed: {t['turn_id']}")
        require(isinstance(t.get('reference'), str) and clean(t['reference']), 'Included turns need nonempty reference text')
        require(finite(t['start']) and finite(t['end']) and 0 <= t['start'] < t['end'] <= sources[t['source_id']]['seconds'], 'Turn boundaries outside source recording')
        corrections = t.get('word_corrections', {})
        require(set(corrections) <= set(t['unit_ids']), 'Word correction belongs to another turn')
        for uid, u in zip(t['unit_ids'], units):
            w = corrections.get(uid, u)
            require(finite(w.get('start')) and finite(w.get('end')) and t['start'] <= w['start'] < w['end'] <= t['end'], f'Word timing outside reviewed turn: {uid}; correct the word timing or turn boundary')
            if uid in corrections:
                require(isinstance(w.get('reason'), str) and w['reason'].strip(), 'Word timing correction requires a reason')
        if clean(t['reference']) != clean(' '.join(u['text'] for u in units)) or corrections:
            require(isinstance(t.get('notes'), str) and t['notes'].strip(), 'Transcript/timing correction requires review notes')
        assigned = set(t['unit_ids'])
        for uid,u in draft['units'].items():
            if uid in assigned or u['source_id'] != t['source_id'] or not clean(u['text']):
                continue
            w = all_corrections.get(uid,u)
            if finite(w['start']) and finite(w['end']) and w['start'] < w['end']:
                require(not max(t['start'],w['start']) < min(t['end'],w['end']), 'Reviewed clip contains an unassigned source word; adjust the boundary or word assignment')
    for sid in sources:
        selected = sorted((t for t in rows if t['source_id'] == sid and t['status'] == 'include'), key=lambda t: t['start'])
        require(all(a['end'] <= b['start'] for a,b in zip(selected, selected[1:])), 'Same-speaker turn clips overlap; split/merge or correct boundaries')
        orders = [draft['units'][u]['order'] for t in selected for u in t['unit_ids']]
        require(orders == sorted(orders), 'Turn order disagrees with source words')
    require(any(t['status'] == 'include' for t in rows), 'No included turns')
    return draft, r


def freeze_turns(draft_path, review_path, out):
    draft, review = validate_review(draft_path, review_path)
    automatic = review.get('preparation_mode') == 'automatic-silero-v1'
    mode = review.get('preparation_mode','manual-review-v1')
    out = Path(out)
    require(not out.exists(), 'Frozen datasets are immutable; choose a new output directory')
    ffmpeg = shutil.which('ffmpeg')
    require(ffmpeg is not None, 'ffmpeg is required for explicit audio resampling')
    converter = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    sources = {s['source_id']: s for s in draft['sources']}
    selected = sorted((t for t in review['turns'] if t['status'] == 'include'), key=lambda t: (t['source_id'], t['start']))
    # WAV master + both PCM derivatives, with headroom; no destructive space cleanup.
    required = sum((t['end']-t['start']+.1)*(sources[t['source_id']]['sample_rate']*4+80000)+80000 for t in selected)
    out.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(out.parent).free > required+16_000_000, 'Insufficient disk space for lossless crops and benchmark derivatives')
    with tempfile.TemporaryDirectory(prefix='.turn-freeze-', dir=out.parent) as temporary:
        stage = Path(temporary)/'dataset'; stage.mkdir()
        for name in ('audio', 'original', 'derivatives/pcm24000'):
            (stage/name).mkdir(parents=True)
        clips, previous_end = [], {}
        for t in selected:
            s = sources[t['source_id']]; rate = s['sample_rate']
            # Neighbor source annotations also bound context when that neighbor was excluded.
            own_units = set(t['unit_ids'])
            neighbors = [u['end'] for uid,u in draft['units'].items() if u['source_id'] == s['source_id'] and finite(u['end']) and u['end'] <= t['start'] and uid not in own_units]
            context_floor = max([0, previous_end.get(s['source_id'], 0), *neighbors])
            start_sample = max(math.floor(max(0, t['start']-.1)*rate), math.ceil(context_floor*rate-1e-8))
            end_sample = math.ceil(t['end']*rate-1e-8)
            x, _ = sf.read(s['audio_path'], start=start_sample, stop=end_sample, dtype='int32', always_2d=True)
            require(len(x) == end_sample-start_sample, 'Short source audio read')
            mono = x[:, s['channel']]
            master = stage/'original'/f"{t['turn_id']}.wav"
            sf.write(master, mono, rate, subtype=s['subtype'])
            command = [ffmpeg, '-v', 'error', '-nostdin', '-i', str(master), '-af',
                       'aresample=16000:filter_size=32:phase_shift=10:linear_interp=0:dither_method=none',
                       '-ac', '1', '-f', 's16le', '-acodec', 'pcm_s16le', 'pipe:1']
            raw = subprocess.run(command, capture_output=True, check=True).stdout
            pcm = np.frombuffer(raw, dtype='<i2').copy()
            boundary_seconds = t['end'] - start_sample/rate
            speech_frames = math.ceil(boundary_seconds*50-1e-8)
            require(abs(len(pcm)-(end_sample-start_sample)*16000/rate) <= 2, 'Unexpected resampling duration')
            require(len(pcm) <= speech_frames*320, 'Resampled speech exceeds frozen boundary')
            pcm = np.pad(pcm, (0, speech_frames*320-len(pcm)+16000))
            target = stage/'audio'/f"{t['turn_id']}.wav"
            sf.write(target, pcm, 16000, subtype='PCM_16')
            converted = stage/'derivatives/pcm24000'/target.name
            sf.write(converted, resample_24k(pcm, speech_frames), 24000, subtype='PCM_16')
            words = []
            for uid in t['unit_ids']:
                u = draft['units'][uid]; w = t.get('word_corrections', {}).get(uid, u)
                words.append(dict(unit_id=uid, text=u['text'], start=w['start']-start_sample/rate,
                                  end=w['end']-start_sample/rate, original_start=u['start'], original_end=u['end']))
            clips.append(dict(clip_id=t['turn_id'], source_id=s['source_id'], conversation_id=s['conversation_id'],
                dependency_group=s['conversation_id'], speaker_id=s['speaker_id'], source_channel=s['channel'],
                condition='private_turn', source_audio=s['audio_path'], source_metadata=s['metadata_path'],
                source_sha256=draft['source_inventory'][s['audio_path']], source_metadata_sha256=draft['source_inventory'][s['metadata_path']],
                source_sample_rate=rate, source_start_sample=start_sample, source_end_sample=end_sample,
                turn_start_seconds=t['start'], turn_end_seconds=t['end'],
                conversation_start_seconds=t['start']+s['offset_seconds'], conversation_end_seconds=t['end']+s['offset_seconds'],
                reference_original=' '.join(draft['units'][u]['text'] for u in t['unit_ids']),
                reference=clean(t['reference']), words=words, unit_ids=t['unit_ids'], entities=None,
                audio='audio/'+target.name, audio_sha256=sha256(target),
                original_audio='original/'+master.name, original_audio_sha256=sha256(master),
                derivative_24000='derivatives/pcm24000/'+converted.name, derivative_24000_sha256=sha256(converted),
                speech_frames=speech_frames, total_frames=speech_frames+50, speech_end_seconds=speech_frames*.02,
                reviewed_speech_end_seconds=None if automatic else boundary_seconds,
                estimated_speech_end_seconds=boundary_seconds if automatic else None,
                preparation_mode=mode, source_samples=end_sample-start_sample,
                submitted_seconds=(speech_frames+50)*.02, appended_silence_ms=1000,
                boundary_rounding_ms=(speech_frames*.02-boundary_seconds)*1000,
                review=dict(boundary_approved=not automatic, transcript_approved=not automatic,
                            reviewed_by=review.get('reviewed_by'), reviewed_at=review.get('reviewed_at'),
                            notes=t.get('notes',''))))
            previous_end[s['source_id']] = t['end']
        write_json(stage/'review.json', review)
        write_json(stage/'draft.json', draft)
        manifest = dict(schema_version=2, dataset_id=DATASET, dataset='Private conversational turns',
            source_revision=digest(draft['source_inventory']), subset='full', split='automatic-turns' if automatic else 'reviewed-turns', language='en',
            turn_schema_version=2 if automatic else 1, listening_review_verified=not automatic,
            preparation_mode=mode, preparation_validated=True,
            boundary_detector=review.get('detector'), transcript_source=review.get('transcript_source','listening_reviewed'),
            review_sha256=sha256(stage/'review.json'),
            draft_sha256=sha256(stage/'draft.json'), source_inventory=draft['source_inventory'],
            preprocessing=dict(sample_rate=16000, channels=1, frame_ms=20, leading_context_ms=100,
                trailing_silence_ms=1000, boundary=('automatic Silero speech end' if automatic else 'reviewed speech end')+', outward frame rounding',
                resampling=command[command.index('-af')+1], converter=converter),
            counts=dict(conversations=len({s['conversation_id'] for s in sources.values()}),
                        speaker_recordings=len(sources), turns=len(clips), excluded_turns=len(review['turns'])-len(clips)),
            exclusions=[t for t in review['turns'] if t['status']=='exclude'], clips=clips)
        write_json(stage/'manifest.json', manifest)
        verify_turn_manifest(stage/'manifest.json')
        stage.rename(out)
    return out/'manifest.json'


def verify_turn_manifest(path):
    path = Path(path); m = load_manifest(path)
    automatic = m.get('preparation_mode') == 'automatic-silero-v1'
    require(m.get('dataset_id') == DATASET and m.get('turn_schema_version') == (2 if automatic else 1) and
            m.get('listening_review_verified') is (not automatic), 'Expected frozen validated turn manifest')
    require(sha256(inside(path.parent, 'review.json')) == m['review_sha256'], 'Frozen review changed')
    require(sha256(inside(path.parent, 'draft.json')) == m['draft_sha256'], 'Frozen draft changed')
    # Freeze-time evidence is self-contained: runs need not access original private paths.
    d = json.loads((path.parent/'draft.json').read_text()); r = json.loads((path.parent/'review.json').read_text())
    require(automatic == (r.get('preparation_mode')=='automatic-silero-v1'), 'Manifest preparation mode differs from evidence')
    if automatic:
        require(m.get('preparation_validated') is True and m.get('boundary_detector')==r.get('detector'), 'Automatic preparation evidence differs')
    validate_review_data(d,r,m['draft_sha256'])
    require(r['draft_sha256'] == m['draft_sha256'] and r['source_inventory'] == m['source_inventory'] == d['source_inventory'], 'Inconsistent source provenance')
    approved = {t['turn_id']: t for t in r['turns'] if t['status']=='include'}
    require(set(approved) == {c['clip_id'] for c in m['clips']}, 'Turn/review coverage differs')
    for c in m['clips']:
        t = approved[c['clip_id']]
        require(t['boundary_approved'] is (not automatic) and t['transcript_approved'] is (not automatic) and
                c['review']['boundary_approved'] is (not automatic) and c['review']['transcript_approved'] is (not automatic) and
                c['reference'] == clean(t['reference']) and c['unit_ids'] == t['unit_ids'] and
                c['turn_start_seconds'] == t['start'] and c['turn_end_seconds'] == t['end'], 'Frozen clip differs from approved review')
        for field in ('original_audio', 'derivative_24000'):
            p = inside(path.parent, c[field])
            require(sha256(p) == c[field+'_sha256'], 'Frozen audio derivative changed')
        info = sf.info(path.parent/c['derivative_24000'])
        require(info.channels == 1 and info.samplerate == 24000 and info.frames == c['total_frames']*480, 'Invalid 24 kHz derivative')
    return m
