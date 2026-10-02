"""Public-only, 16-session stream-ending comparison; no retries or leaderboard writes."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from stt_bench.data import load_manifest, sha256, write_json
from stt_bench.diagnostics import local_probe, validate_preflight
from stt_bench.providers import require_credential, validate
from stt_bench.run import run
from stt_bench.score import aggregate_wer, percentiles, word_errors
from stt_bench.streaming import read_events

SOURCE = Path('datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/full/manifest.json')
KNOWN = ('pipecat-00065230-be8a-6c26-14ec-e531e0539e5b',
         'pipecat-4919946c-1e72-1af5-a14e-8b378cfe7890',
         'pipecat-b980f45a-7289-f63f-0923-2fe102deb8c2')


def prepare(root):
    source = load_manifest(SOURCE)
    lookup = {c['clip_id']: c for c in source['clips']}
    pool = [c for c in source['clips'] if c['clip_id'] not in KNOWN
            and c['condition'] == 'public_anchor' and 1 < c['submitted_seconds'] <= 15]
    pool.sort(key=lambda c: hashlib.sha256(('inworld-stream-end-v2:' + c['clip_id']).encode()).hexdigest())
    clips = [*(lookup[cid] for cid in KNOWN), *pool[:5]]
    assert len(clips) == 8 and all(c['condition'] == 'public_anchor' for c in clips)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'dataset/audio').mkdir(parents=True)
    manifest = {**source, 'clips': clips, 'subset': 'inworld-stream-end-v2'}
    for i, clip in enumerate(clips):
        target = root / 'dataset' / clip['audio']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE.parent / clip['audio'], target)
        write_json(root / 'dataset' / f'clip-{i:02d}.json', {**manifest, 'clips': [clip]})
    write_json(root / 'dataset/manifest.json', manifest)
    c = json.loads(Path('config/models/inworld-stt-1.json').read_text())
    for variant, tail in [('baseline', 50), ('candidate', 0)]:
        frozen = dict(c, transmitted_silence_frames=tail)
        validate(frozen)
        write_json(root / f'{variant}.json', frozen)
    files = [*root.rglob('*.json'), *root.glob('dataset/audio/*'),
             *Path('src/stt_bench').glob('*.py'), Path(__file__)]
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), max_sessions=16,
                attempts_per_clip=1, clip_ids=[c['clip_id'] for c in clips],
                selection='Three historical diagnostic clips plus five deterministic duration-filtered clips',
                baseline_audio_seconds=sum(c['submitted_seconds'] for c in clips),
                candidate_audio_seconds=sum(c['speech_frames'] * .02 for c in clips),
                hashes={str(p.resolve().relative_to(Path.cwd())): sha256(p) for p in files},
                interpretation='Diagnostic sample, not a general WER estimate. Same audio, voice profile and pacing; only transmitted silence tail changes.')
    write_json(root / 'plan.json', plan)
    return plan


def summarize(rows):
    paired = {r['clip_id'] for r in rows if r['variant'] == 'baseline' and r['valid']} & {
        r['clip_id'] for r in rows if r['variant'] == 'candidate' and r['valid']}
    result = dict(planned_sessions=16, attempted_sessions=len(rows), paired_clips=len(paired),
                  observations=rows, variants={})
    for variant in ('baseline', 'candidate'):
        attempted = [r for r in rows if r['variant'] == variant]
        valid = [r for r in attempted if r['clip_id'] in paired]
        result['variants'][variant] = dict(attempted=len(attempted), usable=sum(r['valid'] for r in attempted),
            paired_wer=aggregate_wer([r['word_errors'] for r in valid]),
            paired_without_repetition_example=aggregate_wer([r['word_errors'] for r in valid if r['clip_id'] != KNOWN[2]]),
            last_final_ms=percentiles([r['last_final_ms'] for r in valid if r['last_final_ms'] is not None]),
            transmitted_audio_seconds=sum(r['sent_audio_seconds'] for r in attempted))
    return result


async def live(root):
    plan = json.loads((root / 'plan.json').read_text())
    for filename, digest in plan['hashes'].items():
        if sha256(Path(filename)) != digest:
            raise ValueError('Frozen probe input changed: ' + filename)
    require_credential(json.loads((root / 'candidate.json').read_text()))
    # A failed/uncertain invocation cannot be silently repeated or double billed.
    with (root / 'live-start.json').open('x') as f:
        json.dump({'started_at': datetime.now(timezone.utc).isoformat()}, f)
    await local_probe(root / 'pacing', seconds=10, repeats=3)
    pacing = root / 'pacing/pacing.json'
    validate_preflight(pacing)
    manifest = load_manifest(root / 'dataset/manifest.json')
    rows = []
    for i, clip in enumerate(manifest['clips']):
        for variant in (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline')):
            capture = root / 'captures' / variant / f'{i:02d}'
            await run(root / 'dataset' / f'clip-{i:02d}.json', root / f'{variant}.json', capture,
                      False, pacing_check=pacing, max_attempts=1, stop_on_provider_failure=True)
            a = json.loads((capture / 'outcomes.json').read_text())[0]
            raw = capture / a['raw_file']
            assert sha256(raw) == a['raw_sha256']
            events = read_events(raw)
            t0, last = a['t0_seconds'], a['final_transcript_received_seconds']
            audio = [e for e in events if e['kind'] == 'audio_sent']
            finalize = next((e['time_seconds'] for e in events if e['kind'] == 'finalize_sent'), None)
            rows.append(dict(clip_id=clip['clip_id'], variant=variant, valid=a['valid'],
                exclusion_reasons=a['exclusion_reasons'], transcript=a['transcript'], reference=clip['reference'],
                word_errors=word_errors(clip['reference'], a['transcript']),
                last_final_ms=(last - t0) * 1000 if last is not None and t0 is not None else None,
                sent_audio_seconds=sum(e['bytes'] for e in audio) / 32000,
                audio_frames_after_finalize=sum(e['time_seconds'] > finalize for e in audio) if finalize is not None else None,
                pacing=a['pacing'], raw_file=str(raw), raw_sha256=a['raw_sha256']))
            write_json(root / 'results.json', summarize(rows))
            if 'transport_or_provider_error' in a['exclusion_reasons']:
                raise RuntimeError('Provider failure; stopped without retries or more submissions')
    result = summarize(rows)
    result.update(completed_at=datetime.now(timezone.utc).isoformat(), status='complete',
                  interpretation=plan['interpretation'])
    write_json(root / 'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'live'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.out), indent=2))
    else:
        result = asyncio.run(live(args.out))
        print(json.dumps({k: v for k, v in result.items() if k != 'observations'}, indent=2))
