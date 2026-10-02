"""Bounded public-only A/B experiment; never updates the benchmark dashboard."""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import jiwer

from stt_bench.data import load_manifest, sha256, write_json
from stt_bench.diagnostics import local_probe, validate_preflight
from stt_bench.providers import require_credential, transcript_at, validate
from stt_bench.run import run
from stt_bench.score import NORMALIZATION, aggregate_wer, percentiles, word_errors
from stt_bench.streaming import read_events

MODELS = ('assemblyai-universal-3-5-pro', 'speechmatics-standard',
          'speechmatics-enhanced', 'inworld-stt-1')
SOURCE = Path('datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/full/manifest.json')
TAP_CLIP = 'pipecat-b980f45a-7289-f63f-0923-2fe102deb8c2'
SEED = 'finalization-pilot-20260915-v1'


def profiles(model):
    baseline = json.loads(Path(f'config/models/{model}.json').read_text())
    candidate = deepcopy(baseline)
    if baseline['provider'] == 'assemblyai':
        candidate.update(force_endpoint=True, finalization='manual_at_speech_end')
    elif baseline['provider'] == 'speechmatics':
        candidate.update(force_end_of_utterance=True, max_delay=1.0,
                         finalization='manual_at_speech_end', finalize_ack_supported=True)
    else:
        candidate['voice_profile'] = {'enableVoiceProfile': False}
    for config in (baseline, candidate):
        validate(config)
    return {'baseline': baseline, 'candidate': candidate}


def select_clips(manifest):
    pool = [c for c in manifest['clips'] if 1 < c['submitted_seconds'] <= 20
            and c['condition'] == 'public_anchor' and c['clip_id'] != TAP_CLIP]
    pool.sort(key=lambda c: hashlib.sha256(f"{SEED}:{c['clip_id']}".encode()).hexdigest())
    tap = next(c for c in manifest['clips'] if c['clip_id'] == TAP_CLIP)
    selected = [*pool[:19], tap]
    if len(selected) != 20 or any(c['submitted_seconds'] > 20 for c in selected):
        raise ValueError('Expected exactly 20 short public clips')
    return selected


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    source = json.loads(SOURCE.read_text())
    clips = select_clips(source)
    frozen = root / 'dataset'
    (frozen / 'audio').mkdir(parents=True)
    for c in clips:
        src = SOURCE.parent / c['audio']
        if sha256(src) != c['audio_sha256']:
            raise ValueError('Original audio hash changed')
        shutil.copyfile(src, frozen / c['audio'])
    manifest = {**source, 'subset': 'finalization-pilot', 'clips': clips,
                'selection': {'algorithm': '19 sha256(seed:clip_id) clips with submitted duration (1,20] seconds plus known tap diagnostic',
                              'seed': SEED, 'count': 20}}
    write_json(frozen / 'manifest.json', manifest)
    load_manifest(frozen / 'manifest.json')
    for index, clip in enumerate(clips):
        write_json(frozen / f'clip-{index:02d}.json', {**manifest, 'clips': [clip]})
    (root / 'configs').mkdir()
    for model in MODELS:
        for variant, config in profiles(model).items():
            write_json(root / 'configs' / f'{model}-{variant}.json', config)
    evidence = [SOURCE, *sorted((root / 'configs').glob('*.json')),
                *sorted(frozen.glob('*.json')), *sorted(frozen.glob('audio/*')),
                *sorted(Path('src/stt_bench').glob('*.py')), Path(__file__)]
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), normalization=NORMALIZATION,
                models=list(MODELS), planned_clips_per_variant=20, max_sessions=160,
                max_attempts=1, submitted_seconds_per_variant=sum(c['submitted_seconds'] for c in clips),
                clip_ids=[c['clip_id'] for c in clips], selection=manifest['selection'],
                source_sha256=sha256(SOURCE), manifest_sha256=sha256(frozen / 'manifest.json'),
                hashes={str(p.resolve().relative_to(Path.cwd())): sha256(p) for p in evidence},
                ordering='Adjacent baseline/candidate pairs; alternate which goes first by clip index',
                concurrency='AssemblyAI, Inworld and Speechmatics workers; Speechmatics models serial',
                interpretation='Diagnostic only. Includes one deliberately selected historical failure. Same host paired comparisons; no comparison to old dashboard latency. Speechmatics changes two settings together.',
                deployment='none; dashboard and historical evidence unchanged')
    write_json(root / 'plan.json', plan)
    return plan


def observation(capture, clip, config, variant):
    outcome = json.loads((capture / 'outcomes.json').read_text())[0]
    events = read_events(capture / outcome['raw_file'])
    t0 = outcome['t0_seconds']
    errors = word_errors(clip['reference'], outcome['transcript'])
    alignment = jiwer.process_words(errors['reference_normalized'], errors['hypothesis_normalized'])
    trailing = sum(a.hyp_end_idx-a.hyp_start_idx for a in alignment.alignments[0]
                   if a.type == 'insert' and a.ref_start_idx == errors['reference_words'])
    close = next((e['time_seconds'] for e in events if e['kind'] == 'close_stream_requested'), None)
    sent = [e for e in events if e['kind'] == 'finalize_sent']
    last = outcome['final_transcript_received_seconds']
    return dict(clip_id=clip['clip_id'], variant=variant, valid=outcome['valid'],
                exclusion_reasons=outcome['exclusion_reasons'], transcript=outcome['transcript'],
                reference=clip['reference'], errors=errors, trailing_insertions=trailing,
                last_final_text_ms=(last-t0)*1000 if last is not None and t0 is not None else None,
                last_final_after_close=last > close if last is not None and close is not None else None,
                completion_latency_ms=outcome['completion_latency_ms'],
                finalize_ack_ms=outcome['finalize_latency_ms'],
                finalize_signal_count=len(sent),
                finalize_signal_ms=[(e['time_seconds']-t0)*1000 for e in sent] if t0 is not None else [],
                deadlines={str(ms): word_errors(clip['reference'], transcript_at(events, t0+ms/1000, config)['text'])
                           for ms in (0, 250, 500, 1000)} if t0 is not None else {},
                raw_file=str((capture / outcome['raw_file']).resolve()), raw_sha256=outcome['raw_sha256'])


def summarize(rows):
    by_model = {}
    for model in MODELS:
        current = [r for r in rows if r['model'] == model]
        usable = {v: {r['clip_id'] for r in current if r['variant'] == v and r['valid']}
                  for v in ('baseline', 'candidate')}
        paired = usable['baseline'] & usable['candidate']
        variants = {}
        for variant in usable:
            attempted = [r for r in current if r['variant'] == variant]
            matched = [r for r in attempted if r['clip_id'] in paired]
            variants[variant] = dict(planned=20, attempted=len(attempted), usable=len(usable[variant]),
                failed=sum(not r['valid'] for r in attempted), not_run=20-len(attempted),
                failure_reasons=dict(Counter(reason for r in attempted for reason in r['exclusion_reasons'])),
                paired_wer=aggregate_wer([r['errors'] for r in matched]),
                last_final_text=percentiles([r['last_final_text_ms'] for r in matched if r['last_final_text_ms'] is not None]),
                completion=percentiles([r['completion_latency_ms'] for r in matched if r['completion_latency_ms'] is not None]),
                trailing_insertions=sum(r['trailing_insertions'] for r in matched),
                clips_with_trailing_insertions=sum(r['trailing_insertions'] > 0 for r in matched),
                last_final_after_close=sum(r['last_final_after_close'] is True for r in matched),
                finalize_signals_sent=sum(r['finalize_signal_count'] for r in attempted),
                deadline_wer={str(ms): aggregate_wer([r['deadlines'][str(ms)] for r in matched])
                              for ms in (0, 250, 500, 1000)})
        by_model[model] = dict(paired_clips=len(paired), paired_clip_ids=sorted(paired), variants=variants)
    return dict(normalization=NORMALIZATION, models=by_model, observations=rows,
                limitation='20-clip diagnostic, including a selected historical failure; not a leaderboard or a causal proof of hallucination.')


async def live(root):
    plan = json.loads((root / 'plan.json').read_text())
    for path, expected in plan['hashes'].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'Frozen pilot input changed: {path}')
    for model in MODELS:
        require_credential(profiles(model)['baseline'])
    # An existing start marker prevents accidental double billing after an uncertain interruption.
    with (root / 'live-start.json').open('x') as f:
        json.dump({'started_at': datetime.now(timezone.utc).isoformat()}, f)
    await local_probe(root / 'pacing', seconds=10, repeats=3)
    pacing = root / 'pacing/pacing.json'
    validate_preflight(pacing)
    manifest = load_manifest(root / 'dataset/manifest.json')
    rows = []

    async def worker(models):
        for model in models:
            stopped = False
            for index, clip in enumerate(manifest['clips']):
                variants = ('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')
                for variant in variants:
                    config_path = root / 'configs' / f'{model}-{variant}.json'
                    config = json.loads(config_path.read_text())
                    capture = root / 'captures' / model / variant / f'{index:02d}'
                    print(f'PILOT {model} {variant} clip {index+1}/20', flush=True)
                    await run(root / 'dataset' / f'clip-{index:02d}.json', config_path,
                              capture, False, pacing_check=pacing, max_attempts=1,
                              stop_on_provider_failure=True)
                    row = dict(model=model, **observation(capture, clip, config, variant))
                    rows.append(row)
                    write_json(root / 'results.json', summarize(rows))
                    if 'transport_or_provider_error' in row['exclusion_reasons']:
                        stopped = True
                        print(f'STOPPED {model}: provider/transport error, no retries', flush=True)
                        break
                if stopped:
                    break
    results = await asyncio.gather(worker([MODELS[0]]), worker([MODELS[1], MODELS[2]]),
                                   worker([MODELS[3]]), return_exceptions=True)
    state = dict(finished_at=datetime.now(timezone.utc).isoformat(), sessions=len(rows),
                 status='complete' if len(rows) == 160 and not any(isinstance(r, Exception) for r in results)
                 else 'partial', worker_errors=[type(r).__name__ for r in results if isinstance(r, Exception)])
    write_json(root / 'state.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'live'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        plan = prepare(args.out)
        print(json.dumps({k: plan[k] for k in ('max_sessions', 'submitted_seconds_per_variant', 'manifest_sha256')}))
    else:
        asyncio.run(live(args.out))
