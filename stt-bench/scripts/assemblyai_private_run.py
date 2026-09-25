"""Private AssemblyAI test: private smoke then all eight full sequential streams."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tarfile
import numpy as np
import soundfile as sf
from stt_bench import assemblyai
from stt_bench.credentials import credential
from stt_bench.data import sha256, write_json
from stt_bench.diagnostics import local_probe, validate_preflight
from stt_bench.score import word_errors, aggregate_wer
from stt_bench.streaming import EventLog, read_events, pacing_metrics


def evaluate(raw, clip, config):
    events = read_events(raw)
    snap, result = assemblyai.replay(events, config)
    pacing = pacing_metrics(events)
    stats = next((e['message'] for e in reversed(events)
                  if e['kind'] == 'provider_message' and e['message'].get('type') == 'Termination'), None)
    valid = result['transcript_complete'] and result['model_verified'] and pacing['valid']
    deadlines = {}
    for ms in (0, 250, 500, 1000):
        if result['t0_seconds'] is not None:
            at = assemblyai.replay(events, config, result['t0_seconds'] + ms/1000)[0]
            deadlines[str(ms)] = word_errors(clip['reference'], at['text'])
    return dict(clip_id=clip['clip_id'], valid=valid, transcript=snap['final_text'],
                word_errors=word_errors(clip['reference'], snap['final_text']),
                deadline_word_errors=deadlines, pacing=pacing, protocol=result,
                session_statistics=stats, raw_sha256=sha256(raw),
                errors=[e for e in events if e['kind']=='error'])


def verify_dataset(path):
    m = json.loads(path.read_text())
    if m['mode'] != 'private-longform-v1' or len(m['clips']) != 8:
        raise ValueError('Expected the eight original private recordings')
    for c in m['clips']:
        a = c['audio']['16000']; p = path.parent/a['path']
        if not p.resolve().is_relative_to(path.parent.resolve()) or sha256(p) != a['sha256']:
            raise ValueError('Private audio identity mismatch')
        info = sf.info(p)
        if info.samplerate != 16000 or info.channels != 1 or info.frames != a['samples']:
            raise ValueError('Invalid PCM source')
        if info.frames != (c['speech_frames'] + 50)*320:
            raise ValueError('Invalid private frame count')
        tail, _ = sf.read(p, start=info.frames-16000, dtype='int16')
        if np.any(tail): raise ValueError('Missing silence tail')
    return m


async def main(args):
    if not args.live: raise ValueError('--live required')
    os.environ.update(STT_BENCH_COMPUTE_PROVIDER='vercel-sandbox',STT_BENCH_COMPUTE_REGION='iad1',
                      STT_BENCH_COMPUTE_INSTANCE=args.session_id)
    dataset=Path(args.dataset); manifest=dataset/'manifest.json'
    if sha256(manifest) != args.manifest_sha256: raise ValueError('Manifest hash mismatch')
    m=verify_dataset(manifest)
    config=json.loads(Path('config/models/assemblyai-universal-3-5-pro.json').read_text())
    key, _=credential('assemblyai', env_file=None)
    if not key: raise ValueError('AssemblyAI credential missing')
    root=Path(args.out); root.mkdir(exist_ok=False, parents=True)
    state=dict(status='qualifying', started_at=datetime.now(timezone.utc).isoformat(),
               planned_recordings=8, concurrency=1, max_attempts=1, recordings=[],
               model=config, manifest_sha256=args.manifest_sha256,
               compute=dict(provider='vercel-sandbox',region='iad1',session_id=args.session_id),
               notes=['Private data only; no public provider calls.',m['annotation_review'],
                      'Deadline WER is at recording end, not per-utterance latency.',
                      'Native endpointing; provider-managed default context carryover.',
                      '60 ms AssemblyAI wire packets with packet-aware timing gates; not equivalent to the 20 ms baseline.'])
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        write_json(root/'state.json',state)
    save()
    async def capture(clip, path):
        def audio():
            pcm,_=sf.read(path,dtype='int16'); return pcm.astype('<i2').tobytes()
        payload=await asyncio.to_thread(audio)
        raw=root/(clip['clip_id']+'.jsonl'); log=EventLog(raw)
        try:
            await asyncio.wait_for(assemblyai.transcribe(payload,clip['speech_frames'],config,key,log),
                                   clip['submitted_seconds']*1.1+45)
        except asyncio.CancelledError:
            log.emit('error',error_type='Interrupted')
            raise
        except Exception as exc:
            log.emit('error',error_type=type(exc).__name__)
        finally:
            log.close()
        row=await asyncio.to_thread(evaluate,raw,clip,config)
        write_json(root/(clip['clip_id']+'.json'),row)
        return row
    try:
        await local_probe(root/'pacing',seconds=10,repeats=3)
        validate_preflight(root/'pacing/pacing.json')
        state['status']='smoke_running';save()
        smoke=json.loads((dataset/'smoke.json').read_text())
        state['smoke']=await capture(smoke,dataset/'smoke.wav');save()
        if not state['smoke']['valid']:
            state['status']='smoke_failed';return
        state['status']='running';save()
        for index,clip in enumerate(m['clips']):
            probe=root/f'pacing-recording-{index:02d}'
            await local_probe(probe,seconds=10,repeats=3)
            validate_preflight(probe/'pacing.json')
            state['active_recording']=clip['clip_id'];save()
            row=await capture(clip,dataset/clip['audio']['16000']['path'])
            state['recordings'].append(row);save()
            print(json.dumps(dict(clip=clip['clip_id'],valid=row['valid'],completed=len(state['recordings']))),flush=True)
            if row['errors']:
                state['status']='stopped_on_provider_error';return
        state.pop('active_recording',None)
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        state['finished_at']=datetime.now(timezone.utc).isoformat()
        good=[r for r in state['recordings'] if r['valid']]
        state['usable_recordings']=len(good)
        state['word_errors']=aggregate_wer([r['word_errors'] for r in good])
        state['all_attempts_word_errors']=aggregate_wer([r['word_errors'] for r in state['recordings']])
        state['provider_session_seconds']=sum(r['session_statistics'].get('session_duration_seconds',0)
            for r in [state.get('smoke',{}),*state['recordings']] if r.get('session_statistics'))
        save()
        with tarfile.open(root/'evidence.tar.gz','w:gz',compresslevel=1) as archive:
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.name not in ('evidence.tar.gz','evidence.sha256'):
                    archive.add(p,arcname=str(p.relative_to(root)))
        (root/'evidence.sha256').write_text(sha256(root/'evidence.tar.gz')+'  evidence.tar.gz\n')
        print(json.dumps(dict(status=state['status'],usable=len(good))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true')
    for name in ('dataset','manifest-sha256','session-id','out'):p.add_argument('--'+name,required=True)
    asyncio.run(main(p.parse_args()))
