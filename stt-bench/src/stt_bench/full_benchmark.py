"""Full public/private capture and deterministic offline reduction.

One process owns one stream. The controller owns assignments and retries.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
import re
from pathlib import Path
import tarfile

import numpy as np
import soundfile as sf

from . import gradium, reson8, trial_providers
from .audio_formats import derivative, resample_24k
from .credentials import redact
from .data import sha256, write_json
from .diagnostics import local_probe, validate_preflight
from .full_private_metrics import latency
from .measurement import deadline_observations, summarize_deadlines
from .providers import require_credential, transcribe, validate
from .run import assess
from .score import aggregate_wer, percentiles, word_errors
from .streaming import EventLog, read_events

MODELS = ('smallest-pulse', 'gradium-default', 'reson8-realtime', 'inworld-stt-1')
PUBLIC_LIVE_MODELS = ('gemini-3.8-live', 'gemini-3.8-live-extended-thinking')
PUBLIC = Path('datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/full')
PRIVATE = Path('reports/assemblyai-private-20260914/dataset')
PRIVATE_MANIFEST = Path('workspaces/private-longform-recovery-v2/dataset/manifest.json')
GRADIUM_SESSION_FRAMES=13500


def now():
    return datetime.now(timezone.utc).isoformat()


def load_plan(path, expected=None, runtime=None, runtime_hash=None):
    if expected and sha256(path) != expected:
        raise ValueError('Run plan hash changed')
    p = json.loads(path.read_text())
    models, workers = p['models'], p['workers_per_model']
    legacy = p.get('version', 1) == 1
    public_only = p.get('version') == 3 and p.get('public_only') is True
    if (p.get('version', 1) not in (1, 2, 3)
            or (p.get('version') == 3 and not public_only)
            or not models or len(models) != len(set(models)) or not set(models).issubset(PUBLIC_LIVE_MODELS if public_only else MODELS)
            or p['max_attempts'] != 2 or type(workers) is not int or not 1 <= workers <= 12
            or (legacy and (models != list(MODELS) or workers != 10))):
        raise ValueError('Unexpected model or execution scope')
    if p.get('ramp_after_public_pilot') and (models != ['inworld-stt-1']
            or p.get('private_pilot') is not None
            or p['configs']['inworld-stt-1'].get('transmitted_silence_frames') != 0):
        raise ValueError('Fast ramp requires the corrected Inworld-only configuration')
    ids = [c['clip_id'] for c in p['items']]
    counts = {'public': 1000} if public_only else {'public': 1000, 'private': 8}
    if len(set(ids)) != sum(counts.values()) or len(ids) != sum(counts.values()) or Counter(c['cohort'] for c in p['items']) != counts:
        raise ValueError('Full dataset coverage changed')
    if public_only and (p.get('private_pilot') is not None or p.get('private_manifest_sha256') is not None):
        raise ValueError('Public-only run cannot include a private dataset')
    if runtime:
        if not runtime_hash or sha256(Path(runtime)) != runtime_hash:
            raise ValueError('Runtime amendment hash changed')
        amendment=json.loads(Path(runtime).read_text())
        if amendment['parent_plan_sha256'] != sha256(path) or not set(amendment['code_overrides']).issubset(p['code_hashes']):
            raise ValueError('Runtime amendment cannot change dataset or configuration scope')
        p['code_hashes'].update(amendment['code_overrides'])
        p['runtime_hash']=runtime_hash
        p['private_variants']=amendment.get('private_variants',{})
    return p


def prepare(root, models=MODELS, workers=10, fast_inworld=False, public_only=False):
    if (not models or len(models) != len(set(models)) or not set(models).issubset(PUBLIC_LIVE_MODELS if public_only else MODELS)
            or type(workers) is not int or not 1 <= workers <= 12):
        raise ValueError('Invalid model selection or worker count')
    if fast_inworld and tuple(models) != ('inworld-stt-1',):
        raise ValueError('Fast ramp is limited to the validated Inworld adapter')
    root.mkdir(parents=True, exist_ok=False)
    public = json.loads((PUBLIC/'manifest.json').read_text())
    private = {'clips': []} if public_only else json.loads(PRIVATE_MANIFEST.read_text())
    items = []
    for clip in public['clips']:
        source = PUBLIC/clip['audio']
        if sha256(source) != clip['audio_sha256']:
            raise ValueError('Public source hash mismatch')
        _, converted = derivative(source, clip, PUBLIC/'derivatives/pcm24000')
        identity = converted['conversion']
        tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        items.append({**clip, 'cohort': 'public', 'audio16': str(source),
                      'audio16_sha256': sha256(source),
                      'audio24': str(PUBLIC/'derivatives/pcm24000'/f'{tag}.wav'),
                      'audio24_sha256': converted['output_sha256']})
    for clip in private['clips']:
        source = PRIVATE/clip['audio']['16000']['path']
        if sha256(source) != clip['audio']['16000']['sha256']:
            raise ValueError('Private source hash mismatch')
        items.append({**clip, 'cohort': 'private', 'condition': 'private_longform',
                      'audio16': 'full-input/private/'+source.name,
                      'audio16_sha256': clip['audio']['16000']['sha256'],
                      'audio24': 'full-input/private/'+clip['audio']['24000']['path'],
                      'audio24_sha256': clip['audio']['24000']['sha256']})
    files = sorted([Path('pyproject.toml'), Path('uv.lock'),
                    *Path('src').rglob('*.py'), *Path('config').rglob('*.json'),
                    *Path('scripts').glob('*.mjs'), *Path('scripts').glob('*.py'),
                    *Path('tests').glob('*.py'), *Path('tests').glob('*.mjs')])
    hashes = {str(p): sha256(p) for p in files if '__pycache__' not in str(p)}
    plan = dict(version=3 if public_only else 2, run_id=root.name, created_at=now(), models=list(models),
                workers_per_model=workers, max_workers=workers*len(models), max_attempts=2,
                public_manifest_sha256=sha256(PUBLIC/'manifest.json'),
                private_manifest_sha256=None if public_only else sha256(PRIVATE_MANIFEST),
                private_pilot=None if fast_inworld or public_only else private['clips'][0]['clip_id'],
                ramp_after_public_pilot=fast_inworld, items=items, code_hashes=hashes,
                configs={m: json.loads(Path(f'config/models/{m}.json').read_text()) for m in models})
    if public_only:
        plan['public_only'] = True
    write_json(root/'plan.json', plan)
    load_plan(root/'plan.json')
    with tarfile.open(root/'input.tar.gz', 'w:gz', compresslevel=1) as t:
        for name in hashes:
            t.add(name, arcname=name, recursive=False)
        t.add(root/'plan.json', arcname='full-input/plan.json')
        for clip in private['clips']:
            p = PRIVATE/clip['audio']['16000']['path']
            t.add(p, arcname='full-input/private/'+p.name)
    write_json(root/'input.json', dict(plan_sha256=sha256(root/'plan.json'),
                                      bundle_sha256=sha256(root/'input.tar.gz')))
    print(json.dumps(dict(items=len(items), audio_minutes=sum(c['submitted_seconds'] for c in items)/60)))


def verify_inputs(plan, *, convert=False):
    for name, digest in plan['code_hashes'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Frozen code changed: '+name)
    if convert:
        # All conversions finish before any stream is opened.
        for c in plan['items']:
            if c['cohort'] == 'public':
                derivative(Path(c['audio16']), c, PUBLIC/'derivatives/pcm24000')
            elif not Path(c['audio24']).exists():
                pcm, _ = sf.read(c['audio16'], dtype='int16')
                sf.write(c['audio24'], resample_24k(pcm, c['speech_frames']), 24000, subtype='PCM_16')
    for c in plan['items']:
        for rate in (16, 24):
            p = Path(c[f'audio{rate}'])
            if sha256(p) != c[f'audio{rate}_sha256']:
                raise ValueError('Frozen audio changed: '+c['clip_id'])
            info = sf.info(p)
            if info.channels != 1 or info.samplerate != rate*1000 or info.frames != (c['speech_frames']+50)*rate*20:
                raise ValueError('Invalid full audio shape')


def classify(events):
    errors = [e for e in events if e['kind'] == 'error' or
              (e['kind'] == 'provider_message' and (str(e.get('message', {}).get('type', '')).lower() == 'error'
               or e.get('message', {}).get('error') or e.get('message', {}).get('error_code')
               or e.get('message', {}).get('event') == 'error'))]
    # Never search timestamps/UUIDs for numeric HTTP codes.
    statuses={e.get('http_status') for e in errors}
    details=[]
    for e in errors:
        message=e.get('message',{})
        if isinstance(message,dict):
            statuses.add(message.get('code') if isinstance(message.get('code'),int) else None)
            details.append(json.dumps(message))
        elif isinstance(message,str):details.append(message)
        details.append(str(e.get('error_message','')))
    value=' '.join(details).lower()
    statuses.update(int(x) for x in re.findall(r'\bhttp\s+(\d{3})\b',value))
    if 402 in statuses or any(x in value for x in ('balance_exhausted', 'insufficient_credit', 'insufficient credit', 'credit limit', 'insufficient balance', 'no credits remaining', 'credits exhausted', 'out of credits')):
        return 'credits'
    if statuses.intersection({401,403}) or any(x in value for x in ('unauthorized', 'invalid api', 'authentication', 'required scopes', 'permission denied', 'access denied')):
        return 'authentication'
    if 429 in statuses or any(x in value for x in ('concurrent', 'concurrency', 'too many', 'rate limit', 'resource_exhausted', 'quota exceeded')):
        return 'concurrency'
    if any(x in value for x in ('different model', 'model mismatch', 'different audio or delay')):
        return 'model_identity'
    return 'transient' if errors else None


class SessionLog:
    def __init__(self,parent,index):self.parent,self.index,self.origin=parent,index,parent.origin
    def now(self):return self.parent.now()
    def emit(self,kind,**fields):return self.parent.emit(kind,session_index=self.index,**fields)


async def gradium_sessions(pcm,speech_frames,config,key,log):
    """Consecutive sessions; all source frames occur once, in order.

    Provider-required session tails and reconnect gaps are explicit. Timing gates
    apply inside each session; this is a separate, user-approved private variant.
    """
    frame_bytes=config['sample_rate']//50*2
    windows=[dict(index=i,offset=offset,frames=min(GRADIUM_SESSION_FRAMES,speech_frames-offset))
             for i,offset in enumerate(range(0,speech_frames,GRADIUM_SESSION_FRAMES))]
    log.emit('longform_sessions',variant=config.get('private_variant','gradium-consecutive-v1'),windows=windows,
             session_seconds=GRADIUM_SESSION_FRAMES/50,additional_tail_seconds=len(windows)-1,
             context_policy='Reset at each session; full recording reference scored once; transition gaps reported separately')
    for window in windows:
        session=SessionLog(log,window['index'])
        lo=window['offset']*frame_bytes;hi=lo+window['frames']*frame_bytes
        payload=pcm[lo:hi]+bytes(config['sample_rate']*2)
        session.emit('clip_start')
        try:
            await gradium.transcribe(payload,window['frames'],config,key,session)
        except Exception as exc:
            session.emit('error',error_type=type(exc).__name__,error_message=redact(str(exc),key)[:500])
            raise
        finally:
            session.emit('clip_end')
    log.emit('longform_complete')


def session_assessment(events,clip,config):
    header=next(e for e in events if e['kind']=='longform_sessions')
    assessed=[];snapshots=[];frames={};changes=[];segments={};gaps=[];previous_end=None
    for w in header['windows']:
        ev=[e for e in events if e.get('session_index')==w['index']]
        if not ev:break
        a=assess(ev,config);assessed.append(a)
        for e in ev:
            if e['kind']=='audio_sent' and e['index']<w['frames']:
                frames[w['offset']+e['index']]=e['send_completed_seconds']
        first=next((e['send_completed_seconds'] for e in ev if e['kind']=='audio_sent'),None)
        if previous_end is not None and first is not None:gaps.append(first-previous_end)
        previous_end=next((e['send_completed_seconds'] for e in reversed(ev) if e['kind']=='audio_sent'),None)
        changes.extend((s['time_seconds'],w['index'],s['text']) for s in final_snapshots(ev,config))
    for at,index,text in sorted(changes):
        segments[index]=text
        snapshots.append(dict(time_seconds=at,text=' '.join(segments[i] for i in sorted(segments)).strip()))
    complete=any(e['kind']=='longform_complete' for e in events) and len(assessed)==len(header['windows']) and len(frames)==clip['speech_frames'] and all(a['transcript_complete'] for a in assessed)
    pacing_valid=bool(assessed) and all(a['pacing']['valid'] for a in assessed)
    result=dict(assessed[-1]) if assessed else assess([],config)
    result.update(transcript=' '.join(a['transcript'] for a in assessed),transcript_complete=complete,
                  valid=complete and pacing_valid and all(a['valid'] for a in assessed),
                  model_verified=bool(assessed) and all(a['model_verified'] for a in assessed),
                  pacing=dict(valid=pacing_valid,sessions=[a['pacing'] for a in assessed],
                    gate_reasons=sorted(set(r for a in assessed for r in a['pacing'].get('gate_reasons',[]))),
                    transition_gaps_seconds=gaps,policy='Original gates within each session; reconnect gaps explicit'),
                  exclusion_reasons=sorted(set(r for a in assessed for r in a['exclusion_reasons'])|({'incomplete_source_coverage'} if not complete else set())),
                  transport_variant=header['variant'],session_count=len(assessed),context_resets=max(0,len(assessed)-1))
    return result,snapshots,frames


def final_snapshots(events, config):
    factory = gradium.Protocol if config['provider'] == 'gradium' else reson8.Protocol if config['provider'] == 'reson8' else trial_providers.Protocol
    protocol = factory(config)
    snapshots, previous = [], None
    for e in events:
        if e['kind'] == 'finalize_requested':
            protocol.requested = True
        elif e['kind'] == 'close_stream_requested':
            protocol.closing = True
        elif e['kind'] == 'provider_message':
            try:
                protocol.feed(e['message'])
            except Exception:
                continue
            text = protocol.snapshot()['final_text']
            if text != previous:
                snapshots.append(dict(text=text, time_seconds=e['time_seconds']))
                previous = text
    return snapshots


def evaluate(events, clip, config, number, raw_hash):
    # Historical collectors reserve message for provider JSON. Preserve the raw
    # archive, but normalize our own textual exception field before reduction.
    events=[{**{k:v for k,v in e.items() if k!='message'},'error_message':e['message']}
            if e['kind']=='error' and isinstance(e.get('message'),str) else e for e in events]
    segmented=any(e['kind']=='longform_sessions' for e in events)
    assessment,snapshots,source_frames=session_assessment(events,clip,config) if segmented else (assess(events,config),None,None)
    deadlines = deadline_observations(events, clip['reference'], clip.get('entities'), assessment, 'live', config) if clip['cohort'] == 'public' else None
    private_latency = None
    if clip['cohort'] == 'private' and number == 1:
        frames = source_frames if segmented else {e['index']: e['send_completed_seconds'] for e in events if e['kind'] == 'audio_sent'}
        private_latency = latency(clip, snapshots if segmented else final_snapshots(events, config), frames, assessment['valid'])
    retry_after = max((e.get('retry_after_seconds', 0) for e in events), default=0)
    start = next((e.get('wall_time') for e in events if e['kind']=='clip_start'), None)
    end = next((e.get('wall_time') for e in reversed(events) if e['kind']=='clip_end'), None)
    return dict(clip_id=clip['clip_id'], cohort=clip['cohort'], attempt=number,
                started_at=start, finished_at=end,
                raw_sha256=raw_hash, **assessment, failure_class=classify(events),
                retry_after_seconds=retry_after,
                word_errors=word_errors(clip['reference'], assessment['transcript']),
                deadlines=deadlines, private_latency=private_latency,
                sent_audio_seconds=sum(e.get('bytes', 0) for e in events if e['kind']=='audio_sent')/(config['sample_rate']*2))


async def worker(args):
    plan = load_plan(Path(args.plan), args.plan_hash, args.runtime, args.runtime_hash)
    assignment = json.loads(Path(args.assignment).read_text())
    if assignment['plan_hash'] != args.plan_hash or assignment['model'] not in plan['models']:
        raise ValueError('Assignment identity mismatch')
    if assignment.get('runtime_hash') != plan.get('runtime_hash'):
        raise ValueError('Assignment runtime identity mismatch')
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    config = validate(plan['configs'][assignment['model']])
    for name, digest in plan['code_hashes'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Worker code changed')
    key = require_credential(config)
    os.environ.update(STT_BENCH_COMPUTE_PROVIDER='vercel-sandbox', STT_BENCH_COMPUTE_REGION='iad1', STT_BENCH_COMPUTE_INSTANCE=args.session_id)
    state = dict(status='qualifying', assignment=assignment, started_at=now(), rows=[])
    def save():
        state['updated_at'] = now()
        write_json(root/'state.json', state)
    save()
    try:
        pacing = Path('full-input/pacing.json')
        try:
            receipt = validate_preflight(pacing)
            age = (datetime.now(timezone.utc)-datetime.fromisoformat(receipt['completed_at'])).total_seconds()
            if age + max(c['submitted_seconds'] for c in plan['items']) + 120 > 3600:
                raise ValueError('Qualification will expire')
        except (ValueError, FileNotFoundError):
            probe = root/'pacing'
            await local_probe(probe, seconds=10, repeats=3)
            validate_preflight(probe/'pacing.json')
            pacing.write_bytes((probe/'pacing.json').read_bytes())
        state['pacing'] = json.loads(pacing.read_text())
        save()
        lookup = {c['clip_id']: c for c in plan['items']}
        if len({x['clip_id'] for x in assignment['items']}) != len(assignment['items']):
            raise ValueError('Duplicate assignment')
        for item in assignment['items']:
            clip, number = lookup[item['clip_id']], item['attempt']
            if number not in (1, 2):
                raise ValueError('Attempt limit exceeded')
            rate = config['sample_rate']//1000
            source = Path(clip[f'audio{rate}'])
            if sha256(source) != clip[f'audio{rate}_sha256']:
                raise ValueError('Worker audio hash mismatch')
            pcm, sr = sf.read(source, dtype='int16')
            if sr != config['sample_rate'] or pcm.ndim != 1 or np.any(pcm[-sr:]):
                raise ValueError('Worker audio shape/tail mismatch')
            raw = root/f"{clip['clip_id']}--{number}.jsonl"
            state.update(status='running', active=clip['clip_id']); save()
            log = EventLog(raw)
            log.emit('clip_start', clip_id=clip['clip_id'], attempt=number, model=assignment['model'], plan_hash=args.plan_hash, wall_time=now())
            try:
                segmented=assignment.get('private_variant') in ('gradium-consecutive-v1','gradium-consecutive-v2') and clip['cohort']=='private'
                transport=gradium_sessions if segmented else transcribe
                transport_config={**config,'private_variant':assignment['private_variant']} if segmented else config
                await asyncio.wait_for(transport(pcm.astype('<i2').tobytes(), clip['speech_frames'], transport_config, key, log), 1.2*clip['submitted_seconds']+60)
            except Exception as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                headers = getattr(getattr(exc, 'response', None), 'headers', {})
                try:
                    wait = float(headers.get('Retry-After', 0))
                except (ValueError, TypeError):
                    wait = 0
                log.emit('error', error_type=type(exc).__name__, error_message=redact(str(exc), key)[:500], http_status=status, retry_after_seconds=wait)
            finally:
                log.emit('clip_end', wall_time=now()); log.close()
            # No active sender exists during replay, alignment or compression.
            row = evaluate(read_events(raw), clip, config, number, sha256(raw))
            write_json(root/(raw.stem+'.json'), row)
            state['rows'].append(row)
            state.pop('active', None); save()
            if row['failure_class'] in ('credits', 'authentication', 'model_identity', 'concurrency'):
                break
        state['status'] = 'finished'
    except Exception as exc:
        state.update(status='worker_failed', error_type=type(exc).__name__, error=redact(str(exc), key)[:500])
    finally:
        state['finished_at'] = now(); save()
        with tarfile.open(root/'evidence.tar.gz', 'w:gz', compresslevel=1) as t:
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.name not in ('evidence.tar.gz', 'evidence.sha256'):
                    t.add(p, arcname=str(p.relative_to(root)), recursive=False)
        (root/'evidence.sha256').write_text(sha256(root/'evidence.tar.gz')+'\n')
    print(json.dumps(dict(status=state['status'], completed=len(state['rows']))))


def replay_archive(plan, archive, expected, destination):
    if sha256(archive) != expected:
        raise ValueError('Archive checksum mismatch')
    with tarfile.open(archive) as t:
        state = json.load(t.extractfile('state.json'))
        assignment = state['assignment']
        config = plan['configs'][assignment['model']]
        lookup = {c['clip_id']: c for c in plan['items']}
        allowed = {(i['clip_id'], i['attempt']) for i in assignment['items']}
        seen = set()
        for row_index,saved in enumerate(state['rows']):
            identity = (saved['clip_id'], saved['attempt'])
            if identity not in allowed or identity in seen:
                raise ValueError('Unexpected or duplicate archive result')
            seen.add(identity)
            data = t.extractfile(f'{identity[0]}--{identity[1]}.jsonl').read()
            digest = hashlib.sha256(data).hexdigest()
            events = [json.loads(line) for line in data.splitlines()]
            replayed = evaluate(events, lookup[identity[0]], config, identity[1], digest)
            if replayed != saved:
                differences={k for k in set(replayed)|set(saved) if replayed.get(k)!=saved.get(k)}
                if differences=={'failure_class'} and not saved['valid']:
                    state.setdefault('classification_corrections',[]).append(dict(clip_id=identity[0],attempt=identity[1],previous=saved['failure_class'],corrected=replayed['failure_class']))
                    state['rows'][row_index]=replayed
                else:raise ValueError('Offline replay differs: '+identity[0])
        # A complete raw attempt remains an attempt even if post-capture scoring
        # crashed. Reconstruct it without issuing another provider request.
        reconstructed=[]
        for cid, number in sorted(allowed-seen):
            name=f'{cid}--{number}.jsonl'
            try:
                data=t.extractfile(name).read()
            except KeyError:
                continue
            events=[json.loads(line) for line in data.splitlines()]
            if not events or events[-1]['kind']!='clip_end':
                raise ValueError('Incomplete raw attempt requires explicit interruption recovery')
            row=evaluate(events,lookup[cid],config,number,hashlib.sha256(data).hexdigest())
            state['rows'].append(row);reconstructed.append(dict(clip_id=cid,attempt=number))
        if reconstructed:
            state['reconstructed_from_raw']=reconstructed
            if state.get('error_type')=='AttributeError' and state.get('error')=="'str' object has no attribute 'get'":
                state['original_worker_status']=state['status'];state['status']='finished'
        state['verified'] = True
        write_json(destination, state)
    return state


def combine(plan, states):
    all_rows = {};previous_private=[]
    for state in states:
        if not state.get('verified'):
            raise ValueError('Unverified evidence cannot enter combined scores')
        model = state['assignment']['model']
        for row in state['rows']:
            variant=state['assignment'].get('private_variant','single-session')
            wanted=plan.get('private_variants',{}).get(model,'single-session')
            if row['cohort']=='private' and variant!=wanted:
                previous_private.append(dict(model=model,variant=variant,**row));continue
            key = (model, row['clip_id'], row['attempt'])
            if key in all_rows:
                raise ValueError('Duplicate attempt in merge')
            all_rows[key] = row
    output = {}
    for model in plan.get('models', MODELS):
        records = []
        for clip in plan['items']:
            attempts = [all_rows[(model,clip['clip_id'],n)] for n in (1,2) if (model,clip['clip_id'],n) in all_rows]
            first = next((a for a in attempts if a['attempt']==1), None)
            if attempts and not first:
                raise ValueError('Recovery without original attempt')
            chosen = next((a for a in attempts if a['valid']), None)
            records.append(dict(clip_id=clip['clip_id'], cohort=clip['cohort'], attempts=attempts,
                                status='successful' if chosen else 'failed' if attempts else 'unattempted',
                                selected_attempt=chosen['attempt'] if chosen else None))
        groups = {}
        for cohort in ('combined','public','private'):
            recs = [r for r in records if cohort=='combined' or r['cohort']==cohort]
            chosen = [next(a for a in r['attempts'] if a['attempt']==r['selected_attempt']) for r in recs if r['selected_attempt']]
            firsts = [r['attempts'][0] for r in recs if r['attempts']]
            groups[cohort] = dict(planned=len(recs), successful=len(chosen),
                failed=sum(r['status']=='failed' for r in recs), unattempted=sum(r['status']=='unattempted' for r in recs),
                recovered=sum(r['selected_attempt']==2 for r in recs),
                first_attempt_successful=sum(a['valid'] for a in firsts),
                final_wer=aggregate_wer([a['word_errors'] for a in chosen]),
                first_attempt_valid_wer=aggregate_wer([a['word_errors'] for a in firsts if a['valid']]),
                attempts=sum(len(r['attempts']) for r in recs),
                failed_attempts=sum(not a['valid'] for r in recs for a in r['attempts']),
                pacing_failures=sum(not a['pacing']['valid'] for r in recs for a in r['attempts']),
                failure_classes=dict(Counter(a['failure_class'] for r in recs for a in r['attempts'] if a['failure_class'])),
                submitted_seconds=sum(a['sent_audio_seconds'] for r in recs for a in r['attempts']))
            if cohort != 'combined':
                groups[cohort]['completion_first_attempt'] = percentiles([a['completion_latency_ms'] for a in firsts if a['valid'] and a['completion_latency_ms'] is not None])
        public_rows = []
        for r in records:
            if r['cohort'] == 'public':
                d = r['attempts'][0]['deadlines'] if r['attempts'] else deadline_observations([], '', None, None, 'live', plan['configs'][model])
                public_rows.append(dict(deadlines=d))
        groups['public']['deadlines'] = summarize_deadlines(public_rows)
        metrics = [r['attempts'][0]['private_latency'] for r in records if r['cohort']=='private' and r['attempts']]
        values = [w['delay_ms'] for m in metrics if m for w in m['words']]
        groups['private']['word_finalization'] = dict(**percentiles(values),
            p95_ms=float(np.percentile(values,95)) if values else None,
            reference_words=sum(m['reference_words'] for m in metrics if m),
            exclusions=dict(sum((Counter(m['exclusions']) for m in metrics if m), Counter())))
        output[model] = dict(groups=groups, items=records)
        intervals = []
        for r in records:
            for a in r['attempts']:
                if a.get('started_at') and a.get('finished_at'):
                    intervals.extend([(a['started_at'],1),(a['finished_at'],-1)])
        active = peak = 0
        for _, change in sorted(intervals):
            active += change; peak = max(peak, active)
        output[model]['peak_capture_overlap'] = peak
        prior=[r for r in previous_private if r['model']==model]
        output[model]['previous_private_transport_submitted_seconds']=sum(r['sent_audio_seconds'] for r in prior)
        output[model]['all_variants_submitted_seconds']=groups['combined']['submitted_seconds']+sum(r['sent_audio_seconds'] for r in prior)
    return dict(run_id=plan['run_id'], generated_at=now(), models=output,private_variants=plan.get('private_variants',{}),previous_private_transport_attempts=previous_private,
                methodology='Concurrent streams; public deadlines and private word-finalization timing are distinct. Supplied references not independently listening-verified.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare','verify','worker','replay','report'))
    p.add_argument('--models', nargs='+', choices=(*MODELS, *PUBLIC_LIVE_MODELS), default=list(MODELS))
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--fast-inworld', action='store_true')
    p.add_argument('--public-only', action='store_true')
    p.add_argument('--root'); p.add_argument('--plan',default='full-input/plan.json'); p.add_argument('--plan-hash')
    p.add_argument('--assignment'); p.add_argument('--session-id'); p.add_argument('--out')
    p.add_argument('--archive'); p.add_argument('--archive-hash'); p.add_argument('--convert',action='store_true')
    p.add_argument('--runtime'); p.add_argument('--runtime-hash')
    a=p.parse_args()
    if a.mode=='prepare': prepare(Path(a.root), a.models, a.workers, a.fast_inworld, a.public_only); return
    plan=load_plan(Path(a.plan),a.plan_hash,a.runtime,a.runtime_hash)
    if a.mode=='verify': verify_inputs(plan,convert=a.convert)
    elif a.mode=='worker': asyncio.run(worker(a))
    elif a.mode=='replay': replay_archive(plan,Path(a.archive),a.archive_hash,Path(a.out))
    elif a.mode=='report':
        root=Path(a.root)
        states=[json.loads(f.read_text()) for f in sorted(root.glob('batches/*/verified.json'))]
        report=combine(plan,states)
        controller = root/'controller.json'
        if controller.exists():
            control=json.loads(controller.read_text())
            report['execution']={k:control.get(k) for k in ('status','startedAt','captureFinishedAt','finishedAt','availableWorkersPerModel','errors')}
            report['execution']['models']={m:{k:v.get(k) for k in ('ceiling','peak','reductions','blocked','privateBlocked')} for m,v in control['models'].items()}
            report['execution']['all_compute_stopped']=all(w.get('computeStopped') for w in control['workers'].values()) and control['preparation'].get('computeStopped',False)
        write_json(root/'results.json',report)
        lines=['# Full concurrent STT benchmark','',report['methodology'],'',
               '| Model | Dataset | Successful / planned | Failed | Unattempted | Final WER |',
               '|---|---|---:|---:|---:|---:|']
        for model,data in report['models'].items():
            for cohort,g in data['groups'].items():
                wer=g['final_wer']['wer']
                lines.append(f"| {model} | {cohort} | {g['successful']}/{g['planned']} | {g['failed']} | {g['unattempted']} | "+(f'{wer:.2%}' if wer is not None else 'unavailable')+' |')
        lines += ['', 'Detailed first-attempt deadlines, private word-finalization timing, completion, attempts and exclusions are in results.json.',
                  'Final WER uses valid completed results, allowing one recovery. Missing coverage remains explicit.']
        if plan.get('private_variants'):
            lines += ['', 'Gradium private uses consecutive sessions of up to 270 seconds. All original source frames are sent once in order. Context resets, per-session silence tails and reconnect gaps are explicit; timing gates apply within each session. This is a separate variant from uninterrupted private streams. Earlier transport-validation failures remain in previous_private_transport_attempts and total submitted audio.']
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        tables=[]
        for model,data in report['models'].items():
            rows=[]
            for cohort,g in data['groups'].items():
                wer=g['final_wer']['wer'];error=f'{wer:.2%}' if wer is not None else 'Unavailable'
                rows.append(f"<tr><td>{cohort}</td><td>{g['successful']} / {g['planned']}</td><td>{g['failed']}</td><td>{g['unattempted']}</td><td>{error}</td><td>{g['recovered']}</td></tr>")
            execution=report.get('execution',{}).get('models',{}).get(model,{})
            tables.append(f"<section><h2>{model}</h2><p>Peak overlapping captures: {data['peak_capture_overlap']}. Dispatch ceiling: {execution.get('ceiling','unavailable')}. Block: {html.escape(str(execution.get('blocked') or 'none'))}. Private blocked: {execution.get('privateBlocked',False)}.</p><table><thead><tr><th>Dataset</th><th>Successful / planned</th><th>Failed</th><th>Unattempted</th><th>Final WER</th><th>Recovered</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>")
        status=report.get('execution',{}).get('status','partial')
        page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full concurrent STT results</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#202522;background:#fafaf7}h1{font-size:32px}section{margin-top:36px;overflow:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}p{line-height:1.6}a{color:#206244}</style>'
        page+=f'<h1>Full concurrent STT benchmark</h1><p>Status: <strong>{status}</strong>. Each model: 1,000 public clips and eight intact private recordings.</p><p>{report["methodology"]}</p><p><a href="results.json">Complete metrics and per-item evidence index</a> · <a href="RESULTS.md">Report</a></p>'+''.join(tables)
        page+='<p>Final WER is the ratio of total word errors to reference words among valid completed results. Recovery never replaces first-attempt latency. Missing observations stay unavailable.</p>'
        if plan.get('private_variants'):page+='<p>Gradium private uses consecutive 270-second sessions with context resets, additional session tails, and measured reconnect gaps. It is not an uninterrupted-session latency comparison. Previous transport attempts are retained separately in the JSON.</p>'
        (root/'index.html').write_text(page)


if __name__=='__main__':
    main()
