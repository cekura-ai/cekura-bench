"""Automatic, local speech-end preparation. Never asserts human review."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .data import sha256, write_json
from .turns import clean, finite, load_draft, require

MODE = 'automatic-silero-v1'
RESAMPLING = 'aresample=16000:filter_size=32:phase_shift=10:linear_interp=0:dither_method=none'
SETTINGS = dict(sample_rate=16000, threshold=.5, min_speech_duration_ms=100,
                min_silence_duration_ms=300, speech_pad_ms=100)


class SileroDetector:
    def __init__(self):
        try:
            import torch
            import silero_vad
        except ImportError as e:
            raise ValueError('Install local preparation dependencies: uv sync --extra turn-preparation') from e
        require(importlib.metadata.version('silero-vad') == '6.2.1', 'Expected pinned silero-vad 6.2.1')
        torch.set_num_threads(1)
        self.torch, self.timestamps = torch, silero_vad.get_speech_timestamps
        self.model = silero_vad.load_silero_vad()
        model_path = Path(silero_vad.__file__).parent/'data/silero_vad.jit'
        self.identity = dict(engine='silero-vad', version='6.2.1', model_sha256=sha256(model_path),
                             runtime='torch-cpu', runtime_version=torch.__version__, settings=SETTINGS)

    def __call__(self, samples):
        with self.torch.inference_mode():
            return self.timestamps(self.torch.from_numpy(samples.copy()), self.model,
                sampling_rate=SETTINGS['sample_rate'], threshold=SETTINGS['threshold'],
                min_speech_duration_ms=SETTINGS['min_speech_duration_ms'],
                min_silence_duration_ms=SETTINGS['min_silence_duration_ms'],
                speech_pad_ms=SETTINGS['speech_pad_ms'])


def candidate_bounds(draft, t):
    """Expand segment edges to contain its words, without crossing another segment."""
    s = next(s for s in draft['sources'] if s['source_id'] == t['source_id'])
    units = [draft['units'][uid] for uid in t['unit_ids']]
    if not finite(t['start']) or not finite(t['end']) or not 0 <= t['start'] < t['end'] <= s['seconds']:
        return None, 'invalid_segment_timing'
    if any(not finite(u['start']) or not finite(u['end']) or not 0 <= u['start'] < u['end'] <= s['seconds'] for u in units):
        return None, 'invalid_word_timing'
    if clean(t['reference']) != clean(' '.join(u['text'] for u in units)):
        return None, 'segment_word_text_disagreement'
    if not clean(t['reference']):
        return None, 'empty_reference'
    if 'full_segment_text_disagreement' in t['flags']:
        return None, 'full_segment_text_disagreement'
    if 'same_speaker_overlap' in t['flags']:
        return None, 'same_speaker_overlap'
    start = min(t['start'], *(u['start'] for u in units))
    word_end = max(u['end'] for u in units)
    end = max(t['end'], word_end)
    others = [u for uid,u in draft['units'].items() if u['source_id']==t['source_id'] and uid not in t['unit_ids']]
    if any(finite(u['start']) and finite(u['end']) and max(start,u['start']) < min(end,u['end']) for u in others):
        return None, 'unassigned_word_in_candidate'
    preceding = max([0., *(u['end'] for u in others if finite(u['end']) and u['end'] <= start)])
    following = min([s['seconds'], *(u['start'] for u in others if finite(u['start']) and u['start'] >= end)])
    following = min([following, *(x['start'] for x in draft['turns'] if x['source_id']==t['source_id'] and
                     x['turn_id']!=t['turn_id'] and finite(x['start']) and x['start'] >= end)])
    first = max(0, math.ceil(preceding*16000-1e-8), math.floor(max(0,start-.1)*16000))
    last = math.floor(min(following, end+.3)*16000+1e-8)
    return dict(start=start, word_end=word_end, window_start_sample=first, window_end_sample=last), None


def automatic_preparation(draft_path, out, *, detector=None):
    path, draft = load_draft(draft_path)
    out = Path(out)
    require(not out.exists(), 'Automatic preparation evidence is immutable; use a new path')
    ffmpeg = shutil.which('ffmpeg')
    require(ffmpeg, 'ffmpeg is required for automatic preparation')
    detector = detector or SileroDetector()
    rows = deepcopy(draft['turns'])
    for t in rows:
        t.update(boundary_approved=False, transcript_approved=False, word_corrections={},
                 status='exclude', reason='', notes='')
    converter = subprocess.run([ffmpeg,'-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    source_audio = {}
    for s in draft['sources']:
        # Explicit selection: stereo channels must never be averaged.
        raw = subprocess.run([ffmpeg,'-v','error','-nostdin','-i',s['audio_path'],'-af',
            f"pan=mono|c0=c{s['channel']},"+RESAMPLING, '-f','f32le','pipe:1'],capture_output=True,check=True).stdout
        samples = np.frombuffer(raw,dtype='<f4')
        require(abs(len(samples)-s['seconds']*16000) <= 2, 'VAD resampling changed duration')
        import hashlib
        source_audio[s['source_id']] = dict(pcm_sha256=hashlib.sha256(raw).hexdigest(), samples=len(samples))
        for t in (t for t in rows if t['source_id']==s['source_id']):
            bounds, reason = candidate_bounds(draft,t)
            if reason:
                t['reason'] = reason
                continue
            lo,hi = bounds['window_start_sample'], min(bounds['window_end_sample'],len(samples))
            segments = detector(samples[lo:hi])
            t['automatic'] = dict(**bounds, segments=segments)
            if not segments:
                t['reason'] = 'no_speech_detected'
                continue
            end = (lo+segments[-1]['end'])/16000
            if end < bounds['word_end']:
                t['reason'] = 'vad_end_before_reference_word_end'
                continue
            if end <= bounds['start']:
                t['reason'] = 'vad_end_before_turn_start'
                continue
            t.update(start=bounds['start'],end=end,status='include',reason='',
                     notes='Supplied transcript retained; final speech boundary estimated by Silero VAD. No listening review.')
        print(f"Automatic speech-end detection: {s['source_id']} complete", flush=True)
    result = dict(schema_version=2, preparation_mode=MODE, draft_sha256=sha256(path),
        source_inventory=draft['source_inventory'], prepared_at=datetime.now(timezone.utc).isoformat(),
        listening_review_verified=False, transcript_source='supplied_annotations_not_independently_verified',
        transcription_policy='Supplied segment text; inline non-speech tags excluded from WER.',
        detector=detector.identity, resampling=RESAMPLING, converter=converter, vad_audio=source_audio,
        counts=dict(candidates=len(rows),included=sum(t['status']=='include' for t in rows),
                    excluded=sum(t['status']=='exclude' for t in rows)),
        exclusion_reasons=dict(Counter(t['reason'] for t in rows if t['status']=='exclude')),turns=rows)
    # This also checks source immutability after the potentially long VAD pass.
    load_draft(path)
    from .turns import validate_review_data
    validate_review_data(draft,result,sha256(path))
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,result)
    return out


def validate_automatic(draft, preparation):
    require(preparation.get('schema_version')==2 and preparation.get('preparation_mode')==MODE,
            'Unsupported automatic preparation mode')
    require(preparation.get('listening_review_verified') is False, 'Automatic preparation cannot claim listening review')
    detector = preparation.get('detector',{})
    require(detector.get('engine')=='silero-vad' and detector.get('version')=='6.2.1' and
            detector.get('settings')==SETTINGS and isinstance(detector.get('model_sha256'),str) and
            len(detector['model_sha256'])==64, 'Missing or inconsistent pinned VAD evidence')
    require(preparation.get('resampling')==RESAMPLING and preparation.get('converter'), 'Missing VAD preprocessing provenance')
    require(set(preparation.get('vad_audio',{}))=={s['source_id'] for s in draft['sources']}, 'Missing VAD source evidence')
    candidates = {t['turn_id']:t for t in draft['turns']}
    require(set(candidates)=={t['turn_id'] for t in preparation['turns']}, 'Automatic candidate coverage changed')
    for t in preparation['turns']:
        original = candidates[t['turn_id']]
        require(t['source_id']==original['source_id'] and t['unit_ids']==original['unit_ids'] and
                t['reference']==original['reference'] and not t.get('word_corrections'), 'Automatic preparation changed transcript provenance')
        require(t.get('boundary_approved') is False and t.get('transcript_approved') is False,
                'Automatic preparation cannot claim human approval')
        if t['status']=='exclude':
            continue
        bounds,reason = candidate_bounds(draft,original)
        require(reason is None, 'Unsafe automatic candidate')
        evidence = t.get('automatic',{})
        require(all(evidence.get(k)==v for k,v in bounds.items()), 'Automatic crop evidence changed')
        segments = evidence.get('segments',[])
        span = bounds['window_end_sample']-bounds['window_start_sample']
        require(segments and all(type(x.get('start')) is int and type(x.get('end')) is int and
                0 <= x['start'] < x['end'] <= span for x in segments), 'Invalid VAD speech segments')
        require(all(a['end'] <= b['start'] for a,b in zip(segments,segments[1:])), 'VAD segments out of order')
        expected_end = (bounds['window_start_sample']+segments[-1]['end'])/16000
        require(t['start']==bounds['start'] and t['end']==expected_end and expected_end>=bounds['word_end'],
                'Automatic speech boundary differs from evidence or removes reference words')
