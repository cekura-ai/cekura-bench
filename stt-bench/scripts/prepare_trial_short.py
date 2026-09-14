"""Freeze ten short private excerpts plus the existing ten Pipecat smoke clips."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf
from stt_bench.data import prepare_audio, load_manifest, sha256, write_json


def candidates(clip):
    words = [w for w in clip['words'] if w['text']]
    groups = []; group = []
    for w in words:
        if group and w['start'] - group[-1]['end'] >= .35:
            groups.append(group); group = []
        group.append(w)
    if group:
        groups.append(group)
    valid = [g for g in groups if 3 <= g[-1]['end']-g[0]['start'] <= 14 and len(g) >= 6
             and all(w['timing_valid'] for w in g)
             and all(a['end'] <= b['start'] for a,b in zip(g,g[1:]))]
    return sorted(valid, key=lambda g: hashlib.sha256(f"42:{clip['clip_id']}:{g[0]['start']}".encode()).hexdigest())


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False); (out/'audio').mkdir()
    source_manifest = Path('workspaces/private-longform-v1/dataset/manifest.json')
    private = json.loads(source_manifest.read_text())
    # One excerpt from each of the eight recordings, then a second from the first two.
    selected = [(clip, rank) for rank in range(2) for clip in private['clips']][:10]
    clips=[]
    for clip, rank in selected:
        source = Path('private-dataset')/clip['source_audio']
        if sha256(source) != private['source_inventory'][clip['source_audio']]:
            raise ValueError('Private source hash changed')
        choices=candidates(clip)
        group=choices[rank]
        start=math.floor(max(0,group[0]['start']-.1)*50)/50
        end=math.ceil((group[-1]['end']+.2)*50)/50
        # Read only the excerpt plus the FIR context from the original 48 kHz file.
        start_sample=int(round(start*48000)); end_sample=int(round(end*48000))
        left=max(0,start_sample-60); right=end_sample+60
        audio,rate=sf.read(source,start=left,stop=right,dtype='float64')
        if rate!=48000 or audio.ndim!=1:raise ValueError('Unexpected source format')
        n=np.arange(-60,61,dtype=float)
        kernel=np.sinc(n/3)*np.kaiser(121,5);kernel/=kernel.sum()
        filtered=np.convolve(audio,kernel,mode='full')[60:60+len(audio)]
        y=filtered[start_sample-left:end_sample-left:3]
        pcm=np.clip(np.rint(y*32768),-32768,32767).astype('<i2')
        with tempfile.TemporaryDirectory() as temp:
            raw=Path(temp)/'excerpt.wav';sf.write(raw,pcm,16000,subtype='PCM_16')
            prepared,details=prepare_audio(raw)
        if details['speech_end_seconds'] < group[-1]['end']-start-.04:
            raise ValueError('VAD would remove annotated speech; inspect candidate before selection')
        name=f"private-{clip['clip_id']}-{rank+1:02d}"
        target=out/'audio'/f'{name}.wav';sf.write(target,prepared,16000,subtype='PCM_16')
        clips.append(dict(clip_id=name,source_id=clip['clip_id'],condition='private_short',
            conversation_id=clip['conversation_id'],dependency_group=str(clip['conversation_id']),
            reference=' '.join(w['text'] for w in group),entities=None,audio=f'audio/{name}.wav',
            audio_sha256=sha256(target),source_sha256=private['source_inventory'][clip['source_audio']],
            source_start_seconds=start,source_end_seconds=end,source_sample_rate=48000,
            boundary_review={'method':'pause-bounded supplied word timestamps and automatic VAD',
                             'human_listening_reviewed':False}, **details))
    pipecat_path=Path('datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/smoke/manifest.json')
    pipecat=load_manifest(pipecat_path)
    # Public clips first: an invalid subscription stops before private audio is sent.
    public=[]
    for clip in pipecat['clips']:
        target=out/'audio'/f"{clip['clip_id']}.wav"
        shutil.copyfile(pipecat_path.parent/clip['audio'],target)
        public.append(dict(clip,audio='audio/'+target.name))
    manifest=dict(schema_version=2,dataset_id='trial-short-private-pipecat-v1',
        dataset='private excerpts + pipecat-ai/stt-benchmark-data',subset='10-private-10-pipecat',
        split='diagnostic-short-sample',language='en-US',source_revision=private['revision'],
        private_source_manifest_sha256=sha256(source_manifest),pipecat_source_manifest_sha256=sha256(pipecat_path),
        selection={'private':'seed 42 hash order of pause-bounded 3-14 second spans; one per recording then second from first two',
                   'pipecat':'unchanged existing ten-clip smoke subset'},
        preprocessing={'sample_rate':16000,'channels':1,'frame_ms':20,'trailing_silence_ms':1000,
                       'boundary':'automatic WebRTC VAD; not listening verified'},
        clips=public+clips)
    write_json(out/'manifest.json',manifest);load_manifest(out/'manifest.json')
    print(json.dumps({'manifest':str(out/'manifest.json'),'sha256':sha256(out/'manifest.json'),
        'clips':len(manifest['clips']),'seconds_by_cohort':{c:sum(x['submitted_seconds'] for x in manifest['clips'] if x['condition']==c)
        for c in ('public_anchor','private_short')}}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    prepare(parser.parse_args().out)
