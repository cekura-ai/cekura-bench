"""Recreate only frozen 16 kHz private derivatives, checking original hashes."""
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
import soundfile as sf
from stt_bench.data import sha256


def prepare():
    original=Path('workspaces/private-longform-v1/dataset/manifest.json')
    m=json.loads(original.read_text())
    root=Path('reports/assemblyai-private-20260914');base=root/'dataset'
    base.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(original,base/'manifest.json')
    n=np.arange(-60,61,dtype=float);kernel=np.sinc(n/3)*np.kaiser(121,5);kernel/=kernel.sum()
    for clip in m['clips']:
        source=Path('private-dataset')/clip['source_audio'];a=clip['audio']['16000'];target=base/a['path']
        if target.exists() and sha256(target)==a['sha256']:continue
        if sha256(source)!=m['source_inventory'][clip['source_audio']]:raise ValueError('Private original changed')
        info=sf.info(source)
        if info.samplerate!=48000 or info.channels!=1:raise ValueError('Source format changed')
        with sf.SoundFile(target,'w',samplerate=16000,channels=1,subtype='PCM_16') as output:
            # Exact FIR context, on the same three-sample phase as the frozen transform.
            for start in range(0,info.frames,480000):
                end=min(start+480000,info.frames);left=max(0,start-60);right=min(info.frames,end+60)
                audio,_=sf.read(source,start=left,stop=right,dtype='float64')
                filtered=np.convolve(audio,kernel,mode='full')[60:60+len(audio)]
                y=filtered[start-left:end-left:3]
                output.write(np.clip(np.rint(y*32768),-32768,32767).astype('<i2'))
            padding=clip['speech_frames']*320-output.tell()
            output.write(np.zeros(padding+16000,dtype='<i2'))
        if sha256(target)!=a['sha256']:raise ValueError('Rebuilt audio differs from frozen hash')
        print(clip['clip_id']+' verified',flush=True)
    short=Path('reports/trial-short-20260914/dataset')
    smoke=next(c for c in json.loads((short/'manifest.json').read_text())['clips'] if c['condition']=='private_short')
    if sha256(short/smoke['audio'])!=smoke['audio_sha256']:raise ValueError('Smoke audio changed')
    (base/'smoke.json').write_text(json.dumps(smoke))
    shutil.copyfile(short/smoke['audio'],base/'smoke.wav')
    bundle=root/'dataset.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as t:
        for p in sorted(base.iterdir()):t.add(p,arcname='dataset/'+p.name)
    receipt=dict(bundle_sha256=sha256(bundle),manifest_sha256=sha256(original),
                 recordings=8,audio_seconds=sum(c['submitted_seconds'] for c in m['clips']),
                 smoke_seconds=smoke['submitted_seconds'],bytes=bundle.stat().st_size)
    (root/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':prepare()
