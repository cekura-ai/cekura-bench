#!/usr/bin/env python3
"""Measure accepted-session overlap and new-session rate from saved raw events."""
import argparse
from collections import defaultdict, deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile


def summarize(root):
    state=json.loads((root/'controller.json').read_text())
    intervals=defaultdict(list); starts=defaultdict(list); terminal=defaultdict(int)
    for batch in state['batches'].values():
        if batch['status']!='collected' or not batch.get('verified'):continue
        archive=root/'batches'/batch['id']/'evidence.tar.gz'
        with archive.open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==batch['archiveHash']
        with tarfile.open(archive) as t:
            for member in t:
                if not member.name.endswith('.jsonl') or '/' in member.name:continue
                selected=[]
                for line in t.extractfile(member):
                    # Sender events dominate archives; no transcript content is needed.
                    if b'"kind": "audio_sent"' in line or b'"kind":"audio_sent"' in line:continue
                    e=json.loads(line)
                    if e['kind'] in ('clip_start','clip_end','connection_requested','model_accepted','provider_terminal'):selected.append(e)
                origin=next((e for e in selected if e['kind']=='clip_start' and e.get('wall_time')),None)
                if not origin:continue
                epoch=datetime.fromisoformat(origin['wall_time']).timestamp()-origin['time_seconds']
                end=max(e['time_seconds'] for e in selected)
                accepted={}
                for e in selected:
                    session=e.get('session_index',0);at=epoch+e['time_seconds']
                    if e['kind']=='connection_requested':starts[batch['model']].append(at)
                    elif e['kind']=='model_accepted':
                        assert session not in accepted, 'Duplicate model-accepted event'
                        accepted[session]=at
                    elif e['kind']=='provider_terminal' and session in accepted:
                        intervals[batch['model']].append((accepted.pop(session),at));terminal[batch['model']]+=1
                for at in accepted.values():intervals[batch['model']].append((at,epoch+end))
    output={}
    for model in state['models']:
        points=sorted([(start,1) for start,end in intervals[model]]+[(end,-1) for start,end in intervals[model]])
        active=peak=0
        for _,change in points:active+=change;peak=max(peak,active)
        window=deque();rate=0
        for at in sorted(starts[model]):
            while window and at-window[0]>=60:window.popleft()
            window.append(at);rate=max(rate,len(window))
        gate=state.get('startRate')
        post_gate_rate=None
        if gate:
            cutover=max(e['ackAt'] for key,e in gate['entries'].items() if key.startswith('cutover-'))/1000
            later=deque();post_gate_rate=0
            for at in sorted(t for t in starts[model] if t>=cutover):
                while later and at-later[0]>=60:later.popleft()
                later.append(at);post_gate_rate=max(post_gate_rate,len(later))
        output[model]={'post_gate_peak_session_starts_in_60_seconds':post_gate_rate,
                       'post_gate_rate_limit_respected':None if post_gate_rate is None else post_gate_rate<=gate['limit'],
                       'peak_accepted_session_overlap':peak,'peak_session_starts_in_60_seconds':rate,
                       'requested_sessions':len(starts[model]),'accepted_sessions':len(intervals[model]),
                       'terminal_confirmed_sessions':terminal[model],
                       'definition':'Server acceptance to terminal confirmation, or capture end for failed sessions. Includes recovery and prior transport variants. Wall-clock alignment uses sandbox UTC clocks.'}
    (root/'stream-concurrency.json').write_text(json.dumps(output,indent=2)+'\n')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(summarize(p.parse_args().root),indent=2))
