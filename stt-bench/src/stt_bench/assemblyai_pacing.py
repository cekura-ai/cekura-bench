"""Actual AssemblyAI wire packets; source frames are never reported as wire sends."""
import time
import numpy as np
from .timing import deadline_timer, wait_until

POLICY='assemblyai-60ms-v1'
DESCRIPTION=('60 ms wire packets; final packet of each speech/tail phase may be 80 or 100 ms; '
             'packet-duration minus 2 ms to plus 20 ms send gaps; <=2% span drift; '
             'send duration <=40 ms; exactly 1000 ms silence; unchanged source audio')


def packet_frames(count):
    if type(count) is not int or count<3:raise ValueError('At least 60 ms required per phase')
    result=[]
    while count>=6:
        result.append(3);count-=3
    result.append(count)
    return result


async def stream_audio(pcm,speech_frames,send_audio,finalize,log,*,sample_rate=16000):
    if sample_rate!=16000 or len(pcm)!=(speech_frames+50)*640:
        raise ValueError('Expected original 16 kHz audio with exactly one second silence')
    plan=[(n,'speech') for n in packet_frames(speech_frames)]+[(n,'silence') for n in packet_frames(50)]
    log.emit('transport_packetization',policy=POLICY,speech_frames=speech_frames,tail_frames=50,source_frame_ms=20)
    async with deadline_timer() as timer:
        start=log.now();offset=0;last_start=None;last_complete=None;t0=None
        for index,(frames,phase) in enumerate(plan):
            seconds=frames*.02
            ideal=start+(offset+frames)*.02
            due=ideal if last_start is None else max(ideal,last_start+seconds-.001,last_complete+.001)
            await wait_until(log.origin+due,time.perf_counter,timer=timer)
            at=log.now();await send_audio(pcm[offset*640:(offset+frames)*640]);completed=log.now()
            log.emit('audio_sent',at=at,index=index,bytes=frames*640,sample_rate=sample_rate,phase=phase,
                     source_frames=frames,ideal_seconds=ideal,scheduled_seconds=due,
                     wakeup_delay_ms=max(0,at-due)*1000,send_duration_ms=(completed-at)*1000,
                     send_completed_seconds=completed)
            offset+=frames;last_start=at;last_complete=completed
            if offset==speech_frames:
                t0=completed;log.emit('speech_end',at=t0,index=index);await finalize(t0)
        log.emit('audio_complete',t0_seconds=t0)


def pacing_metrics(events):
    headers=[e for e in events if e['kind']=='transport_packetization']
    packets=[e for e in events if e['kind']=='audio_sent']
    reasons=[]
    if len(headers)!=1 or headers[0].get('policy')!=POLICY:
        return dict(valid=False,frames=len(packets),gate_reasons=['invalid_packetization_header'])
    h=headers[0]
    try:
        plan=[(n,'speech') for n in packet_frames(h['speech_frames'])]+[(n,'silence') for n in packet_frames(50)]
    except (ValueError,KeyError):return dict(valid=False,frames=len(packets),gate_reasons=['invalid_source_frames'])
    if h.get('tail_frames')!=50 or h.get('source_frame_ms')!=20 or len(packets)!=len(plan):
        reasons.append('incomplete_packet_coverage')
    if len(packets)<2:return dict(valid=False,frames=len(packets),gate_reasons=reasons+['too_few_packets'])
    fields=('time_seconds','ideal_seconds','scheduled_seconds','send_completed_seconds','send_duration_ms','wakeup_delay_ms')
    if any(type(e.get(k)) not in (int,float) or not np.isfinite(e[k]) for e in packets for k in fields):
        return dict(valid=False,frames=len(packets),gate_reasons=reasons+['invalid_timing_fields'])
    for i,e in enumerate(packets):
        if i>=len(plan) or (e.get('index'),e.get('source_frames'),e.get('phase'),e.get('bytes'),e.get('sample_rate'))!=(i,plan[i][0],plan[i][1],plan[i][0]*640,16000):
            reasons.append('invalid_packet_shape');break
    sent=np.array([e['time_seconds'] for e in packets]);ideal=np.array([e['ideal_seconds'] for e in packets])
    gaps=np.diff(sent)*1000
    expected_gaps=np.array([n*.02 for n,_ in plan[1:len(packets)]])*1000
    if len(expected_gaps)!=len(gaps):reasons.append('unexpected_extra_packets')
    else:
        if np.any(gaps<expected_gaps-2-1e-6):reasons.append('send_gap_below_packet_duration_minus_2ms')
        if np.any(gaps>expected_gaps+20+1e-6):reasons.append('send_gap_above_packet_duration_plus_20ms')
        if np.any(np.abs(np.diff(ideal)*1000-expected_gaps)>1e-6):reasons.append('invalid_ideal_schedule')
    expected=sum(n for n,_ in plan[1:])*.02;actual=float(sent[-1]-sent[0])
    if not .98<=actual/expected<=1.02:reasons.append('span_drift_above_2pct')
    durations=[(e['send_completed_seconds']-e['time_seconds'])*1000 for e in packets]
    if min(sent)<0 or np.any(sent<ideal-1e-9) or any(e['scheduled_seconds']>e['time_seconds'] or e['wakeup_delay_ms']<0 for e in packets) or min(durations)<0 or np.any(gaps<=0) or any(a['send_completed_seconds']>b['time_seconds'] for a,b in zip(packets,packets[1:])):
        reasons.append('invalid_timestamp_order')
    if max(durations)>40 or any(e['send_duration_ms']>40 for e in packets):reasons.append('send_duration_above_40ms')
    if any(abs(d-e['send_duration_ms'])>1e-6 for d,e in zip(durations,packets)):reasons.append('inconsistent_send_duration')
    ends=[e for e in events if e['kind']=='speech_end']
    speech_packets=len(packet_frames(h['speech_frames']))
    if len(ends)!=1 or len(packets)<speech_packets or abs(ends[0]['time_seconds']-packets[speech_packets-1]['send_completed_seconds'])>1e-6:
        reasons.append('invalid_speech_end')
    if not any(e['kind']=='audio_complete' for e in events):reasons.append('audio_incomplete')
    return dict(valid=not reasons,gate_reasons=reasons,policy=POLICY,frames=len(packets),
        wire_packet_ms=60,terminal_packet_ms=[n*20 for n,_ in plan if n!=3],
        silence_frames=50,expected_span_seconds=expected,actual_span_seconds=actual,actual_over_ideal=actual/expected,
        interval_ms_min=float(min(gaps)),interval_ms_max=float(max(gaps)),
        interval_ms_p50=float(np.percentile(gaps,50)),interval_ms_p90=float(np.percentile(gaps,90)),
        max_schedule_lag_ms=float(max((sent-ideal)*1000)),send_timing_evidence='complete',
        send_duration_ms_max=max(durations),wakeup_delay_ms_max=max(e['wakeup_delay_ms'] for e in packets),
        diagnosis='Provider-required packet duration differs from the 20 ms baseline; actual wire timings are measured.')
