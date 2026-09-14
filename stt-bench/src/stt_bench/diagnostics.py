"""Local duplex streaming diagnostics and an audit of saved send timing."""
import asyncio
from collections import Counter
import contextlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect

from .data import FRAME_BYTES, write_json, sha256
from .streaming import EventLog, pacing_metrics, read_events, stream_audio
from .timing import process_activity_identity, scheduler_backend


def audit_pacing(run_dir, out):
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    clips = {c['clip_id']: c for c in manifest['clips']}
    rows = []
    for path in sorted((run_dir / 'raw').glob('*.jsonl')):
        events = read_events(path, allow_truncated_final=True)
        start = next((e for e in events if e['kind'] == 'clip_start'), {})
        clip = clips.get(start.get('clip_id'), {})
        metrics = pacing_metrics(events)
        rows.append(dict(raw_file=path.name, clip_id=start.get('clip_id'), attempt=start.get('attempt', 1),
                         submitted_seconds=clip.get('submitted_seconds'), **metrics))
    report = dict(attempts=len(rows), valid_attempts=sum(r['valid'] for r in rows),
                  gate_reasons=dict(Counter(reason for r in rows for reason in r.get('gate_reasons', []))), rows=rows,
                  interpretation='Observed client timing only. Send waits cannot separate remote backpressure from network or local stalls.')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'pacing.json', report)
    return report


def pacing_identity():
    owned_loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = owned_loop = asyncio.new_event_loop()
    try:
        selector = getattr(loop, '_selector', None)
        runtime = dict(event_loop=f'{type(loop).__module__}.{type(loop).__qualname__}',
                       selector=f'{type(selector).__module__}.{type(selector).__qualname__}',
                       scheduler_backend=scheduler_backend(),
                       process_activity=process_activity_identity())
    finally:
        if owned_loop is not None:
            owned_loop.close()
    context = {key: os.environ.get('STT_BENCH_' + key.upper()) for key in
               ('compute_provider', 'compute_region', 'compute_instance')}
    if any(context.values()):
        if not all(context.values()):
            raise ValueError('Remote compute provenance requires provider, region and instance')
        runtime['execution_environment'] = context
    return dict(host=platform.node(), system=platform.platform(), python=platform.python_version(),
                runtime=runtime,
                source_hashes={name: sha256(Path(__file__).parent / name)
                               for name in ('streaming.py', 'diagnostics.py', 'probe_server.py', 'timing.py',
                                            'macos_timer.py', 'macos_activity.py')})


def validate_preflight(path):
    if path is None:
        raise ValueError('Live runs require --pacing-check from a passing local probe (at least 3 trials, 10 seconds each)')
    report = json.loads(path.read_text())
    if report.get('probe_version') != 4 or report.get('receiver_process') != 'independent':
        raise ValueError('Legacy pacing check lacks independent duplex qualification; run a fresh probe')
    if report.get('mode') != 'local_websocket_probe' or report.get('identity') != pacing_identity():
        raise ValueError('Pacing check belongs to different code, runtime or host; run a fresh probe')
    completed = datetime.fromisoformat(report['completed_at'])
    if completed.tzinfo is None:
        raise ValueError('Pacing check completion time must include a timezone')
    age = (datetime.now(timezone.utc) - completed).total_seconds()
    trials = report.get('trials', [])
    failures = []
    if not 0 <= age <= 3600:
        failures.append('completion must be within the past hour')
    if report.get('seconds', 0) < 10 or len(trials) < 3 or len(trials) != report.get('repeats'):
        failures.append('at least three complete trials of ten seconds are required')
    if any(report.get(field) != 0 for field in ('injected_send_stall_ms', 'receiver_delay_ms',
                                               'receiver_stall_ms', 'client_stall_ms')):
        failures.append('fault injections must all be zero')
    for index, trial in enumerate(trials, 1):
        if not all(trial.get(field) for field in ('valid', 'send_pacing_valid', 'duplex_valid',
                                                   'client_delivery_valid', 'fixture_valid')):
            reasons = trial.get('gate_reasons') or ['missing or failed duplex qualification']
            failures.append(f"trial {index}: {', '.join(reasons)}")
    if failures:
        raise ValueError('Pacing check failed: ' + '; '.join(failures))
    return dict(path=str(path.resolve()), sha256=sha256(path), completed_at=report['completed_at'])


def duplex_metrics(events, expected_frames):
    """Qualify actual emission coverage and delivery, independent of timer phase.

    Scheduled-deadline lateness remains diagnostic. A late heartbeat can still
    arrive within the permitted actual emission gap when the preceding heartbeat
    was also late; absolute timer phase adds no delivery requirement.
    """
    messages = [e for e in events if e['kind'] == 'local_receiver_reply']
    acknowledgments = [e for e in messages if e['message'].get('type') == 'audio_ack']
    heartbeats = [e for e in messages if e['message'].get('type') == 'heartbeat']
    summaries = [e['message'] for e in messages if e['message'].get('type') == 'summary']
    summary = summaries[-1] if summaries else {}
    received = summary.get('received_seconds', [])
    expected = list(range(expected_frames))
    indexes = summary.get('indexes', [])
    ack_indexes = [e['message'].get('index') for e in acknowledgments]
    fixture_reasons, client_reasons = [], []
    if (len(summaries) != 1 or summary.get('frames') != expected_frames
            or len(received) != expected_frames or summary.get('invalid_frames')):
        fixture_reasons.append('receiver_frame_inventory_mismatch')
    if indexes != expected:
        fixture_reasons.append('receiver_frame_order_invalid')
    if ack_indexes != expected:
        client_reasons.append('receiver_frame_order_invalid')
    if received != [e['message'].get('received_seconds') for e in acknowledgments]:
        client_reasons.append('receiver_acknowledgment_mismatch')
    if not heartbeats:
        client_reasons.append('independent_heartbeat_missing')
    emitted = summary.get('emitted_heartbeats')
    if not isinstance(emitted, list) or not emitted or not all(isinstance(h, dict) for h in emitted):
        fixture_reasons.append('heartbeat_emission_inventory_missing')
        client_reasons.append('heartbeat_delivery_unverifiable')
        emitted = []
    elif emitted != [e['message'] for e in heartbeats]:
        client_reasons.append('heartbeat_emission_inventory_mismatch')
    if any(e['kind'] == 'probe_error' for e in events):
        fixture_reasons.append('probe_execution_error')
    starts = [e.get('absolute_seconds') for e in events if e['kind'] == 'probe_audio_started']
    finishes = [e.get('absolute_seconds') for e in events if e['kind'] == 'probe_audio_finished']
    if len(starts) != 1 or len(finishes) != 1:
        fixture_reasons.append('heartbeat_coverage_unavailable')
    output = dict(receiver_frames=len(received),
                  receiver_frame_order_valid=indexes == expected and ack_indexes == expected,
                  receiver_gap_ms_max=None, heartbeat_messages=len(heartbeats),
                  heartbeat_send_lateness_ms_max=None, heartbeat_coverage_gap_ms_max=None,
                  heartbeat_emission_coverage_gap_ms_max=None,
                  emitted_heartbeat_messages=len(emitted),
                  message_receipt_delay_ms_max=None, message_receipt_delay_ms_p99=None,
                  reply_delay_ms_max=None, reply_delay_ms_p99=None)
    qualification_complete = False

    def finish():
        if not qualification_complete:
            if not fixture_reasons:
                fixture_reasons.append('fixture_qualification_unavailable')
            if not client_reasons:
                client_reasons.append('client_delivery_qualification_unavailable')
        return {**output, 'fixture_valid': not fixture_reasons,
                'client_delivery_valid': not client_reasons,
                'fixture_gate_reasons': fixture_reasons, 'client_gate_reasons': client_reasons,
                'duplex_valid': not fixture_reasons and not client_reasons,
                'duplex_gate_reasons': list(dict.fromkeys(fixture_reasons + client_reasons))}

    fixture_timestamps = list(received) + starts + finishes
    fixture_timestamps += [h.get(key) for h in emitted for key in ('sent_seconds', 'scheduled_seconds')]
    client_timestamps = [e.get('received_absolute_seconds') for e in acknowledgments + heartbeats]
    client_timestamps += [e['message'].get('received_seconds') for e in acknowledgments]
    client_timestamps += [e['message'].get('sent_seconds') for e in acknowledgments + heartbeats]
    client_timestamps += [e['message'].get('scheduled_seconds') for e in heartbeats]
    def invalid_timestamps(values):
        return any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in values)
    if invalid_timestamps(fixture_timestamps):
        fixture_reasons.append('invalid_duplex_timestamps')
    if invalid_timestamps(client_timestamps):
        client_reasons.append('invalid_duplex_timestamps')
    if 'invalid_duplex_timestamps' in fixture_reasons + client_reasons:
        return finish()  # Invalid raw evidence must never leak NaN into JSON metrics.
    gaps = [(b - a) * 1000 for a, b in zip(received, received[1:])]
    heartbeat_late = [(h['sent_seconds'] - h['scheduled_seconds']) * 1000 for h in emitted]
    message_delays = [(e['received_absolute_seconds'] - e['message']['sent_seconds']) * 1000
        for e in acknowledgments + heartbeats]
    reply_delays = [(e['received_absolute_seconds'] - e['message'][
        'received_seconds' if e['message']['type'] == 'audio_ack' else 'sent_seconds']) * 1000
        for e in acknowledgments + heartbeats]
    if any(not math.isfinite(value) or value < 0 for value in gaps + heartbeat_late):
        fixture_reasons.append('invalid_duplex_timestamps')
    if any(not math.isfinite(value) or value < 0 for value in message_delays + reply_delays):
        client_reasons.append('invalid_duplex_timestamps')
    if 'invalid_duplex_timestamps' in fixture_reasons + client_reasons:
        return finish()
    sequences = [h.get('sequence') for h in emitted]
    invalid_sequence = any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                           for value in sequences)
    if sequences and sequences[0] != 0:
        invalid_sequence = True
    if not invalid_sequence:
        for old, new in zip(emitted, emitted[1:]):
            # Mirror the child policy: a late heartbeat skips expired slots, never
            # emits a burst. Unexplained sequence gaps mean messages were lost.
            skipped = max(1, math.floor((old['sent_seconds'] - old['scheduled_seconds']) / .020) + 1)
            if (new['sequence'] - old['sequence'] != skipped
                    or abs(new['scheduled_seconds'] - old['scheduled_seconds'] - skipped * .020) > 1e-6):
                invalid_sequence = True
    if invalid_sequence:
        fixture_reasons.append('heartbeat_sequence_invalid')
    client_heartbeat_times = [e['received_absolute_seconds'] for e in heartbeats]
    if any(b < a for a, b in zip(client_heartbeat_times, client_heartbeat_times[1:])):
        client_reasons.append('invalid_duplex_timestamps')
        return finish()
    emitted_times = [h['sent_seconds'] for h in emitted]
    if any(b < a for a, b in zip(emitted_times, emitted_times[1:])):
        fixture_reasons.append('invalid_duplex_timestamps')
        return finish()
    coverage_gaps, emission_gaps = [], []
    if len(starts) == 1 and len(finishes) == 1:
        if finishes[0] <= starts[0]:
            fixture_reasons.append('invalid_duplex_timestamps')
            return finish()
        coverage = [starts[0]] + [at for at in client_heartbeat_times
                                   if starts[0] < at < finishes[0]] + [finishes[0]]
        coverage_gaps = [(b - a) * 1000 for a, b in zip(coverage, coverage[1:])]
        emission_coverage = [starts[0]] + [at for at in emitted_times
                                          if starts[0] < at < finishes[0]] + [finishes[0]]
        emission_gaps = [(b - a) * 1000 for a, b in zip(emission_coverage, emission_coverage[1:])]
        if max(emission_gaps) > 40:
            fixture_reasons.append('heartbeat_emission_coverage_gap_above_40ms')
    # Actual source coverage above proves the stimulus cadence. Timer phase is
    # neither an additional cadence constraint nor a client receipt measurement.
    if message_delays and max(message_delays) > 40:
        client_reasons.append('message_receipt_delay_above_40ms')
    output.update(receiver_gap_ms_max=max(gaps) if gaps else None,
                  heartbeat_send_lateness_ms_max=max(heartbeat_late) if heartbeat_late else None,
                  heartbeat_coverage_gap_ms_max=max(coverage_gaps) if coverage_gaps else None,
                  heartbeat_emission_coverage_gap_ms_max=max(emission_gaps) if emission_gaps else None,
                  message_receipt_delay_ms_max=max(message_delays) if message_delays else None,
                  message_receipt_delay_ms_p99=float(np.percentile(message_delays, 99)) if message_delays else None,
                  # Receiver processing plus message delivery, retained as a diagnostic.
                  reply_delay_ms_max=max(reply_delays) if reply_delays else None,
                  reply_delay_ms_p99=float(np.percentile(reply_delays, 99)) if reply_delays else None)
    qualification_complete = True
    return finish()


async def _finish_process(process):
    """Reap normal exits and terminate failed children without leaving orphans."""
    try:
        return await asyncio.wait_for(process.communicate(), 3)
    except TimeoutError:
        if process.returncode is None:
            process.terminate()
        try:
            return await asyncio.wait_for(process.communicate(), 3)
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            return await process.communicate()


async def local_probe(out, *, seconds=10.0, repeats=3, receiver_delay_ms=0, stall_ms=0,
                      receiver_stall_ms=0, client_stall_ms=0):
    if (not math.isfinite(seconds) or seconds <= 0 or repeats < 1
            or any(not math.isfinite(value) or value < 0
                   for value in (receiver_delay_ms, stall_ms, receiver_stall_ms, client_stall_ms))):
        raise ValueError('Positive duration/repeats and nonnegative delays required')
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    speech_frames = max(1, round(seconds / .02))
    pcm = b'\0' * FRAME_BYTES * (speech_frames + 50)
    for trial in range(repeats):
        log = EventLog(out / f'probe-{trial+1}.jsonl')
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'stt_bench.probe_server',
                '--receiver-stall-ms', str(receiver_stall_ms),
                '--receiver-delay-ms', str(receiver_delay_ms),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            line = await asyncio.wait_for(process.stdout.readline(), 10)
            if not line:
                raise RuntimeError('Independent receiver exited before announcing its port; see probe_process_exit')
            port = json.loads(line)['port']
            log.emit('probe_process_started', pid=process.pid, port=port)
            async with connect(f'ws://127.0.0.1:{port}', compression=None,
                               open_timeout=5, close_timeout=2, max_size=8 * 1024 * 1024) as ws:
                receiver = sender = None
                try:
                    async def receive():
                        injected = False
                        async for raw in ws:
                            received_at = time.perf_counter()  # Before parsing or logging.
                            message = json.loads(raw)
                            log.emit('local_receiver_reply', at=received_at - log.origin,
                                     received_absolute_seconds=received_at, message=message)
                            if (client_stall_ms and not injected and message.get('type') == 'audio_ack'
                                    and message.get('index') == 4):
                                injected = True
                                log.emit('injected_client_stall', milliseconds=client_stall_ms)
                                time.sleep(client_stall_ms / 1000)
                    receiver = asyncio.create_task(receive())
                    count = 0
                    async def send(frame):
                        nonlocal count
                        index = count
                        count += 1
                        if index == 0:
                            log.emit('probe_audio_started', absolute_seconds=time.perf_counter())
                        if index == 4 and stall_ms:
                            await asyncio.sleep(stall_ms / 1000)
                        await ws.send(index.to_bytes(8, 'little') + frame[8:])
                    async def finalize(t0):
                        log.emit('local_finalize', t0_seconds=t0)
                    sender = asyncio.create_task(stream_audio(pcm, speech_frames, send, finalize, log))
                    done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                    if receiver in done:
                        await receiver
                        raise RuntimeError('Independent receiver closed before streaming completed')
                    await sender
                    log.emit('probe_audio_finished', absolute_seconds=time.perf_counter())
                    await ws.send(json.dumps({'type': 'finish'}))
                    await asyncio.wait_for(receiver,
                        max(5, (speech_frames + 50) * receiver_delay_ms / 1000 + 2))
                finally:
                    for task in (sender, receiver):
                        if task is not None:
                            if not task.done():
                                task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await task
        except Exception as exc:
            log.emit('probe_error', error=f'{type(exc).__name__}: {exc}')
        finally:
            try:
                if process is not None:
                    _, stderr = await _finish_process(process)
                    log.emit('probe_process_exit', returncode=process.returncode,
                             stderr=stderr.decode('utf-8', errors='replace'))
                    if process.returncode != 0:
                        log.emit('probe_error', error=f'Independent receiver exited with status {process.returncode}')
            finally:
                log.close()
        events = read_events(out / f'probe-{trial+1}.jsonl')
        pacing = pacing_metrics(events)
        duplex = duplex_metrics(events, speech_frames + 50)
        reasons = pacing.get('gate_reasons', []) + duplex['duplex_gate_reasons']
        if not pacing['valid'] and not pacing.get('gate_reasons'):
            reasons.insert(0, pacing.get('reason', 'invalid_send_pacing'))
        rows.append({**pacing, **duplex, 'trial': trial+1,
                     'send_pacing_valid': pacing['valid'],
                     'send_gate_reasons': pacing.get('gate_reasons', []),
                     'valid': pacing['valid'] and duplex['duplex_valid'], 'gate_reasons': reasons})
    result = dict(mode='local_websocket_probe', probe_version=4, receiver_process='independent',
                  seconds=seconds, repeats=repeats,
                  identity=pacing_identity(), completed_at=datetime.now(timezone.utc).isoformat(),
                  receiver_delay_ms=receiver_delay_ms, injected_send_stall_ms=stall_ms,
                  receiver_stall_ms=receiver_stall_ms, client_stall_ms=client_stall_ms, trials=rows,
                  interpretation='Sender timing, child stimulus coverage, and actual send-to-client delivery qualify separately. Aggregate receipt gaps are diagnostic; terminal heartbeat inventories prevent missing-message blind spots. No provider accuracy or remote network evidence.')
    write_json(out / 'pacing.json', result)
    return result
