"""Opt-in offline calibration of authored Track 2; never rewrites v1 results.

Energy thresholds are declared robustness probes, not learned speech labels.
The pinned VAD can veto a measurement; it cannot manufacture an acoustic stop.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from turn_taking_bench.author import service_bridge
from turn_taking_bench.runner import harness_git_state
from turn_taking_bench.s2s.authored_metrics import summarize as legacy_summary
from turn_taking_bench.s2s.authored_vad import SpeechVeto

SCHEMA = 'turn-taking-bench/authored-acoustic-calibration/2'
THRESHOLDS_DBFS = (-60.0, -50.0, -40.0)
FRAME_S = .01
MAX_SPREAD_MS = 10.0
NOTES = [
    'Experimental offline calibration; unchanged v1 results remain separate. No ranking or causal interruption accuracy is established.',
    'Fixed -60/-50/-40 dBFS thresholds probe sensitivity. Disagreement in overlap, stop outcome, or stop timing beyond 10 ms is an explicit exclusion.',
    'Pinned Silero probabilities only veto uncertain energy observations. They are not exact speech labels and never supply stop timestamps.',
    'Stop evidence still requires 200 ms of observed low energy and the full 1.2 s observation after exact PCM injection. Cached padding remains part of that stimulus.',
    'A reported stop range includes the threshold spread and +/-10 ms frame uncertainty. Outcomes crossing the one-second boundary within that range are excluded.',
    'Speech confidence requires two consecutive complete 32 ms VAD chunks ending within the last 128 ms before injection, using at most two seconds of reconstructed context.',
    'Two consecutive VAD-negative chunks with energetic audio before the candidate stop veto the observation. Without an energy stop, the check includes the full 1.2 s observation tail. VAD mistakes can exclude real speech; coverage must remain visible.',
    'Two speech-positive VAD chunks ending in the final 100 ms of the claimed 200 ms quiet run also veto a stop. Classifier hangover can exclude a genuine stop; it never moves the acoustic timestamp.',
    'Natural endings, internal pauses, model/client behavior, and semantic backchannel continuation remain unresolved. This is an acoustic observation, not a causal score.',
    'Recorded control/provider/pacing exclusions are preserved, not re-derived. Bootstrap clusters remain whole original conversations; fewer than two scored clusters yields no interval.',
    'Aggregate stop times summarize threshold medians. Bootstrap intervals represent conversation sampling only; detector/frame uncertainty is reported separately as ranges.',
]


def policy():
    return {'schema': SCHEMA, 'thresholds_dbfs': list(THRESHOLDS_DBFS), 'frame_ms': 10,
            'onset_frames': 3, 'quiet_frames': 20, 'stop_window_ms': 1000,
            'required_observation_ms': 1200, 'max_stop_spread_ms': MAX_SPREAD_MS,
            'latency_uncertainty_ms': 10, 'vad_threshold': .5,
            'vad_threshold_rule': 'probability > 0.5',
            'vad_context_max_s': 2, 'vad_overlap_history_ms': 128,
            'vad_positive_chunks': 2, 'vad_negative_chunks': 2,
            'quiet_vad_veto_window_ms': [100, 200],
            'no_stop_vad_observation_ms': 1200,
            'boundary': 'first transmitted PCM sample; not an exact speech onset',
            'publication_permitted': False}


def _void(record, reason, evidence):
    return {**record, 'void': reason, 'verdict': None, 'hit': None, 'fp': None,
            'stop_ms': None, 'stop_interval_ms': None, 'measurement': evidence}


def _played(adapter, begin, end):
    rate = adapter.agent_timeline.rate
    count = int(math.floor((end-begin) / FRAME_S + 1e-8)) * round(rate*FRAME_S)
    values = np.zeros(count, dtype='<i2')
    raw = np.frombuffer(adapter.agent_pcm, dtype='<i2')
    for chunk in adapter.agent_timeline.chunks:
        left, right = adapter.agent_timeline.playout_span(chunk.seq)
        left, right = max(left, begin), min(right, end)
        if right <= left:
            continue
        dst = max(0, round((left-begin)*rate))
        src = chunk.first_sample + max(0, round((left-adapter.agent_timeline.playout_span(chunk.seq)[0])*rate))
        length = min(round((right-left)*rate), len(values)-dst, chunk.last_sample-src)
        if length > 0:
            values[dst:dst+length] = raw[src:src+length]
    return values


def score_event(record, adapter, veto):
    evidence = {'policy': policy(), 'clock': 'reference_client_playout',
                'causal_yield_verified': False}
    if record.get('void'):
        return _void(record, str(record['void']), evidence)
    if record.get('kind') not in ('barge_in', 'backchannel'):
        raise ValueError('unknown authored event kind')
    start, end, observed = (record.get(k) for k in ('caller_start_s', 'caller_end_s', 'observation_end_s'))
    finite = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if not all(finite(v) for v in (start, end, observed)):
        return _void(record, 'missing_or_invalid_actual_timing', evidence)
    if start < 0 or end <= start:
        return _void(record, 'invalid_caller_timing', evidence)
    if observed < start + 1.2 - 1e-8:
        return _void(record, 'insufficient_stop_observation', evidence)
    next_start = record.get('next_caller_start_s')
    if next_start is not None:
        if not finite(next_start):
            return _void(record, 'missing_or_invalid_actual_timing', evidence)
        if start <= next_start < start + 1.2 - 1e-8:
            return _void(record, 'next_caller_contaminates_observation', evidence)
    timeline = adapter.agent_timeline
    if timeline.rate <= 0 or len(adapter.agent_pcm) != timeline.n_samples*2:
        return _void(record, 'invalid_agent_pcm_timeline', evidence)
    if timeline.rate % 100:
        return _void(record, 'unsupported_agent_10ms_frame_rate', evidence)
    if not timeline.chunks:
        return _void(record, 'no_agent_audio', evidence)
    played = _played(adapter, start-.1, start+1.2)
    energy = service_bridge.detector().frame_energy_db(played, timeline.rate)
    candidates = []
    for threshold in THRESHOLDS_DBFS:
        loud = energy > threshold
        overlap = any(bool(loud[i:i+3].all()) for i in (8, 9, 10) if i+3 <= len(loud))
        stop = next((i for i in range(10, len(loud)-20+1) if not loud[i:i+20].any()), None) if overlap else None
        stop_ms = None if stop is None else (stop-10)*10.0
        candidates.append({'threshold_dbfs': threshold, 'overlap': overlap,
                           'stop_ms': stop_ms, 'within_window': stop_ms is not None and stop_ms <= 1000})
    evidence['threshold_candidates'] = candidates
    if not any(c['overlap'] for c in candidates):
        return _void(record, 'no_agent_speech_overlap', evidence)
    if not all(c['overlap'] for c in candidates):
        return _void(record, 'detector_threshold_sensitive_overlap', evidence)
    if len({c['within_window'] for c in candidates}) != 1:
        return _void(record, 'detector_threshold_sensitive_stop', evidence)
    times = [c['stop_ms'] for c in candidates if c['within_window']]
    if times and max(times)-min(times) > MAX_SPREAD_MS:
        return _void(record, 'detector_threshold_sensitive_latency', evidence)
    interval = [max(0, min(times)-10), max(times)+10] if times else None
    evidence['stop_interval_ms'] = interval
    if interval and interval[0] <= 1000 < interval[1]:
        return _void(record, 'detector_success_window_boundary', evidence)

    # Reconstruct only observed playout; discarded queued PCM cannot supply VAD evidence.
    context_start = max(0, start-2)
    context = _played(adapter, context_start, start+1.2)
    confidence = veto.analyze(context.tobytes(), timeline.rate, start_s=context_start)
    evidence['speech_veto'] = confidence
    past = [c for c in confidence['chunks'] if start-.128-1e-8 <= c['end_s'] <= start+1e-8]
    if not any(a['probability'] > .5 and b['probability'] > .5 for a, b in zip(past, past[1:])):
        return _void(record, 'speech_confidence_unavailable_at_injection', evidence)
    if times:
        quiet_start = start + min(times)/1000
        quiet_tail = [c for c in confidence['chunks']
                      if quiet_start+.1-1e-8 <= c['end_s'] <= quiet_start+.2+1e-8]
        if len(quiet_tail) < 2:
            return _void(record, 'speech_confidence_unavailable_for_quiet', evidence)
        if any(a['probability'] > .5 and b['probability'] > .5 for a, b in zip(quiet_tail, quiet_tail[1:])):
            return _void(record, 'energy_quiet_speech_classifier_disagreement', evidence)
    limit = start + (min(times)/1000 if times else 1.2)
    after = [c for c in confidence['chunks'] if c['start_s'] >= start-1e-8 and c['end_s'] <= limit+1e-8]
    for first, last in zip(after, after[1:]):
        if first['probability'] > .5 or last['probability'] > .5:
            continue
        left = max(10, int(math.floor((first['start_s']-(start-.1))/FRAME_S)))
        right = min(len(energy), int(math.ceil((last['end_s']-(start-.1))/FRAME_S)))
        if bool((energy[left:right] > THRESHOLDS_DBFS[-1]).any()):
            return _void(record, 'energy_speech_classifier_disagreement', evidence)
    within = bool(times)
    barge = record['kind'] == 'barge_in'
    passed = within if barge else not within
    return {**record, 'void': None, 'verdict': 'pass' if passed else 'fail',
            'hit': within if barge else None, 'fp': within if not barge else None,
            'stop_ms': float(np.median(times)) if times else None,
            'stop_interval_ms': interval, 'measurement': evidence}


def summarize(rows):
    result = legacy_summary(rows)
    result.update(schema=SCHEMA, notes=list(NOTES), ranking_permitted=False,
                  publication_permitted=False, policy=policy())
    successful = [r['stop_interval_ms'] for r in rows if r.get('kind') == 'barge_in'
                  and not r.get('void') and r.get('verdict') == 'pass']
    result['detector_latency_ranges_ms'] = {
        name: [float(np.percentile([r[0] for r in successful], percentile)),
               float(np.percentile([r[1] for r in successful], percentile))] if successful else None
        for name, percentile in (('p50', 50), ('p90', 90))}
    return result


def calibrate_run(run_dir, out, model_path):
    """Read an immutable v1 run, write a separate v2 diagnostic report, no provider calls."""
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    if out.exists() or out == run_dir or run_dir in out.parents:
        raise FileExistsError('calibration output must be a fresh directory outside the source run')
    info = json.loads((run_dir/'run.json').read_text())
    if info.get('schema') != 'turn-taking-bench/authored-live-run/1' or info.get('status') != 'complete':
        raise ValueError('calibration requires a completed supported v1 authored run')
    veto = SpeechVeto(Path(model_path))  # validated locally; never downloaded
    audio = service_bridge.audio()
    rows, changes, sources = [], [], {}
    # Exclusions and original source identities are carried into the sidecar.
    def source(path):
        sources[str(path.relative_to(run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    source(run_dir/'run.json')
    for conv in info['conversations']:
        directory = run_dir/f"{conv['conversation_id']}-r{conv['repeat']}"
        result_path = directory/'result.json'
        source(result_path)
        events = json.loads(result_path.read_text())['events']
        adapter = None
        if any(not r.get('void') for r in events):
            source(directory/'timelines.json'); source(directory/'agent.wav')
            timeline = json.loads((directory/'timelines.json').read_text())
            pcm, rate = audio.read_wav(directory/'agent.wav')
            recorded = audio.AudioTimeline.from_json(timeline['agent'])
            if recorded.rate != rate:
                raise ValueError('agent WAV rate does not match timeline')
            adapter = SimpleNamespace(agent_pcm=pcm, agent_timeline=recorded)
        for record in events:
            calibrated = score_event(record, adapter, veto)
            rows.append(calibrated)
            changes.append({'conversation_id': record['conversation_id'], 'event_id': record['event_id'],
                            'repeat': record.get('repeat'), 'legacy_verdict': record.get('verdict'),
                            'legacy_void': record.get('void'), 'legacy_stop_ms': record.get('stop_ms'),
                            'calibrated_verdict': calibrated['verdict'], 'calibrated_void': calibrated['void'],
                            'calibrated_stop_interval_ms': calibrated['stop_interval_ms']})
    summary = summarize(rows)
    summary.update(source_run=str(run_dir), provider=info['provider'], model=info['model'],
                   harness=harness_git_state(), original_run_provenance=info.get('harness'),
                   source_hashes=sources, speech_veto=veto.provenance,
                   legacy_scored=sum(not bool(r['legacy_void']) for r in changes),
                   original_results_unchanged=True, control_exclusions_preserved=True,
                   live_anchor_calibrated=False)
    for name, digest in sources.items():
        if hashlib.sha256((run_dir/name).read_bytes()).hexdigest() != digest:
            raise ValueError('source artifact changed during calibration')
    out.mkdir(parents=True, exist_ok=False)
    (out/'calibration.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (out/'calibrated-events.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    (out/'comparison.json').write_text(json.dumps(changes, indent=2, allow_nan=False)+'\n')
    from turn_taking_bench.s2s.authored_metrics import render_markdown
    text = (render_markdown(summary)
            .replace('# Authored Track 2 — experimental', '# Offline acoustic calibration — experimental')
            .replace('Yield p50 / p90', 'Stop midpoint p50 / p90'))
    text += '\nDetector uncertainty ranges for stop percentiles (not bootstrap confidence intervals): ' + json.dumps(summary['detector_latency_ranges_ms']) + '\n'
    (out/'CALIBRATION.md').write_text(text)
    return summary
