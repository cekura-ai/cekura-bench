"""Independent probe qualification from recorded events, without timing runs."""
import copy
from datetime import datetime, timezone
import json

import pytest

from stt_bench.diagnostics import duplex_metrics, pacing_identity, validate_preflight


def observations():
    received = [10.0, 10.02, 10.04]
    messages = [dict(kind='local_receiver_reply', received_absolute_seconds=at + .002,
                     message=dict(type='audio_ack', index=i, received_seconds=at, sent_seconds=at))
                for i, at in enumerate(received)]
    for sequence in range(5):
        messages.append(dict(kind='local_receiver_reply', received_absolute_seconds=10.007 + sequence * .020,
                             message=dict(type='heartbeat', sequence=sequence,
                                          scheduled_seconds=10.005 + sequence * .020,
                                          sent_seconds=10.006 + sequence * .020)))
    messages.extend([dict(kind='probe_audio_started', absolute_seconds=10.0),
                     dict(kind='probe_audio_finished', absolute_seconds=10.101)])
    messages.append(dict(kind='local_receiver_reply', received_absolute_seconds=10.045,
                         message=dict(type='summary', frames=3, indexes=[0, 1, 2],
                                      received_seconds=received, invalid_frames=[],
                                      emitted_heartbeats=copy.deepcopy([
                                          e['message'] for e in messages
                                          if e.get('message', {}).get('type') == 'heartbeat']))))
    return messages


def test_independent_duplex_metrics_accept_complete_fast_receipts():
    result = duplex_metrics(observations(), 3)
    assert result['duplex_valid']
    assert result['receiver_frame_order_valid']
    assert result['receiver_frames'] == 3
    assert result['receiver_gap_ms_max'] == pytest.approx(20)
    assert result['heartbeat_send_lateness_ms_max'] == pytest.approx(1)
    assert result['message_receipt_delay_ms_max'] == pytest.approx(2)
    assert result['client_delivery_valid'] and result['fixture_valid']


def test_emitter_jitter_and_client_delivery_budgets_are_not_added():
    events = observations()
    heartbeats = [e for e in events if e.get('message', {}).get('type') == 'heartbeat']
    # A 38 ms source gap plus a 15 ms delivery-delay change creates a 53 ms
    # receipt gap. Both components independently meet their stated limits.
    heartbeats[1]['message']['sent_seconds'] = 10.044
    heartbeats[1]['received_absolute_seconds'] = 10.060
    heartbeats[2]['received_absolute_seconds'] = 10.062
    events[-1]['message']['emitted_heartbeats'] = copy.deepcopy([e['message'] for e in heartbeats])
    result = duplex_metrics(events, 3)
    assert result['heartbeat_coverage_gap_ms_max'] == pytest.approx(53)
    assert result['heartbeat_emission_coverage_gap_ms_max'] == pytest.approx(38)
    assert result['message_receipt_delay_ms_max'] == pytest.approx(16)
    assert result['duplex_valid']


def test_timer_phase_over_20ms_is_valid_when_actual_emission_coverage_is_complete():
    scheduled = [1.000, 1.020, 1.060]
    sent = [1.002, 1.041, 1.061]
    sequences = [0, 1, 3]  # The 21 ms late heartbeat legitimately skips a slot.
    emitted = [dict(type='heartbeat', sequence=sequence, scheduled_seconds=deadline, sent_seconds=at)
               for sequence, deadline, at in zip(sequences, scheduled, sent)]
    audio_received = [1.000, 1.020, 1.040]
    events = [dict(kind='local_receiver_reply', received_absolute_seconds=at + .001,
                   message=dict(type='audio_ack', index=index, received_seconds=at, sent_seconds=at))
              for index, at in enumerate(audio_received)]
    events += [dict(kind='local_receiver_reply', received_absolute_seconds=h['sent_seconds'] + .001,
                    message=copy.deepcopy(h)) for h in emitted]
    events += [dict(kind='probe_audio_started', absolute_seconds=1.000),
               dict(kind='probe_audio_finished', absolute_seconds=1.080),
               dict(kind='local_receiver_reply', received_absolute_seconds=1.081,
                    message=dict(type='summary', frames=3, indexes=[0, 1, 2],
                                 received_seconds=audio_received, invalid_frames=[],
                                 emitted_heartbeats=emitted))]
    result = duplex_metrics(events, 3)
    assert result['heartbeat_send_lateness_ms_max'] == pytest.approx(21)
    assert result['heartbeat_emission_coverage_gap_ms_max'] == pytest.approx(39)
    assert result['message_receipt_delay_ms_max'] == pytest.approx(1)
    assert result['fixture_valid'] and result['client_delivery_valid'] and result['duplex_valid']


def test_actual_emission_gap_over_80ms_still_rejects_fixture():
    events = observations()
    heartbeats = [e for e in events if e.get('message', {}).get('type') == 'heartbeat']
    heartbeats[-1]['message']['sent_seconds'] += .080
    heartbeats[-1]['received_absolute_seconds'] += .080
    events[-1]['message']['emitted_heartbeats'][-1] = copy.deepcopy(heartbeats[-1]['message'])
    next(e for e in events if e['kind'] == 'probe_audio_finished')['absolute_seconds'] = 10.201
    result = duplex_metrics(events, 3)
    assert result['heartbeat_emission_coverage_gap_ms_max'] >= 80
    assert result['client_delivery_valid']
    assert not result['fixture_valid']
    assert 'heartbeat_emission_coverage_gap_above_40ms' in result['fixture_gate_reasons']


@pytest.mark.parametrize('position', [0, 2, -1])
def test_inventory_detects_missing_first_middle_and_final_heartbeat(position):
    events = observations()
    heartbeats = [e for e in events if e.get('message', {}).get('type') == 'heartbeat']
    events.remove(heartbeats[position])
    result = duplex_metrics(events, 3)
    assert not result['client_delivery_valid']
    assert result['fixture_valid']
    assert 'heartbeat_emission_inventory_mismatch' in result['client_gate_reasons']


def test_emitter_stopping_early_is_fixture_failure_even_if_every_message_arrives():
    events = observations()
    heartbeats = [e for e in events if e.get('message', {}).get('type') == 'heartbeat']
    events = [e for e in events if e not in heartbeats[3:]]
    events[-1]['message']['emitted_heartbeats'] = copy.deepcopy([e['message'] for e in heartbeats[:3]])
    result = duplex_metrics(events, 3)
    assert result['client_delivery_valid']
    assert not result['fixture_valid']
    assert 'heartbeat_emission_coverage_gap_above_40ms' in result['fixture_gate_reasons']


def test_missing_terminal_emission_inventory_cannot_qualify():
    events = observations()
    del events[-1]['message']['emitted_heartbeats']
    result = duplex_metrics(events, 3)
    assert not result['duplex_valid']
    assert 'heartbeat_emission_inventory_missing' in result['fixture_gate_reasons']


def test_reconciled_but_invalid_source_sequence_is_fixture_failure():
    events = observations()
    heartbeat = next(e for e in events if e.get('message', {}).get('type') == 'heartbeat')
    heartbeat['message']['sequence'] = 4
    events[-1]['message']['emitted_heartbeats'][0]['sequence'] = 4
    result = duplex_metrics(events, 3)
    assert result['client_delivery_valid']
    assert not result['fixture_valid']
    assert 'heartbeat_sequence_invalid' in result['fixture_gate_reasons']


def test_child_processing_delay_is_not_client_delivery_delay():
    events = observations()
    for event in events[:3]:
        event['message']['sent_seconds'] += .080
        event['received_absolute_seconds'] += .080
    result = duplex_metrics(events, 3)
    assert result['duplex_valid']
    assert result['message_receipt_delay_ms_max'] == pytest.approx(2)
    assert result['reply_delay_ms_max'] == pytest.approx(82)


@pytest.mark.parametrize('fault,reason', [
    ('missing_summary', 'receiver_frame_inventory_mismatch'),
    ('reorder', 'receiver_frame_order_invalid'),
    ('missing_heartbeat', 'independent_heartbeat_missing'),
    ('child_stall', 'heartbeat_emission_coverage_gap_above_40ms'),
    ('client_stall', 'message_receipt_delay_above_40ms'),
    ('process_error', 'probe_execution_error'),
])
def test_duplex_faults_cannot_hide_behind_valid_send_timing(fault, reason):
    events = observations()
    heartbeats = [e for e in events if e.get('message', {}).get('type') == 'heartbeat']
    if fault == 'missing_summary':
        events.pop()
    elif fault == 'reorder':
        events[-1]['message']['indexes'] = [0, 2, 1]
    elif fault == 'missing_heartbeat':
        events = [e for e in events if e not in heartbeats]
    elif fault == 'child_stall':
        heartbeats[-1]['message']['sent_seconds'] += .08
        heartbeats[-1]['received_absolute_seconds'] += .08
        events[-1]['message']['emitted_heartbeats'][-1] = copy.deepcopy(heartbeats[-1]['message'])
        next(e for e in events if e['kind'] == 'probe_audio_finished')['absolute_seconds'] = 10.201
    elif fault == 'client_stall':
        heartbeats[-1]['received_absolute_seconds'] += .08
    elif fault == 'process_error':
        events.append(dict(kind='probe_error', error='child disconnected'))
    result = duplex_metrics(events, 3)
    assert not result['duplex_valid']
    assert reason in result['duplex_gate_reasons']


@pytest.mark.parametrize('only_one', [False, True])
def test_lost_middle_heartbeats_cannot_pass(only_one):
    events = observations()
    heartbeats = [e for e in events if e.get('message', {}).get('type') == 'heartbeat']
    removed = heartbeats[1:] if only_one else heartbeats[1:4]
    events = [e for e in events if e not in removed]
    result = duplex_metrics(events, 3)
    assert not result['duplex_valid']
    assert 'heartbeat_emission_inventory_mismatch' in result['client_gate_reasons']
    assert result['fixture_valid']


@pytest.mark.parametrize('field', ['received_absolute_seconds', 'sent_seconds', 'scheduled_seconds'])
def test_nonfinite_timestamps_fail_without_nan_in_metrics(field):
    events = observations()
    heartbeat = next(e for e in events if e.get('message', {}).get('type') == 'heartbeat')
    target = heartbeat if field == 'received_absolute_seconds' else heartbeat['message']
    target[field] = float('nan')
    result = duplex_metrics(events, 3)
    assert not result['duplex_valid']
    assert 'invalid_duplex_timestamps' in result['duplex_gate_reasons']
    json.dumps(result, allow_nan=False)


def report():
    return dict(mode='local_websocket_probe', probe_version=4, receiver_process='independent',
                identity=pacing_identity(), completed_at=datetime.now(timezone.utc).isoformat(),
                seconds=10, repeats=3, receiver_delay_ms=0, injected_send_stall_ms=0,
                receiver_stall_ms=0, client_stall_ms=0,
                trials=[dict(valid=True, send_pacing_valid=True, duplex_valid=True,
                             client_delivery_valid=True, fixture_valid=True, gate_reasons=[])
                        for _ in range(3)])


def test_preflight_requires_independent_receiver_and_names_failed_trial(tmp_path):
    path = tmp_path / 'pacing.json'
    current = report()
    path.write_text(json.dumps(current))
    assert validate_preflight(path)['sha256']
    legacy = copy.deepcopy(current)
    legacy.pop('probe_version')
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match='Legacy.*independent'):
        validate_preflight(path)
    current['trials'][1].update(valid=False, duplex_valid=False,
                                gate_reasons=['message_receipt_delay_above_40ms'])
    path.write_text(json.dumps(current))
    with pytest.raises(ValueError, match='trial 2: message_receipt_delay_above_40ms'):
        validate_preflight(path)


@pytest.mark.parametrize('field', ['receiver_delay_ms', 'injected_send_stall_ms',
                                  'receiver_stall_ms', 'client_stall_ms'])
def test_preflight_rejects_every_injection_mode(tmp_path, field):
    path = tmp_path / 'pacing.json'
    current = report()
    current[field] = 80
    path.write_text(json.dumps(current))
    with pytest.raises(ValueError, match='fault injections'):
        validate_preflight(path)


@pytest.mark.parametrize('changed', ['timer_source', 'activity_source', 'timer_backend', 'activity_options'])
def test_preflight_rejects_changed_timer_implementation(tmp_path, changed):
    path = tmp_path / 'pacing.json'
    current = report()
    if changed == 'timer_source':
        assert 'macos_timer.py' in current['identity']['source_hashes']
        current['identity']['source_hashes']['macos_timer.py'] = 'different'
    elif changed == 'activity_source':
        current['identity']['source_hashes']['macos_activity.py'] = 'different'
    elif changed == 'activity_options':
        current['identity']['runtime']['process_activity'] = {'options': 0}
    else:
        current['identity']['runtime']['scheduler_backend'] = 'different'
    path.write_text(json.dumps(current))
    with pytest.raises(ValueError, match='different code, runtime or host'):
        validate_preflight(path)
