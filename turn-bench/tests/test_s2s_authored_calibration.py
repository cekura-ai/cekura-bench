import json
from types import SimpleNamespace

import numpy as np
import pytest

from turn_bench.author import service_bridge
from turn_bench.s2s.authored_calibration import calibrate_run, score_event, summarize

RATE = 16000


class Veto:
    provenance = {'model': 'test boundary classifier', 'threshold': .5}

    def __init__(self, negative_after=None):
        self.negative_after = negative_after

    def analyze(self, pcm, rate, *, start_s=0):
        count = len(pcm)//2
        return {'chunks': [{'start_s': start_s+i/rate, 'end_s': start_s+(i+512)/rate,
                            'probability': .01 if self.negative_after is not None and start_s+i/rate >= self.negative_after else .99}
                           for i in range(0, count-511, 512)], 'provenance': self.provenance}


def fixture(*, cut=None, residual=None, amplitude=10000):
    t = np.arange(3*RATE)/RATE
    values = amplitude*np.sin(2*np.pi*220*t)
    if residual is not None:
        rng = np.random.default_rng(31)
        values[t >= .7] = rng.normal(0, 32768*10**(residual/20), sum(t >= .7))
    pcm = np.rint(values).astype('<i2').tobytes()
    timeline = service_bridge.audio().AudioTimeline(RATE)
    timeline.record_pcm(pcm, 0)
    if cut is not None:
        timeline.cut(cut)
    return SimpleNamespace(agent_pcm=pcm, agent_timeline=timeline)


def event(kind='barge_in', **changes):
    return {'kind': kind, 'conversation_id': 'c', 'event_id': 'e', 'repeat': 1,
            'caller_start_s': .5, 'caller_end_s': .8, 'observation_end_s': 1.7,
            'void': None, **changes}


@pytest.mark.parametrize('kind,cut,verdict', [('barge_in', .7, 'pass'), ('backchannel', None, 'pass')])
def test_actual_playout_cut_and_continuation_preserve_operational_timing(kind, cut, verdict):
    row = score_event(event(kind), fixture(cut=cut), Veto(cut))
    assert row['void'] is None and row['verdict'] == verdict
    assert row['measurement']['causal_yield_verified'] is False
    if cut:
        assert row['stop_ms'] == 200 and row['stop_interval_ms'] == [190, 210]
    summary = summarize([row])
    assert summary['ranking_permitted'] is False and summary['publication_permitted'] is False


@pytest.mark.parametrize('audio,veto,reason', [
    (lambda: fixture(residual=-45), lambda: Veto(.7), 'detector_threshold_sensitive_stop'),
    (lambda: fixture(residual=-30), lambda: Veto(.7), 'energy_speech_classifier_disagreement'),
    (lambda: fixture(), lambda: Veto(0), 'speech_confidence_unavailable_at_injection'),
    (lambda: fixture(amplitude=260), lambda: Veto(), 'detector_threshold_sensitive_overlap'),
    (lambda: fixture(cut=1.5), lambda: Veto(), 'detector_success_window_boundary'),
    (lambda: fixture(cut=.7), lambda: Veto(), 'energy_quiet_speech_classifier_disagreement'),
])
def test_uncertain_audio_is_counted_not_scored(audio, veto, reason):
    row = score_event(event(), audio(), veto())
    assert row['void'] == reason and row['verdict'] is None and row['stop_ms'] is None
    summary = summarize([row])
    assert summary['scored'] == 0 and summary['voids'] == {reason: 1}
    assert row['measurement']['threshold_candidates']


def test_low_residual_noise_does_not_extend_a_verified_speech_stop():
    row = score_event(event(), fixture(residual=-65), Veto(.7))
    assert row['verdict'] == 'pass' and row['stop_ms'] == 200


def test_confirmation_tail_can_veto_residual_noise_without_inventing_a_stop():
    # Classifier rejection arrives after the 1 s success window, within the
    # already-required 1.2 s observation. It excludes; it cannot assert a hit.
    row = score_event(event(), fixture(residual=-30), Veto(1.52))
    assert row['void'] == 'energy_speech_classifier_disagreement'
    assert row['stop_ms'] is None and row['hit'] is None


@pytest.mark.parametrize('changes,reason', [
    ({'observation_end_s': .9}, 'insufficient_stop_observation'),
    ({'next_caller_start_s': 1.1}, 'next_caller_contaminates_observation'),
    ({'void': 'provider_session_failed'}, 'provider_session_failed'),
    ({'caller_start_s': -.1}, 'invalid_caller_timing'),
])
def test_existing_validity_exclusions_survive(changes, reason):
    row = score_event(event(**changes), fixture(cut=.7), Veto())
    assert row['void'] == reason and row['verdict'] is None


def test_sidecar_is_reproducible_and_source_results_are_immutable(tmp_path, monkeypatch):
    source = tmp_path/'source'
    directory = source/'c-r1'
    directory.mkdir(parents=True)
    adapter = fixture(cut=.7)
    service_bridge.audio().write_wav(directory/'agent.wav', adapter.agent_pcm, RATE)
    (directory/'timelines.json').write_text(json.dumps({'agent': adapter.agent_timeline.as_json()}))
    original = {**event(), 'verdict': 'pass', 'stop_ms': 200}
    (directory/'result.json').write_text(json.dumps({'events': [original]}))
    (source/'run.json').write_text(json.dumps({'schema': 'turn-bench/authored-live-run/1',
        'status': 'complete', 'provider': 'fake', 'model': 'scripted',
        'conversations': [{'conversation_id': 'c', 'repeat': 1}]}))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr('turn_bench.s2s.authored_calibration.SpeechVeto', lambda _: Veto(.7))
    first = calibrate_run(source, tmp_path/'first', tmp_path/'unused')
    second = calibrate_run(source, tmp_path/'second', tmp_path/'unused')
    assert first['source_hashes'] == second['source_hashes']
    assert (tmp_path/'first'/'calibrated-events.json').read_bytes() == (tmp_path/'second'/'calibrated-events.json').read_bytes()
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert first['legacy_scored'] == first['scored'] == 1
    assert first['live_anchor_calibrated'] is False
    with pytest.raises(FileExistsError):
        calibrate_run(source, source/'nested', tmp_path/'unused')
    payload = json.loads((source/'run.json').read_text())
    payload['schema'] = 'unknown'
    (source/'run.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='supported v1'):
        calibrate_run(source, tmp_path/'invalid', tmp_path/'unused')
    assert not (tmp_path/'invalid').exists()
