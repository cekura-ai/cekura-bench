import asyncio
import json

import numpy as np
import pytest

from turn_bench.author import service_bridge
from turn_bench.s2s import default_config
from turn_bench.s2s.authored_runner import run_conversation, replay_authored

service_bridge._ensure_path()
from service.adapters.fake import FakeAdapter
from service.caller import Clip

RATE = 24000


def clip(name, speech_s=0.2, lead_s=0.1):
    t = np.arange(int(RATE * speech_s)) / RATE
    pcm = np.concatenate([np.zeros(int(RATE * lead_s)), 8000 * np.sin(2*np.pi*220*t), np.zeros(int(RATE * 0.05))]).astype('<i2').tobytes()
    return Clip(name, pcm, RATE, name)


def conversation(kind='barge_in', *, followup=False):
    items = [{'id':'c1','speaker':'caller','kind':'turn','after':None,'gap_s':0.1},
             {'id':'a1','speaker':'agent','kind':'turn','trigger_caller_id':'c1'},
             {'id':'x1','speaker':'caller','kind':kind,'into':'a1','offset_s':0.3}]
    if followup:
        # Ordinary turns without explicit `after` must wait for the live agent
        # reply instead of becoming accidental additional barge-ins.
        items = items[:2] + [{'id':'c2','speaker':'caller','kind':'turn','after':None,'gap_s':0.1},
                            {'id':'a2','speaker':'agent','kind':'turn','trigger_caller_id':'c2'},
                            {'id':'x1','speaker':'caller','kind':kind,'into':'a2','offset_s':0.3}]
    return {'conversation_id':'1','spec_id':'TEST','timing':{'start_s':0.0},'items':items,'void':None}


def execute(tmp_path, monkeypatch, spec, **kwargs):
    clips = {i['id']:clip(i['id'], lead_s=0.12 if i['id']=='x1' else 0.1)
             for i in spec['items'] if i['speaker']=='caller'}
    monkeypatch.setattr('turn_bench.s2s.authored_runner.load_clips',lambda *_: clips)
    records = asyncio.run(run_conversation(spec, tmp_path, default_config('fake'), tmp_path/'result',
                      adapter_factory=FakeAdapter, adapter_kwargs=kwargs,
                      onset_timeout_s=3, quiet_timeout_s=4, timeout_s=12))
    return records, json.loads((tmp_path/'result'/'result.json').read_text())


@pytest.mark.parametrize('kind,yields,expected', [('barge_in',True,'pass'), ('backchannel',False,'pass')])
def test_fake_live_caller_has_exact_injection_and_scored_audio(tmp_path, monkeypatch, kind, yields, expected):
    records, result = execute(tmp_path, monkeypatch, conversation(kind), reply_ms=2500,
                              endpoint_ms=150, yields_to_barge_in=yields)
    assert len(records)==1 and records[0]['void'] is None
    event=records[0]
    assert event['verdict']==expected
    assert 0 <= event['injection_lateness_ms'] <= 25
    assert event['detected_speech_start_sample'] > event['caller_first_sample']
    assert event['boundary_definition'].startswith('first transmitted PCM sample')
    assert event['observation_end_s'] >= event['caller_start_s'] + 1.2
    assert result['prompt_sha256'] and result['model']=='scripted'
    assert (tmp_path/'result'/'agent.wav').is_file()
    (tmp_path/'result').rename(tmp_path/'1-r1')
    (tmp_path/'run.json').write_text(json.dumps({'provider':'fake','model':'scripted','conversations':[{'conversation_id':'1','repeat':1}]}))
    replay = replay_authored(tmp_path, tmp_path/'replay')
    assert replay['scored'] == 1 and replay['ranking_permitted'] is False
    row = replay['interruption_handling'] if kind == 'barge_in' else replay['backchannel_handling']
    assert row['value'] == 1.0


def test_implicit_next_caller_turn_waits_for_first_live_reply(tmp_path, monkeypatch):
    records, result=execute(tmp_path,monkeypatch,conversation(followup=True),reply_ms=800,endpoint_ms=150)
    events=[json.loads(line) for line in (tmp_path/'result'/'events.jsonl').read_text().splitlines()]
    caller_starts=[x['t'] for x in events if x.get('kind')=='caller.audio.start']
    first_end=next(x['t'] for x in events if x.get('kind')=='agent.audio.end')
    assert caller_starts[1] >= first_end + 0.05
    assert records[0]['void'] is None


def test_missing_anchor_counts_event_instead_of_guessing(tmp_path,monkeypatch):
    monkeypatch.setattr('turn_bench.s2s.authored_runner._anchor', lambda *a, **k: empty())
    records,result=execute(tmp_path,monkeypatch,conversation(),reply_ms=1200,endpoint_ms=150)
    assert len(records)==1 and records[0]['void']=='no_live_agent_speech_anchor'
    assert records[0]['caller_start_s'] is None and records[0]['verdict'] is None


async def empty():
    return None


@pytest.mark.parametrize('valid', [False, True])
def test_optional_track1_report_validated_before_output_and_handles_missing_scores(tmp_path, valid):
    from turn_bench.s2s import prompt_sha256
    from turn_bench.s2s.authored_runner import run_authored
    from turn_bench.runner import turnbench_provenance

    track1 = tmp_path / 'track1'
    track1.mkdir()
    report = {'conversations_scored': 0, 'blocks': {'caller': {'scorecard': {
        'pause_handling': {'value': None}, 'turn_taking': {'value': 0.5}}}}} if valid else {}
    (track1 / 'report.json').write_text(json.dumps(report))
    (track1 / 'run.json').write_text(json.dumps({
        'model': {'name': 's2s:fake', 'provenance': {
            'model': 'scripted', 'prompt_sha256': prompt_sha256()}},
        'turnbench': turnbench_provenance()}))
    plan = {'conversations': [], 'cache_dir': str(tmp_path), 'counts': {}}
    out = tmp_path / 'out'
    if not valid:
        with pytest.raises(ValueError, match='caller scorecard fields'):
            run_authored(plan, out, provider='fake', track1=track1)
        assert not out.exists()
    else:
        run_authored(plan, out, provider='fake', track1=track1)
        assert '| n/a | 0.500 |' in (out / 'authored-scorecard.md').read_text()
        saved = json.loads((out / 'authored-scorecard.json').read_text())
        assert saved['track1']['caller'] == report['blocks']['caller']['scorecard']
        assert saved['overall'] is None and saved['ranking_permitted'] is False
