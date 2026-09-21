"""Offline regression tests for timing semantics and public export boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from benchmark_uncertainty import add_uncertainty
from benchmark_publication import public_data, timing_contracts, verify_public_directory
from stt_bench.gradium import Protocol
from stt_bench.providers import validate, reduce_events
from stt_bench.streaming import transmitted_silence_frames


def config():
    c = json.loads(Path('config/models/gradium-default.json').read_text())
    c['transcript_reconstruction'] = 'gradium-append-only-text-v2'
    return c


def event(at, **message):
    return dict(kind='provider_message', time_seconds=at, message=message)


def test_gradium_text_receipt_is_not_audio_stop_or_stream_completion():
    c = config()
    events = [dict(kind='model_accepted', time_seconds=0, model='default'),
        dict(kind='speech_end', time_seconds=1), dict(kind='finalize_requested', time_seconds=1),
        event(1.2, type='text', text='hello'), event(1.3, type='end_text', stop_s=.9),
        event(1.35, type='text', text='world'), event(1.4, type='flushed', flush_id=1),
        dict(kind='audio_complete', time_seconds=2), dict(kind='close_stream_requested', time_seconds=2),
        event(2.3, type='end_of_stream'), dict(kind='provider_terminal', time_seconds=2.31)]
    a = reduce_events(events, c)
    assert a['transcript'] == 'hello world' and a['transcript_complete']
    assert a['final_transcript_received_seconds'] == 1.35
    assert a['timing_observations']['last_text_arrival_ms'] == pytest.approx(350)
    assert a['timing_observations']['server_completion_ms'] == pytest.approx(1300)
    assert a['timing_observations']['harness_completion_ms'] == pytest.approx(1310)
    old = reduce_events(events, dict(c, transcript_reconstruction='gradium-segment-finality-v1'))
    assert old['transcript'] == a['transcript']
    assert old['final_transcript_received_seconds'] == 2.3
    assert not reduce_events(events[:-2], c)['transcript_complete']
    assert reduce_events(events[:-2], c)['timing_observations']['last_text_arrival_ms'] is None


def test_gradium_rejects_unclosed_segment_even_with_append_semantics():
    p = Protocol(config()); p.feed(dict(type='text', text='one')); p.feed(dict(type='text', text='two'))
    assert p.unsupported


def test_gradium_zero_tail_is_explicit_and_completion_basis_matches():
    c = config()
    with pytest.raises(ValueError): validate(dict(c, transmitted_silence_frames=0))
    experiment = dict(c, transmitted_silence_frames=0, stream_ending_profile='gradium-no-tail-v1',
                      completion_basis='end_of_stream_after_speech')
    validate(experiment)
    assert transmitted_silence_frames(experiment) == 0
    with pytest.raises(ValueError): validate(dict(experiment, completion_basis='end_of_stream_after_tail'))
    with pytest.raises(ValueError): transmitted_silence_frames(dict(experiment, provider='reson8'))
    # Preserve the explicitly versioned policy owned by the parallel Reson8 fix.
    assert transmitted_silence_frames(dict(provider='reson8',transmitted_silence_frames=0,
        transport_profile='reson8-stop-after-flush-v2')) == 0


def ranking_fixture():
    ids = ['conversation-01-A', 'conversation-01-B', 'conversation-02-A', 'conversation-02-B']
    def counts(n): return dict(substitutions=n, insertions=0, deletions=0, reference_words=10)
    records = {mid: [dict(id=cid, cohort=cohort, counts=counts(n)) for cohort, cids in
                    [('pipecat', ['p1', 'p2']), ('private', ids)] for cid in cids]
               for mid, n in [('a', 1), ('b', 2)]}
    data = dict(models=[dict(id=mid, rank=i, ranking_score=dict(wer=n/10))
                       for i, (mid, n) in enumerate([('a', 1), ('b', 2)], 1)],
                ranking=dict(datasets=dict(pipecat=dict(clip_ids=['p1', 'p2']), private=dict(clip_ids=ids))))
    return data, records


def test_bootstrap_keeps_private_pairs_and_is_reproducible():
    d, r = ranking_fixture(); again = deepcopy(d)
    add_uncertainty(d, r, iterations=100); add_uncertainty(again, r, iterations=100)
    assert d == again
    assert d['ranking']['uncertainty']['private_groups'] == 2
    assert d['models'][0]['ranking_uncertainty']['upper_rank'] == 1
    assert d['models'][1]['ranking_uncertainty']['lower_rank'] == 2
    r['b'].pop()
    with pytest.raises(ValueError): add_uncertainty(again, r, iterations=100)


def test_public_export_removes_private_text_and_paths_without_mutating_source():
    d = dict(models=[], clip_review=dict(includes_private=True, manifests=[], clips=[
        dict(id='conversation-01-A', cohort='private', reference='CONFIDENTIAL'),
        dict(id='pipecat-public', cohort='pipecat', reference='public')]),
        turns=dict(models=[dict(id='gradium-default', proof='turn-proof/gradium-default/index.html',
             recovery_attempts=[dict(source='/Users/owner/private-dataset/file')])]))
    clean = public_data(d)
    assert 'CONFIDENTIAL' not in json.dumps(clean)
    assert 'CONFIDENTIAL' in json.dumps(d)
    assert clean['turns']['models'][0]['proof'] is None


def test_public_asset_gate_rejects_stale_private_assets(tmp_path):
    (tmp_path/'index.html').write_text('<html>safe</html>')
    verify_public_directory(tmp_path)
    p = tmp_path/'turn-proof'; p.mkdir(); (p/'manifest.json').write_text('{}')
    with pytest.raises(ValueError): verify_public_directory(tmp_path)


def test_harness_completion_not_offered_as_provider_latency():
    m = dict(id='openai-gpt-4o-transcribe', finalization_contract=dict(group='signal_at_speech_end'),
             timings=dict(completion=dict(n=10, p50_ms=1000.03)))
    d = dict(models=[m]); timing_contracts(d); timing_contracts(d)
    assert m['timings']['completion']['n'] == 0
    assert m['timings']['harness_completion']['p50_ms'] == 1000.03

@pytest.mark.parametrize('tail', [0, 50])
def test_gradium_sender_transmits_exact_policy_and_waits_for_real_completion(tmp_path, tail):
    import asyncio
    from stt_bench.gradium import exchange
    from stt_bench.streaming import EventLog, read_events
    c=config();c['transmitted_silence_frames']=tail
    if tail==0:c.update(stream_ending_profile='gradium-no-tail-v1',completion_basis='end_of_stream_after_speech')
    class Socket:
        def __init__(self):self.queue=asyncio.Queue();self.sent=[]
        def __aiter__(self):return self
        async def __anext__(self):return json.dumps(await self.queue.get())
        async def send(self,raw):
            m=json.loads(raw);self.sent.append(m)
            if m['type']=='setup':await self.queue.put(dict(type='ready',model_name=c['expected_resolved_model'],sample_rate=24000,delay_in_frames=c['delay_in_frames']))
            if m['type']=='flush':
                await self.queue.put(dict(type='text',text='hello'))
                await self.queue.put(dict(type='flushed',flush_id=1))
            if m['type']=='end_of_stream':await self.queue.put(dict(type='end_of_stream'))
    async def work():
        ws=Socket();log=EventLog(tmp_path/'events.jsonl')
        try:await exchange(ws,bytes(55*960),5,c,log)
        finally:log.close()
        assert sum(m['type']=='audio' for m in ws.sent)==5+tail
        assert ws.sent[-1]['type']=='end_of_stream'
        events=read_events(tmp_path/'events.jsonl');a=reduce_events(events,c)
        assert a['transcript']=='hello' and a['transcript_complete']
        assert a['timing_observations']['transmitted_silence_frames']==tail
        assert a['timing_observations']['server_completion_ms'] is not None
    asyncio.run(work())


def test_public_zip_refuses_private_proof_even_if_review_flag_is_false(tmp_path):
    from benchmark_clip_review import share_zip
    page=tmp_path/'index.html';page.write_text('public page')
    proof=tmp_path/'turn-proof';proof.mkdir()
    with pytest.raises(ValueError,match='explicit private-review'):
        share_zip(page,dict(clips=[],includes_private=False),proof)


def test_exported_numeric_evidence_never_includes_normalized_private_text():
    from benchmark_uncertainty import records_from_review
    d={'models':[{'id':'a'}],'clip_review':{'clips':[{'id':'conversation-01-A','cohort':'private',
       'results':{'a':{'counts':{'substitutions':1,'insertions':0,'deletions':0,'reference_words':4,
                               'reference_normalized':'SECRET'}}}}]}}
    assert 'SECRET' not in json.dumps(records_from_review(d))


def test_public_directory_rejects_linked_private_directory(tmp_path):
    private=tmp_path/'internal';private.mkdir()
    public=tmp_path/'public';public.mkdir()
    (public/'audio').symlink_to(private,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlinks'):
        verify_public_directory(public)


def test_public_zip_checks_clips_even_if_private_flag_is_false(tmp_path):
    from benchmark_clip_review import share_zip
    page=tmp_path/'index.html';page.write_text('public page')
    with pytest.raises(ValueError,match='private recordings'):
        share_zip(page,dict(includes_private=False,clips=[dict(cohort='private',audio='audio/private/clip.flac')]))


def test_default_gradium_uses_corrected_receipt_semantics_and_retains_tail():
    for name in ('config/models/gradium-default.json','config/profiles/private-turns-v1/gradium-default.json'):
        c=json.loads(Path(name).read_text());validate(c)
        assert c['transcript_reconstruction']=='gradium-append-only-text-v2'
        assert transmitted_silence_frames(c)==50
