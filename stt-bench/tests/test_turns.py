import asyncio
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stt_bench.data import sha256, write_json
from stt_bench.turns import prepare_turns, freeze_turns, validate_review, verify_turn_manifest
from stt_bench.turn_runner import controlled_profile, prepare_run_plan, run_turns
from stt_bench.turn_metrics import measure, transcript_timeline


def inputs(root, stereo=False, segments=1):
    root.mkdir()
    speakers=[]
    for i,label in enumerate(('A','B')):
        text='yes please'
        detail=dict(speaker_label=label,speaker_id=label,starts_at_seconds=i*.08 if not stereo else 0,
                    transcript=' '.join([text]*segments),segments=[dict(start=.1+n*.6,end=.5+n*.6,text=text,
                        words=[dict(word='yes',start=.1+n*.6,end=.2+n*.6),dict(word='please',start=.3+n*.6,end=.5+n*.6)]) for n in range(segments)])
        write_json(root/f'{label}.json',detail)
        x=(np.sin(np.arange(16000*(segments+1))*2*np.pi*(300+i*200)/16000)*10000).astype('int16')
        if not stereo:sf.write(root/f'{label}.wav',x,16000,subtype='PCM_16')
        speakers.append(dict(label=label,audio_file='stereo.wav' if stereo else f'{label}.wav',metadata_file=f'{label}.json',**({'channel':i} if stereo else {})))
    if stereo:sf.write(root/'stereo.wav',np.column_stack([x,-x]),16000,subtype='PCM_16')
    write_json(root/'input.json',dict(conversations=[dict(conversation_id='conversation-fixture',speakers=speakers)]))
    return root


def approved(draft):
    path=draft.parent/'review.json';r=json.loads(path.read_text());r.update(reviewed_by='Synthetic fixture test',reviewed_at='2026-09-15T00:00:00Z')
    for t in r['turns']:t.update(boundary_approved=True,transcript_approved=True)
    write_json(path,r);return path


@pytest.fixture
def frozen(tmp_path):
    source=inputs(tmp_path/'source')
    draft=prepare_turns(source,tmp_path/'draft')
    manifest=freeze_turns(draft,approved(draft),tmp_path/'frozen')
    return manifest


def test_stereo_channel_identity_and_lossless_crops(tmp_path):
    src=inputs(tmp_path/'source',True);d=prepare_turns(src,tmp_path/'draft');m=freeze_turns(d,approved(d),tmp_path/'out')
    manifest=verify_turn_manifest(m)
    stereo,rate=sf.read(src/'stereo.wav',dtype='int32',always_2d=True)
    for c in manifest['clips']:
        x,sr=sf.read(m.parent/c['original_audio'],dtype='int32')
        assert sr==rate
        np.testing.assert_array_equal(x,stereo[c['source_start_sample']:c['source_end_sample'],c['source_channel']])
        y,sr=sf.read(m.parent/c['audio'],dtype='int16');assert sr==16000 and not y[-16000:].any()
        z,sr=sf.read(m.parent/c['derivative_24000'],dtype='int16');assert sr==24000 and not z[-24000:].any()
        assert len(z)/sr==pytest.approx(len(y)/16000)


def test_explicit_stereo_mapping_required(tmp_path):
    src=inputs(tmp_path/'source',True);p=src/'input.json';m=json.loads(p.read_text());m['conversations'][0]['speakers'][0].pop('channel');write_json(p,m)
    with pytest.raises(ValueError,match='mapping'):prepare_turns(src,tmp_path/'draft')


def test_offsets_affect_conversation_not_source_crop(frozen):
    m=json.loads(frozen.read_text());a,b=m['clips']
    assert a['source_start_sample']==b['source_start_sample']==0
    assert b['conversation_start_seconds']-a['conversation_start_seconds']==pytest.approx(.08)
    assert 'cross_speaker_overlap' in json.loads((frozen.parent/'draft.json').read_text())['turns'][0]['flags']


def test_unreviewed_and_stale_sources_rejected(tmp_path):
    src=inputs(tmp_path/'source');d=prepare_turns(src,tmp_path/'draft')
    with pytest.raises(ValueError,match='Reviewer'):freeze_turns(d,d.parent/'review.json',tmp_path/'out')
    r=approved(d);j=json.loads(r.read_text());j['turns'][0]['boundary_approved']=False;write_json(r,j)
    with pytest.raises(ValueError,match='not listening reviewed'):freeze_turns(d,r,tmp_path/'out')
    (src/'A.json').write_text('{}')
    with pytest.raises(ValueError,match='Source changed'):validate_review(d,r)


def test_word_accounting_split_merge_exclude(tmp_path):
    d=prepare_turns(inputs(tmp_path/'source'),tmp_path/'draft');r=approved(d);j=json.loads(r.read_text())
    a=j['turns'][0];right=deepcopy(a);right.update(turn_id='split-right',unit_ids=a['unit_ids'][1:],start=.3,reference='please')
    a.update(unit_ids=a['unit_ids'][:1],end=.2,reference='yes');j['turns'].insert(1,right);write_json(r,j)
    validate_review(d,r)
    # Adjacent clips keep short one-word replies and bound leading context.
    manifest=freeze_turns(d,r,tmp_path/'frozen');m=json.loads(manifest.read_text())
    assert m['counts']['turns']==3
    assert m['clips'][1]['source_start_sample']>=int(.2*16000)
    right['unit_ids']=a['unit_ids'];write_json(r,j)
    with pytest.raises(ValueError,match='exactly once'):validate_review(d,r)
    right['unit_ids']=json.loads((d.parent/'draft.json').read_text())['turns'][0]['unit_ids'][1:]
    right.update(status='exclude',reason='');write_json(r,j)
    with pytest.raises(ValueError,match='reason'):validate_review(d,r)
    right['reason']='Synthetic excluded example';write_json(r,j);validate_review(d,r)


def test_invalid_word_timing_needs_explicit_correction(tmp_path):
    src=inputs(tmp_path/'source');p=src/'A.json';j=json.loads(p.read_text());j['segments'][0]['words'][0]['start']=-1;write_json(p,j)
    d=prepare_turns(src,tmp_path/'draft');r=approved(d)
    assert 'invalid_word_timing' in json.loads(d.read_text())['turns'][0]['flags']
    with pytest.raises(ValueError,match='Word timing'):validate_review(d,r)
    j=json.loads(r.read_text());t=j['turns'][0];t['word_corrections'][t['unit_ids'][0]]=dict(start=.1,end=.2,reason='Synthetic timestamp repair');t['notes']='Corrected invalid annotation';write_json(r,j)
    validate_review(d,r)


def test_frozen_derivative_and_review_are_bound(frozen):
    m=json.loads(frozen.read_text());path=frozen.parent/m['clips'][0]['derivative_24000'];path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='derivative changed'):verify_turn_manifest(frozen)


def test_profiles_preserve_tuning_and_flag_exceptions():
    for path in Path('config/models').glob('*.json'):
        base=json.loads(path.read_text());p=controlled_profile(base)
        for k in ('max_delay','max_delay_mode','mode','language_codes','transmitted_silence_frames','voice_profile'):
            assert p.get(k)==base.get(k)
        if p['provider']=='google':assert p['turn_finalization_class']=='provider_exception'
        if p['provider']=='assemblyai':assert p['force_endpoint']
    from stt_bench.turn_runner import validate_profile
    for p in Path('config/profiles/private-turns-v1').glob('*.json'):validate_profile(json.loads(p.read_text()))


CONFIG=controlled_profile(json.loads(Path('config/models/deepgram-nova-3.json').read_text()))


def msg(at,text,final=False,start=0,duration=1):
    return dict(kind='provider_message',time_seconds=at,message=dict(type='Results',is_final=final,
        start=start,duration=duration,channel_index=[0,1],channel=dict(alternatives=[dict(transcript=text)])))


def events():
    return [dict(kind='connection_open',time_seconds=0),dict(kind='audio_sent',time_seconds=2),
            msg(2.4,'yes'),dict(kind='speech_end',time_seconds=6),msg(6.2,'yes please',True),
            dict(kind='provider_terminal',time_seconds=9)]


def assessment(valid=True):return dict(valid=valid,pacing=dict(valid=valid))


def test_ttft_ttfs_and_delayed_close():
    result=measure(events(),CONFIG,assessment())
    assert result['ttft_ms']==pytest.approx(400);assert result['ttfs_ms']==pytest.approx(200)
    assert result['first_text_kind']=='partial'
    assert result['final_text_received_seconds']==6.2


def test_final_first_duplicates_late_revisions_and_negative():
    ev=events();ev[2]=msg(2.4,'yes',True,duration=.2)
    ev[4]=msg(6.2,'please',True,start=.3,duration=.7)
    ev += [msg(7,'please',True,start=.3,duration=.7),msg(7.2,'please now',True,start=.3,duration=.7)]
    r=measure(ev,CONFIG,assessment());assert r['first_text_kind']=='final';assert r['ttfs_ms']==pytest.approx(1200)
    assert r['final_text']=='yes please now'
    r=measure(ev[:-1],CONFIG,assessment());assert r['ttfs_ms']==pytest.approx(200)
    ev=[dict(kind='audio_sent',time_seconds=0),msg(.3,'yes',True),dict(kind='speech_end',time_seconds=1)]
    r=measure(ev,CONFIG,assessment());assert r['ttfs_signed_ms']==-700;assert r['ttfs_ms']==0 and r['final_text_before_boundary']


@pytest.mark.parametrize('condition',['invalid','empty','exception','missing_end','dry'])
def test_unavailable_is_not_zero(condition):
    ev=events();cfg=dict(CONFIG);a=assessment();dry=False
    if condition=='invalid':a=assessment(False)
    if condition=='empty':ev=[e for e in ev if e['kind']!='provider_message']
    if condition=='exception':cfg['turn_finalization_class']='provider_exception'
    if condition=='missing_end':ev=[e for e in ev if e['kind']!='speech_end']
    if condition=='dry':dry=True
    r=measure(ev,cfg,a,dry_run=dry);assert r['ttfs_ms'] is None
    if condition=='exception':assert r['observed_speech_end_to_final_ms']==pytest.approx(200)


def test_run_plan_deterministic_and_not_authorization(frozen,tmp_path):
    cfg=Path('config/profiles/private-turns-v1/deepgram-nova-3.json')
    plan=prepare_run_plan(frozen,[cfg],tmp_path/'plan.json')
    assert plan['planned_sessions']==2 and len(plan['smoke_clip_ids'])==2 and not plan['provider_calls_authorized']


def test_dry_run_resume_and_replay(frozen,tmp_path,monkeypatch):
    import stt_bench.run as runner
    from stt_bench.score import score
    called=[]
    async def local_sender(pcm,speech_frames,send,finalize,log,**kwargs):
        called.append(1)
        for i in range(speech_frames+50):
            at=(i+1)*.02
            log.emit('audio_sent',at=at,index=i,bytes=640,sample_rate=16000,
                     phase='speech' if i<speech_frames else 'silence',ideal_seconds=at,send_completed_seconds=at)
            if i==speech_frames-1:log.emit('speech_end',at=at,index=i)
        log.emit('audio_complete',at=(speech_frames+50)*.02)
    monkeypatch.setattr(runner,'stream_audio',local_sender)
    cfg=Path('config/profiles/private-turns-v1/deepgram-nova-3.json');out=tmp_path/'run'
    result=asyncio.run(run_turns(frozen,cfg,out,dry_run=True))
    assert len(called)==2 and len(result)==2 and all(x['valid'] for x in result)
    asyncio.run(run_turns(frozen,cfg,out,dry_run=True,resume=True));assert len(called)==2
    report=score(out,tmp_path/'report');assert report['ttft']['n']==0 and report['accuracy']['wer'] is None
    assert report['counts']['attempted']==2 and (tmp_path/'report/index.html').exists()
    changed=tmp_path/'changed.json';c=json.loads(cfg.read_text());c['close_timeout_seconds']+=1;write_json(changed,c)
    with pytest.raises(ValueError,match='matching'):asyncio.run(run_turns(frozen,changed,out,dry_run=True,resume=True))


def test_smoke_failure_blocks_full_and_resume_does_not_retry(tmp_path,monkeypatch):
    import stt_bench.run as runner
    src=inputs(tmp_path/'source',segments=6);draft=prepare_turns(src,tmp_path/'draft')
    manifest=freeze_turns(draft,approved(draft),tmp_path/'dataset')
    monkeypatch.setattr(runner,'validate_preflight',lambda p:{'synthetic':True})
    monkeypatch.setattr(runner,'require_credential',lambda config:'synthetic-no-provider')
    calls=[]
    async def simulated_empty_complete(pcm,speech_frames,config,key,log):
        calls.append(1)
        for i in range(speech_frames+50):
            at=(i+1)*.02;log.emit('audio_sent',at=at,index=i,bytes=640,sample_rate=16000,
                phase='speech' if i<speech_frames else 'silence',ideal_seconds=at,send_completed_seconds=at)
        end=(speech_frames+50)*.02
        log.emit('speech_end',at=speech_frames*.02)
        log.emit('audio_complete',at=end)
        log.emit('close_stream_requested',at=end+.01)
        log.emit('provider_message',at=end+.02,message={'type':'Metadata','model_info':{
            config['expected_model_uuid']:{'version':config['version']}}})
    monkeypatch.setattr(runner,'transcribe',simulated_empty_complete)
    cfg=Path('config/profiles/private-turns-v1/deepgram-nova-3.json');out=tmp_path/'run'
    results=asyncio.run(run_turns(manifest,cfg,out,authorized_private_manifest_sha256=sha256(manifest)))
    assert len(calls)==len(results)==10  # Two full-stage turns were not sent.
    assert all(x['turn_timing']['ttft_status']=='no_text' for x in results)
    asyncio.run(run_turns(manifest,cfg,out,resume=True,authorized_private_manifest_sha256=sha256(manifest)))
    assert len(calls)==10


def test_pacing_and_completion_both_required_for_timing():
    a={'valid':False,'pacing':{'valid':True}}
    r=measure(events(),CONFIG,a)
    assert r['ttft_status']==r['ttfs_status']=='invalid_attempt'
    assert r['ttfs_ms'] is None and r['first_text_received_seconds']==2.4


def test_grouped_latency_uncertainty_uses_conversations():
    from stt_bench.turn_report import latency_interval
    rows=[dict(dependency_group='call-a',turn_timing={'ttft_ms':v}) for v in (100,200,300)]
    assert latency_interval(rows,'ttft_ms')['status']=='unavailable_insufficient_groups'
    rows.append(dict(dependency_group='call-b',turn_timing={'ttft_ms':500}))
    ci=latency_interval(rows,'ttft_ms');assert ci['groups']==2 and ci['lower_ms']<=250<=ci['upper_ms']
    assert ci==latency_interval(rows,'ttft_ms')


def test_no_entity_annotations_need_no_entity_attestation(tmp_path):
    from stt_bench.review import export_review, load_review
    path=tmp_path/'manifest.json';write_json(path,dict(clips=[dict(clip_id='one',audio='one.wav',audio_sha256='audio-hash',reference='yes',condition='private_turn',entities=None)]))
    r=export_review(path,tmp_path/'review');j=json.loads(r.read_text());j['transcription_policy']='Synthetic fixture';j['clips'][0].update(reviewed_by='Test',reviewed_at='2026-09-15',reference_listened_verified=True,boundary_listened_verified=True)
    write_json(r,j);assert load_review(path,r)[1]['status']=='verified'


def test_48khz_source_conversion_and_end_of_recording(tmp_path):
    src=inputs(tmp_path/'source')
    for label in ('A','B'):
        path=src/f'{label}.json';j=json.loads(path.read_text())
        j['segments'][0]['end']=.503;j['segments'][0]['words'][-1]['end']=.503;write_json(path,j)
        x=(np.sin(np.arange(24144)*2*np.pi*440/48000)*.3)
        sf.write(src/f'{label}.wav',x,48000,subtype='PCM_24')
    d=prepare_turns(src,tmp_path/'draft');m=freeze_turns(d,approved(d),tmp_path/'frozen')
    for c in json.loads(m.read_text())['clips']:
        assert c['source_end_sample']==24144
        assert c['speech_frames']==26
        assert c['boundary_rounding_ms']==pytest.approx(17)
        assert c['words'][-1]['end']<=c['speech_end_seconds']


def test_changed_transcript_needs_notes_and_review_hash_is_stale(tmp_path):
    d=prepare_turns(inputs(tmp_path/'source'),tmp_path/'draft');r=approved(d);j=json.loads(r.read_text())
    j['turns'][0]['reference']='yes thank you';write_json(r,j)
    with pytest.raises(ValueError,match='notes'):validate_review(d,r)
    j['turns'][0]['notes']='Synthetic corrected transcript';j['draft_sha256']='bad';write_json(r,j)
    with pytest.raises(ValueError,match='different draft'):validate_review(d,r)


def test_capture_audio_start_and_packet_speech_end(tmp_path,monkeypatch):
    from stt_bench import assemblyai_pacing
    from stt_bench.streaming import EventLog, read_events
    async def no_wait(*args,**kwargs):pass
    monkeypatch.setattr(assemblyai_pacing,'wait_until',no_wait)
    log=EventLog(tmp_path/'events.jsonl')
    async def scenario():
        await assemblyai_pacing.stream_audio(bytes(60*640),10,no_wait,no_wait,log)
    asyncio.run(scenario());log.close();ev=read_events(tmp_path/'events.jsonl')
    sends=[e for e in ev if e['kind']=='audio_sent'];start=next(e for e in ev if e['kind']=='audio_start')
    end=next(e for e in ev if e['kind']=='speech_end')
    assert start['time_seconds']==sends[0]['time_seconds']
    assert end['time_seconds']==[e for e in sends if e['phase']=='speech'][-1]['send_completed_seconds']


def test_unassigned_audible_word_cannot_hide_inside_turn(tmp_path):
    d=prepare_turns(inputs(tmp_path/'source'),tmp_path/'draft');r=approved(d);j=json.loads(r.read_text())
    a=j['turns'][0];excluded=deepcopy(a);excluded.update(turn_id='excluded-word',status='exclude',reason='Synthetic omission',unit_ids=a['unit_ids'][1:])
    a['unit_ids']=a['unit_ids'][:1];a['reference']='yes';j['turns'].append(excluded);write_json(r,j)
    with pytest.raises(ValueError,match='unassigned source word'):validate_review(d,r)


@pytest.mark.parametrize('name,partial,final',[
 ('assemblyai-universal-3-5-pro',{'type':'Turn','turn_order':0,'transcript':'yes','end_of_turn':False},
  {'type':'Turn','turn_order':0,'transcript':'yes please','end_of_turn':True}),
 ('deepgram-flux-en',{'type':'TurnInfo','turn_index':0,'event':'Update','transcript':'yes'},
  {'type':'TurnInfo','turn_index':0,'event':'EndOfTurn','transcript':'yes please'}),
 ('speechmatics-standard',{'message':'AddPartialTranscript','metadata':{'transcript':'yes','start_time':0,'end_time':1}},
  {'message':'AddTranscript','metadata':{'transcript':'yes please','start_time':0,'end_time':4}}),
 ('gemini-3.5-transcribe-live',{'serverContent':{'interimInputTranscription':{'text':'yes'}}},
  {'serverContent':{'inputTranscription':{'text':'yes please'}}}),
 ('smallest-pulse',{'type':'transcription','transcript':'yes','is_final':False},
  {'type':'transcription','transcript':'yes please','is_final':True}),
 ('gradium-default',{'type':'text','text':'yes please','stream_id':0},
  {'type':'end_text','stream_id':0}),
])
def test_provider_independent_timing_from_native_messages(name,partial,final):
    cfg=controlled_profile(json.loads((Path('config/models')/(name+'.json')).read_text()))
    ev=[dict(kind='audio_sent',time_seconds=0),dict(kind='provider_message',time_seconds=.4,message=partial),
        dict(kind='speech_end',time_seconds=4),dict(kind='provider_message',time_seconds=4.2,message=final)]
    r=measure(ev,cfg,assessment());assert r['ttft_ms']==pytest.approx(400) and r['ttfs_ms']==pytest.approx(200)
    assert r['first_text_kind']=='partial'
