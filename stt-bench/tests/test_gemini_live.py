import json
from pathlib import Path
import pytest
from stt_bench import gemini_live
from stt_bench.providers import reduce_events, transcript_at, validate
from stt_bench.full_benchmark import load_plan


def config(model='gemini-3.8-live-extended-thinking'):
    return json.loads(Path(f'config/models/{model}.json').read_text())


def event(content, at):
    return dict(kind='provider_message', time_seconds=at, message={'serverContent': content})


def test_reasoning_completion_does_not_end_at_filler_or_generation_complete():
    c = config()
    events = [dict(kind='model_accepted', time_seconds=0, model=c['model']),
              event({'inputTranscription': {'text': 'yes'}}, .1),
              event({'inputTranscription': {'text': ' yes.'}, 'outputTranscription': {'text': 'Ignore this reply'}}, .2),
              dict(kind='speech_end', time_seconds=1), dict(kind='finalize_requested', time_seconds=1),
              event({'turnComplete': True, 'interactionStatus': 'IN_PROGRESS'}, 1.1),
              event({'generationComplete': True}, 1.2),
              dict(kind='audio_complete', time_seconds=2),
              dict(kind='provider_terminal', time_seconds=2.1)]
    result = reduce_events(events, c)
    assert result['transcript'] == 'yes yes.'
    assert not result['transcript_complete']
    assert result['finalize_ack_received_seconds'] is None
    assert transcript_at(events, .15, c)['text'] == 'yes'
    events += [event({'turnComplete': True, 'interactionStatus': 'IDLE'}, 2.2),
               dict(kind='provider_terminal', time_seconds=2.3)]
    result = reduce_events(events, c)
    assert result['transcript_complete']
    assert result['finalize_ack_received_seconds'] == 2.2


def test_standard_waits_for_turn_complete_and_ignores_initial_idle():
    p = gemini_live.Protocol(config('gemini-3.8-live'))
    p.feed({'serverContent': {'turnComplete': True}})
    assert not p.ack
    p.requested = True
    p.feed({'serverContent': {'generationComplete': True}})
    assert not p.ack
    p.feed({'serverContent': {'turnComplete': True}})
    assert p.ack and p.terminal


def test_model_specific_setup_and_validation():
    for model in gemini_live.MODELS:
        c = validate(config(model))
        setup = gemini_live.Protocol(c).setup()['setup']
        assert setup['generationConfig']['responseModalities'] == ['AUDIO']
        assert setup['inputAudioTranscription'] == {}
        if model.endswith('thinking'):
            assert setup['generationConfig']['thinkingConfig']['thinkingLevel'] == 'LOW'
            c['thinking_level'] = 'MINIMAL'
        else:
            assert 'thinkingConfig' not in setup['generationConfig']
            c['thinking_level'] = 'LOW'
        with pytest.raises(ValueError):
            validate(c)


def test_public_plan_rejects_private_clips_and_missing_coverage(tmp_path):
    data = dict(version=3, public_only=True, models=sorted(gemini_live.MODELS),
                max_attempts=2, workers_per_model=10,
                private_manifest_sha256=None, private_pilot=None,
                items=[dict(clip_id=str(i), cohort='public') for i in range(1000)])
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(data))
    assert len(load_plan(path)['items']) == 1000
    data['items'][0]['cohort'] = 'private'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='coverage'):
        load_plan(path)
    data['items'] = data['items'][1:]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='coverage'):
        load_plan(path)
