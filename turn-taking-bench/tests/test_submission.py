import json

import pytest
from turnbench.submission import (
    SCHEMA_VERSION,
    ConversationPrediction,
    Submission,
    validate_coverage,
    validate_event_times,
)

from turn_taking_bench.adapters.rms_vad import RmsVad
from turn_taking_bench.runner import speaker_events, stream_speaker
from tests.test_rms_vad import SAMPLE_RATE, silence_tone_silence


def build_prediction(conversation_id="20"):
    model = RmsVad()
    speakers = {}
    for speaker in (1, 2):
        run = stream_speaker(model, silence_tone_silence(), SAMPLE_RATE)
        speakers[speaker], _deduped = speaker_events(run.events)
    return ConversationPrediction(
        conversation_id=conversation_id, speaker_1=speakers[1], speaker_2=speakers[2]
    )


def test_predictions_json_round_trips_through_turnbench_validation():
    submission = Submission(schema_version=SCHEMA_VERSION, predictions=[build_prediction()])
    text = submission.model_dump_json(indent=2)
    parsed = Submission.model_validate(json.loads(text))  # pydantic schema, extra="forbid"
    validate_coverage(parsed, ["20"])
    validate_event_times(parsed.predictions[0], duration_s=3.0)
    assert parsed.predictions[0].speaker_1.eot == [101 * 0.02]
    assert parsed.predictions[0].speaker_2.interruption == [51 * 0.02]


def test_events_past_the_audio_fail_turnbench_validation():
    prediction = build_prediction()
    with pytest.raises(ValueError, match="past the end of the audio"):
        validate_event_times(prediction, duration_s=1.5)
