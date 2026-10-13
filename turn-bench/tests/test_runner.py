import numpy as np
import pytest

from turn_bench.model import FRAME_S, Event
from turn_bench.runner import CausalityError, EventOrderError, speaker_events, stream_speaker


class ScriptedModel:
    """Emits the events it is told to, at the frame index it is told to."""

    def __init__(self, script):
        self.script = script  # {frame_index: [Event, ...]}
        self.frames = 0

    def reset(self, sample_rate):
        self.frames = 0

    def push(self, frame, t_end_s):
        events = self.script.get(self.frames, [])
        self.frames += 1
        return events

    def flush(self, t_end_s):
        return self.script.get("flush", [])


def test_event_past_audio_heard_is_a_hard_error():
    # Frame 4 ends at 0.1 s; committing at 0.12 s would need audio not yet pushed.
    model = ScriptedModel({4: [Event("eot", 0.12)]})
    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(CausalityError):
        stream_speaker(model, samples, 16000)


def test_flush_cannot_commit_past_the_end_of_audio():
    model = ScriptedModel({"flush": [Event("eot", 1.5)]})
    samples = np.zeros(16000, dtype=np.float32)  # 1.0 s of audio
    with pytest.raises(CausalityError):
        stream_speaker(model, samples, 16000)


def test_frames_are_whole_20ms_and_tail_is_dropped():
    model = ScriptedModel({4: [Event("eot", 5 * FRAME_S)]})
    samples = np.zeros(16000 + 100, dtype=np.float32)
    run = stream_speaker(model, samples, 16000)
    assert run.frames == 50
    assert run.audio_time_s == 50 * FRAME_S
    assert run.dropped_tail_samples == 100
    assert run.events == [Event("eot", 5 * FRAME_S)]


def test_speaker_events_dedupes_repeats_and_rejects_going_backwards():
    events = [Event("eot", 0.5), Event("eot", 0.5), Event("interruption", 0.3), Event("eot", 0.9)]
    speaker, deduped = speaker_events(events)
    assert speaker.eot == [0.5, 0.9]
    assert speaker.interruption == [0.3]
    assert deduped == {"eot": 1, "interruption": 0}
    with pytest.raises(EventOrderError):
        speaker_events([Event("eot", 0.9), Event("eot", 0.5)])
