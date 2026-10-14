"""Reading onsets off a recorded output timeline, with no provider involved."""

import numpy as np

from turn_taking_bench.author import service_bridge
from turn_taking_bench.s2s.onsets import Segment, active_at, active_mask, merge_segments, output_onsets, speech_segments

RATE = 24000


def tone(seconds: float, amplitude: float = 0.25) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return amplitude * np.sin(2 * np.pi * 220.0 * t)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(RATE * seconds))


def pcm(*parts: np.ndarray) -> bytes:
    return service_bridge.audio().to_pcm(np.concatenate(parts) * 32767.0)


def test_speech_segments_find_sustained_runs_and_merge_short_gaps():
    audio = pcm(silence(0.3), tone(1.0), silence(0.3), tone(0.5), silence(1.0), tone(0.5))
    # 10 ms frames: runs at 30-130, 160-210, 310-360; the 300 ms gap merges, the 1 s gap does not.
    segments = speech_segments(audio, RATE, merge_gap_s=0.5)
    assert [(s.start_frame, s.end_frame) for s in segments] == [(30, 210), (310, 360)]
    assert speech_segments(audio, RATE, merge_gap_s=1.5) == [Segment(30, 360)]
    assert merge_segments([Segment(0, 10), Segment(12, 20), Segment(40, 50)], gap_frames=5) == [Segment(0, 20), Segment(40, 50)]


def _timeline_and_events(chunks):
    """``chunks``: (t_wall, pcm_bytes, new_response) -> (timeline, events, agent_pcm)."""
    audio = service_bridge.audio()
    ev = service_bridge.events()
    timeline = audio.AudioTimeline(RATE)
    events = []
    buffer = bytearray()
    for t_wall, data, new_response in chunks:
        chunk = timeline.record_pcm(data, t_wall)
        if new_response:
            events.append(ev.Event(t_wall, ev.AGENT_AUDIO_START, {"sample": chunk.first_sample}))
        buffer.extend(data)
    return timeline, events, bytes(buffer)


def test_first_onset_is_arrival_anchored_and_later_ones_follow_playout():
    # Response 1: one 1.3 s chunk arriving at t=1.0 that opens with 300 ms of silence.
    first = pcm(silence(0.3), tone(1.0))
    chunks = [(1.0, first, True)]
    # Response 2: 20 ms chunks every 5 ms (faster than real time) from t=5.0:
    # 0.5 s of speech, 1.0 s of silence, 0.5 s of speech.
    second = pcm(tone(0.5), silence(1.0), tone(0.5))
    step = int(RATE * 0.02) * 2
    for index, start in enumerate(range(0, len(second), step)):
        chunks.append((5.0 + index * 0.005, second[start : start + step], index == 0))
    timeline, events, agent_pcm = _timeline_and_events(chunks)

    onsets = output_onsets(agent_pcm, timeline, events, merge_gap_s=0.5)
    assert [(o.response, o.segment, o.anchor) for o in onsets] == [(0, 0, "arrival"), (1, 0, "arrival"), (1, 1, "playout")]
    assert onsets[0].t == 1.3  # arrival + lead silence
    assert onsets[0].lead_silence_ms == 300.0 and onsets[0].first_chunk_ms == 1300.0
    assert onsets[1].t == 5.0
    # The player is still draining the first 1.5 s of response 2 when the second
    # segment's samples arrive: audible at 5.0 + 1.5, not at their arrival (~5.4).
    assert abs(onsets[2].t - 6.5) < 0.011
    assert onsets[2].chunk_arrival_t < 5.5

    merged = output_onsets(agent_pcm, timeline, events, merge_gap_s=1.5)
    assert [(o.response, o.segment) for o in merged] == [(0, 0), (1, 0)]


def test_segment_cut_before_playout_is_not_an_onset():
    second = pcm(tone(0.5), silence(1.0), tone(0.5))
    step = int(RATE * 0.02) * 2
    chunks = [(5.0 + index * 0.005, second[start : start + step], index == 0) for index, start in enumerate(range(0, len(second), step))]
    timeline, events, agent_pcm = _timeline_and_events(chunks)
    timeline.cut(5.8)  # the provider's interrupt signal: everything queued is discarded
    onsets = output_onsets(agent_pcm, timeline, events, merge_gap_s=0.5)
    assert [(o.response, o.segment) for o in onsets] == [(0, 0)]


def test_caller_activity_mask_marks_loud_frames_only():
    caller = pcm(silence(0.5), tone(1.0), silence(0.5))
    mask = active_mask(caller, RATE)
    assert not active_at(mask, 0.2)
    assert active_at(mask, 1.0)
    assert not active_at(mask, 1.6)
    assert not active_at(mask, 99.0)
    # A short gap is bridged only when asked to (TurnBench's 0.5 s merge).
    gappy = pcm(tone(0.5), silence(0.3), tone(0.5))
    assert not active_at(active_mask(gappy, RATE), 0.65)
    assert active_at(active_mask(gappy, RATE, merge_gap_s=0.5), 0.65)
