"""Barge-in stop time is the last moment the listener heard the interrupted speech."""

from __future__ import annotations

from service.adapters.base import SessionConfig
from service.adapters.fake import FakeAdapter
from service.caller import Clip, Utterance
from service.events import Clock, EventLog
from service.metrics import barge_in

RATE = 24000


def _adapter() -> FakeAdapter:
    clock = Clock()
    return FakeAdapter(model="scripted", api_key="unused", log=EventLog(clock), config=SessionConfig())


def _utterance(start_s: float, end_s: float) -> Utterance:
    clip = Clip(name="c", pcm=b"", rate=RATE, text="x")
    return Utterance(clip=clip, first_sample=int(start_s * RATE), last_sample=int(end_s * RATE),
                     speech_start_sample=int(start_s * RATE), speech_end_sample=int(end_s * RATE), t_start=start_s, t_end=end_s)


def _caller_line(adapter: FakeAdapter, seconds: float) -> None:
    # The caller streams 20 ms chunks from t=0, so caller sample k is at k/rate seconds.
    for i in range(int(seconds / 0.02)):
        adapter.caller_timeline.record(int(RATE * 0.02), i * 0.02)


def test_speech_that_resumes_during_the_interruption_counts_until_it_ends():
    adapter = _adapter()
    _caller_line(adapter, 3.0)
    adapter.agent_timeline.record(int(RATE * 0.3), 0.0)    # heard 0.0–0.3
    adapter.agent_timeline.record(int(RATE * 0.4), 0.6)    # resumes 0.6–1.0, while the caller is still talking
    adapter.agent_timeline.record(int(RATE * 1.0), 2.0)    # the reply to the interruption; not interrupted speech
    result = barge_in(adapter, _utterance(0.4, 1.2))
    assert result.stop_ms == 600.0 and result.stopped


def test_an_agent_already_quiet_at_onset_that_stays_quiet_stops_in_zero():
    adapter = _adapter()
    _caller_line(adapter, 3.0)
    adapter.agent_timeline.record(int(RATE * 0.3), 0.0)
    adapter.agent_timeline.record(int(RATE * 1.0), 2.0)
    result = barge_in(adapter, _utterance(0.4, 1.2))
    assert result.stop_ms == 0.0 and result.discarded_ms == 0.0
