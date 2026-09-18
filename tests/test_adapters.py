"""Each adapter's event handling, driven by frames recorded off the live wire.

No network: the frames below are the shapes each provider actually sent, so
the tests pin the mapping from a provider's vocabulary onto ours. When a
provider renames an event, this is where it should fail first.
"""

from __future__ import annotations

import base64

import pytest

from lane_a import events as ev
from lane_a.adapters.base import SessionConfig, TurnDetection
from lane_a.adapters.gemini_live import GeminiLiveAdapter, _flatten_usage_metadata
from lane_a.adapters.grok_realtime import GrokRealtimeAdapter
from lane_a.adapters.openai_realtime import OpenAIRealtimeAdapter
from lane_a.audio import AudioTimeline


def adapter(cls, **config):
    log = ev.EventLog(ev.Clock())
    return cls(api_key="unused", log=log, config=SessionConfig(**config))


AUDIO = base64.b64encode(b"\x00\x01" * 480).decode()


class TestGemini:
    def test_fragments_become_one_transcript_per_turn(self):
        a = adapter(GeminiLiveAdapter)
        a._handle({"serverContent": {"outputTranscription": {"text": "I can"}}})
        a._handle({"serverContent": {"outputTranscription": {"text": " help."}}})
        a._handle({"serverContent": {"inputTranscription": {"text": " Hi"}}})
        assert a.agent_text == []
        a._handle({"serverContent": {"turnComplete": True}, "usageMetadata": {"totalTokenCount": 5}})
        assert a.agent_text == ["I can help."]
        assert a.caller_text == ["Hi"]
        done = a.log.last(ev.RESPONSE_DONE)
        assert done and done.data["usage"] == {"totalTokenCount": 5}

    def test_audio_parts_land_on_the_timeline_and_thoughts_are_kept_apart(self):
        a = adapter(GeminiLiveAdapter)
        a._handle({"serverContent": {"modelTurn": {"parts": [{"text": "**Plan**", "thought": True}]}}})
        a._handle({"serverContent": {"modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": AUDIO}}]}}})
        assert a.agent_timeline.n_samples == 480
        assert a.log.first(ev.AGENT_THOUGHT) is not None
        assert a.agent_text == []

    def test_a_tool_call_carries_the_name_for_its_reply(self):
        a = adapter(GeminiLiveAdapter)
        a._handle({"toolCall": {"functionCalls": [{"id": "fc-1", "name": "lookup_patient", "args": {"phone": "2025550188"}}]}})
        assert a.tool_calls == [{"name": "lookup_patient", "call_id": "fc-1", "arguments": {"phone": "2025550188"}}]
        assert a._call_names["fc-1"] == "lookup_patient"

    def test_usage_modality_rows_become_a_dict(self):
        flat = _flatten_usage_metadata({
            "promptTokenCount": 474, "thoughtsTokenCount": 139,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 473}, {"modality": "AUDIO", "tokenCount": 1}],
        })
        assert flat == {"promptTokenCount": 474, "thoughtsTokenCount": 139, "promptTokensDetails": {"TEXT": 473, "AUDIO": 1}}

    def test_unsupported_configurations_are_named_up_front(self):
        assert GeminiLiveAdapter.unsupported_reason(SessionConfig(turn_detection=TurnDetection("semantic_vad")))
        assert GeminiLiveAdapter.unsupported_reason(SessionConfig(turn_detection=TurnDetection("server_vad", threshold=0.6)))
        assert GeminiLiveAdapter.unsupported_reason(SessionConfig(turn_detection=TurnDetection("server_vad", silence_duration_ms=500))) is None

    def test_manual_mode_brackets_the_utterance_not_the_session(self):
        a = adapter(GeminiLiveAdapter, turn_detection=TurnDetection("manual"))
        assert a._open_window_next is False
        a.note_utterance_start()
        assert a._open_window_next is True

    def test_setup_declares_text_in_audio_out_for_the_text_arm(self):
        a = adapter(GeminiLiveAdapter, modality="text")
        setup = a._setup_payload()
        assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
        assert "inputAudioTranscription" not in setup


class TestGrok:
    def test_cumulative_input_drafts_are_not_transcripts(self):
        a = adapter(GrokRealtimeAdapter)
        a._handle({"type": "conversation.item.input_audio_transcription.completed", "status": "in_progress", "transcript": "Hi, I'd"})
        a._handle({"type": "conversation.item.input_audio_transcription.updated", "transcript": "Hi, I'd like"})
        a._handle({"type": "conversation.item.input_audio_transcription.completed", "status": "completed", "transcript": "Hi, I'd like to book."})
        assert a.caller_text == ["Hi, I'd like to book."]

    def test_manual_mode_sends_an_explicit_null(self):
        a = adapter(GrokRealtimeAdapter, turn_detection=TurnDetection("manual"))
        payload = a._session_payload()
        assert "turn_detection" in payload and payload["turn_detection"] is None

    def test_pings_are_not_evidence(self):
        a = adapter(GrokRealtimeAdapter)
        a._handle({"type": "ping"})
        assert a.log.raw_frames == 0


class TestReferenceClient:
    def test_speech_started_clears_playback_that_is_still_queued(self):
        """Delivered faster than realtime, a reply keeps playing after the last byte lands."""
        a = adapter(OpenAIRealtimeAdapter)
        a._handle({"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x00\x01" * 48000).decode()})
        assert a.agent_speaking
        a._handle({"type": "input_audio_buffer.speech_started", "audio_start_ms": 10})
        started = a.log.last(ev.VAD_SPEECH_START)
        assert started.data["cleared_playback"] is True
        assert a.agent_timeline.playout_end() == pytest.approx(started.t)
        assert a.agent_timeline.cuts == [started.t]

    def test_a_cancelled_response_is_an_interrupt(self):
        a = adapter(OpenAIRealtimeAdapter)
        a._handle({"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x00\x01" * 48000).decode()})
        a._handle({"type": "response.done", "response": {"status": "cancelled", "usage": {}}})
        assert a.log.first(ev.AGENT_INTERRUPTED) is not None
        assert len(a.agent_timeline.cuts) == 1


class TestPlayoutModel:
    def test_chunks_arriving_early_queue_behind_the_previous_one(self):
        t = AudioTimeline(1000)
        t.record(1000, 0.0)      # one second of audio at t=0
        t.record(1000, 0.2)      # the next second arrives 200 ms later
        assert t.playout_start(1) == 1.0
        assert t.playout_end() == 2.0
        assert t.time_of_sample(1000) == 0.2          # arrival view, for onset
        assert t.playout_time_of_sample(1000) == 1.0  # listener view, for everything after

    def test_a_cut_ends_what_is_playing_and_drops_what_is_queued(self):
        t = AudioTimeline(1000)
        t.record(1000, 0.0)
        t.record(1000, 0.2)
        t.cut(0.5)
        assert t.playout_span(0) == (0.0, 0.5)
        assert t.playout_span(1) == (0.5, 0.5)
        assert t.discarded_s() == pytest.approx(1.5)
        t.record(500, 0.6)
        assert t.playout_span(2) == (0.6, 1.1)

    def test_cuts_survive_the_round_trip_through_json(self):
        t = AudioTimeline(1000)
        t.record(1000, 0.0)
        t.record(1000, 0.2)
        t.cut(0.5)
        t.record(500, 0.6)
        back = AudioTimeline.from_json(t.as_json())
        assert back.cuts == t.cuts
        assert [back.playout_span(i) for i in range(3)] == [t.playout_span(i) for i in range(3)]
