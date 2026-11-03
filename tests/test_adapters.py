"""Each adapter's event handling, driven by frames recorded off the live wire.

No network: the frames below are the shapes each provider actually sent, so
the tests pin the mapping from a provider's vocabulary onto ours. When a
provider renames an event, this is where it should fail first.
"""

from __future__ import annotations

import base64

import pytest

from service import events as ev
from service.adapters.base import AdapterError, SessionConfig, ToolSpec, TurnDetection
from service.adapters.deepgram_voice_agent import DeepgramVoiceAgentAdapter, split_model
from service.adapters.gemini_live import GeminiLiveAdapter, _flatten_usage_metadata
from service.adapters.grok_realtime import GrokRealtimeAdapter
from service.adapters.higgs_realtime import HiggsRealtimeAdapter
from service.adapters.openai_realtime import OpenAIRealtimeAdapter
from service.audio import AudioTimeline


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


class TestHiggs:
    def test_session_names_the_model_and_boson_transcription(self):
        payload = adapter(HiggsRealtimeAdapter, transcribe_input=True)._session_payload()
        assert payload["model"] == "higgs-realtime"
        assert payload["audio"]["input"]["transcription"]["model"] == "higgs-stt-3.1"
        assert payload["audio"]["input"]["turn_detection"]["type"] == "server_vad"

    def test_semantic_turn_detection_is_passed_through(self):
        payload = adapter(HiggsRealtimeAdapter, turn_detection=TurnDetection("semantic_vad"))._session_payload()
        assert payload["audio"]["input"]["turn_detection"]["type"] == "semantic_vad"


class TestDeepgramVoiceAgent:
    def test_settings_name_the_text_model_voice_and_client_side_tools(self):
        tool = ToolSpec(name="lookup_patient", description="Find a patient", parameters={"type": "object", "properties": {}})
        a = adapter(DeepgramVoiceAgentAdapter, instructions="p", voice="aura-2-orion-en", tools=(tool,), temperature=0.2)
        settings = a._settings_payload()
        assert settings["audio"]["input"] == {"encoding": "linear16", "sample_rate": 16000}
        assert settings["audio"]["output"]["sample_rate"] == 24000
        think = settings["agent"]["think"]
        assert think["provider"] == {"type": "open_ai", "model": "gpt-4.1", "temperature": 0.2}
        assert think["prompt"] == "p"
        assert think["functions"] == [{"name": "lookup_patient", "description": "Find a patient", "parameters": {"type": "object", "properties": {}}}]
        assert settings["agent"]["listen"]["provider"]["model"] == "flux-general-en"
        assert settings["agent"]["speak"]["provider"]["model"] == "aura-2-orion-en"

    def test_the_opening_line_is_history_not_a_spoken_greeting(self):
        agent = adapter(DeepgramVoiceAgentAdapter, first_message="Thanks for calling.")._settings_payload()["agent"]
        assert "greeting" not in agent
        assert agent["context"]["messages"] == [{"type": "History", "role": "assistant", "content": "Thanks for calling."}]

    def test_only_the_first_slash_splits_the_model_id(self):
        assert split_model("groq/openai/gpt-oss-20b") == ("groq", "openai/gpt-oss-20b")
        with pytest.raises(AdapterError):
            split_model("gpt-4.1")

    def test_reply_sentences_become_one_transcript_and_the_caller_is_heard(self):
        a = adapter(DeepgramVoiceAgentAdapter)
        a._handle({"type": "ConversationText", "role": "user", "content": "I'd like to book."})
        a._handle({"type": "ConversationText", "role": "assistant", "content": "Sure."})
        a._handle({"type": "ConversationText", "role": "assistant", "content": "What day works?"})
        assert a.agent_text == []
        a._on_agent_audio(b"\x00\x01" * 480)
        a._handle({"type": "AgentAudioDone"})
        assert a.agent_text == ["Sure. What day works?"]
        assert a.caller_text == ["I'd like to book."]
        assert a.log.last(ev.AGENT_AUDIO_END) is not None
        assert a.log.last(ev.RESPONSE_DONE) is not None

    def test_a_tool_call_carries_the_name_for_its_reply(self):
        a = adapter(DeepgramVoiceAgentAdapter)
        a._handle({"type": "FunctionCallRequest", "functions": [
            {"id": "fc-1", "name": "lookup_patient", "arguments": "{\"phone\": \"2025550188\"}", "client_side": True},
        ]})
        assert a.tool_calls == [{"name": "lookup_patient", "call_id": "fc-1", "arguments": {"phone": "2025550188"}}]
        assert a._call_names["fc-1"] == "lookup_patient"

    def test_a_reply_ended_by_the_caller_starting_is_the_provider_yielding(self):
        a = adapter(DeepgramVoiceAgentAdapter)
        a._on_agent_audio(b"\x00\x01" * 48000)
        a._handle({"type": "UserStartedSpeaking"})
        assert a.log.last(ev.VAD_SPEECH_START).data["cleared_playback"] is True
        assert a.log.first(ev.AGENT_INTERRUPTED) is None
        a._handle({"type": "AgentAudioDone"})
        assert a.log.first(ev.AGENT_INTERRUPTED) is not None
        assert a.log.last(ev.RESPONSE_DONE).data["status"] == "cancelled"
        assert a.agent_timeline.cuts[0] == a.log.last(ev.VAD_SPEECH_START).t
        a._handle({"type": "EndOfTurn", "trigger": "model"})
        assert a.log.last(ev.VAD_SPEECH_END).data["trigger"] == "model"

    def test_speech_start_between_replies_is_not_a_cancel(self):
        a = adapter(DeepgramVoiceAgentAdapter)
        a._on_agent_audio(b"\x00\x01" * 480)
        a._handle({"type": "AgentAudioDone"})
        a._handle({"type": "UserStartedSpeaking"})
        a._on_agent_audio(b"\x00\x01" * 480)
        a._handle({"type": "AgentAudioDone"})
        assert a.log.first(ev.AGENT_INTERRUPTED) is None

    def test_unsupported_configurations_are_named_up_front(self):
        def refused(*args, **kwargs):
            return DeepgramVoiceAgentAdapter.unsupported_reason(SessionConfig(turn_detection=TurnDetection(*args, **kwargs)))

        assert refused("manual")
        assert refused("server_vad", threshold=0.6)
        assert refused("server_vad", silence_duration_ms=200)
        assert refused("server_vad", silence_duration_ms=500) is None
        assert refused("semantic_vad") is None
        assert refused() is None

    def test_a_silence_duration_becomes_flux_eot_timeout_and_defaults_send_none(self):
        listen = adapter(DeepgramVoiceAgentAdapter, turn_detection=TurnDetection("server_vad", silence_duration_ms=500))._settings_payload()["agent"]["listen"]["provider"]
        assert listen["eot_timeout_ms"] == 500
        assert "eot_timeout_ms" not in adapter(DeepgramVoiceAgentAdapter)._settings_payload()["agent"]["listen"]["provider"]


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
