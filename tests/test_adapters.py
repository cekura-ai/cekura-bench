"""Each adapter's event handling, driven by frames recorded off the live wire.

No network: the frames below are the shapes each provider actually sent, so
the tests pin the mapping from a provider's vocabulary onto ours. When a
provider renames an event, this is where it should fail first.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from service import events as ev
from service.adapters.base import SessionConfig, TurnDetection
from service.adapters.gemini_live import GeminiLiveAdapter, _flatten_usage_metadata
from service.adapters.grok_realtime import GrokRealtimeAdapter
from service.adapters.higgs_realtime import HiggsRealtimeAdapter
from service.adapters.openai_realtime import OpenAIRealtimeAdapter
from service.adapters.smallest_hydra import SmallestHydraAdapter
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


class TestHydra:
    def test_session_configure_carries_rate_voice_and_tools_but_no_turn_settings(self):
        from mock_tools.spec import ToolSpec

        tool = ToolSpec(name="lookup", description="d", parameters={"type": "object", "properties": {}})
        payload = adapter(SmallestHydraAdapter, voice="sterling", tools=(tool,))._session_payload()
        assert payload["output_audio_sample_rate"] == 24000
        assert payload["generate_initial_response"] is False
        assert payload["voice"] == "sterling"
        assert payload["tools"][0]["name"] == "lookup"
        assert "turn_detection" not in payload and "audio" not in payload

    def test_only_the_default_server_vad_cell_runs(self):
        ok = SessionConfig(turn_detection=TurnDetection("server_vad", silence_duration_ms=500))
        assert SmallestHydraAdapter.unsupported_reason(ok) is None
        for detection in (TurnDetection("manual"), TurnDetection("semantic_vad"),
                          TurnDetection("server_vad", silence_duration_ms=200)):
            assert SmallestHydraAdapter.unsupported_reason(SessionConfig(turn_detection=detection))
        assert SmallestHydraAdapter.unsupported_reason(SessionConfig(modality="text"))

    def test_the_cell_record_says_no_turn_setting_was_applied(self):
        from service.adapters import smallest_hydra

        assert adapter(SmallestHydraAdapter).state()["session"]["turn_detection_applied"] == smallest_hydra.TURN_DETECTION_APPLIED

    def test_configured_is_the_acknowledgement(self):
        a = adapter(SmallestHydraAdapter)
        a._handle({"type": "session.configured", "session": {"voice": "sterling", "output_audio_sample_rate": 24000}})
        assert a._configured.is_set()
        assert a.session_ack["output_audio_sample_rate"] == 24000

    async def test_the_reply_text_hydra_streams_is_the_agent_transcript(self):
        a = adapter(SmallestHydraAdapter)
        a._handle({"type": "response.output_audio.delta", "response_id": "r1", "delta": AUDIO})
        for delta in ("Thanks, James.", "What's the reason for your visit with Dr.", "Lee?"):
            a._handle({"type": "response.output_audio_transcript.delta", "response_id": "r1", "delta": delta})
        assert a.agent_text == []  # recorded when the response ends, not per fragment
        a._handle({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
        side, ref, pcm, _rate, text = a._transcribe_queue.get_nowait()
        assert (side, ref, pcm) == ("agent", "r1", b"")
        assert text == "Thanks, James. What's the reason for your visit with Dr. Lee?"

    async def test_a_reply_with_audio_but_no_text_falls_back_to_the_transcriber(self):
        a = adapter(SmallestHydraAdapter)
        a._handle({"type": "response.output_audio.delta", "response_id": "r1", "delta": AUDIO})
        assert a.transcripts_pending is False
        a._handle({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
        assert a.transcripts_pending is True
        side, ref, pcm, rate, text = a._transcribe_queue.get_nowait()
        assert (side, ref, rate, text) == ("agent", "r1", 24000, None) and len(pcm) == 960

    def test_a_tool_only_response_queues_nothing(self):
        a = adapter(SmallestHydraAdapter)
        a._handle({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
        assert a._transcribe_queue.empty() and a.transcripts_pending is False

    async def test_caller_turns_are_cut_from_our_own_stream_and_noise_is_kept_for_the_next(self):
        a = adapter(SmallestHydraAdapter, transcribe_input=True)
        a.caller_pcm.extend(b"\x00" * 100)
        a._handle({"type": "conversation.item.done", "item": {"id": "u0", "type": "message", "role": "user", "status": "incomplete"}})
        assert a._transcribe_queue.empty()
        a.caller_pcm.extend(b"\x00" * 60)
        a._handle({"type": "conversation.item.done", "item": {"id": "u1", "type": "message", "role": "user", "status": "completed"}})
        side, ref, pcm, rate, text = a._transcribe_queue.get_nowait()
        assert (side, ref, len(pcm), rate, text) == ("caller", "u1", 160, 16000, None)

    async def test_the_silence_before_a_caller_turn_is_not_transcribed(self):
        """The transcriber invents words in long silence, so a turn starts a second before it was heard."""
        from service.adapters import smallest_hydra

        a = adapter(SmallestHydraAdapter, transcribe_input=True)
        second = 16000 * 2
        a.caller_pcm.extend(b"\x00" * 30 * second)
        a._handle({"type": "input_audio_buffer.speech_started", "item_id": "u1"})
        a.caller_pcm.extend(b"\x00" * 2 * second)
        a._handle({"type": "input_audio_buffer.speech_started", "item_id": "u1"})  # restated
        a._handle({"type": "conversation.item.done", "item": {"id": "u1", "type": "message", "role": "user", "status": "completed"}})
        _, ref, pcm, _, _ = a._transcribe_queue.get_nowait()
        assert (ref, len(pcm)) == ("u1", int((smallest_hydra.PREROLL_S + 2) * second))

    async def test_closing_waits_for_transcripts_of_speech_already_received(self):
        a = adapter(SmallestHydraAdapter)

        def slow(pcm, rate):
            import time

            time.sleep(0.1)
            return "last words"

        a._transcribe = slow
        a._transcriber = asyncio.create_task(a._transcribe_worker())
        a._queue_transcription("caller", "u9", b"\x00\x00", 16000)
        await a.close()
        assert a.caller_text == ["last words"]
        await asyncio.sleep(0)
        assert a._transcriber.done()

    async def test_transcripts_land_in_turn_order_and_name_their_source(self):
        a = adapter(SmallestHydraAdapter)
        heard = iter(["transcribed", "caller said"])
        a._transcribe = lambda pcm, rate: next(heard)
        worker = asyncio.create_task(a._transcribe_worker())
        a._queue_transcription("agent", "r1", b"\x00\x00", 24000)
        a._queue_transcription("agent", "r2", b"", 24000, text="sent by hydra")
        a._queue_transcription("caller", "u1", b"\x00\x00", 16000)
        while a.transcripts_pending:
            await asyncio.sleep(0.01)
        worker.cancel()
        assert a.agent_text == ["transcribed", "sent by hydra"]
        assert a.caller_text == ["caller said"]
        assert [e.data["transcriber"] for e in a.log.of_kind(ev.AGENT_TRANSCRIPT)] == ["pulse-pro", "provider"]
        assert [e.data["transcriber"] for e in a.log.of_kind(ev.CALLER_TRANSCRIPT)] == ["pulse-pro"]

    async def test_several_tool_results_ask_for_one_reply(self):
        a = adapter(SmallestHydraAdapter)
        sent = []

        async def capture(payload):
            sent.append(payload["type"] if payload["type"] != "conversation.item.create" else payload["item"]["type"])

        a._send_json = capture
        await a._send_tool_result("c1", {"ok": 1})
        await a._send_tool_result("c2", {"ok": 2})
        await asyncio.sleep(0.3)
        assert sent == ["function_call_output", "function_call_output", "response.create"]

    def test_the_key_is_scrubbed_from_errors(self):
        a = SmallestHydraAdapter(api_key="sk/secret+1", log=ev.EventLog(ev.Clock()))
        assert "secret" not in a._redact("bad uri wss://x?api_key=sk%2Fsecret%2B1 and sk/secret+1")


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
