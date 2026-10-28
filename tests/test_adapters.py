"""Each adapter's event handling, driven by frames recorded off the live wire.

No network: the frames below are the shapes each provider actually sent, so
the tests pin the mapping from a provider's vocabulary onto ours. When a
provider renames an event, this is where it should fail first.
"""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock

import pytest

from service import events as ev
from service.adapters.base import AdapterError, SessionConfig, TurnDetection
from service.adapters.elevenlabs_realtime import ElevenLabsRealtimeAdapter
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


class TestElevenLabs:
    @pytest.mark.parametrize("model", ["eleven_realtime_v1_max", "eleven_realtime_v1_mini"])
    def test_session_selects_model_and_preserves_audio_and_turn_settings(self, model):
        a = ElevenLabsRealtimeAdapter(
            model=model, api_key="unused", log=ev.EventLog(ev.Clock()),
            config=SessionConfig(voice="brian", instructions="Answer briefly.",
                                 turn_detection=TurnDetection("server_vad", silence_duration_ms=500)),
        )
        payload = a._session_payload()
        assert payload["model"] == model
        assert payload["instructions"] == "Answer briefly."
        assert payload["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert payload["audio"]["output"] == {
            "format": {"type": "audio/pcm", "rate": 24000}, "voice": "brian",
        }
        assert payload["audio"]["input"]["turn_detection"] == {
            "type": "server_vad", "silence_duration_ms": 500,
        }

    def test_manual_and_text_controls(self):
        payload = adapter(ElevenLabsRealtimeAdapter, turn_detection=TurnDetection("manual"),
                          modality="text", transcribe_input=False)._session_payload()
        assert payload["audio"]["input"]["turn_detection"] is None
        assert payload["output_modalities"] == ["text"]
        assert "output" not in payload["audio"]
        assert "transcription" not in payload["audio"]["input"]

    @pytest.mark.parametrize("model", ["eleven_realtime_v1_max", "eleven_realtime_v1_mini"])
    def test_handshake_uses_eleven_header_and_records_applied_session(self, monkeypatch, model):
        async def run():
            a = ElevenLabsRealtimeAdapter(model=model, api_key="unused", log=ev.EventLog(ev.Clock()))
            applied = {"id": "eleven-session", "model": model}
            ws = AsyncMock()
            ws.recv.side_effect = [
                json.dumps({"type": "session.created", "session": {"id": "eleven-session"}}),
                json.dumps({"type": "rate_limits.updated", "rate_limits": []}),
                json.dumps({"type": "session.updated", "session": applied}),
            ]
            connect = AsyncMock(return_value=ws)
            monkeypatch.setattr("service.adapters.elevenlabs_realtime.websockets.connect", connect)
            await a.connect()
            connect.assert_awaited_once_with(
                a.url, additional_headers={"xi-api-key": "unused"}, open_timeout=20.0, max_size=None,
            )
            assert json.loads(ws.send.call_args_list[0].args[0])["session"]["model"] == model
            assert a.session_id == "eleven-session"
            assert a.session_ack == applied
            assert a.log.first(ev.SESSION_CONFIGURED) is not None
            await a.close()
        asyncio.run(run())

    @pytest.mark.parametrize("failure", ["connect", "initial", "update", "timeout", "update_timeout", "malformed"])
    def test_failed_handshake_raises_adapter_error_and_closes(self, monkeypatch, failure):
        async def run():
            a = adapter(ElevenLabsRealtimeAdapter)
            ws = AsyncMock()
            error = json.dumps({"type": "error", "error": {"message": "refused"}})
            ws.recv.side_effect = {
                "initial": [error],
                "update": [json.dumps({"type": "session.created", "session": {"id": "s"}}), error],
                "timeout": [asyncio.TimeoutError()],
                "update_timeout": [json.dumps({"type": "session.created", "session": {"id": "s"}}),
                                   asyncio.TimeoutError()],
                "malformed": ["not json"],
                "connect": [],
            }[failure]
            connect = AsyncMock(return_value=ws)
            if failure == "connect":
                connect.side_effect = OSError("connection refused")
            monkeypatch.setattr("service.adapters.elevenlabs_realtime.websockets.connect", connect)
            with pytest.raises(AdapterError):
                await a.connect()
            if failure != "connect":
                ws.close.assert_awaited_once()
            assert a.closed.is_set()
        asyncio.run(run())

    def test_half_samples_are_carried_without_changing_audio(self):
        async def run():
            a = adapter(ElevenLabsRealtimeAdapter)
            a._ws = AsyncMock()
            for chunk in (b"a", b"bc", b"d", b"ef"):
                await a._send_audio_chunk(chunk)
            sent = [base64.b64decode(json.loads(c.args[0])["audio"]) for c in a._ws.send.call_args_list]
            assert sent == [b"ab", b"cd", b"ef"]
            await a._send_audio_chunk(b"g")
            await a.close()
            assert a._audio_carry == b""
        asyncio.run(run())

    def test_vendor_events_preserve_audio_vad_tools_and_unreported_usage(self):
        a = adapter(ElevenLabsRealtimeAdapter)
        a._handle({"type": "response.audio.delta", "delta": AUDIO})
        a._handle({"type": "input_audio_buffer.speech_started", "audio_start_ms": 10})
        a._handle({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 20})
        a._handle({"type": "response.function_call_arguments.done", "name": "lookup_patient",
                   "call_id": "fc-1", "arguments": '{"phone":"2025550188"}'})
        a._handle({"type": "output_audio_buffer.cleared"})
        a._handle({"type": "response.done", "response": {"status": "completed", "usage": None}})
        assert a.agent_timeline.n_samples == 480
        assert a.log.first(ev.VAD_SPEECH_START) is not None
        assert a.log.first(ev.VAD_SPEECH_END) is not None
        assert a.tool_calls == [{"name": "lookup_patient", "call_id": "fc-1",
                                 "arguments": {"phone": "2025550188"}}]
        assert a.log.last(ev.RESPONSE_DONE).data["usage"] is None
        assert a.log.first(ev.SESSION_ERROR) is None

    @pytest.mark.parametrize("kind", ["response.output_audio_transcript.done", "response.audio_transcript.done"])
    def test_delivery_tags_are_removed_from_completed_transcripts(self, kind):
        a = adapter(ElevenLabsRealtimeAdapter)
        payload = {"type": kind, "transcript": "[happy] Hello [laughs softly] there [123]."}
        a._handle(payload)
        assert a.agent_text == ["Hello there [123]."]
        assert payload["transcript"].startswith("[happy]")

    def test_registry_names_the_adapter_credential_and_voice(self):
        from service.registry import PROVIDERS

        entry = PROVIDERS["elevenlabs-realtime"]
        assert entry.adapter is ElevenLabsRealtimeAdapter
        assert entry.default_model == "eleven_realtime_v1_max"
        assert entry.credential_env == "ELEVENLABS_REALTIME_API_KEY"
        assert entry.default_voice == "brian"


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
