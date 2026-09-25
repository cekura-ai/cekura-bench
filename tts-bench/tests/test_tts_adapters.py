"""Protocol handling that decides a timing: which server message ends an utterance, which one acknowledges a cancel, what goes on the wire."""

from __future__ import annotations

from tts_bench.adapters.base import TTSConfig
from tts_bench.adapters.deepgram_flux import DeepgramFluxAdapter
from tts_bench.adapters.elevenlabs_dialogue import ElevenLabsDialogueAdapter
from tts_bench.common.events import Clock, EventLog
from tts_bench.registry import PROVIDERS, lineup
from tts_bench.runner import RunSpec, Runner
from tts_bench.probes import OneShot


def adapter(cls, model: str, voice: str):
    clock = Clock()
    instance = cls(TTSConfig(model=model, voice=voice), "unused", EventLog(clock), clock)
    sent: list[dict] = []

    async def capture(payload):
        sent.append(payload)

    instance._send_json = capture
    return instance, sent


class TestDeepgramFlux:
    async def test_the_early_flushed_does_not_end_the_turn_speech_metadata_does(self):
        flux, _ = adapter(DeepgramFluxAdapter, "flux-haley-en", "flux-haley-en")
        synthesis = await flux.open_context("main")
        await flux.send("main", "Hello there.")
        await flux.finish("main")
        flux._on_message({"type": "SpeechStarted", "speech_id": "s"})
        flux._on_message({"type": "Flushed", "speech_id": "s"})
        assert not synthesis.done.is_set()
        flux._on_binary(b"\x01\x00" * 480)
        flux._on_message({"type": "SpeechMetadata", "speech_id": "s", "audio_duration_ms": 20})
        assert synthesis.done.is_set() and synthesis.meta["audio_duration_ms"] == 20

    async def test_interrupt_is_the_cancel_and_speech_interrupted_acknowledges_it(self):
        flux, sent = adapter(DeepgramFluxAdapter, "flux-haley-en", "flux-haley-en")
        synthesis = await flux.open_context("main")
        await flux.send("main", "A long answer.")
        await flux.cancel("main")
        assert sent[-1] == {"type": "Interrupt"}
        flux._on_message({"type": "SpeechInterrupted", "audio_played_ms": 880})
        assert synthesis.t_cancel_ack is not None and synthesis.done.is_set()

    def test_the_url_is_v2(self):
        flux, _ = adapter(DeepgramFluxAdapter, "flux-haley-en", "flux-haley-en")
        assert flux.url.startswith("wss://api.deepgram.com/v2/speak?model=flux-haley-en")


class TestElevenLabsDialogue:
    async def test_the_whole_text_goes_in_one_flushed_frame(self):
        dialogue, sent = adapter(ElevenLabsDialogueAdapter, "eleven_v3_conversational", "v")
        await dialogue.open_context("main")
        await dialogue.send("main", "Thanks for calling. ")
        await dialogue.send("main", "How can I help?")
        assert sent == []
        await dialogue.finish("main")
        assert sent == [{"inputs": [{"text": "Thanks for calling. How can I help?", "voice_id": "v"}], "flush": True}]

    async def test_the_turn_ends_on_its_final_audio_marker(self):
        dialogue, _ = adapter(ElevenLabsDialogueAdapter, "eleven_v3_conversational", "v")
        synthesis = await dialogue.open_context("main")
        dialogue._on_message({"audio": "AAAA"})
        assert not synthesis.done.is_set()
        dialogue._on_message({"is_final_audio_for_turn": True})
        assert synthesis.done.is_set() and synthesis.pcm

    def test_what_the_protocol_cannot_do_is_declared(self):
        reason = ElevenLabsDialogueAdapter.unsupported_reason(TTSConfig("eleven_v3_conversational", "v"), ("streamed_input",))
        assert reason == "elevenlabs-dialogue has no streamed input"
        assert not ElevenLabsDialogueAdapter.supports_cancel and not ElevenLabsDialogueAdapter.supports_continuation


class TestLineup:
    def test_each_model_runs_with_its_own_voice_and_names_its_run(self, tmp_path):
        spec = RunSpec(provider="gemini", probes=[OneShot()], model="gemini-3.8-flash-lite-tts", store=str(tmp_path))
        runner = Runner(spec, "unused")
        assert runner.config.voice == "Kore"
        assert "-gemini-3.8-flash-lite-tts-" in runner.root.name
        assert PROVIDERS["cartesia"].voice_for("sonic-3.5") == PROVIDERS["cartesia"].voice_for("sonic-3.6")

    def test_every_lineup_entry_is_unique_and_names_a_voice(self):
        pairs = [(key, m.model) for key, m in lineup()]
        assert len(pairs) == len(set(pairs)) and all(m.voice for _, m in lineup())
        assert ("elevenlabs-dialogue", "eleven_v3_conversational") in pairs and ("deepgram-flux", "flux-haley-en") in pairs


class TestInworld:
    async def test_setup_waits_for_the_acknowledgement_and_the_flush_ends_the_utterance(self):
        import asyncio

        from tts_bench.adapters.inworld import InworldAdapter

        inworld, sent = adapter(InworldAdapter, "inworld-tts-2-flash", "Brooke")
        opening = asyncio.create_task(inworld.open_context("main"))
        await asyncio.sleep(0.01)
        assert not opening.done() and sent[0]["create"]["audioConfig"]["audioEncoding"] == "PCM"
        inworld._on_message({"result": {"contextId": "main", "contextCreated": {}, "status": {"code": 0}}})
        synthesis = await opening
        await inworld.send("main", "Hello ")
        await inworld.finish("main")
        assert sent[1:] == [{"send_text": {"text": "Hello "}, "contextId": "main"}, {"flush_context": {}, "contextId": "main"}]
        riff = b"RIFF" + b"\x00" * 32 + b"data" + b"\x00" * 4 + b"\x01\x00" * 8
        import base64

        inworld._on_message({"result": {"contextId": "main", "audioChunk": {"audioContent": base64.b64encode(riff).decode()}, "status": {"code": 0}}})
        assert bytes(synthesis.pcm) == b"\x01\x00" * 8                          # a stray WAV header never reaches the timeline
        inworld._on_message({"result": {"contextId": "main", "flushCompleted": {}, "status": {"code": 0}}})
        assert synthesis.done.is_set()

    def test_close_context_is_not_offered_as_a_cancel(self):
        from tts_bench.adapters.inworld import InworldAdapter

        assert InworldAdapter.unsupported_reason(TTSConfig("inworld-tts-2-flash", "Brooke"), ("cancel",)) == "inworld has no cancel"


class TestXai:
    async def test_text_clear_is_the_cancel_and_audio_clear_acknowledges_it(self):
        from tts_bench.adapters.xai import XaiTTSAdapter

        xai, sent = adapter(XaiTTSAdapter, "grok-tts", "carina")
        synthesis = await xai.open_context("main")
        await xai.send("main", "A long answer.")
        await xai.finish("main")
        await xai.cancel("main")
        assert [m["type"] for m in sent] == ["text.delta", "text.done", "text.clear"]
        xai._on_message({"type": "audio.clear"})
        assert synthesis.t_cancel_ack is not None and synthesis.done.is_set()
        assert "voice=carina" in xai.url and "codec=pcm" in xai.url


class TestSmallest:
    async def test_frames_continue_one_context_and_complete_does_not_end_it(self):
        from tts_bench.adapters.smallest import SmallestAdapter

        smallest, sent = adapter(SmallestAdapter, "lightning_v3.1_pro", "kelsey")
        synthesis = await smallest.open_context("main")
        await smallest.send("main", "Thanks ")
        await smallest.finish("main")
        assert [(m["context_id"], m["continue"], m["complete_backoff_ms"]) for m in sent] == [("main", True, 0), ("main", False, 0)]
        smallest._on_message({"status": "chunk", "data": {"audio": "AAAA"}})
        smallest._on_message({"status": "complete"})
        assert not synthesis.done.is_set() and synthesis.meta["completes"] == 1   # ends on quiet: complete is per segment


class TestSoniox:
    async def test_a_stream_is_a_context_and_terminated_ends_or_acknowledges_it(self):
        from tts_bench.adapters.soniox import SonioxAdapter

        soniox, sent = adapter(SonioxAdapter, "tts-rt-v2", "Emma")
        synthesis = await soniox.open_context("main")
        await soniox.send("main", "Hello ")
        await soniox.finish("main")
        assert sent[0]["stream_id"] == "main" and sent[0]["audio_format"] == "pcm_s16le"
        assert sent[1:] == [{"stream_id": "main", "text": "Hello ", "text_end": False}, {"stream_id": "main", "text": "", "text_end": True}]
        soniox._on_message({"stream_id": "main", "audio": "AAAA", "audio_end": True})
        soniox._on_message({"stream_id": "main", "terminated": True})
        assert synthesis.done.is_set() and synthesis.t_cancel_ack is None

        second = await soniox.open_context("next")
        await soniox.send("next", "Long text ")
        await soniox.cancel("next")
        assert sent[-1] == {"stream_id": "next", "cancel": True}
        soniox._on_message({"stream_id": "next", "terminated": True})
        assert second.t_cancel_ack is not None

    def test_the_key_in_the_configuration_frame_never_reaches_the_raw_log(self):
        from tts_bench.adapters.soniox import SonioxAdapter

        soniox, _ = adapter(SonioxAdapter, "tts-rt-v2", "Emma")
        assert soniox._logged({"api_key": "secret", "stream_id": "s"})["api_key"] == "<redacted>"


class TestRawLog:
    def test_nested_audio_is_elided_wherever_the_protocol_puts_it(self):
        from tts_bench.adapters._ws import WebSocketAdapter

        blob = "A" * 200
        frame = {"result": {"audioChunk": {"audioContent": blob}}, "data": {"audio": blob}, "type": "audio.delta", "delta": blob,
                 "text": "short"}
        out = WebSocketAdapter._without_audio(frame)
        assert out["result"]["audioChunk"]["audioContent"] == "<200 b64 chars>"
        assert out["data"]["audio"] == "<200 b64 chars>" and out["delta"] == "<200 b64 chars>" and out["text"] == "short"


class TestDeepInfra:
    def test_the_openai_shape_on_deepinfras_host(self):
        from tts_bench.adapters.deepinfra import DeepInfraTTSAdapter

        deepinfra, _ = adapter(DeepInfraTTSAdapter, "Qwen/Qwen3-TTS", "Vivian")
        assert f"{deepinfra.base}{deepinfra.speech_path}" == "https://api.deepinfra.com/v1/openai/audio/speech"
        assert deepinfra._prewarm_url() == "https://api.deepinfra.com/models/Qwen/Qwen3-TTS"
