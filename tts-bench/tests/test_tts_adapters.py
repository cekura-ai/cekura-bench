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
