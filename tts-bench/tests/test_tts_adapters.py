"""Protocol handling that decides a timing: which server message ends an utterance, which one acknowledges a cancel, what goes on the wire."""

from __future__ import annotations

from tts_bench.adapters.base import TTSConfig
from tts_bench.adapters.deepgram_flux import DeepgramFluxAdapter
from tts_bench.adapters.elevenlabs_dialogue import ElevenLabsDialogueAdapter
from tts_bench.adapters.elevenlabs_dialogue_multi import ElevenLabsDialogueMultiAdapter
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


class TestElevenLabsDialogueMulti:
    async def test_text_goes_in_pieces_per_context_and_the_flush_ends_input(self):
        multi, sent = adapter(ElevenLabsDialogueMultiAdapter, "eleven_v4_turbo", "v")
        await multi.open_context("a")
        await multi.send("a", "Thanks for ")
        await multi.send("a", "calling.")
        await multi.finish("a")
        assert sent == [
            {"context_id": "a", "voices": ["v"]},
            {"context_id": "a", "inputs": [{"text": "Thanks for ", "voice_id": "v"}]},
            {"context_id": "a", "inputs": [{"text": "calling.", "voice_id": "v"}]},
            {"context_id": "a", "flush": True},
        ]

    async def test_audio_and_the_end_of_turn_are_routed_by_context(self):
        multi, _ = adapter(ElevenLabsDialogueMultiAdapter, "eleven_v4_turbo", "v")
        a = await multi.open_context("a")
        b = await multi.open_context("b")
        multi._on_message({"audio": "AAAA", "context_id": "b"})
        multi._on_message({"is_final_audio_for_turn": True, "context_id": "b"})
        assert b.done.is_set() and b.pcm and not a.done.is_set() and not a.pcm

    async def test_close_context_is_the_cancel_and_its_final_acknowledges_it(self):
        multi, sent = adapter(ElevenLabsDialogueMultiAdapter, "eleven_v4_turbo", "v")
        synthesis = await multi.open_context("a")
        await multi.send("a", "A long answer.")
        await multi.finish("a")
        await multi.cancel("a")
        assert sent[-1] == {"context_id": "a", "close_context": True}
        multi._on_message({"is_final_audio_for_turn": True, "context_id": "a"})
        assert synthesis.t_cancel_ack is not None and synthesis.done.is_set()

    def test_v4_is_on_the_dialogue_endpoint_and_waits_for_its_done_signal(self):
        multi, _ = adapter(ElevenLabsDialogueMultiAdapter, "eleven_v4", "v")
        assert multi.url == "wss://api.elevenlabs.io/v1/text-to-dialogue/multi-stream-input?model_id=eleven_v4&output_format=pcm_24000"
        assert not ElevenLabsDialogueMultiAdapter.ends_on_quiet
        assert ElevenLabsDialogueMultiAdapter.unsupported_reason(TTSConfig("eleven_v4", "v"), ("streamed_input", "cancel", "continuation")) is None


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
        assert ("elevenlabs-dialogue-multi", "eleven_v4_turbo") in pairs and ("elevenlabs-dialogue-multi", "eleven_v4") in pairs


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


class TestGradium:
    async def test_setup_waits_for_ready_and_only_end_of_stream_ends_the_utterance(self):
        import asyncio

        from tts_bench.adapters.gradium import GradiumAdapter
        from tts_bench.probes import Continuation

        gradium, sent = adapter(GradiumAdapter, "default", "NbpkqMVS3CJeq2j8")
        opening = asyncio.create_task(gradium.open_context("main"))
        await asyncio.sleep(0.01)
        assert not opening.done() and gradium.contexts["main"].t0 is None
        assert sent == [{"type": "setup", "model_name": "default", "voice_id": "NbpkqMVS3CJeq2j8",
                         "output_format": "pcm_24000", "close_ws_on_eos": False, "client_req_id": "main"}]
        gradium._on_message({"type": "ready", "client_req_id": "main", "request_id": "r1", "sample_rate": 24000})
        synthesis = await opening
        for frame in Continuation.frames("Hello there. How can I help?"):
            await gradium.send("main", frame)
        await gradium.finish("main")
        assert sent[1:] == [{"type": "text", "text": "Hello there. ", "client_req_id": "main"},
                            {"type": "text", "text": "How can I help?", "client_req_id": "main"},
                            {"type": "end_of_stream", "client_req_id": "main"}]
        gradium._on_message({"type": "audio", "client_req_id": "main", "audio": "AQABAA=="})
        gradium._on_message({"type": "text", "client_req_id": "main", "text": "Hello"})
        gradium._on_message({"type": "flushed", "client_req_id": "main"})
        assert bytes(synthesis.pcm) == b"\x01\x00" * 2 and not synthesis.done.is_set()
        gradium._on_message({"type": "end_of_stream", "client_req_id": "main"})
        assert synthesis.done.is_set() and synthesis.t0 is not None and synthesis.meta["ready"]["request_id"] == "r1"

    async def test_interleaved_responses_go_to_their_own_context(self):
        import asyncio

        from tts_bench.adapters.gradium import GradiumAdapter

        gradium, _ = adapter(GradiumAdapter, "default", "v")
        a = asyncio.create_task(gradium.open_context("a"))
        b = asyncio.create_task(gradium.open_context("b"))
        await asyncio.sleep(0.01)
        gradium._on_message({"type": "ready", "client_req_id": "b"})
        second = await b
        assert not a.done()
        gradium._on_message({"type": "ready", "client_req_id": "a"})
        first = await a
        gradium._on_message({"type": "audio", "client_req_id": "b", "audio": "AQABAA=="})
        gradium._on_message({"type": "end_of_stream", "client_req_id": "b"})
        assert second.done.is_set() and second.pcm and not first.done.is_set() and not first.pcm
        gradium._on_message({"type": "error", "client_req_id": "a", "message": "Internal server error", "code": 1011})
        assert first.error == "Internal server error" and second.error is None

    async def test_setup_errors_and_a_closed_socket_release_the_ready_wait(self):
        import asyncio

        import pytest

        from tts_bench.adapters.base import AdapterError
        from tts_bench.adapters.gradium import GradiumAdapter

        for frame in ({"type": "error", "message": "Invalid API key", "code": 1008}, None):
            gradium, _ = adapter(GradiumAdapter, "default", "v")
            opening = asyncio.create_task(gradium.open_context("main"))
            await asyncio.sleep(0.01)
            if frame is None:
                gradium._on_closed("connection lost")
            else:
                gradium._on_message(frame)
            with pytest.raises(AdapterError, match="gradium refused context main"):
                await asyncio.wait_for(opening, 1.0)

    async def test_account_errors_are_refusals_but_policy_errors_on_input_are_failures(self):
        import asyncio

        from tts_bench.adapters.gradium import GradiumAdapter
        from tts_bench.probes import ProbeResult
        from tts_bench.runner import _refused

        for error in ("API key is revoked or expired", "Missing API key", "Missing subscription", "Insufficient balance",
                      "Quota exceeded", "Rate limit exceeded", "Invalid audio format"):
            gradium, _ = adapter(GradiumAdapter, "default", "v")
            opening = asyncio.create_task(gradium.open_context("main"))
            await asyncio.sleep(0.01)
            gradium._on_message({"type": "ready", "client_req_id": "main"})
            synthesis = await opening
            await gradium.send("main", "Hello.")
            gradium._on_message({"type": "error", "client_req_id": "main", "message": error, "code": 1008})
            result = _refused(ProbeResult("one_shot", verdict="fail", syntheses=[synthesis]))
            assert synthesis.done.is_set() and bool(result.void) == (error != "Invalid audio format")
            assert result.verdict == ("fail" if error == "Invalid audio format" else None)

    async def test_pcm_rates_match_the_setup_and_cancel_is_a_declared_exclusion(self):
        import asyncio

        from tts_bench.adapters.gradium import GradiumAdapter

        for rate in GradiumAdapter.native_rates:
            gradium, sent = adapter(GradiumAdapter, "default", "v")
            gradium.config = TTSConfig("default", "v", sample_rate=rate)
            opening = asyncio.create_task(gradium.open_context("main"))
            await asyncio.sleep(0.01)
            assert sent[0]["output_format"] == ("pcm" if rate == 48000 else f"pcm_{rate}")
            gradium._on_message({"type": "ready", "client_req_id": "main"})
            await opening
        assert gradium.url == "wss://api.gradium.ai/api/speech/tts" and gradium._headers() == {"x-api-key": "unused"}
        assert GradiumAdapter.unsupported_reason(gradium.config, ("cancel",)) == "gradium has no cancel"
        assert GradiumAdapter.unsupported_reason(gradium.config, ("streamed_input", "continuation")) is None
        assert GradiumAdapter.native_mulaw_8k and not GradiumAdapter.ends_on_quiet

    async def test_a_reusable_socket_streams_two_syntheses_without_logging_the_key(self, tmp_path):
        import base64
        import json

        import websockets

        from tts_bench.adapters.gradium import GradiumAdapter

        seen: list[dict] = []
        headers: dict = {}
        pcm = b"\x01\x00" * 240

        async def serve(ws):
            headers.update(ws.request.headers)
            async for raw in ws:
                frame = json.loads(raw)
                seen.append(frame)
                stamp = {"client_req_id": frame["client_req_id"]}
                if frame["type"] == "setup":
                    await ws.send(json.dumps({"type": "ready", "sample_rate": 24000, **stamp}))
                elif frame["type"] == "text":
                    await ws.send(json.dumps({"type": "audio", "audio": base64.b64encode(pcm).decode(), **stamp}))
                elif frame["type"] == "end_of_stream":
                    await ws.send(json.dumps({"type": "end_of_stream", **stamp}))

        clock = Clock()
        log = EventLog(clock, raw_path=tmp_path / "raw.jsonl")
        gradium = GradiumAdapter(TTSConfig("default", "NbpkqMVS3CJeq2j8"), "secret-key", log, clock)
        async with websockets.serve(serve, "127.0.0.1", 0) as server:
            gradium.url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            async with gradium:
                for context_id in ("first", "repeat"):
                    synthesis = await gradium.open_context(context_id)
                    assert synthesis.t0 is None
                    await gradium.send(context_id, "Hello ")
                    await gradium.send(context_id, "there.")
                    await gradium.finish(context_id)
                    done = await gradium.wait(context_id, 5.0)
                    assert bytes(done.pcm) == pcm * 2 and done.meta["ended_by"] == "provider"
        assert [frame["type"] for frame in seen] == ["setup", "text", "text", "end_of_stream"] * 2
        assert headers["x-api-key"] == "secret-key"
        raw = (tmp_path / "raw.jsonl").read_text()
        assert "secret-key" not in raw and "secret-key" not in str(log.events) and "<640 b64 chars>" in raw


class TestRime:
    async def test_concatenable_text_waits_for_one_flush_and_done_ends_the_batch(self):
        from tts_bench.adapters.rime import RimeAdapter
        from tts_bench.probes import Continuation

        rime, sent = adapter(RimeAdapter, "coda", "astra")
        synthesis = await rime.open_context("main")
        assert synthesis.t0 is None and sent == []
        for frame in Continuation.frames("Hello there. How can I help?"):
            await rime.send("main", frame)
        await rime.finish("main")
        assert sent == [{"text": "Hello there. ", "contextId": "main"},
                        {"text": "How can I help?", "contextId": "main"}, {"operation": "flush"}]
        rime._on_message({"type": "chunk", "contextId": "main", "data": "AQABAA=="})
        rime._on_message({"type": "timestamps", "contextId": "main", "word_timestamps": {"words": ["Hello"]}})
        assert bytes(synthesis.pcm) == b"\x01\x00" * 2 and not synthesis.done.is_set()
        rime._on_message({"type": "done", "contextId": "main"})
        assert synthesis.done.is_set() and synthesis.t_done is not None

    async def test_errors_are_recorded_and_clear_is_not_offered_as_a_cancel(self):
        from tts_bench.adapters.rime import RimeAdapter
        from tts_bench.registry import PENDING

        rime, _ = adapter(RimeAdapter, "mistv3", "astra")
        synthesis = await rime.open_context("main")
        rime._on_message({"type": "error", "message": "Speaker not found"})
        assert synthesis.error == "Speaker not found" and synthesis.done.is_set()
        for model in PROVIDERS["rime"].models:
            for rate in RimeAdapter.native_rates:
                rime.config = TTSConfig(model.model, model.voice, sample_rate=rate)
                assert f"modelId={model.model}" in rime.url and f"samplingRate={rate}" in rime.url
                assert "audioFormat=pcm" in rime.url and "segment=never" in rime.url
                assert RimeAdapter.unsupported_reason(rime.config, ("streamed_input", "continuation")) is None
        assert rime.url.startswith("wss://users-ws.rime.ai/ws3?") and rime._headers() == {"Authorization": "Bearer unused"}
        assert RimeAdapter.unsupported_reason(rime.config, ("cancel",)) == "rime has no cancel"
        assert RimeAdapter.native_mulaw_8k and not RimeAdapter.ends_on_quiet and "rime" not in PENDING

    async def test_a_reusable_socket_keeps_repeat_audio_separate_and_the_key_out_of_logs(self, tmp_path):
        import base64
        import json

        import websockets

        from tts_bench.adapters.rime import RimeAdapter

        seen: list[dict] = []
        headers: dict = {}
        pcm = b"\x01\x00" * 240

        async def serve(ws):
            headers.update(ws.request.headers)
            context_id = None
            async for raw in ws:
                frame = json.loads(raw)
                seen.append(frame)
                if "text" in frame:
                    context_id = frame["contextId"]
                elif frame.get("operation") == "flush":
                    await ws.send(json.dumps({"type": "chunk", "contextId": context_id, "data": base64.b64encode(pcm).decode()}))
                    await ws.send(json.dumps({"type": "done", "contextId": context_id}))

        clock = Clock()
        log = EventLog(clock, raw_path=tmp_path / "raw.jsonl")
        rime = RimeAdapter(TTSConfig("coda", "astra"), "secret-key", log, clock)
        async with websockets.serve(serve, "127.0.0.1", 0) as server:
            rime.base = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            async with rime:
                for context_id in ("first", "repeat"):
                    synthesis = await rime.open_context(context_id)
                    assert synthesis.t0 is None
                    await rime.send(context_id, "Hello ")
                    await rime.send(context_id, "there.")
                    await rime.finish(context_id)
                    done = await rime.wait(context_id, 5.0)
                    assert bytes(done.pcm) == pcm and done.meta["ended_by"] == "provider"
        assert len(seen) == 6 and headers["authorization"] == "Bearer secret-key"
        raw = (tmp_path / "raw.jsonl").read_text()
        assert "secret-key" not in raw and "secret-key" not in str(log.events) and "<640 b64 chars>" in raw


class TestFishAudio:
    async def test_start_is_before_t0_and_only_finish_ends_the_utterance(self):
        import msgpack

        from tts_bench.adapters.fish_audio import FishAudioAdapter
        from tts_bench.registry import SARAH

        fish, sent = adapter(FishAudioAdapter, "s2.1-pro", SARAH)
        synthesis = await fish.open_context("main")
        assert synthesis.t0 is None and sent == [{"event": "start", "request": {
            "text": "", "reference_id": SARAH, "format": "pcm", "sample_rate": 24000,
            "latency": "balanced", "condition_on_previous_chunks": True}}]
        await fish.send("main", "Hello ")
        await fish.send("main", "there.")
        await fish.finish("main")
        assert sent[1:] == [{"event": "text", "text": "Hello "}, {"event": "text", "text": "there."}, {"event": "stop"}]
        fish._on_binary(msgpack.packb({"event": "audio", "audio": b"\x01\x00" * 2}, use_bin_type=True))
        fish._on_binary(msgpack.packb({"event": "future_event"}, use_bin_type=True))
        assert bytes(synthesis.pcm) == b"\x01\x00" * 2 and not synthesis.done.is_set()
        fish._on_binary(msgpack.packb({"event": "finish", "reason": "stop"}, use_bin_type=True))
        assert synthesis.done.is_set() and synthesis.t_done is not None

    async def test_finish_error_is_a_failure_and_cancel_and_mulaw_are_excluded(self):
        from tts_bench.adapters.fish_audio import FishAudioAdapter
        from tts_bench.registry import PENDING, SARAH

        fish, sent = adapter(FishAudioAdapter, "s2.1-pro", SARAH)
        synthesis = await fish.open_context("main")
        fish._on_message({"event": "finish", "reason": "error"})
        assert synthesis.error and synthesis.done.is_set() and synthesis.t_done is None
        assert fish.url == "wss://api.fish.audio/v1/tts/live"
        assert fish._headers() == {"Authorization": "Bearer unused", "model": "s2.1-pro"}
        for rate in FishAudioAdapter.native_rates:
            fish, sent = adapter(FishAudioAdapter, "s2.1-pro", SARAH)
            fish.config = TTSConfig("s2.1-pro", SARAH, sample_rate=rate, options=(("latency", "low"),))
            await fish.open_context("main")
            assert sent[0]["request"]["sample_rate"] == rate and sent[0]["request"]["latency"] == "low"
            assert FishAudioAdapter.unsupported_reason(fish.config, ("streamed_input", "continuation")) is None
        assert FishAudioAdapter.unsupported_reason(fish.config, ("cancel",)) == "fish-audio has no cancel"
        assert FishAudioAdapter.unsupported_reason(TTSConfig("s2.1-pro", SARAH, sample_rate=48000)) == "fish-audio does not emit 48000 Hz PCM"
        assert not FishAudioAdapter.native_mulaw_8k and not FishAudioAdapter.ends_on_quiet and "fish-audio" not in PENDING

    async def test_messagepack_streams_two_syntheses_on_two_sockets_without_logging_the_key(self, tmp_path):
        import msgpack
        import websockets

        from tts_bench.adapters.fish_audio import FishAudioAdapter
        from tts_bench.registry import SARAH

        seen: list[list[dict]] = []
        headers: list[dict] = []
        pcm = b"\x01\x00" * 240

        async def serve(ws):
            frames: list[dict] = []
            seen.append(frames)
            headers.append(dict(ws.request.headers))
            async for raw in ws:
                assert isinstance(raw, bytes)
                frame = msgpack.unpackb(raw, raw=False)
                frames.append(frame)
                if frame["event"] == "text":
                    await ws.send(msgpack.packb({"event": "audio", "audio": pcm}, use_bin_type=True))
                elif frame["event"] == "stop":
                    await ws.send(msgpack.packb({"event": "finish", "reason": "stop"}, use_bin_type=True))
                    await ws.close()

        clock = Clock()
        log = EventLog(clock, raw_path=tmp_path / "raw.jsonl")
        fish = FishAudioAdapter(TTSConfig("s2.1-pro", SARAH), "secret-key", log, clock)
        async with websockets.serve(serve, "127.0.0.1", 0) as server:
            fish.url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            async with fish:
                for context_id in ("first", "repeat"):
                    synthesis = await fish.open_context(context_id)
                    assert synthesis.t0 is None
                    await fish.send(context_id, "Hello ")
                    await fish.send(context_id, "there.")
                    await fish.finish(context_id)
                    done = await fish.wait(context_id, 5.0)
                    assert bytes(done.pcm) == pcm * 2 and done.meta["ended_by"] == "provider"
        assert [[frame["event"] for frame in frames] for frames in seen] == [["start", "text", "text", "stop"]] * 2
        assert all(h["authorization"] == "Bearer secret-key" and h["model"] == "s2.1-pro" for h in headers)
        raw = (tmp_path / "raw.jsonl").read_text()
        assert "secret-key" not in raw and "secret-key" not in str(log.events) and "<480 bytes>" in raw


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


class TestEndOfUtterance:
    async def test_a_pause_does_not_end_a_protocol_that_says_when_it_is_done(self):
        import asyncio

        from tts_bench.adapters.openai_tts import OpenAITTSAdapter

        openai, _ = adapter(OpenAITTSAdapter, "gpt-4o-mini-tts", "alloy")
        synthesis = await openai.open_context("main")
        synthesis.t_first_text = synthesis.t_input_done = openai.clock.now()
        openai._on_audio("main", b"\x01\x00" * 480)
        waiting = asyncio.create_task(openai.wait("main", 5.0, quiet_s=0.1))
        await asyncio.sleep(0.4)                                   # a stall four times the quiet window
        assert not waiting.done()
        openai._on_audio("main", b"\x01\x00" * 480)
        openai._on_done("main")
        finished = await waiting
        assert finished.meta["ended_by"] == "provider" and len(finished.pcm) == 1920

    async def test_a_protocol_without_a_done_message_still_ends_on_quiet(self):
        from tts_bench.adapters.smallest import SmallestAdapter

        smallest, _ = adapter(SmallestAdapter, "lightning_v3.1_pro", "kelsey")
        synthesis = await smallest.open_context("main")
        synthesis.t_first_text = synthesis.t_input_done = smallest.clock.now()
        smallest._on_audio("main", b"\x01\x00" * 480)
        finished = await smallest.wait("main", 5.0, quiet_s=0.1)
        assert finished.meta["ended_by"] == "quiet"


class TestAzureSpeech:
    def test_the_model_rides_on_the_voice_name_and_the_text_is_escaped(self):
        from tts_bench.adapters.azure_speech import AzureSpeechAdapter

        azure, _ = adapter(AzureSpeechAdapter, "MAI-Voice-2.1-Flash", "en-US-Harper")
        body = azure.ssml('Press Ctrl+Shift+T & say "5 < 6".')
        assert '<voice name="en-US-Harper:MAI-Voice-2.1-Flash">' in body and 'xml:lang="en-US"' in body
        assert "Ctrl+Shift+T &amp; say \"5 &lt; 6\"." in body and "express-as" not in body

    def test_the_region_is_the_host_and_east_us_is_the_default(self):
        from tts_bench.adapters.azure_speech import AzureSpeechAdapter
        from tts_bench.common.events import Clock, EventLog

        azure, _ = adapter(AzureSpeechAdapter, "MAI-Voice-2.1", "en-US-Harper")
        assert f"{azure.base}{azure.speech_path}" == "https://eastus2.tts.speech.microsoft.com/cognitiveservices/v1"
        clock = Clock()
        west = AzureSpeechAdapter(TTSConfig(model="MAI-Voice-2.1", voice="en-US-Harper", options=(("region", "westus2"),)),
                                  "unused", EventLog(clock), clock)
        assert west.base == "https://westus2.tts.speech.microsoft.com"

    async def test_one_request_streams_pcm_and_the_key_never_reaches_the_log(self, tmp_path):
        from aiohttp import web

        from tts_bench.adapters.azure_speech import AzureSpeechAdapter

        seen: dict = {}

        async def voices(request):
            return web.json_response([])

        async def speak(request):
            seen["headers"], seen["body"] = dict(request.headers), await request.text()
            response = web.StreamResponse(headers={"X-RequestId": "r1"})
            await response.prepare(request)
            for _ in range(3):
                await response.write(b"\x01\x00" * 240)
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_get("/cognitiveservices/voices/list", voices)
        app.router.add_post("/cognitiveservices/v1", speak)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            clock = Clock()
            log = EventLog(clock, raw_path=tmp_path / "raw.jsonl")
            azure = AzureSpeechAdapter(TTSConfig(model="MAI-Voice-2.1-Flash", voice="en-US-Harper"), "secret-key", log, clock)
            azure.base = f"http://127.0.0.1:{port}"
            async with azure:
                synthesis = await azure.open_context("main")
                await azure.send("main", "Your total is $42.")
                await azure.finish("main")
                done = await azure.wait("main", 5.0)
        finally:
            await runner.cleanup()
        assert len(done.pcm) == 1440 and done.meta["ended_by"] == "provider" and synthesis is done
        assert seen["headers"]["X-Microsoft-OutputFormat"] == "raw-24khz-16bit-mono-pcm"
        assert seen["headers"]["Ocp-Apim-Subscription-Key"] == "secret-key" and "$42." in seen["body"]
        assert done.meta["headers"]["x-requestid"] == "r1"
        raw = (tmp_path / "raw.jsonl").read_text()
        assert "secret-key" not in raw and "secret-key" not in str(log.events)
        assert "$42" not in raw and '"input": "<18 chars>"' in raw
