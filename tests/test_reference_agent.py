"""The agent bench reference agent's wiring, without a provider or a phone line.

What is worth testing offline is everything that decides whether a call is
scored fairly: the sample rate handed to each provider, the tools the model is
shown, the answers it gets back, and the greeting it is told to say. The parts
that need a network -- that the model connects, speaks and calls tools -- are
verified by running it, not by mocking it.

Skipped automatically unless Pipecat is installed, so the service bench test run stays
dependency-free.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pipecat", reason="reference agent needs pipecat-ai")

AGENT_DIR = Path(__file__).resolve().parent.parent / "reference-agents" / "pipecat-s2s"
sys.path.insert(0, str(AGENT_DIR))

import bot  # noqa: E402


def asked(**overrides) -> bot.Settings:
    """The session body a call arrives with.

    Configuration reaches this agent from the session that starts it, so the
    tests configure it the way the platform does rather than by setting the
    environment the deployment happens to have.
    """
    return bot.Settings({"agent_dir": "appointments", **overrides})


class TestProviderTable:
    def test_openai_realtime_runs_the_pipeline_at_24k(self):
        """This service does not resample and accepts 24 kHz only.

        At any other rate it hears the caller at the wrong speed, transcribes
        badly, and the run looks like a model failure rather than a wiring one.
        """
        assert bot.PROVIDERS["openai-realtime"].input_rate == 24000

    def test_every_provider_declares_a_rate_a_model_and_a_credential(self):
        for name, provider in bot.PROVIDERS.items():
            assert provider.input_rate in (8000, 16000, 24000), name
            assert provider.default_model and provider.default_voice, name
            assert provider.credential_env, name

    def test_every_credential_variable_is_documented(self):
        """An operator can only set a variable they can find.

        The names are the vendors' own rather than a house style -- Bedrock's
        API key carries no ``_API_KEY`` suffix -- so nothing about the spelling
        can be asserted. What can be asserted is that each one is written down
        where someone looking for it will look.
        """
        readme = (bot.REPO_ROOT / "reference-agents" / "pipecat-s2s" / "README.md").read_text()
        for name, provider in bot.PROVIDERS.items():
            for variable in provider.credential_env:
                assert variable in readme, f"{name}: {variable} is not in the README"

    def test_every_provider_names_an_importable_module(self):
        """The pre-import is only worth having if the module names are right.

        A typo here costs nothing visible: the warm-up logs and moves on, and the
        SDK loads later instead -- inside the window being measured, which is the
        one place the cost does not show up as itself.
        """
        import importlib.util

        for name, provider in bot.PROVIDERS.items():
            assert importlib.util.find_spec(provider.module), f"{name}: {provider.module}"

    def test_an_unknown_provider_is_refused_by_name(self):
        assert "openai-realtime" in bot.PROVIDERS
        assert "pipecat-cascade" not in bot.PROVIDERS


class TestOpeningTurn:
    def test_the_greeting_is_quoted_verbatim(self):
        messages = bot.opening_messages("Thanks for calling Acme.")
        assert messages[0]["role"] == "user"
        assert "exactly" in messages[0]["content"]
        assert '"Thanks for calling Acme."' in messages[0]["content"]

    def test_inner_double_quotes_are_folded(self):
        """They would close the quoted span the instruction opens."""
        content = bot.opening_messages('Hello, this is "Acme" calling.')[0]["content"]
        assert "'Acme'" in content
        assert content.count('"') == 2, content

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_no_greeting_falls_back_to_an_improvised_one(self, empty):
        content = bot.opening_messages(empty)[0]["content"]
        assert "exactly" not in content and "Greet" in content


class TestTools:
    def test_the_model_is_shown_every_published_tool(self):
        server = bot.load_agent(asked())
        schema = bot.build_tools(server)
        shown = {t.name for t in schema.standard_tools}
        assert set(server.tool_names) <= shown
        assert all(t.description for t in schema.standard_tools)

    def test_the_model_can_also_end_and_hand_over_the_call(self):
        """Two things the agent must do rather than look up.

        Neither returns a record, so neither is in the published tables. But a
        scored call is judged on whether it terminated appropriately, and an
        agent with no way to hang up fails that for a reason having nothing to
        do with the model.
        """
        shown = {t.name for t in bot.build_tools(bot.load_agent(asked())).standard_tools}
        assert {"end_call", "transfer_call"} <= shown

    def test_call_control_is_not_confused_with_the_contract(self):
        """The published tables answer lookups; these two are not lookups."""
        server = bot.load_agent(asked())
        assert not (set(bot.CALL_CONTROL) & set(server.tool_names))

    async def test_a_registered_handler_answers_from_the_contract(self):
        server = bot.load_agent(asked())
        registered = {}

        class StubLLM:
            def register_function(self, name, handler, **_kwargs):
                registered[name] = handler

        trace = bot.ToolTrace()
        bot.register_tools(StubLLM(), server, trace)
        assert set(registered) == set(server.tool_names) | set(bot.CALL_CONTROL)

        answers = []

        class Params:
            function_name = "lookup_patient"
            arguments = {"phone": "2025550188"}

            async def result_callback(self, result):
                answers.append(result)

        await registered["lookup_patient"](Params())
        assert answers and answers[0]["patient_id"] == "p_1002"

    async def test_an_unknown_record_is_reported_as_a_miss(self):
        server = bot.load_agent(asked())
        registered = {}

        class StubLLM:
            def register_function(self, name, handler, **_kwargs):
                registered[name] = handler

        trace = bot.ToolTrace()
        bot.register_tools(StubLLM(), server, trace)
        answers = []

        class Params:
            function_name = "lookup_patient"
            arguments = {"phone": "4045550000"}

            async def result_callback(self, result):
                answers.append(result)

        await registered["lookup_patient"](Params())
        # A number nothing in the table resembles is answered by the contract's
        # own "no patient found" row, not by an invented patient.
        assert "patient_id" not in answers[0]
        assert answers[0]["match"] is False
        assert server.calls[-1].matched is False
        assert trace.as_metadata()["tool_calls_matched"] == 0, "only an exact call counts as a hit"

    @pytest.mark.asyncio
    async def test_every_tool_call_is_recorded_for_the_run(self):
        """Tool evidence has to be the same shape whoever answered the call.

        The framework traces tool calls unevenly across providers -- two of the
        five emit arguments and results, three emit no model-level spans at all
        -- so a board comparing tool use cannot read it from there. This side
        of the call is ours, so it is recorded here instead.
        """
        server = bot.MockToolServer(suite="appointments")
        registered = {}

        class StubLLM:
            def register_function(self, name, handler, **_kwargs):
                registered[name] = handler

        trace = bot.ToolTrace()
        bot.register_tools(StubLLM(), server, trace)

        class Params:
            function_name = "lookup_patient"
            arguments = {"phone": "2025550188"}
            llm = None

            async def result_callback(self, result):
                pass

        await registered["lookup_patient"](Params())
        metadata = trace.as_metadata()
        assert metadata["tool_call_count"] == 1
        assert metadata["tool_calls_matched"] == 1
        call = metadata["tool_calls"][0]
        assert call["name"] == "lookup_patient"
        assert call["arguments"] == {"phone": "2025550188"}
        assert call["output"], "the answer is the evidence; a name alone proves nothing"
        assert call["answered_ms"] >= call["requested_ms"], "a call cannot be answered before it is made"


class TestAgentDefinition:
    def test_the_contracts_own_prompt_and_greeting_are_used(self):
        server = bot.load_agent(asked())
        assert server.system_prompt and server.first_message
        assert server.suite == "appointments"


class TestBuildRecord:
    """A phone call cannot be replayed, so the build that answered it must be recorded."""

    def test_it_names_the_configuration_a_call_cannot_be_re_run_without(self):
        server = bot.load_agent(asked())
        record = bot.build_record(
            "openai-realtime", bot.PROVIDERS["openai-realtime"], "gpt-realtime-2.1", "marin", server, asked()
        )
        # The rate is the one that silently ruins a call: these services do not
        # resample, so a wrong rate reads as a bad model rather than bad wiring.
        assert record["pipeline_sample_rate"] == bot.PROVIDERS["openai-realtime"].input_rate
        assert record["pipecat_version"] != "unknown", "the framework version is part of the agent"
        assert record["system_prompt_sha256"] and record["first_message_sha256"]
        assert "lookup_patient" in record["tools"]

    def test_a_changed_prompt_changes_the_record(self):
        server = bot.load_agent(asked())
        provider = bot.PROVIDERS["openai-realtime"]
        before = bot.build_record("openai-realtime", provider, "m", "v", server, asked())
        server.system_prompt = server.system_prompt + " Answer briefly."
        after = bot.build_record("openai-realtime", provider, "m", "v", server, asked())
        assert before["system_prompt_sha256"] != after["system_prompt_sha256"]

    def test_every_declared_disclosure_reaches_the_record(self):
        """A provider that declares an extra must actually put it on the record.

        One loop rather than a test per provider: the failure this guards against
        is a sixth provider added with a disclosure that never lands, and a test
        naming the five that exist could not catch it.
        """
        server = bot.load_agent(asked())
        for name, provider in bot.PROVIDERS.items():
            record = bot.build_record(
                name, provider, provider.default_model, provider.default_voice, server, asked()
            )
            for field in provider.discloses(asked()):
                assert record.get(field), f"{name} declares {field} but the record has no value"

    def test_a_provider_with_nothing_to_disclose_carries_no_empty_field(self):
        """Absent rather than empty.

        A provider that reasons for itself has no backend model. An empty string
        would read as a field that went unrecorded, which is a different claim.
        """
        server = bot.load_agent(asked())
        plain = bot.build_record(
            "openai-realtime", bot.PROVIDERS["openai-realtime"], "gpt-realtime-2.1", "marin", server, asked()
        )
        assert "s2s_backend_model" not in plain
        assert "aws_region" not in plain


class TestConfiguredByTheSession:
    """One image answers for every row, so the session decides what is measured."""

    def test_the_session_decides_the_provider_and_the_model(self):
        settings = bot.Settings({"s2s_provider": "gemini-live", "s2s_model": "models/x"})
        assert settings.get("s2s_provider") == "gemini-live"
        assert settings.get("s2s_model") == "models/x"

    def test_the_environment_is_the_fallback(self, monkeypatch):
        """A deployment may carry a default, and a laptop has nothing else."""
        monkeypatch.setenv("S2S_PROVIDER", "grok-realtime")
        assert bot.Settings({}).get("s2s_provider") == "grok-realtime"
        assert bot.Settings({"s2s_provider": "gemini-live"}).get("s2s_provider") == "gemini-live"

    def test_a_credential_is_never_taken_from_the_session(self, monkeypatch):
        """Credentials stay in the environment.

        A key sent with the request is copied into every log, trace and session
        record that quotes the body, so this agent reads keys from one place
        only -- and a session that supplies one must not be able to change that.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "from-the-environment")
        settings = bot.Settings({"openai_api_key": "from-the-session"})
        assert settings.get("openai_api_key") != "from-the-session"
        assert bot._credential(("OPENAI_API_KEY",), "openai-realtime") == "from-the-environment"

    def test_a_scenario_variable_cannot_change_what_was_measured(self):
        """The platform flattens a scenario's own variables into the same body.

        An unfiltered read would let a fixture field decide the provider or the
        agent definition -- a scored run against the wrong contract, with
        nothing in the record saying so.
        """
        settings = bot.Settings({"agent_dir": "medicare", "s2s_voice": "ash", "caller_name": "x"})
        assert settings.get("caller_name") is None
        assert settings.get("agent_dir") == "medicare" and settings.get("s2s_voice") == "ash"

    def test_an_unset_agent_definition_is_refused(self, monkeypatch):
        monkeypatch.delenv("AGENT_DIR", raising=False)
        with pytest.raises(ValueError, match="agent_dir"):
            bot.load_agent(bot.Settings({}))

    def test_the_record_says_which_side_configured_the_call(self, monkeypatch):
        """Two rows configured differently are not obviously two rows otherwise."""
        monkeypatch.setenv("S2S_PROVIDER", "openai-realtime")
        provider = bot.PROVIDERS["openai-realtime"]
        from_session = bot.build_record(
            "openai-realtime", provider, "m", "v", bot.load_agent(asked()),
            asked(s2s_provider="openai-realtime"),
        )
        from_image = bot.build_record(
            "openai-realtime", provider, "m", "v", bot.load_agent(asked()), asked()
        )
        assert from_session["config_source"] == "session"
        assert from_image["config_source"] == "environment"

    def test_every_provider_sdk_is_loaded_before_a_call_arrives(self):
        """The provider is unknown until the session names it.

        So every SDK is imported at start-up, not the one this process will
        use -- otherwise the import lands inside the window a first response is
        timed in, and a container start reads as a slow model.
        """
        needed = {e.module for e in bot.PROVIDERS.values()} | {e.module for e in bot.TEXT_MODELS.values()}
        missing = needed - set(sys.modules)
        assert not missing, f"not pre-imported: {sorted(missing)}"

    def test_the_tracer_is_loaded_before_a_call_arrives_too(self):
        """It is imported by ``create_task``, which runs per call.

        Left to load itself it is a tenth of a second of import inside the first
        call on every worker, and several times that on a container whose page
        cache is cold -- charged to the first response of a scored call.
        """
        assert "cekura.pipecat" in sys.modules

    def test_a_record_says_which_worker_answered_and_how_many_it_had(self):
        """A worker serves many calls, and the first one is not like the rest."""
        record = bot.build_record(
            "openai-realtime", bot.PROVIDERS["openai-realtime"], "m", "v",
            bot.load_agent(asked()), asked(),
        )
        assert record["worker_instance"] == bot.INSTANCE
        assert isinstance(record["worker_call"], int)


class TestCredentialForms:
    """Bedrock's two credential forms, only one of which can reach this model."""

    def test_an_access_key_pair_signs_with_sigv4(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secretpart")
        built = bot._nova_sonic("AKIAEXAMPLE", "amazon.nova-2-sonic-v1:0", "matthew", "hi", asked())
        assert built.__class__.__name__ == "AWSNovaSonicLLMService"

    def test_an_api_key_is_not_a_credential_for_this_model(self):
        """AWS excludes the bidirectional stream from bearer authentication.

        A key that opens no session must not be listed as one that does: it
        would satisfy every start-up check and fail on the first call.
        """
        assert "AWS_BEARER_TOKEN_BEDROCK" not in bot.PROVIDERS["nova-sonic"].credential_env

    def test_the_default_region_is_one_the_model_is_served_in(self):
        """Four regions serve it; a default outside them fails as a denial.

        An unserved region and an ungranted one raise the same error, so a
        default that is merely plausible costs a debugging session to rule out.
        """
        served = {"us-east-1", "us-west-2", "eu-north-1", "ap-northeast-1"}
        assert bot.aws_region(asked()) in served


class TestQwenRealtime:
    """The one provider whose protocol we speak ourselves."""

    def _service(self, **overrides):
        from qwen_realtime import QwenRealtimeLLMService

        return QwenRealtimeLLMService(
            api_key="k", workspace_id="ws-1", instructions="hi", **overrides
        )

    def test_the_endpoint_names_the_region_and_the_workspace(self):
        """A key authenticates against one region, so the two must agree."""
        assert "ap-southeast-1" in self._service()._url
        assert "ws-1" in self._service()._url
        assert "cn-beijing" in self._service(region="beijing")._url

    def test_an_unknown_region_is_refused(self):
        with pytest.raises(ValueError):
            self._service(region="mars")

    def test_tools_are_sent_in_the_nested_form(self):
        """Qwen takes chat-completions' shape, not the realtime protocols' flat one.

        A tool in the wrong shape is ignored rather than rejected, so the model
        would fail a scenario for having no tools rather than for anything
        about the model.
        """
        server = bot.MockToolServer(suite="appointments")
        service = self._service(tools=bot.build_tools(server))
        encoded = service._encoded_tools()
        assert encoded, "the published contract produced no tools"
        for tool in encoded:
            assert tool["type"] == "function"
            assert "name" in tool["function"], "name must sit under function"
            assert "name" not in tool, "the flat realtime shape is not accepted here"

    def test_the_two_sample_rates_differ(self):
        """It listens at 16 kHz and speaks at 24 kHz.

        One rate for both plays the model's voice at the wrong speed, which
        reads as a bad model rather than as bad wiring.
        """
        import qwen_realtime

        assert qwen_realtime.INPUT_SAMPLE_RATE == bot.PROVIDERS["qwen-realtime"].input_rate
        assert qwen_realtime.OUTPUT_SAMPLE_RATE != qwen_realtime.INPUT_SAMPLE_RATE

    def test_a_missing_workspace_is_refused(self):
        """No default: a guessed workspace fails as a DNS error, not a setting."""
        with pytest.raises(ValueError):
            bot.qwen_workspace(asked())


class TestCascadeCounterparts:
    """The comparison the board exists to make: native against the pipeline it replaces."""

    def test_every_native_provider_has_a_counterpart(self):
        """A native row with nothing to compare against cannot answer the question.

        Nova Sonic and GPT-Live are allowed to stand alone for now -- one is
        credential-blocked and the other delegates to a backend, so its fair
        pairing is a cascade on that same backend rather than a vendor default.
        """
        paired = {c.counterpart_to for c in bot.TEXT_MODELS.values() if c.counterpart_to}
        unpaired = set(bot.PROVIDERS) - paired - {"nova-sonic", "gpt-live"}
        assert not unpaired, f"native providers with no cascade counterpart: {sorted(unpaired)}"

    def test_a_counterpart_names_a_provider_that_exists(self):
        for key, text in bot.TEXT_MODELS.items():
            if text.counterpart_to is not None:
                assert text.counterpart_to in bot.PROVIDERS, key

    def test_both_stacks_describe_the_same_job(self):
        """If these differ, the two agents were not given the same work to do.

        That is the assumption every native-versus-cascade comparison rests on,
        and it is the one that can break silently: a prompt or tool list that
        drifted between the two paths would produce a difference that looks like
        architecture and is not.
        """
        server = bot.load_agent(asked())
        native = bot.build_record(
            "openai-realtime", bot.PROVIDERS["openai-realtime"], "gpt-realtime-2.1", "marin", server, asked()
        )
        cascade = bot.cascade_record(
            "cascade-openai", bot.TEXT_MODELS["cascade-openai"], "gpt-4.1", server, asked()
        )
        for field in ("agent_definition", "system_prompt_sha256", "first_message_sha256", "tools"):
            assert native[field] == cascade[field], field
        assert native["stack"] == "native" and cascade["stack"] == "cascade"

    def test_a_cascade_row_names_all_three_services(self):
        """A latency column is mostly the endpointer and the voice, not the text model.

        A row naming only its text model would hide the two components doing
        most of what is being measured.
        """
        server = bot.load_agent(asked())
        record = bot.cascade_record(
            "cascade-baseline", bot.TEXT_MODELS["cascade-baseline"], "gpt-4.1", server, asked()
        )
        assert record["stt_model"] and record["llm_model"] and record["tts_model"]

    def test_every_cascade_names_an_importable_module(self):
        import importlib.util

        for key, text in bot.TEXT_MODELS.items():
            assert importlib.util.find_spec(text.module), f"{key}: {text.module}"

    def test_a_configuration_name_is_never_ambiguous(self):
        """One name selects one stack. An overlap would make a run unattributable."""
        assert not (set(bot.PROVIDERS) & set(bot.TEXT_MODELS))


class TestContractRoot:
    """Where the agent looks for the published contract.

    The repository nests this file three directories below the contract; the
    deployed image flattens both into one working directory. A root derived by
    counting parent directories is therefore correct in exactly one of the two
    places, and wrong in the one that runs the benchmark -- which is how a
    deployment reached a session and then died looking for `/agent-definitions`.
    """

    def test_flattened_layout_resolves_beside_the_agent(self, tmp_path):
        """What the deployed image looks like: agent and contract in one directory."""
        (tmp_path / "agent-definitions" / "appointments").mkdir(parents=True)
        assert bot._repo_root(tmp_path / "bot.py") == tmp_path

    def test_repository_layout_resolves_three_directories_up(self):
        assert bot._repo_root(AGENT_DIR / "bot.py") == AGENT_DIR.parent.parent

    def test_resolved_root_holds_the_contract(self):
        assert (bot.REPO_ROOT / "agent-definitions" / "appointments").is_dir()

    def test_the_agent_loads_its_contract_from_that_root(self):
        server = bot.load_agent(asked())
        assert server.system_prompt
        assert server.tool_specs


class TestCallerTranscription:
    """Both halves of the conversation have to reach the record.

    A realtime service that is not asked to transcribe the caller runs perfectly
    well and produces a transcript containing only the agent, which reads as a
    caller who never spoke. Tool calls travel separately, so a run can show
    resolved tools and still carry no speech -- which is why this is checked
    rather than assumed from a passing call.
    """

    def test_openai_realtime_asks_for_caller_transcription(self):
        service = bot.PROVIDERS["openai-realtime"].build(
            "test-key", "gpt-realtime-2.1", "marin", "instructions", asked()
        )
        audio = service._settings.session_properties.audio
        assert audio.input is not None and audio.input.transcription is not None

    def test_grok_names_its_own_transcription_model(self):
        service = bot.PROVIDERS["grok-realtime"].build(
            "test-key", "grok-voice-latest", "eve", "instructions", asked()
        )
        audio = service._settings.session_properties.audio
        assert audio.input.transcription.model == bot.GROK_TRANSCRIBE_MODEL


class TestCallerTurnsReachTheRecord:
    """Transcription is necessary and not sufficient: a turn also has to end.

    A realtime service endpoints on its own server and only *proposes* the
    boundary; the proposal becomes a turn only if a strategy adopts it, and the
    default strategies watch for local voice activity instead. Left at the
    defaults the caller's text is aggregated and never handed over, so the
    record shows an agent talking to nobody -- with tools resolved, turns
    counted and a clean verdict, which is why none of those caught it.
    """

    def test_realtime_defers_to_the_service_for_turn_boundaries(self):
        from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

        params = bot.user_aggregator_params(realtime=True)
        assert isinstance(params.user_turn_strategies, ExternalUserTurnStrategies)

    def test_cascade_leaves_room_for_its_speech_to_text_recommendation(self):
        # Naming strategies here would override the ones the speech-to-text
        # service recommends when it announces itself.
        assert bot.user_aggregator_params(realtime=False).user_turn_strategies is None

    def test_every_row_carries_the_speech_clock(self):
        # Response time is measured from the instant the caller fell silent, and
        # only a voice-activity detector marks it. No detector, no latency cell.
        for realtime in (True, False):
            assert bot.user_aggregator_params(realtime=realtime).vad_analyzer is not None


class TestSpeechClockRate:
    """The detector has to survive the rate the fastest providers open at."""

    def test_detector_holds_its_own_rate_when_the_pipeline_differs(self):
        vad = bot.BenchVAD()
        vad.set_sample_rate(24000)  # what openai-realtime and gpt-live open at
        assert vad.sample_rate == bot.VAD_RATE

    @pytest.mark.asyncio
    async def test_pipeline_audio_is_converted_before_analysis(self):
        vad = bot.BenchVAD()
        vad.set_sample_rate(24000)
        # A quarter second of silence at the pipeline's rate. Unconverted this
        # raises inside the detector rather than returning a state.
        assert await vad.analyze_audio(b"\x00\x00" * 6000) is not None


class TestCallerWordsAreExported:
    """The caller reaches the context, and has to reach the exported transcript too.

    In realtime mode the framework reports the end of a caller's turn with no
    text attached -- the service is often still transcribing when the boundary
    is announced -- and delivers the finalized text later on a separate event.
    The tracing SDK subscribes only to the boundary and drops it when it carries
    no text, which is right for a cascade and lossy for every native speech
    model. The call itself is unaffected, which is why a run scored a hundred
    with no caller in it.

    This drives the real aggregator through the frame order a realtime service
    produces -- boundary first, transcript after -- and asks the SDK's own
    recorder what it kept.
    """

    @staticmethod
    async def _run(register_fix: bool):
        import asyncio

        from cekura.pipecat._speech_timing import SpeechTimingObserver
        from cekura.pipecat.tracer import TranscriptCapture
        from pipecat.frames.frames import (
            Frame,
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            ProposedUserStartedSpeakingFrame,
            ProposedUserStoppedSpeakingFrame,
            StartFrame,
            TranscriptionFrame,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.utils.time import time_now_iso8601

        class FakeRealtimeService(FrameProcessor):
            def __init__(self):
                super().__init__()
                self.done = asyncio.Event()

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, StartFrame):
                    self.create_task(self._script())
                await self.push_frame(frame, direction)

            async def _script(self):
                # The service endpoints on its own server and only proposes the
                # boundary; the caller's words arrive afterwards, upstream.
                await asyncio.sleep(0.05)
                await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
                await asyncio.sleep(0.05)
                await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
                await asyncio.sleep(0.05)
                await self.push_frame(
                    TranscriptionFrame("i need to book an appointment", "", time_now_iso8601()),
                    FrameDirection.UPSTREAM,
                )
                await asyncio.sleep(0.05)
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame("sure, what day works"))
                await self.push_frame(LLMFullResponseEndFrame())
                await asyncio.sleep(1.0)
                self.done.set()

        context = LLMContext([{"role": "system", "content": "be brief"}])
        pair = LLMContextAggregatorPair(
            context,
            realtime_service_mode=True,
            user_params=bot.user_aggregator_params(realtime=True),
        )
        timing = SpeechTimingObserver()
        capture = TranscriptCapture(context, timing)

        # What the SDK itself subscribes to.
        @pair.user().event_handler("on_user_turn_started")
        async def _started(aggregator, strategy):
            await capture.on_user_turn_started()

        @pair.user().event_handler("on_user_turn_stopped")
        async def _stopped(aggregator, strategy, message):
            await capture.on_user_turn_stopped(message)

        if register_fix:
            bot.capture_caller_turns(SimpleNamespace(_transcript_capture=capture), pair.user())

        service = FakeRealtimeService()
        task = PipelineTask(
            Pipeline([pair.user(), service, pair.assistant()]),
            params=PipelineParams(),
            observers=[timing],
        )
        runner = PipelineRunner(handle_sigint=False)
        running = asyncio.create_task(runner.run(task))
        await asyncio.wait_for(service.done.wait(), timeout=30)
        await task.stop_when_done()
        await asyncio.wait_for(running, timeout=30)

        return context, [entry.get("role") for entry in capture.session_transcript]

    @pytest.mark.asyncio
    async def test_the_framework_does_put_the_caller_in_the_context(self):
        # Establishes that nothing upstream is losing the caller: the words are
        # there, so anything missing downstream is ours to fix.
        context, _ = await self._run(register_fix=False)
        assert "user" in [m.get("role") for m in context.messages if isinstance(m, dict)]

    @pytest.mark.asyncio
    async def test_without_the_late_event_the_caller_is_dropped(self):
        _, roles = await self._run(register_fix=False)
        assert "user" not in roles

    @pytest.mark.asyncio
    async def test_the_caller_is_exported(self):
        _, roles = await self._run(register_fix=True)
        assert "user" in roles


class TestTheRunCanBeIdentifiedFromItsOwnLogs:
    """The configuration line has to land inside the window the run captures.

    The tracing SDK starts capturing when the task is created and not before, so
    a line logged earlier reaches the container's output and never the run
    record. A log read months later against a single result has to say which
    provider, model, contract and commit produced it.
    """

    @staticmethod
    def _source() -> str:
        return (AGENT_DIR / "bot.py").read_text()

    def test_the_configuration_line_is_logged_after_capture_opens(self):
        source = self._source()
        assert source.index("create_task(pipeline") < source.index(
            'logger.info("agent bench reference agent: {}", record)'
        )

    def test_it_is_logged_exactly_once(self):
        # Twice would mean the pre-build line was left behind, doubling every
        # configuration line in the container output.
        assert self._source().count(
            'logger.info("agent bench reference agent: {}", record)'
        ) == 1

    def test_a_build_that_fails_still_says_what_it_was_building(self):
        # Both branches keep a small line ahead of the build, outside the window.
        assert self._source().count('logger.info("building {} on {}", name, model)') == 2


class TestControlTokensNeverReachTheRecord:
    """A tokenizer artifact is not speech and must not be scored as speech.

    One of these services streams control tokens into the transcript of what it
    said. There is no audio behind them -- the model never uttered them -- but
    on the record they read as the agent saying something incoherent, and any
    word-level comparison counts them as words.
    """

    @staticmethod
    async def _through(text: str) -> str:
        from pipecat.frames.frames import TTSTextFrame
        from pipecat.processors.frame_processor import FrameDirection

        processor = bot.DropControlTokens()
        seen = []

        async def capture(frame, direction=None):
            seen.append(frame)

        processor.push_frame = capture
        await processor.process_frame(
            TTSTextFrame(text, aggregated_by="test"), FrameDirection.DOWNSTREAM
        )
        return seen[-1].text

    @pytest.mark.asyncio
    async def test_the_artifact_is_removed(self):
        assert await self._through("<ctrl46><ctrl46>") == ""

    @pytest.mark.asyncio
    async def test_speech_around_it_survives(self):
        assert await self._through("your appointment <ctrl46>is booked") == "your appointment is booked"

    @pytest.mark.asyncio
    async def test_ordinary_text_is_untouched(self):
        # A filter on a benchmark's transcript is a filter on its evidence.
        for text in ("Hello <there>", "1 < 2 and 3 > 2", "a\n\nb"):
            assert await self._through(text) == text


class TestTheRecordDoesNotDependOnAGoodbye:
    """The record is finished as it goes, because the last moment may not come.

    When the agent ends the call itself -- how a scenario normally finishes --
    the transport tears down before it reports a departed caller, so a handler
    hung on that event never runs. Everything but the configuration was lost
    that way: the tool trace and everything the call consumed.
    """

    @pytest.mark.asyncio
    async def test_recording_a_tool_call_rewrites_the_record(self):
        server = bot.MockToolServer(suite="appointments")
        registered = {}

        class StubLLM:
            def register_function(self, name, handler, **_kwargs):
                registered[name] = handler

        trace = bot.ToolTrace()
        rewrites = []
        trace.on_change = lambda: rewrites.append(len(trace.as_metadata()["tool_calls"]))
        bot.register_tools(StubLLM(), server, trace)

        class Params:
            function_name = "lookup_patient"
            arguments = {"phone": "2025550188"}

            async def result_callback(self, result):
                pass

        await registered["lookup_patient"](Params())
        assert rewrites == [1], "the record must be rewritten as each call lands"

    def test_ending_the_call_is_itself_a_recorded_call(self):
        # Which is what makes the rewrite reach the end of a normal scenario.
        source = (AGENT_DIR / "bot.py").read_text()
        assert 'trace.record("end_call"' in source

    def test_the_rewrite_is_wired_to_the_trace(self):
        source = (AGENT_DIR / "bot.py").read_text()
        assert "trace.on_change = publish" in source
        assert source.index("def publish") < source.index("trace.on_change = publish")


class TestWhoDecidesTheCallersTurns:
    """Not every realtime service announces a turn boundary, and three do not.

    Their API exposes an interruption event and no turn start or end. Following
    an announcement that never comes leaves a pipeline that keeps a conversation
    context with no turn boundary at all: the transcript still arrives, so the
    run looks ordinary, but a caller who speaks across the end of the agent's
    reply can go unanswered. Those rows run a local detector instead, which the
    framework's own guidance prescribes for exactly this case.
    """

    SILENT = ("gemini-live", "nova-sonic", "qwen-realtime")

    def test_the_services_that_announce_nothing_are_marked(self):
        for name in self.SILENT:
            assert bot.PROVIDERS[name].turns == "local", name

    def test_every_other_service_is_followed(self):
        for name, provider in bot.PROVIDERS.items():
            if name not in self.SILENT:
                assert provider.turns == "provider", name

    def test_an_announced_boundary_is_followed(self):
        from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

        params = bot.user_aggregator_params(realtime=True, turns="provider")
        assert isinstance(params.user_turn_strategies, ExternalUserTurnStrategies)

    def test_a_silent_service_gets_a_local_detector(self):
        from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

        params = bot.user_aggregator_params(realtime=True, turns="local")
        assert params.user_turn_strategies is not None
        assert not isinstance(params.user_turn_strategies, ExternalUserTurnStrategies)
        assert params.user_turn_strategies.stop, "a local row needs something to end a turn"

    def test_a_cascade_is_unaffected(self):
        assert bot.user_aggregator_params(realtime=False, turns="local").user_turn_strategies is None

    def test_the_record_says_which_it_was(self):
        # Two rows whose turns were decided by different things are not
        # measuring the same thing, and a reader has to be able to see it.
        record = bot.build_record(
            "gemini-live", bot.PROVIDERS["gemini-live"], "m", "v",
            bot.MockToolServer(suite="appointments"), asked(),
        )
        assert record["turn_source"] == "local"
