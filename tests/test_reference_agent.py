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

import inspect
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


class StubLLM:
    """As much of an ``LLMService`` as ``register_tools`` binds to.

    One copy: the handlers it collects are the subject of several tests, and a
    per-test copy meant every new framework call in ``register_tools`` had to
    be stubbed four times.
    """

    def __init__(self):
        self.registered = {}

    def register_function(self, name, handler, **_kwargs):
        self.registered[name] = handler

    def event_handler(self, _name):
        return lambda fn: fn


def a_call(**overrides):
    """One tool call, shaped the way the framework hands it to a handler.

    Answers land on ``.answers`` so a test can read what the handler replied.
    """
    answers = []

    class Params:
        tool_call_id = "call-1"
        function_name = "lookup_patient"
        arguments = {}

        async def result_callback(self, result):
            answers.append(result)

    for name, value in overrides.items():
        setattr(Params, name, value)
    call = Params()
    call.answers = answers
    return call


def gemini_service(service_class, **attributes):
    """The Gemini service the row builds, with the network cut.

    Takes the model from the provider table so a model pin cannot leave the
    tests exercising a name the harness no longer runs.
    """
    service = service_class(
        api_key="k",
        settings=service_class.Settings(
            model=bot.PROVIDERS["gemini-live"].default_model, system_instruction="p"
        ),
    )

    async def quiet(*_args, **_kwargs):
        return None

    service._connect = quiet
    service._disconnect = quiet
    service._process_completed_function_calls = quiet
    service._create_initial_response = quiet
    for name, value in attributes.items():
        setattr(service, name, value)
    return service


def make_frame(cls, **fields):
    """A frame carrying only the fields this version of the framework declares.

    Frame signatures move between releases; a test that names a field the
    installed version dropped fails for the wrong reason.
    """
    import dataclasses

    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in fields.items() if k in names})


def spy_on_pushes(assistant) -> list:
    """Record the direction of every context frame the assistant pushes, while
    still letting the real push run so its bookkeeping happens."""
    from pipecat.processors.frame_processor import FrameDirection

    pushed: list = []
    real = assistant.push_context_frame

    async def record(direction=FrameDirection.DOWNSTREAM):
        pushed.append(direction)
        await real(direction)

    assistant.push_context_frame = record
    return pushed


async def narrate_a_tool_call(assistant, *, cancel_on_interruption=True):
    """The frames a narrated tool call produces, in the pipeline's own order.

    The agent narrates while the tool runs, so the result frame reaches the
    aggregator behind the narration's audio, with the agent still speaking.
    This is the shared prefix: what follows it -- a barge-in, or the agent
    simply finishing -- is what each test is about. One copy, because the
    sequence *is* the fixture, and two copies drift apart silently.
    """
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        FunctionCallFromLLM,
        FunctionCallInProgressFrame,
        FunctionCallResultFrame,
        FunctionCallsStartedFrame,
        LLMTextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection

    call = {"function_name": "lookup", "tool_call_id": "call-1", "arguments": {"id": 1}}
    # What the response-start frame records; that frame needs a running
    # pipeline's task manager, which a unit test does not have.
    assistant._assistant_turn_start_timestamp = "t0"
    for frame in [
        BotStartedSpeakingFrame(),
        make_frame(LLMTextFrame, text="Let me look that up for you."),
        make_frame(FunctionCallsStartedFrame,
                   function_calls=[make_frame(FunctionCallFromLLM, **call, context=None)]),
        make_frame(FunctionCallInProgressFrame, **call, cancel_on_interruption=cancel_on_interruption),
        make_frame(FunctionCallResultFrame, **call, result={"found": True},
                   run_llm=None, properties=None),
    ]:
        await assistant.process_frame(frame, FrameDirection.DOWNSTREAM)


def record_for(provider_key, **overrides):
    """The published record for a row, built the way ``run_bot`` builds it."""
    provider = bot.PROVIDERS[provider_key]
    settings = asked(**overrides)
    return bot.build_record(
        provider_key, provider, provider.default_model, provider.default_voice,
        bot.load_agent(settings), settings,
    )


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
        llm = StubLLM()
        trace = bot.ToolTrace()
        bot.register_tools(llm, server, trace, lambda _reason: None)
        assert set(llm.registered) == set(server.tool_names) | set(bot.CALL_CONTROL)

        call = a_call(arguments={"phone": "2025550188"})
        await llm.registered["lookup_patient"](call)
        assert call.answers and call.answers[0]["patient_id"] == "p_1002"

    async def test_an_unknown_record_is_reported_as_a_miss(self):
        server = bot.load_agent(asked())
        llm = StubLLM()
        trace = bot.ToolTrace()
        bot.register_tools(llm, server, trace, lambda _reason: None)

        call = a_call(arguments={"phone": "4045550000"})
        await llm.registered["lookup_patient"](call)
        # A number nothing in the table resembles is answered by the contract's
        # own "no patient found" row, not by an invented patient.
        answers = call.answers
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
        llm = StubLLM()
        trace = bot.ToolTrace()
        bot.register_tools(llm, server, trace, lambda _reason: None)
        await llm.registered["lookup_patient"](a_call(arguments={"phone": "2025550188"}, llm=None))
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

    The agent now collects its log from the call's first line, but the SDK's
    own capture -- the fallback if that handover ever fails -- opens only when
    the task is created. The configuration line stays after that point so it
    reaches the record either way: a log read months later against a single
    result has to say which provider, model, contract and commit produced it.
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
        llm = StubLLM()
        trace = bot.ToolTrace()
        rewrites = []
        trace.on_change = lambda: rewrites.append(len(trace.as_metadata()["tool_calls"]))
        bot.register_tools(llm, server, trace, lambda _reason: None)
        await llm.registered["lookup_patient"](a_call(arguments={"phone": "2025550188"}))
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

    def test_a_cascade_says_its_turns_came_from_its_transcriber(self):
        # Three answers, not two. A cascade follows the strategies its
        # speech-to-text service recommends, so the endpointing figure beside
        # its reply time belongs to that service rather than to the text model
        # the row is about -- and a row that left this blank would be read as
        # if it were comparable with a realtime service's own endpointing.
        record = bot.cascade_record(
            "cascade-openai", bot.TEXT_MODELS["cascade-openai"], "gpt-4.1",
            bot.MockToolServer(suite="appointments"), asked(),
        )
        assert record["turn_source"] == "stt"

    def test_every_row_says_whose_endpointing_it_reports(self):
        # The latency column is split into reply and endpointing, and the
        # second half cannot be read without knowing whose it is. A row missing
        # this field is one nobody can place.
        server = bot.MockToolServer(suite="appointments")
        rows = [
            bot.build_record(name, provider, "m", "v", server, asked())
            for name, provider in bot.PROVIDERS.items()
        ] + [
            bot.cascade_record(key, text, "m", server, asked())
            for key, text in bot.TEXT_MODELS.items()
        ]
        for record in rows:
            assert record["turn_source"] in {"provider", "local", "stt"}, record["config"]


class TestEveryLineIsOnTheCallClock:
    """A log line is placed by the time since the call started, not by the wall clock.

    A result raises questions like "what happened around the caller's second
    turn", and a wall-clock timestamp answers none of them. The stamp is the
    same shape the platform's own testing agent puts on its lines, so the two
    sides of a call read on one clock.
    """

    def test_the_stamp_is_minutes_and_seconds(self):
        bot.start_call_clock("clock-test")
        started = bot._CLOCK.get()
        assert bot.call_stamp(now=started) == "[00:00]"
        assert bot.call_stamp(now=started + 125.7) == "[02:05]"

    def test_every_logged_line_carries_it(self):
        from loguru import logger

        bot.start_call_clock("clock-test-2")
        seen = []
        sink = logger.add(lambda m: seen.append(m.record["message"]), level="DEBUG")
        try:
            logger.debug("hello from the test")
        finally:
            logger.remove(sink)
        assert any(line.startswith("[00:00] hello from the test") for line in seen), seen


class TestTheWholeCallLogIsKept:
    """The record ships the log from the call's first line, at DEBUG.

    The SDK's own capture opens when the task is created and keeps INFO and
    above, so the framework's account of each turn -- all at DEBUG -- never left
    the container, and nothing logged while the pipeline was being built did
    either. This collection replaces it and is handed to the SDK so the record
    carries it without a second exporter.
    """

    def _log_of(self, call_id: str) -> "bot.CallLog":
        bot.start_call_clock(call_id)
        return bot.CallLog(call_id)

    def test_debug_lines_are_kept(self):
        from loguru import logger

        log = self._log_of("keep-debug")
        try:
            logger.debug("framework detail")
            logger.info("headline")
        finally:
            log.close()
        levels = [line["level"] for line in log.lines]
        assert "DEBUG" in levels and "INFO" in levels
        assert all(line["message"].startswith("[00:0") for line in log.lines)

    def test_another_calls_lines_are_not_mixed_in(self):
        from loguru import logger

        first = self._log_of("call-a")
        try:
            logger.info("line for a")
            bot.start_call_clock("call-b")
            logger.info("line for b")
        finally:
            first.close()
        assert [line["message"].split("] ", 1)[1] for line in first.lines] == ["line for a"]

    def test_a_long_line_is_cut_and_says_so(self):
        from loguru import logger

        log = self._log_of("long-line")
        try:
            logger.info("x" * (bot.CallLog.MAX_CHARS + 50))
        finally:
            log.close()
        assert "more chars]" in log.lines[-1]["message"]
        assert len(log.lines[-1]["message"]) < bot.CallLog.MAX_CHARS + 40

    def test_the_collection_is_bounded_and_marks_where_it_stopped(self, monkeypatch):
        from loguru import logger

        monkeypatch.setattr(bot.CallLog, "MAX_LINES", 3)
        log = self._log_of("bounded")
        try:
            for i in range(6):
                logger.info("line {}", i)
        finally:
            log.close()
        assert len(log.lines) == 4  # three kept, one marker
        assert "log capture reached 3 lines" in log.lines[-1]["message"]
        assert log.dropped == 3

    def test_it_is_handed_to_the_sdk_in_place_of_its_own(self):
        from loguru import logger

        sdk_lines: list = []
        sdk_sink = logger.add(lambda m: sdk_lines.append(m.record["message"]), level="INFO")
        tracer = SimpleNamespace(_log_sink_id=sdk_sink, _session_logs=sdk_lines)

        log = self._log_of("handover")
        log.hand_to(tracer)
        try:
            logger.debug("after the handover")
        finally:
            logger.remove(tracer._log_sink_id)
        assert tracer._session_logs is log.lines
        assert any("after the handover" in line["message"] for line in log.lines)
        # The SDK's sink is gone: nothing more lands in its old list.
        assert sdk_lines == []
        # The id is kept, not surrendered: the sink lives on the process-wide
        # logger, and if the SDK never finalises, close() is the only thing that
        # will take it down. Removing it twice is harmless.
        assert log._sink_id is not None
        log.close()
        log.close()


class TestTheCallTellsItsOwnStory:
    """One INFO line per thing that happened, on the call clock.

    The framework's DEBUG lines are the right record for debugging the
    framework and the wrong one for reading a call. These are the lines a
    reader wants first: turn boundaries, what each side said, tools asked and
    answered, interruptions, errors, how the pipeline ended.
    """

    @staticmethod
    def _narrate(*frames):
        import asyncio

        from loguru import logger
        from pipecat.observers.base_observer import FramePushed
        from pipecat.processors.frame_processor import FrameDirection

        narrator = bot.CallNarrator()
        seen: list[str] = []
        sink = logger.add(lambda m: seen.append(m.record["message"]), level="DEBUG")

        async def run():
            for frame in frames:
                push = FramePushed(source=None, destination=None, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0)
                await narrator.on_push_frame(push)
                # Broadcast frames arrive at every processor; the story has one line each.
                await narrator.on_push_frame(push)

        try:
            asyncio.run(run())
        finally:
            logger.remove(sink)
        return narrator, [line.split("] ", 1)[1] for line in seen]

    def test_turns_speech_and_tools_are_narrated_once_each(self):
        from pipecat.frames.frames import (
            FunctionCallInProgressFrame,
            FunctionCallResultFrame,
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            TranscriptionFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        )

        narrator, lines = self._narrate(
            UserStartedSpeakingFrame(),
            TranscriptionFrame("book me in", "", "t"),
            UserStoppedSpeakingFrame(),
            LLMFullResponseStartFrame(),
            FunctionCallInProgressFrame(function_name="lookup_patient", tool_call_id="c1", arguments={"phone": "1"}, cancel_on_interruption=False),
            FunctionCallResultFrame(function_name="lookup_patient", tool_call_id="c1", arguments={"phone": "1"}, result={"ok": True}),
            LLMFullResponseEndFrame(),
        )
        assert lines == [
            "caller turn 1 starts",
            "caller transcript: \"book me in\"",
            "caller turn 1 ends",
            "agent response 1 starts",
            'tool lookup_patient requested',
            'tool lookup_patient answered',
            "agent response 1 ends",
        ]
        assert (narrator.caller_turns, narrator.agent_turns) == (1, 1)

    def test_a_broadcast_pair_is_one_event_not_two(self):
        # The framework constructs a separate frame for each direction and links
        # them. Counting by frame identity alone counted every turn twice, and
        # which frames are broadcast differs by provider -- so the inflation was
        # not even a constant factor across rows.
        from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

        def pair(cls):
            downstream, upstream = cls(), cls()
            downstream.broadcast_sibling_id = upstream.id
            upstream.broadcast_sibling_id = downstream.id
            return downstream, upstream

        narrator, lines = self._narrate(*pair(UserStartedSpeakingFrame), *pair(UserStoppedSpeakingFrame))
        assert lines == ["caller turn 1 starts", "caller turn 1 ends"]
        assert narrator.caller_turns == 1

    def test_only_an_interruption_over_live_audio_is_a_barge_in(self):
        # In realtime mode the aggregator broadcasts an interruption at the
        # start of every caller turn, whether or not the agent was saying
        # anything, so counting those would report a barge-in per turn on every
        # row. A barge-in is one that lands while the agent is speaking.
        from pipecat.frames.frames import (
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            InterruptionFrame,
        )

        narrator, lines = self._narrate(
            InterruptionFrame(),                     # ordinary turn start, agent silent
            BotStartedSpeakingFrame(),
            InterruptionFrame(),                     # the caller cuts in
            BotStoppedSpeakingFrame(),
            InterruptionFrame(),                     # silent again
        )
        assert narrator.barge_ins == 1
        assert [line for line in lines if "barge-in" in line] == [
            "barge-in 1: the caller spoke over the agent"
        ]

    def test_errors_and_endings_are_narrated(self):
        from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame

        narrator, lines = self._narrate(ErrorFrame("boom"), EndFrame(), CancelFrame())
        assert lines[0].startswith("error frame: ") and "boom" in lines[0]
        assert "pipeline ending" in lines[1] and "pipeline cancelled" in lines[2]
        assert narrator.errors == 1


class TestEventsATransportLacksAreSkippedQuietly:
    """Asking a transport for an event it never emits must not put a warning in every call."""

    def test_an_unknown_event_is_skipped_without_a_warning(self):
        from loguru import logger
        from pipecat.utils.base_object import BaseObject

        transport = BaseObject()
        seen = []
        sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
        try:
            bot._on_transport_event(transport, "on_participant_joined", lambda *_: None)
        finally:
            logger.remove(sink)
        assert seen == []


class TestToolRowsAfterTheLastTurnAreExported:
    """A tool call that ends the call has to reach the exported transcript.

    The exporter copies the context into the transcript when an assistant turn
    ends, and no turn ends after the call is over -- so ``end_call`` and
    ``transfer_call``, which are always the last thing the agent does, were
    written to the context and never exported. A scenario is scored on whether
    they were made. The SDK snapshots the transcript inside its finalisation,
    on every path that ends a call, and that snapshot is where the rows still in
    the context are swept across.
    """

    @staticmethod
    async def _run(finish: bool):
        import asyncio

        from cekura.pipecat._speech_timing import SpeechTimingObserver
        from cekura.pipecat.tracer import TranscriptCapture
        from pipecat.frames.frames import (
            EndTaskFrame,
            Frame,
            FunctionCallInProgressFrame,
            FunctionCallResultFrame,
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            StartFrame,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        class FakeRealtimeService(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, StartFrame):
                    self.create_task(self._script())
                await self.push_frame(frame, direction)

            async def _script(self):
                await asyncio.sleep(0.05)
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame("thanks for calling, goodbye"))
                await self.push_frame(LLMFullResponseEndFrame())
                await asyncio.sleep(0.2)
                # The agent hangs up: a tool call, its result, and the end of the task.
                await self.push_frame(FunctionCallInProgressFrame(
                    function_name="end_call", tool_call_id="c9", arguments={}, cancel_on_interruption=False,
                ))
                await asyncio.sleep(0.05)
                await self.push_frame(FunctionCallResultFrame(
                    function_name="end_call", tool_call_id="c9", arguments={}, result={"status": "ending_call"},
                ))
                await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

        context = LLMContext([{"role": "user", "content": "greet"}])
        pair = LLMContextAggregatorPair(
            context, realtime_service_mode=True, user_params=bot.user_aggregator_params(realtime=True),
        )
        capture = TranscriptCapture(context, SpeechTimingObserver())

        @pair.assistant().event_handler("on_assistant_turn_stopped")
        async def _stopped(aggregator, message):
            await capture.on_assistant_turn_stopped(message)

        published = []
        if finish:
            bot.finish_record(
                SimpleNamespace(_transcript_capture=capture), context,
                publish=lambda: published.append("record"), summarise=lambda: published.append("summary"),
            )

        task = PipelineTask(Pipeline([pair.user(), FakeRealtimeService(), pair.assistant()]), params=PipelineParams())
        await asyncio.wait_for(PipelineRunner(handle_sigint=False).run(task), timeout=30)
        # What the SDK does at finalisation.
        snapshot = capture.to_dict()
        rows = [(e.get("role"), "tool_calls" in e, e.get("tool_call_id")) for e in snapshot["transcript"]]
        return rows, published

    @pytest.mark.asyncio
    async def test_the_context_has_the_rows_and_the_export_did_not(self):
        rows, _ = await self._run(finish=False)
        assert ("tool", False, "c9") not in rows

    @pytest.mark.asyncio
    async def test_the_snapshot_sweeps_them_in_and_finishes_the_record(self):
        rows, published = await self._run(finish=True)
        assert ("assistant", True, None) in rows, "the model's request for end_call"
        assert ("tool", False, "c9") in rows, "and its answer"
        assert published == ["record", "summary"]


class TestARestatedTranscriptIsNotRepeatedSpeech:
    """One service restates the whole turn on every final; the rest send one.

    The aggregator appends each final, which is right for every service that
    sends one per turn and triples the caller's words for the one that does
    not. The transcript is what a judge reads and what a word-level comparison
    counts, so that row was being scored against a conversation nobody had.

    The correction is a statement about transcripts, not about a vendor: pass on
    only the part of a final that is new. For a service sending one final per
    turn it does nothing, which is the test of whether it is fair.
    """

    @staticmethod
    async def _through(*texts, restart_between=False):
        import asyncio

        from pipecat.frames.frames import Frame, StartFrame, TranscriptionFrame, UserStartedSpeakingFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.utils.time import time_now_iso8601

        kept: list[str] = []

        class Sink(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TranscriptionFrame):
                    kept.append(frame.text)

        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask

        filter_ = bot.RestatementIsNotSpeech()

        class Source(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, StartFrame):
                    self.create_task(self._script())
                await self.push_frame(frame, direction)

            async def _script(self):
                await asyncio.sleep(0.05)
                await self.push_frame(UserStartedSpeakingFrame())
                for text in texts:
                    if text is None:
                        await self.push_frame(UserStartedSpeakingFrame())
                        continue
                    await self.push_frame(TranscriptionFrame(text, "", time_now_iso8601()))
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.3)
                await self.queue_frame_and_wait_for_end()

            async def queue_frame_and_wait_for_end(self):
                pass

        task = PipelineTask(Pipeline([Source(), filter_, Sink()]), params=PipelineParams())
        runner = PipelineRunner(handle_sigint=False)
        running = asyncio.create_task(runner.run(task))
        await asyncio.sleep(1.0)
        await task.stop_when_done()
        await asyncio.wait_for(running, timeout=30)
        return kept, filter_

    @pytest.mark.asyncio
    async def test_a_cumulative_turn_is_reassembled_once(self):
        # The exact shape one service sent for a real caller turn.
        kept, filter_ = await self._through(
            "Hi, I'd like to book",
            "Hi, I'd like to book a new appointment.",
            "Hi, I'd like to book a new appointment.",
        )
        # What the aggregator would append is what the service last reported.
        assert " ".join(kept) == "Hi, I'd like to book a new appointment."
        assert filter_.restatements == 2

    @pytest.mark.asyncio
    async def test_one_final_per_turn_is_untouched(self):
        # Every other service on the board. A correction that changed these rows
        # would be a patch aimed at one vendor rather than a fix.
        kept, filter_ = await self._through("I need to book an appointment.")
        assert kept == ["I need to book an appointment."]
        assert filter_.restatements == 0

    @pytest.mark.asyncio
    async def test_a_new_turn_starts_again(self):
        # The same words in a later turn are new speech, not a restatement.
        kept, _ = await self._through("Yes, that's correct.", None, "Yes, that's correct.")
        assert kept == ["Yes, that's correct.", "Yes, that's correct."]

    @pytest.mark.asyncio
    async def test_genuinely_new_speech_survives(self):
        kept, _ = await self._through("No.", "Wait, actually yes.")
        assert " ".join(kept) == "No. Wait, actually yes."

    @pytest.mark.asyncio
    async def test_punctuation_does_not_make_a_restatement_look_new(self):
        # The same service rewrites "July 8th." as "July 8th if possible." on
        # the next pass. Compared byte for byte that is a fresh sentence, and
        # the whole thing is appended again.
        kept, filter_ = await self._through(
            "July 8th.", "July 8th if possible.", "July 8th if possible.",
        )
        assert " ".join(kept) == "July 8th. if possible."
        assert filter_.restatements == 2

    @pytest.mark.asyncio
    async def test_a_shorter_rewrite_is_not_appended(self):
        # It also revises downwards: "Perfect, got it." becomes "Perfect, got."
        # The longer version stands; appending the shorter one would say it twice.
        kept, filter_ = await self._through("Perfect, got it.", "Perfect, got.")
        assert " ".join(kept) == "Perfect, got it."
        assert filter_.restatements == 1


class TestTheSpeechHalfOfACostIsRecorded:
    """One provider bills its speech model by the second, and only logs it.

    Its token counts come from the *backend* text model it delegates to, which
    is the cheaper half. A row priced from those alone understates the call by
    most of it, and would sit on a board beside rows that are complete.
    """

    @pytest.mark.asyncio
    async def test_the_seconds_reach_the_record(self):
        reported = []

        class LiveService:
            async def _report_usage(self, usage):
                reported.append(usage)

        meter = bot.UsageMeter()
        service = LiveService()
        bot.capture_live_audio(service, meter)
        # Reported cumulatively during the call, so the last figure is the call's.
        await service._report_usage(SimpleNamespace(seconds=42.0))
        await service._report_usage(SimpleNamespace(seconds=104.0))
        assert meter.as_metadata()["usage"]["live_audio_seconds"] == 104.0
        assert len(reported) == 2, "the service's own reporting must still happen"

    @pytest.mark.asyncio
    async def test_a_service_that_bills_by_the_token_is_untouched(self):
        class TokenService:
            pass

        meter = bot.UsageMeter()
        bot.capture_live_audio(TokenService(), meter)
        assert "live_audio_seconds" not in meter.as_metadata()["usage"]


class TestSpeechTokensAreKeptApartFromText:
    """A row billed at one rate for speech and another for text needs both counts."""

    @staticmethod
    def _event(delta_in, delta_out, total_in, total_out):
        return {"usageEvent": {"details": {
            "delta": {"input": {"speechTokens": delta_in[0], "textTokens": delta_in[1]},
                      "output": {"speechTokens": delta_out[0], "textTokens": delta_out[1]}},
            "total": {"input": {"speechTokens": total_in[0], "textTokens": total_in[1]},
                      "output": {"speechTokens": total_out[0], "textTokens": total_out[1]}},
        }}}

    @pytest.mark.asyncio
    async def test_the_running_speech_totals_reach_the_record(self):
        handled = []

        class SpeechService:
            async def _handle_usage_event(self, event_json):
                handled.append(event_json)

        meter = bot.UsageMeter()
        service = SpeechService()
        bot.capture_speech_tokens(service, meter)
        await service._handle_usage_event(self._event((40, 900), (0, 0), (40, 900), (0, 0)))
        await service._handle_usage_event(self._event((25, 0), (120, 30), (65, 900), (120, 30)))
        usage = meter.as_metadata()["usage"]
        assert usage["input_audio_tokens"] == 65
        assert usage["output_audio_tokens"] == 120
        assert len(handled) == 2, "the service's own reporting must still happen"

    @pytest.mark.asyncio
    async def test_a_service_without_the_split_is_untouched(self):
        class TokenService:
            pass

        meter = bot.UsageMeter()
        bot.capture_speech_tokens(TokenService(), meter)
        assert "input_audio_tokens" not in meter.as_metadata()["usage"]

    def test_the_framework_still_collapses_the_split(self):
        # Tripwire: if the framework starts reporting speech apart, this wrapper
        # becomes a second writer of the same fields and should go.
        from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService

        source = inspect.getsource(AWSNovaSonicLLMService._handle_usage_event)
        assert "input_audio_tokens" not in source
        assert 'get("speechTokens", 0) + input_tokens.get("textTokens", 0)' in source

    def test_every_call_captures_both(self):
        source = inspect.getsource(bot.run_bot)
        assert "capture_live_audio(llm, meter)" in source
        assert "capture_speech_tokens(llm, meter)" in source


class TestARewrittenWordIsStillTheSameTurn:
    """The same service also corrects words as it goes, not only adds them.

    "Hi, I'd like to book an" becomes "Hi, I'd like to book a new appointment."
    Compared word for word those diverge at the article, so a strict test reads
    the second as a new sentence and appends the whole thing again. Two versions
    that agree on most of their words are the same words, rewritten.
    """

    @staticmethod
    def _run(*finals: str) -> str:
        """What the aggregator would end up with, without a pipeline."""
        import asyncio

        from pipecat.frames.frames import TranscriptionFrame, UserStartedSpeakingFrame
        from pipecat.processors.frame_processor import FrameDirection
        from pipecat.utils.time import time_now_iso8601

        filter_ = bot.RestatementIsNotSpeech()
        kept: list[str] = []

        async def push(frame, direction=FrameDirection.UPSTREAM):
            if isinstance(frame, TranscriptionFrame):
                kept.append(frame.text)

        filter_.push_frame = push

        async def drive():
            await filter_.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            for text in finals:
                await filter_.process_frame(
                    TranscriptionFrame(text, "", time_now_iso8601()), FrameDirection.UPSTREAM
                )

        asyncio.run(drive())
        return " ".join(kept)

    def test_a_corrected_word_does_not_repeat_the_sentence(self):
        said = self._run(
            "Hi, I'd like to book an",
            "Hi, I'd like to book a new appointment.",
            "Hi, I'd like to book a new appointment.",
        )
        # The stray article survives -- what was already reported cannot be
        # withdrawn -- but the sentence is said once, not three times.
        assert said.count("appointment") == 1, said
        assert said.lower().count("hi") == 1, said

    def test_two_different_utterances_are_both_kept(self):
        said = self._run("I need to cancel.", "Actually, let me reschedule instead.")
        assert said == "I need to cancel. Actually, let me reschedule instead."


class TestTheMiddleOutcomeSurvivesTheExport:
    """A record found after speech bent an argument is neither a hit nor a miss,
    and the payload has to say so: a bare matched flag hides it."""

    @staticmethod
    def _trace():
        import bot

        trace = bot.ToolTrace()
        trace.record("lookup_patient", {"phone": "1"}, False, {}, 0.0, "fuzzy")
        trace.record("book_appointment", {"id": "2"}, True, {}, 0.0, "exact")
        return trace.as_metadata()["tool_calls"]

    def test_a_nearest_record_is_reported_as_such(self):
        assert self._trace()[0]["resolution"] == "fuzzy"

    def test_it_is_not_flattened_into_the_matched_flag(self):
        fuzzy = self._trace()[0]
        assert fuzzy["matched"] is False and fuzzy["resolution"] != "none"

    def test_an_outright_match_still_says_exact(self):
        assert self._trace()[1]["resolution"] == "exact"


class TestHowLongTheAgentTookToAnswer:
    """Timed from our own detector hearing the caller stop, because that instant
    is the same on every row -- a service that decides its own turns is measured
    from where a service whose turns this pipeline decides is measured."""

    @staticmethod
    def _narrator():
        import bot

        return bot.CallNarrator()

    def test_a_greeting_is_not_a_reply(self):
        narrator = self._narrator()
        narrator._agent_speaking = False
        assert narrator.timing() == {}

    def test_the_interval_is_reported_with_its_spread(self):
        narrator = self._narrator()
        narrator.replies = [0.4, 0.9, 1.1, 3.0]
        reply = narrator.timing()["reply"]
        assert (reply["count"], reply["p50_ms"], reply["max_ms"]) == (4, 1100, 3000)

    def test_endpointing_is_reported_beside_it_not_inside_it(self):
        narrator = self._narrator()
        narrator.replies, narrator.endpointing = [1.0], [0.6]
        timing = narrator.timing()
        assert timing["reply"]["p50_ms"] == 1000 and timing["endpointing"]["p50_ms"] == 600

    def test_a_call_with_no_reply_exports_nothing_rather_than_zero(self):
        assert "reply" not in self._narrator().timing()

    @pytest.mark.asyncio
    async def test_every_turn_is_kept_with_when_in_the_call_it_happened(self):
        # The spread is a summary; the turns are the measurement, and a later
        # reading lines them up with the platform's own per-turn figures.
        import asyncio

        from pipecat.frames.frames import (
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            UserStoppedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
        )
        from pipecat.observers.base_observer import FramePushed
        from pipecat.processors.frame_processor import FrameDirection

        bot.start_call_clock("turns")
        narrator = self._narrator()

        async def push(frame):
            await narrator.on_push_frame(FramePushed(
                source=None, destination=None, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0))

        for _ in range(2):
            await push(VADUserStoppedSpeakingFrame())
            await asyncio.sleep(0.02)
            await push(UserStoppedSpeakingFrame())
            await asyncio.sleep(0.03)
            await push(BotStartedSpeakingFrame())
            await push(BotStoppedSpeakingFrame())

        timing = narrator.timing()
        replies, endpointing = timing["reply"]["turns"], timing["endpointing"]["turns"]
        assert len(replies) == len(endpointing) == timing["reply"]["count"] == 2
        assert all(turn["ms"] >= 45 for turn in replies)
        assert all(15 <= turn["ms"] < replies[0]["ms"] for turn in endpointing)
        assert replies[0]["caller_stopped_at_s"] < replies[1]["caller_stopped_at_s"]
        assert replies[0]["caller_stopped_at_s"] == endpointing[0]["caller_stopped_at_s"]


class TestEveryRowIsCheckedWhetherOrNotItLooksWrong:
    """The checks that run on every call, in the order that matters: before
    anyone has decided the row is worth examining.

    A fault is otherwise only ever found in a row someone thought to look at,
    and the row nobody looks at is the one that looks fine. These say whether
    the call was delivered and answered as a call -- not whether the model was
    any good at it."""

    @staticmethod
    def _narrator():
        import bot

        return bot.CallNarrator()

    def test_a_steady_call_reports_that_the_checks_ran(self):
        narrator = self._narrator()
        narrator.replies = [1.0, 1.2, 0.9, 1.1]
        narrator.caller_turns, narrator.agent_turns = 4, 4
        assert narrator.integrity()["checks"] == ["ok"]

    def test_replies_that_grow_through_a_call_are_named(self):
        # A slow model is slow evenly. A session falling behind the audio it is
        # sent gets slower as the call goes on, and only the second half shows it.
        narrator = self._narrator()
        narrator.replies = [1.0, 1.2, 12.0, 20.0]
        narrator.caller_turns, narrator.agent_turns = 4, 4
        report = narrator.integrity()
        assert "replies_drifting" in report["checks"]
        assert report["reply_drift_ms"] > 2000

    def test_a_uniformly_slow_model_is_not_a_broken_row(self):
        narrator = self._narrator()
        narrator.replies = [8.0, 8.2, 8.1, 8.3]
        narrator.caller_turns, narrator.agent_turns = 4, 4
        assert narrator.integrity()["checks"] == ["ok"]

    def test_caller_turns_that_drew_no_reply_are_named(self):
        narrator = self._narrator()
        narrator.caller_turns, narrator.agent_turns = 10, 2
        assert "turns_unanswered" in narrator.integrity()["checks"]
        assert narrator.integrity()["answered"] == "2/10"

    def test_a_call_where_neither_side_spoke_is_named(self):
        assert "silent_call" in self._narrator().integrity()["checks"]

    def test_audio_arriving_slower_than_the_clock_is_named(self):
        clock = bot.AudioClock()
        clock.add(-4.0)  # four seconds of audio short of the time that has passed
        narrator = bot.CallNarrator(clock)
        narrator.caller_turns, narrator.agent_turns = 4, 4
        report = narrator.integrity()
        assert "audio_in_starved" in report["checks"]
        assert report["audio_in_drift_ms"] == -4000

    def test_one_call_does_not_read_the_clock_of_the_one_before_it(self):
        """The accumulator is a process-level object; the narrator is per call.

        A narrator that reached for the global rather than being handed one
        would report the previous call's audio on this call's row."""
        busy = bot.AudioClock()
        busy.add(600.0)
        fresh = bot.CallNarrator()
        fresh.caller_turns, fresh.agent_turns = 4, 4
        assert "audio_in_drift_ms" not in fresh.integrity()

    @staticmethod
    def _clock(ticks: list[float]) -> "bot.AudioClock":
        stamps = iter(ticks)
        return bot.AudioClock(now=lambda: next(stamps))

    def test_the_tail_after_the_caller_stops_is_not_missing_caller_audio(self):
        """A call ends with the agent talking, tools finishing and the transport
        closing. Caller audio has rightly stopped by then, so a reading taken at
        the end would report the whole tail as audio that never arrived."""
        clock = self._clock([0.0, 0.02, 0.04])
        for _ in range(3):
            clock.add(0.02)
        narrator = bot.CallNarrator(clock)
        narrator.caller_turns, narrator.agent_turns = 4, 4
        # No clock is read after the last buffer, so no tail can enter the figure.
        assert narrator.integrity()["checks"] == ["ok"]

    def test_audio_missing_while_the_caller_is_still_speaking_is_still_named(self):
        """The counterpart: a gap that opens between two caller buffers is the one
        worth reporting, and taking the reading early must not hide it."""
        clock = self._clock([0.0, 1.0, 8.0])
        for _ in range(3):
            clock.add(1.0)  # the third buffer is five seconds late
        narrator = bot.CallNarrator(clock)
        narrator.caller_turns, narrator.agent_turns = 4, 4
        report = narrator.integrity()
        assert "audio_in_starved" in report["checks"]
        assert report["audio_in_drift_ms"] == -5000

    def test_a_live_call_delivers_one_second_of_audio_per_second(self):
        import bot

        clock = bot.AudioClock()
        assert clock.drift() is None
        clock.add(1.0)
        # Wall time has barely moved, so a second of audio is a second ahead.
        assert 0.9 < clock.drift() <= 1.0


class TestAToolResultReachesTheModelThatAskedForIt:
    """A realtime service learns what a tool returned from the context frame the
    assistant aggregator pushes upstream, and from nothing else. A caller who
    talks over the agent while that push is waiting must not cost the model the
    answer -- otherwise it asks for the same tool again, and the call is scored
    with a duplicate."""

    @staticmethod
    def _assistant(realtime: bool = True):
        from pipecat.processors.aggregators.llm_context import LLMContext

        return bot.BenchAggregators(LLMContext([]), realtime_service_mode=realtime).assistant()

    @staticmethod
    async def _narrated_tool_call_then_barge_in(assistant) -> None:
        """The caller talks over the narration: the user half announces the
        turn, broadcasts an interruption, and the transport reports the agent
        has stopped. The caller's turn ends a moment later."""
        from pipecat.frames.frames import (
            BotStoppedSpeakingFrame,
            InterruptionFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection

        await narrate_a_tool_call(assistant, cancel_on_interruption=False)
        for frame in [
            UserStartedSpeakingFrame(),
            InterruptionFrame(),
            BotStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
        ]:
            await assistant.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_a_result_interrupted_by_the_caller_is_delivered_once_the_turn_ends(self):
        from pipecat.processors.frame_processor import FrameDirection

        assistant = self._assistant()
        pushed = spy_on_pushes(assistant)
        await self._narrated_tool_call_then_barge_in(assistant)
        # One downstream push records the cut-off narration; exactly one upstream
        # push tells the model what the tool returned.
        assert pushed.count(FrameDirection.UPSTREAM) == 1
        assert pushed[-1] is FrameDirection.UPSTREAM
        assert assistant._push_context_on_bot_stopped_speaking is False
        results = [m for m in assistant.context.get_messages() if m.get("role") == "tool"]
        assert [m.get("tool_call_id") for m in results] == ["call-1"]

    async def test_a_cascade_row_leaves_the_delivery_to_the_half_that_owns_it(self):
        """There the user half pushes the context at turn end regardless, so a
        second push here would answer the same turn twice."""
        from pipecat.processors.frame_processor import FrameDirection

        assistant = self._assistant(realtime=False)
        pushed = spy_on_pushes(assistant)
        await self._narrated_tool_call_then_barge_in(assistant)
        assert FrameDirection.UPSTREAM not in pushed
        assert assistant._push_context_on_bot_stopped_speaking is False

    async def test_the_framework_still_needs_this(self):
        """The one test that should fail on a framework upgrade.

        This works around a defect: when the caller interrupts, the stock
        aggregator records the cut-off utterance with a downstream push that
        clears the pending delivery, then resets, and never retries. If that
        stops being true the workaround is not merely unnecessary, it pushes a
        second time and the row records an answer the caller never prompted --
        so this asserts the defect is still there, and fails loudly when it is
        not."""
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
        )
        from pipecat.processors.frame_processor import FrameDirection

        stock = LLMContextAggregatorPair(LLMContext([]), realtime_service_mode=True).assistant()
        pushed = spy_on_pushes(stock)
        await self._narrated_tool_call_then_barge_in(stock)
        assert FrameDirection.UPSTREAM not in pushed, (
            "the framework now delivers a tool result interrupted by a barge-in; "
            "delete DeliversToolResults and BenchAggregators"
        )


class TestAResumedGeminiSessionIsNotToldOldResultsAgain:
    """The service resumes a dropped connection from a handle the server gave
    it, so the server still holds every tool result already delivered. The
    reconnect must not make the next context frame deliver them all again."""

    @staticmethod
    def _service(service_class):
        svc = service_class(api_key="unused")
        svc._completed_tool_calls = {"call-1"}
        svc._tool_call_id_to_name = {"call-1": "lookup"}
        connected: list = []

        async def disconnect():  # what the framework's own disconnect does to these
            svc._completed_tool_calls = set()
            svc._tool_call_id_to_name = {}

        async def connect(session_resumption_handle=None):
            connected.append(session_resumption_handle)

        svc._disconnect, svc._connect = disconnect, connect
        return svc, connected

    async def test_a_resume_keeps_what_was_already_delivered(self):
        svc, connected = self._service(bot.gemini_service_class())
        svc._session_resumption_handle = "handle"
        await svc._reconnect()
        assert connected == ["handle"]
        assert svc._completed_tool_calls == {"call-1"}
        assert svc._tool_call_id_to_name == {"call-1": "lookup"}

    async def test_a_fresh_reconnect_starts_clean_as_before(self):
        """Without a handle the framework re-seeds the history itself, and marks
        those results delivered as it goes; nothing to carry over."""
        svc, connected = self._service(bot.gemini_service_class())
        await svc._reconnect()
        assert connected == [None]
        assert svc._completed_tool_calls == set()

    async def test_the_framework_still_needs_this(self):
        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

        svc, _ = self._service(GeminiLiveLLMService)
        svc._session_resumption_handle = "handle"
        await svc._reconnect()
        assert svc._completed_tool_calls == set(), (
            "the framework now keeps delivered tool results across a resumed session; "
            "delete gemini_service_class"
        )


class TestARowSaysWhereItDiffersFromTheOthers:
    """One pipeline answers every row, but the session opened at the top of it is
    not identical, because these services do not offer the same contract. What
    differs is the part of a result that is ours rather than the model's, so it
    travels with the score."""

    def test_every_native_row_declares_the_same_set(self):
        expected = {"endpointing", "service_vad", "interruptions",
                    "caller_transcription", "input_rate", "result_delivery"}
        for key in bot.PROVIDERS:
            assert set(record_for(key)["divergences"]) == expected, key

    def test_a_cascade_row_declares_it_too(self):
        server = bot.load_agent(asked())
        for key, text in bot.TEXT_MODELS.items():
            record = bot.cascade_record(key, text, "m", server, asked())
            assert record["divergences"]["endpointing"] == "stt", key

    def test_the_declared_detector_is_the_one_the_service_is_built_with(self):
        # The declaration and the builder are two places, so they can disagree.
        # This is the check that stops them.
        provider = bot.PROVIDERS["gemini-live"]
        assert provider.service_vad is False
        service = provider.build("k", provider.default_model, provider.default_voice, "p", {})
        assert service._vad_disabled is True

    def test_the_declared_interruption_owner_is_the_one_in_force(self):
        for key, provider in bot.PROVIDERS.items():
            if provider.turns != "provider":
                continue
            strategies = bot.user_aggregator_params(
                realtime=True, turns=provider.turns, interruptions=provider.interruptions
            ).user_turn_strategies
            declared = record_for(key)["divergences"]["interruptions"]
            assert strategies.enable_interruptions is (declared == "pipeline"), key


class TestVendorDefaultsAreExplicitAndOnTheRecord:
    """A setting the vendor applies when nothing is sent is still a setting.

    Three rows were running on values nobody had written down: a reasoning
    effort, a detector threshold, a pause before answering. Each is now sent
    explicitly and disclosed, so a record says what was measured without a
    reader having to know what the vendor's default was that month.
    """

    def test_no_default_model_is_an_alias(self):
        for key, provider in bot.PROVIDERS.items():
            assert "latest" not in provider.default_model, key

    def test_grok_sends_its_reasoning_effort_and_detector_settings(self):
        provider = bot.PROVIDERS["grok-realtime"]
        service = provider.build("k", provider.default_model, provider.default_voice, "p", asked())
        properties = service._settings.session_properties
        assert properties.reasoning.effort == bot.GROK_REASONING
        assert properties.turn_detection.type == "server_vad"
        for name, value in bot.GROK_VAD.items():
            assert getattr(properties.turn_detection, name) == value, name
        record = record_for("grok-realtime")
        assert record["grok_reasoning"] == bot.GROK_REASONING
        assert "threshold 0.85" in record["grok_vad"]

    def test_openai_sends_its_best_reasoning_effort_and_says_so(self):
        provider = bot.PROVIDERS["openai-realtime"]
        service = provider.build("k", provider.default_model, provider.default_voice, "p", asked())
        assert service._settings.session_properties.reasoning.effort == bot.OPENAI_REASONING == "high"
        assert record_for("openai-realtime")["openai_reasoning"] == "high"

    def test_gemini_runs_its_thinking_model_at_a_named_level(self):
        provider = bot.PROVIDERS["gemini-live"]
        assert provider.default_model.endswith("-extended-thinking")
        service = provider.build("k", provider.default_model, provider.default_voice, "p", asked())
        assert service._settings.thinking.thinking_level.value == bot.GEMINI_THINKING
        # The framework reads the model id to decide the turn waits for the
        # background reasoning to finish, and to not replace the level.
        assert service._expects_interaction_status
        assert service._resolved_thinking_config().thinking_level.value == "HIGH"
        assert record_for("gemini-live")["gemini_thinking_level"] == "HIGH"

    def test_the_live_backend_reasons_at_a_named_effort(self):
        provider = bot.PROVIDERS["gpt-live"]
        service = provider.build("k", provider.default_model, provider.default_voice, "p", asked())
        backend = service._delegation.settings
        assert (backend.model, backend.reasoning.effort) == ("gpt-6-sol", "low")
        record = record_for("gpt-live")
        assert (record["s2s_backend_model"], record["s2s_backend_reasoning"]) == ("gpt-6-sol", "low")

    def test_qwen_runs_its_audio_model(self):
        provider = bot.PROVIDERS["qwen-realtime"]
        assert (provider.default_model, provider.default_voice) == ("qwen-audio-3.0-realtime-plus", "longanqian")
        assert provider.caller_transcription == "automatic"

    def test_nova_sends_its_endpointing_sensitivity(self, monkeypatch):
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secretpart")
        provider = bot.PROVIDERS["nova-sonic"]
        service = provider.build("AKIAEXAMPLE", provider.default_model, provider.default_voice, "p", asked())
        assert service._settings.endpointing_sensitivity == bot.NOVA_ENDPOINTING
        record = record_for("nova-sonic")
        assert record["nova_endpointing"] == bot.NOVA_ENDPOINTING

    def test_the_live_model_is_told_when_to_delegate_and_the_backend_gets_the_prompt(self):
        provider = bot.PROVIDERS["gpt-live"]
        service = provider.build("k", provider.default_model, provider.default_voice, "AGENT PROMPT", asked())
        live = service._settings.system_instruction
        assert live.startswith("AGENT PROMPT\n\n")
        assert live.endswith(bot.DELEGATION_PROMPT)
        assert "ended" in bot.DELEGATION_PROMPT and "transfer" in bot.DELEGATION_PROMPT
        backend = service._delegation.settings
        assert backend.system_instruction == "AGENT PROMPT"
        assert backend.model == bot.backend_model(asked())
        record = record_for("gpt-live")
        assert record["prompt_addendum"] == "gpt-live-delegation"
        # The name alone would let the section be rewritten with every digest
        # on the record unchanged, so the section is hashed too -- and it is
        # not the prompt digest, which covers the prompt this row shares.
        assert record["prompt_addendum_sha256"] != record["system_prompt_sha256"]
        assert record["prompt_addendum_sha256"] == bot._digest(bot.DELEGATION_PROMPT)

    def test_every_other_row_gets_the_prompt_unchanged(self, monkeypatch):
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secretpart")
        for key in ("openai-realtime", "gemini-live", "grok-realtime", "nova-sonic"):
            provider = bot.PROVIDERS[key]
            service = provider.build("AKIAEXAMPLE", provider.default_model, provider.default_voice, "AGENT PROMPT", asked())
            assert service._settings.system_instruction == "AGENT PROMPT", key

    def test_the_record_separates_who_ended_the_turn_from_who_decided_to_answer(self):
        # Nova runs its own detector over the whole call and answers on it; the
        # pipeline's local turn only feeds the context. Gemini, with its detector
        # switched off, answers on the pipeline's word. Same turn_source, different
        # endpointing -- and the difference is a second and a half of reply time.
        server = bot.load_agent(asked())
        nova = bot.build_record("nova-sonic", bot.PROVIDERS["nova-sonic"], "m", "v", server, asked())
        gemini = bot.build_record("gemini-live", bot.PROVIDERS["gemini-live"], "m", "v", server, asked())
        assert nova["turn_source"] == "local" and nova["divergences"]["endpointing"] == "provider"
        assert gemini["turn_source"] == "local" and gemini["divergences"]["endpointing"] == "local"

    def test_the_row_that_takes_its_result_off_the_frame_says_so(self):
        # GPT-Live never reads a tool result out of the context: the service
        # takes it off the frame and answers at once, so neither the wait for
        # the agent to stop speaking nor the immediate push describes it.
        assert bot.PROVIDERS["gpt-live"].results == "service"
        assert record_for("gpt-live")["divergences"]["result_delivery"] == "service"


class TestAConfigurationReconnectOpensANewGeminiSession:
    """A resumed session keeps the setup it was opened with, tools included.

    The framework connects before the context carrying the tools arrives and
    reconnects to apply them. If the server has already issued a resumption
    handle by then, that reconnect resumes the tool-less session and the model
    spends the call unable to call anything. The handle is dropped for that one
    reconnect and kept for every later one.

    These drive the real ``_reconnect`` and read the handle the connection was
    actually opened with, because that is the thing the session is made of --
    a test that stubs the reconnect can only assert that we called ourselves.
    """

    @staticmethod
    def _service(service_class):
        service = gemini_service(service_class)
        service._session_resumption_handle = "handle-from-the-first-connection"
        resumed = []

        async def connect(session_resumption_handle=None):
            resumed.append(session_resumption_handle)

        service._connect = connect
        return service, resumed

    @staticmethod
    def _context_carrying_tools():
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema
        from pipecat.processors.aggregators.llm_context import LLMContext

        return LLMContext(
            [{"role": "assistant", "content": "hello"}],
            tools=ToolsSchema(standard_tools=[FunctionSchema("t", "d", {}, [])]),
        )

    async def test_the_first_context_opens_a_new_session(self):
        service, resumed = self._service(bot.gemini_service_class())
        await service._handle_context(self._context_carrying_tools())
        assert resumed == [None]

    async def test_the_framework_still_resumes_there(self):
        # Tripwire: the day the framework opens a new session for a
        # configuration change, this override is redundant and should go.
        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
        service, resumed = self._service(GeminiLiveLLMService)
        await service._handle_context(self._context_carrying_tools())
        assert resumed == ["handle-from-the-first-connection"]

    async def test_a_later_reconnect_still_resumes(self):
        # The mid-call error path: the session is under way, and resuming is
        # the whole point -- a new session there would lose the conversation.
        from pipecat.processors.aggregators.llm_context import LLMContext

        service, resumed = self._service(bot.gemini_service_class())
        service._context = LLMContext([])
        await service._reconnect()
        assert resumed == ["handle-from-the-first-connection"]

    async def test_a_handle_is_not_discarded_for_a_reconnect_that_never_happens(self):
        # A context with nothing to apply does not reconnect, so the handle it
        # arrived with has to survive for the next mid-call error to use.
        from pipecat.processors.aggregators.llm_context import LLMContext

        service, resumed = self._service(bot.gemini_service_class())
        await service._handle_context(LLMContext([{"role": "assistant", "content": "hi"}]))
        assert resumed == []
        assert service._session_resumption_handle == "handle-from-the-first-connection"


class TestAResultIsDeliveredWhileTheAgentIsStillSpeaking:
    """The wait for the agent to stop is the window in which a barge-in withdraws the call.

    A service whose context handling only forwards the result can have it at
    once. One that opens a new response on it keeps the framework's timing, so
    its narration is not cut short.
    """

    @staticmethod
    def _assistant(deliver_immediately: bool):
        from pipecat.processors.aggregators.llm_context import LLMContext

        return bot.BenchAggregators(
            LLMContext([]), realtime_service_mode=True, deliver_immediately=deliver_immediately
        ).assistant()

    @staticmethod
    async def _narrated_tool_call(assistant):
        """Upstream pushes before the agent stops speaking, and after."""
        from pipecat.frames.frames import BotStoppedSpeakingFrame
        from pipecat.processors.frame_processor import FrameDirection

        pushes = spy_on_pushes(assistant)

        def upstream():
            return [d for d in pushes if d is FrameDirection.UPSTREAM]

        await narrate_a_tool_call(assistant)
        before = upstream()
        await assistant.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        return before, upstream()

    async def test_an_immediate_row_delivers_before_the_agent_stops(self):
        before, after = await self._narrated_tool_call(self._assistant(True))
        assert len(before) == 1 and len(after) == 1

    async def test_a_row_left_to_the_framework_delivers_after(self):
        before, after = await self._narrated_tool_call(self._assistant(False))
        assert len(before) == 0 and len(after) == 1

    def test_the_rows_that_only_forward_a_result_are_the_immediate_ones(self):
        assert {k for k, p in bot.PROVIDERS.items() if p.results == "immediate"} == {"gemini-live", "nova-sonic"}
        for key in ("openai-realtime", "grok-realtime"):
            assert bot.PROVIDERS[key].results == "after_speech", key


class TestAWithdrawnGeminiToolCallIsClosedAndMarked:
    """The service withdraws a call it is no longer waiting for; the framework has no branch for it."""

    @staticmethod
    def _message(ids):
        return SimpleNamespace(
            server_content=None, tool_call=None, session_resumption_update=None,
            usage_metadata=None, setup_complete=None, go_away=None,
            tool_call_cancellation=SimpleNamespace(ids=ids),
        )

    @staticmethod
    def _service(service_class, still_running=()):
        from pipecat.processors.aggregators.llm_context import LLMContext

        service = gemini_service(service_class)
        service._context = LLMContext([])
        service._tool_call_id_to_name = {"fc-1": "lookup"}
        seen = {"frames": [], "events": []}

        async def broadcast(frame_cls, **kwargs):
            seen["frames"].append((frame_cls.__name__, kwargs))

        async def cancel_tasks(predicate, **_kwargs):
            running = [SimpleNamespace(tool_call_id=tid, function_name="lookup")
                       for tid in still_running]
            return [item for item in running if predicate(item)]

        async def event(name, *args):
            seen["events"].append((name, args))

        service.broadcast_frame = broadcast
        service._cancel_function_call_tasks = cancel_tasks
        service._call_event_handler = event
        return service, seen

    async def test_the_withdrawn_call_gets_no_result_and_the_pipeline_is_told(self):
        service, seen = self._service(bot.gemini_service_class())
        await service._handle_server_message(self._message(["fc-1"]))
        assert "fc-1" in service._completed_tool_calls
        assert seen["frames"] == [
            ("FunctionCallCancelFrame",
             {"function_name": "lookup", "tool_call_id": "fc-1", "run_llm": False}),
        ]
        assert [name for name, _ in seen["events"]] == ["on_function_calls_cancelled"]
        assert seen["events"][0][1][0][0].tool_call_id == "fc-1"

    async def test_a_call_withdrawn_while_it_was_still_running_is_left_to_the_framework(self):
        # The framework's own helper settles a running call. Broadcasting a
        # second cancellation for it would settle it twice.
        service, seen = self._service(bot.gemini_service_class(), still_running=["fc-1"])
        await service._handle_server_message(self._message(["fc-1"]))
        assert "fc-1" in service._completed_tool_calls
        assert seen["frames"] == [] and seen["events"] == []

    async def test_the_framework_still_ignores_the_message(self):
        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
        service, seen = self._service(GeminiLiveLLMService)
        await service._handle_server_message(self._message(["fc-1"]))
        assert "fc-1" not in service._completed_tool_calls and seen["frames"] == []

    def test_the_trace_marks_a_withdrawn_call_and_keeps_it(self):
        trace = bot.ToolTrace()
        rewrites = []
        trace.record("lookup", {"id": 1}, True, {"found": True}, 0.0, "exact", tool_call_id="fc-1")
        trace.record("lookup", {"id": 1}, True, {"found": True}, 5.0, "exact", tool_call_id="fc-2")
        trace.on_change = lambda: rewrites.append(1)
        trace.cancel(["fc-1", "fc-2"])
        meta = trace.as_metadata()
        assert meta["tool_call_count"] == 2 and meta["tool_calls_cancelled"] == 2
        assert rewrites == [1], "a withdrawn batch rewrites the record once, not once per call"

    def test_a_call_withdrawn_before_it_returned_is_not_on_the_record(self):
        # Nothing to mark, and nothing to count: the handler was cancelled
        # mid-run, so the call never reached the trace.
        trace = bot.ToolTrace()
        rewrites = []
        trace.on_change = lambda: rewrites.append(1)
        trace.cancel(["fc-9"])
        assert trace.as_metadata()["tool_calls_cancelled"] == 0 and rewrites == []


class TestAHangUpEndsTheCallOnceTheGoodbyeHasPlayed:
    """A hang-up has to reach the caller as a call that ended -- after the
    goodbye has been heard, and not long after.

    Each runs the processor in a real pipeline whose only way to end is the
    hang-up, so a cancel that is pushed and never acted on fails by not
    finishing. The waits are shortened on the instance, which is where the
    processor reads them.
    """

    QUIET = 0.05
    MOST = 0.6

    async def _call(self, script, leave=None, order=None):
        """Run the script against a live pipeline; return what the hang-up did and when."""
        import asyncio
        import time

        from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        hangup = bot.HangsUpOnceHeard(leave=leave)
        hangup.QUIET_SECS, hangup.MAX_WAIT_SECS = self.QUIET, self.MOST
        marks: dict[str, float] = {}
        ended: list[str] = []

        class Source(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, StartFrame):
                    self.create_task(self._run())
                await self.push_frame(frame, direction)

            async def _run(self):
                await asyncio.sleep(0.02)
                await script(self, hangup, lambda name: marks.setdefault(name, time.monotonic()))

        class Sink(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, (CancelFrame, EndFrame)):
                    ended.append(type(frame).__name__)
                    if order is not None:
                        order.append(type(frame).__name__)
                await self.push_frame(frame, direction)

        task = PipelineTask(Pipeline([Source(), hangup, Sink()]), params=PipelineParams())
        # Timed rather than trusted: when the wait below runs out, the runner
        # swallows the cancellation and returns as if the call had ended.
        await asyncio.wait_for(PipelineRunner(handle_sigint=False).run(task), timeout=5)
        finished = time.monotonic()
        assert hangup.hung_up_at is not None, "the call ended without the agent hanging up"
        assert finished - hangup.hung_up_at < 0.5, "the hang-up was decided and the call stayed up"
        assert ended == ["CancelFrame"], "the call must end by cancelling, not by draining"
        marks["hung_up"] = hangup.hung_up_at
        marks["left"] = hangup.left_at
        return marks

    async def test_a_goodbye_still_playing_is_heard_out(self):
        import asyncio

        from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame

        async def script(source, hangup, mark):
            await source.push_frame(BotStartedSpeakingFrame())
            hangup.hang_up("end_call")
            await asyncio.sleep(0.2)
            mark("stopped")
            await source.push_frame(BotStoppedSpeakingFrame())

        marks = await self._call(script)
        waited = marks["hung_up"] - marks["stopped"]
        assert self.QUIET <= waited < self.QUIET + 0.1, "hung up mid-goodbye, or sat on the line after it"

    async def test_a_hang_up_after_the_goodbye_ends_the_call_at_once(self):
        async def script(source, hangup, mark):
            mark("asked")
            hangup.hang_up("end_call")

        marks = await self._call(script)
        assert marks["hung_up"] - marks["asked"] < self.QUIET + 0.1

    async def test_a_reply_that_starts_after_the_hang_up_is_heard_out(self):
        # A model often answers its own tool's result: "done, goodbye".
        import asyncio

        from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame

        async def script(source, hangup, mark):
            hangup.hang_up("end_call")
            await asyncio.sleep(self.QUIET / 3)
            await source.push_frame(BotStartedSpeakingFrame())
            await asyncio.sleep(0.2)
            mark("stopped")
            await source.push_frame(BotStoppedSpeakingFrame())

        marks = await self._call(script)
        assert marks["hung_up"] >= marks["stopped"] + self.QUIET

    async def test_a_goodbye_that_never_ends_does_not_hold_the_line(self):
        from pipecat.frames.frames import BotStartedSpeakingFrame

        async def script(source, hangup, mark):
            await source.push_frame(BotStartedSpeakingFrame())
            mark("asked")
            hangup.hang_up("end_call")

        marks = await self._call(script)
        assert self.MOST <= marks["hung_up"] - marks["asked"] < self.MOST + 0.1

    async def test_the_room_is_left_before_the_teardown_starts(self):
        # The cancel reaches the transport only after the realtime service has
        # closed its own connection, seconds on some rows; the caller should not
        # wait for that.
        order: list[str] = []

        async def leave():
            order.append("left")

        async def script(source, hangup, mark):
            hangup.hang_up("end_call")

        marks = await self._call(script, leave=leave, order=order)
        assert order == ["left", "CancelFrame"]
        assert marks["left"] is not None and marks["left"] - marks["hung_up"] < 0.1

    async def test_a_room_that_cannot_be_left_early_still_ends_the_call(self):
        async def leave():
            raise RuntimeError("the transport refused")

        async def script(source, hangup, mark):
            hangup.hang_up("end_call")

        marks = await self._call(script, leave=leave)
        assert marks["left"] is None, "a failed leave must not be reported as the caller let go"

    def test_only_a_transport_whose_room_is_shared_is_left_early(self):
        assert bot.leaves_the_room(SimpleNamespace(_client=SimpleNamespace(leave=None))) is None

    def test_the_framework_leaves_a_shared_room_once_and_ignores_a_surplus_release(self):
        """What leaving early depends on, asserted against the framework itself.

        Both transport halves hold the room through one shared client, the room
        is left when the second hold is released, and a release with nothing
        held does nothing -- so the halves' own releases after an early leave
        are harmless. If either stops being true, leaving early either leaves
        nothing or leaves twice, and this should fail before a call does."""
        import asyncio

        from pipecat.transports.daily import transport as daily
        from pipecat.utils.shared import acquires, releases

        source = inspect.getsource(daily.DailyTransportClient)
        assert '@acquires("room")\n    async def join' in source
        assert '@releases("room")\n    async def leave' in source

        class Room:
            def __init__(self):
                self.left = 0

            @acquires("room")
            async def join(self):
                pass

            @releases("room")
            async def leave(self):
                self.left += 1

        async def early_leave_then_both_halves():
            room = Room()
            await room.join()
            await room.join()
            await room.leave()
            await room.leave()  # left here
            await room.leave()  # the input half's own release
            await room.leave()  # the output half's
            return room.left

        assert asyncio.run(early_leave_then_both_halves()) == 1

    def test_the_waits_are_the_ones_a_caller_can_live_with(self):
        # Long enough to ride out a pause inside a sentence; short enough that a
        # caller does not hear a line that has gone quiet but not closed.
        assert 0.5 <= bot.HangsUpOnceHeard.QUIET_SECS <= 1.5
        assert bot.HangsUpOnceHeard.MAX_WAIT_SECS <= 10

    async def test_both_closing_tools_hang_up_and_are_still_on_the_record(self):
        server = bot.load_agent(asked())
        llm = StubLLM()
        trace = bot.ToolTrace()
        closed: list[str] = []
        bot.register_tools(llm, server, trace, closed.append)
        await llm.registered["end_call"](a_call(function_name="end_call"))
        await llm.registered["transfer_call"](a_call(function_name="transfer_call"))
        assert closed == ["end_call", "transfer_call"]
        assert [c["name"] for c in trace.as_metadata()["tool_calls"]] == ["end_call", "transfer_call"]

    def test_it_sits_where_the_goodbye_is_played(self):
        # Upstream of the output transport the processor would see the agent's
        # audio before the caller does, and hang up on a goodbye still queued.
        import re

        source = (AGENT_DIR / "bot.py").read_text()
        assert len(re.findall(r"transport\.output\(\),\s*hangup,", source)) == 2, "both rows' pipelines"

    def test_a_hang_up_that_did_not_end_the_call_is_named(self):
        import time

        held = bot.CallNarrator(hangup=SimpleNamespace(hung_up_at=time.monotonic() - 5.0, left_at=None))
        held.caller_turns, held.agent_turns = 4, 4
        report = held.integrity()
        assert "hangup_held" in report["checks"] and report["hangup_tail_ms"] >= 5000

        prompt = bot.CallNarrator(hangup=SimpleNamespace(hung_up_at=time.monotonic(), left_at=None))
        prompt.caller_turns, prompt.agent_turns = 4, 4
        assert prompt.integrity()["checks"] == ["ok"]

        # The caller is let go when the room is left, whatever the teardown does after.
        left_early = bot.CallNarrator(
            hangup=SimpleNamespace(hung_up_at=time.monotonic() - 5.0, left_at=time.monotonic() - 4.98)
        )
        left_early.caller_turns, left_early.agent_turns = 4, 4
        report = left_early.integrity()
        assert report["checks"] == ["ok"] and report["hangup_tail_ms"] < 100

        no_hang_up = bot.CallNarrator(hangup=SimpleNamespace(hung_up_at=None, left_at=None))
        no_hang_up.caller_turns, no_hang_up.agent_turns = 4, 4
        assert "hangup_tail_ms" not in no_hang_up.integrity()
