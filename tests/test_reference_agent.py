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

import sys
from pathlib import Path

import pytest

pytest.importorskip("pipecat", reason="reference agent needs pipecat-ai")

AGENT_DIR = Path(__file__).resolve().parent.parent / "reference-agents" / "pipecat-s2s"
sys.path.insert(0, str(AGENT_DIR))

import bot  # noqa: E402
from nova_bearer import BearerTokenNovaSonic  # noqa: E402


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
        server = bot.load_agent()
        schema = bot.build_tools(server)
        assert {t.name for t in schema.standard_tools} == set(server.tool_names)
        assert all(t.description for t in schema.standard_tools)

    async def test_a_registered_handler_answers_from_the_contract(self):
        server = bot.load_agent()
        registered = {}

        class StubLLM:
            def register_function(self, name, handler, **_kwargs):
                registered[name] = handler

        bot.register_tools(StubLLM(), server)
        assert set(registered) == set(server.tool_names)

        answers = []

        class Params:
            function_name = "lookup_patient"
            arguments = {"phone": "2025550188"}

            async def result_callback(self, result):
                answers.append(result)

        await registered["lookup_patient"](Params())
        assert answers and answers[0]["patient_id"] == "p_1002"

    async def test_an_unknown_record_is_reported_as_a_miss(self):
        server = bot.load_agent()
        registered = {}

        class StubLLM:
            def register_function(self, name, handler, **_kwargs):
                registered[name] = handler

        bot.register_tools(StubLLM(), server)
        answers = []

        class Params:
            function_name = "lookup_patient"
            arguments = {"phone": "4045550000"}

            async def result_callback(self, result):
                answers.append(result)

        await registered["lookup_patient"](Params())
        assert answers[0] == {"result": "no_match"}
        assert server.calls[-1].matched is False


class TestAgentDefinition:
    def test_the_contracts_own_prompt_and_greeting_are_used(self):
        server = bot.load_agent()
        assert server.system_prompt and server.first_message
        assert server.suite == "appointments"


class TestBuildRecord:
    """A phone call cannot be replayed, so the build that answered it must be recorded."""

    def test_it_names_the_configuration_a_call_cannot_be_re_run_without(self):
        server = bot.load_agent()
        record = bot.build_record(
            "openai-realtime", bot.PROVIDERS["openai-realtime"], "gpt-realtime-2.1", "marin", server
        )
        # The rate is the one that silently ruins a call: these services do not
        # resample, so a wrong rate reads as a bad model rather than bad wiring.
        assert record["pipeline_sample_rate"] == bot.PROVIDERS["openai-realtime"].input_rate
        assert record["pipecat_version"] != "unknown", "the framework version is part of the agent"
        assert record["system_prompt_sha256"] and record["first_message_sha256"]
        assert "lookup_patient" in record["tools"]

    def test_a_changed_prompt_changes_the_record(self):
        server = bot.load_agent()
        provider = bot.PROVIDERS["openai-realtime"]
        before = bot.build_record("openai-realtime", provider, "m", "v", server)
        server.system_prompt = server.system_prompt + " Answer briefly."
        after = bot.build_record("openai-realtime", provider, "m", "v", server)
        assert before["system_prompt_sha256"] != after["system_prompt_sha256"]

    def test_every_declared_disclosure_reaches_the_record(self):
        """A provider that declares an extra must actually put it on the record.

        One loop rather than a test per provider: the failure this guards against
        is a sixth provider added with a disclosure that never lands, and a test
        naming the five that exist could not catch it.
        """
        server = bot.load_agent()
        for name, provider in bot.PROVIDERS.items():
            record = bot.build_record(
                name, provider, provider.default_model, provider.default_voice, server
            )
            for field in provider.discloses():
                assert record.get(field), f"{name} declares {field} but the record has no value"

    def test_a_provider_with_nothing_to_disclose_carries_no_empty_field(self):
        """Absent rather than empty.

        A provider that reasons for itself has no backend model. An empty string
        would read as a field that went unrecorded, which is a different claim.
        """
        server = bot.load_agent()
        plain = bot.build_record(
            "openai-realtime", bot.PROVIDERS["openai-realtime"], "gpt-realtime-2.1", "marin", server
        )
        assert "s2s_backend_model" not in plain
        assert "aws_region" not in plain


class TestCredentialForms:
    """Bedrock issues an API key or an access-key pair, and both must reach the model."""

    def test_an_access_key_pair_signs_with_sigv4(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secretpart")
        built = bot._nova_sonic("ignored", "amazon.nova-2-sonic-v1:0", "matthew", "hi")
        assert not isinstance(built, BearerTokenNovaSonic), "a pair must sign with SigV4"

    def test_an_api_key_alone_takes_the_bearer_path(self, monkeypatch):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        built = bot._nova_sonic("abcdefghij", "amazon.nova-2-sonic-v1:0", "matthew", "hi")
        assert isinstance(built, BearerTokenNovaSonic)

    def test_a_half_set_pair_does_not_sign_with_sigv4(self, monkeypatch):
        """An id with no secret is not a credential, and must not look like one."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        built = bot._nova_sonic("abcdefghij", "amazon.nova-2-sonic-v1:0", "matthew", "hi")
        assert isinstance(built, BearerTokenNovaSonic)


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
        server = bot.load_agent()
        native = bot.build_record(
            "openai-realtime", bot.PROVIDERS["openai-realtime"], "gpt-realtime-2.1", "marin", server
        )
        cascade = bot.cascade_record(
            "cascade-openai", bot.TEXT_MODELS["cascade-openai"], "gpt-4.1", server
        )
        for field in ("agent_definition", "system_prompt_sha256", "first_message_sha256", "tools"):
            assert native[field] == cascade[field], field
        assert native["stack"] == "native" and cascade["stack"] == "cascade"

    def test_a_cascade_row_names_all_three_services(self):
        """A latency column is mostly the endpointer and the voice, not the text model.

        A row naming only its text model would hide the two components doing
        most of what is being measured.
        """
        server = bot.load_agent()
        record = bot.cascade_record(
            "cascade-baseline", bot.TEXT_MODELS["cascade-baseline"], "gpt-4.1", server
        )
        assert record["stt_model"] and record["llm_model"] and record["tts_model"]

    def test_every_cascade_names_an_importable_module(self):
        import importlib.util

        for key, text in bot.TEXT_MODELS.items():
            assert importlib.util.find_spec(text.module), f"{key}: {text.module}"

    def test_a_configuration_name_is_never_ambiguous(self):
        """One name selects one stack. An overlap would make a run unattributable."""
        assert not (set(bot.PROVIDERS) & set(bot.TEXT_MODELS))
