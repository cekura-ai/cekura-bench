"""The Lane B reference agent's wiring, without a provider or a phone line.

What is worth testing offline is everything that decides whether a call is
scored fairly: the sample rate handed to each provider, the tools the model is
shown, the answers it gets back, and the greeting it is told to say. The parts
that need a network -- that the model connects, speaks and calls tools -- are
verified by running it, not by mocking it.

Skipped automatically unless Pipecat is installed, so the Lane A test run stays
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
            # The variable is whatever the vendor documents, not a house style:
            # Bedrock's own name for its API key carries no _API_KEY suffix, and
            # renaming it here would mean a key that works everywhere else is
            # unset for this agent alone.
            assert provider.credential_env.isupper(), name
            assert " " not in provider.credential_env, name

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

    def test_the_delegating_provider_discloses_its_backend(self):
        """One provider does not do its own reasoning, and the row must say so.

        Its conversational model is named in every record. The model that
        actually answers a reasoned question is a second one, and a record
        naming only the first would describe a configuration nobody ran.
        """
        server = bot.load_agent()
        record = bot.build_record("gpt-live", bot.PROVIDERS["gpt-live"], "gpt-live-1", "marin", server)
        assert record["s2s_backend_model"], "a delegating provider must name its backend"

        plain = bot.build_record(
            "openai-realtime", bot.PROVIDERS["openai-realtime"], "gpt-realtime-2.1", "marin", server
        )
        # Absent rather than empty: a provider that reasons for itself has no
        # backend, and an empty string would read as one that went unrecorded.
        assert "s2s_backend_model" not in plain

    def test_the_bedrock_provider_records_the_region_it_ran_in(self):
        """A Bedrock key is scoped by region and a model is served in some regions only.

        Two runs of one model id in two regions are two different calls, and a
        refusal in one looks like a refusal in the other without this field.
        """
        server = bot.load_agent()
        record = bot.build_record(
            "nova-sonic", bot.PROVIDERS["nova-sonic"], "amazon.nova-2-sonic-v1:0", "matthew", server
        )
        assert record["aws_region"]


class TestCredentialForms:
    """Bedrock issues an API key or an access-key pair, and both must reach the model."""

    def test_a_pair_is_recognised_by_its_separator(self):
        from nova_bearer import BearerTokenNovaSonic

        pair = bot._nova_sonic("AKIAEXAMPLE:secretpart", "amazon.nova-2-sonic-v1:0", "matthew", "hi")
        assert not isinstance(pair, BearerTokenNovaSonic), "a pair must sign with SigV4"

    def test_a_bare_token_takes_the_bearer_path(self):
        from nova_bearer import BearerTokenNovaSonic

        token = bot._nova_sonic("abcdefghij", "amazon.nova-2-sonic-v1:0", "matthew", "hi")
        assert isinstance(token, BearerTokenNovaSonic)
