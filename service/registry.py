"""Which providers the service bench can measure, and what a row about them must disclose.

Adapters are keyed by **wire protocol**, so adding a model is a table entry and
adding a protocol is an adapter. That ratio is the whole cost argument for direct
adapters: most providers ship several models on one protocol, so protocol work
amortises across the table.

``discloses`` is not documentation. Where a provider delegates part of the work to
another model, its cost and its latency are still part of what the caller
experiences, and a row that quietly omits them is not comparable to a row for a
service that delegates nothing. Naming the delegation in the registry keeps that
out of a footnote and inside the published cell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Type

from service.adapters.base import RealtimeAdapter
from service.adapters.deepgram_voice_agent import THINK_KEY_ENV, DeepgramVoiceAgentAdapter
from service.adapters.fake import FakeAdapter
from service.adapters.gemini_live import GeminiLiveAdapter
from service.adapters.grok_realtime import GrokRealtimeAdapter
from service.adapters.higgs_realtime import HiggsRealtimeAdapter
from service.adapters.openai_realtime import OpenAIRealtimeAdapter


@dataclass(frozen=True)
class ProviderEntry:
    key: str
    adapter: Type[RealtimeAdapter]
    default_model: str
    credential_env: str
    default_voice: str | None = None
    discloses: tuple[str, ...] = field(default_factory=tuple)
    # Credentials the adapter reads from the environment besides its own key.
    extra_credentials: tuple[str, ...] = ()


PROVIDERS: dict[str, ProviderEntry] = {
    "openai-realtime": ProviderEntry(
        key="openai-realtime",
        adapter=OpenAIRealtimeAdapter,
        default_model="gpt-realtime-2.1",
        credential_env="OPENAI_API_KEY",
        default_voice="marin",
    ),
    "gemini-live": ProviderEntry(
        key="gemini-live",
        adapter=GeminiLiveAdapter,
        default_model="gemini-2.5-flash-native-audio-preview-12-2025",
        credential_env="GEMINI_AUTHORIZATION",
        default_voice="Kore",
        discloses=("reasons before replying by default; thought tokens are counted in usage",),
    ),
    "grok-realtime": ProviderEntry(
        key="grok-realtime",
        adapter=GrokRealtimeAdapter,
        default_model="grok-voice-think-fast-2.0",
        credential_env="XAI_API_KEY",
        default_voice="eve",
    ),
    "higgs-realtime": ProviderEntry(
        key="higgs-realtime",
        adapter=HiggsRealtimeAdapter,
        default_model="higgs-realtime",
        credential_env="BOSON_API_KEY",
        default_voice="default",
    ),
    "deepgram-voice-agent": ProviderEntry(
        key="deepgram-voice-agent",
        adapter=DeepgramVoiceAgentAdapter,
        default_model="open_ai/gpt-4.1",
        credential_env="DEEPGRAM_API_KEY",
        default_voice="aura-2-thalia-en",
        extra_credentials=(THINK_KEY_ENV,),
        discloses=(
            "a managed pipeline, not one speech model: Flux (flux-general-en) hears the caller, "
            "the named text model replies and an Aura voice speaks; latency covers all three",
            "the text model runs on our own OpenAI account through Deepgram's bring-your-own endpoint: "
            "Deepgram's hosted models cut prompts past 25,000 characters, and the medicare contract's is longer",
            "Deepgram bills per minute and OpenAI bills the text model's tokens on our account; "
            "the service bench records no cost for this row",
            "a fixed silence duration is sent as Flux's eot_timeout_ms, a ceiling: Flux may end a turn sooner",
        ),
    ),
    # Not a provider: a scripted agent with known reply timing, registered so the
    # whole harness can be run end to end with no API key and no network. Anyone
    # checking that our runner produces artifacts a number can be recomputed from
    # should start here rather than take our word for it.
    "fake": ProviderEntry(
        key="fake",
        adapter=FakeAdapter,
        default_model="scripted",
        credential_env="SERVICE_FAKE_KEY",
    ),
}

# Not adapted here yet: gpt-live-1 (a separate /v1/live product with a delegated
# backend), qwen-omni (needs a DashScope key) and nova-sonic (needs Bedrock).


def load_extra_credentials(entry: ProviderEntry, env_file: str | None = None) -> list[str]:
    """Put ``entry``'s extra credentials into this process's environment, from
    ``env_file`` when they are not already set there. Returns the names found
    nowhere. Values stay in-process and are never printed."""
    missing = [name for name in entry.extra_credentials if not os.environ.get(name)]
    if missing and env_file:
        from dotenv import dotenv_values

        values = dotenv_values(env_file)
        for name in missing:
            if values.get(name):
                os.environ[name] = values[name]
    return [name for name in entry.extra_credentials if not os.environ.get(name)]
