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

from dataclasses import dataclass, field
from typing import Type

from service.adapters.base import RealtimeAdapter
from service.adapters.fake import FakeAdapter
from service.adapters.gemini_live import GeminiLiveAdapter
from service.adapters.grok_realtime import GrokRealtimeAdapter
from service.adapters.openai_realtime import OpenAIRealtimeAdapter


@dataclass(frozen=True)
class ProviderEntry:
    key: str
    adapter: Type[RealtimeAdapter]
    default_model: str
    credential_env: str
    default_voice: str | None = None
    discloses: tuple[str, ...] = field(default_factory=tuple)


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
