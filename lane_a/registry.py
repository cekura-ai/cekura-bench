"""Which providers Lane A can measure, and what a row about them must disclose.

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

from lane_a.adapters.base import RealtimeAdapter
from lane_a.adapters.fake import FakeAdapter
from lane_a.adapters.gemini_live import GeminiLiveAdapter
from lane_a.adapters.grok_realtime import GrokRealtimeAdapter
from lane_a.adapters.openai_realtime import OpenAIRealtimeAdapter


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
        credential_env="LANE_A_FAKE_KEY",
    ),
}

# Verified unavailable or not yet adapted, kept here so a gap is explicit rather
# than looking like an oversight:
#   gpt-live-1      -- refused on /v1/realtime; separate product at /v1/live/sessions
#                      with a delegated backend text model. Needs its own adapter
#                      and its row must disclose backend cost.
#   qwen-omni       -- needs a DashScope key. Fireworks cannot serve it as S2S:
#                      audio inference is deprecated there, Qwen3-Omni is served
#                      through Chat Completions, and their websocket audio is ASR
#                      only, which would make it a cascade, not an S2S row.
#   nova-sonic      -- needs AWS credentials and Bedrock model access.
PENDING = ("gpt-live-1", "qwen-omni", "nova-sonic")
