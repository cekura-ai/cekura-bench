"""Which TTS services the bench can measure, keyed by wire protocol.

Adding a model on a protocol the bench already speaks is a table entry; adding
a protocol is an adapter. Every row published about a provider carries what its
adapter excludes from t0 and which features it lacks, so a gap in the table is
a declared exclusion rather than a silent omission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from tts_bench.adapters.base import TTSAdapter
from tts_bench.adapters.cartesia import CartesiaAdapter
from tts_bench.adapters.deepgram import DeepgramSpeakAdapter
from tts_bench.adapters.elevenlabs import ElevenLabsAdapter
from tts_bench.adapters.fake import FakeTTSAdapter
from tts_bench.adapters.gemini_tts import GeminiTTSAdapter
from tts_bench.adapters.openai_tts import OpenAITTSAdapter


@dataclass(frozen=True)
class ProviderEntry:
    key: str
    adapter: Type[TTSAdapter]
    default_model: str
    default_voice: str
    credential_env: str
    notes: str = ""


PROVIDERS: dict[str, ProviderEntry] = {
    "fake": ProviderEntry("fake", FakeTTSAdapter, "scripted", "tone", "FAKE_KEY_UNUSED"),
    "elevenlabs": ProviderEntry(
        "elevenlabs", ElevenLabsAdapter, "eleven_flash_v2_5", "21m00Tcm4TlvDq8ikWAM", "ELEVENLABS_API_KEY",
        notes="multi-context websocket; flush per context; close_context is the cancel",
    ),
    "cartesia": ProviderEntry(
        "cartesia", CartesiaAdapter, "sonic-3", "a0e99841-438c-4a64-b679-ae501e7d6091", "CARTESIA_API_KEY",
        notes="websocket with continue/cancel per context_id",
    ),
    "deepgram": ProviderEntry(
        "deepgram", DeepgramSpeakAdapter, "aura-2-thalia-en", "aura-2-thalia-en", "DEEPGRAM_API_KEY",
        notes="one utterance per websocket; Speak/Flush/Clear; the voice is the model",
    ),
    "openai": ProviderEntry(
        "openai", OpenAITTSAdapter, "gpt-4o-mini-tts", "alloy", "OPENAI_API_KEY",
        notes="HTTP streaming response; whole text per request; no cancel short of closing the connection",
    ),
    "gemini": ProviderEntry(
        "gemini", GeminiTTSAdapter, "gemini-2.5-flash-preview-tts", "Kore", "GEMINI_AUTHORIZATION",
        notes="HTTP generateContent; audio in one or a few parts; whole text per request",
    ),
}

# Protocols not yet spoken. Listed so the gap is a fact, not an oversight.
PENDING = ("rime", "inworld", "hume", "lmnt", "azure-speech", "google-cloud-tts", "polly", "smallest", "minimax")
