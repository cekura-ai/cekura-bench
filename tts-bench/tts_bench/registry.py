"""Which TTS services the bench can measure, keyed by wire protocol.

Adding a model on a protocol the bench already speaks is a table entry; adding
a protocol is an adapter. Every row published about a provider carries what its
adapter excludes from t0 and which features it lacks, so a gap in the table is
a declared exclusion rather than a silent omission.

``models`` is the lineup a campaign runs on each protocol, with one fixed voice
per model; the first entry is the default. Any other model or voice the
protocol serves still runs with ``--model`` / ``--voice``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from tts_bench.adapters.base import TTSAdapter
from tts_bench.adapters.cartesia import CartesiaAdapter
from tts_bench.adapters.deepgram import DeepgramSpeakAdapter
from tts_bench.adapters.deepgram_flux import DeepgramFluxAdapter
from tts_bench.adapters.deepinfra import DeepInfraTTSAdapter
from tts_bench.adapters.elevenlabs import ElevenLabsAdapter
from tts_bench.adapters.elevenlabs_dialogue import ElevenLabsDialogueAdapter
from tts_bench.adapters.fake import FakeTTSAdapter
from tts_bench.adapters.gemini_tts import GeminiTTSAdapter
from tts_bench.adapters.inworld import InworldAdapter
from tts_bench.adapters.openai_tts import OpenAITTSAdapter
from tts_bench.adapters.smallest import SmallestAdapter
from tts_bench.adapters.soniox import SonioxAdapter
from tts_bench.adapters.xai import XaiTTSAdapter


@dataclass(frozen=True)
class Model:
    model: str
    voice: str


@dataclass(frozen=True)
class ProviderEntry:
    key: str
    adapter: Type[TTSAdapter]
    credential_env: str
    models: tuple[Model, ...]
    notes: str = ""

    @property
    def default_model(self) -> str:
        return self.models[0].model

    def voice_for(self, model: str) -> str:
        """The lineup's voice for ``model``; the default model's voice for a model outside the lineup."""
        return next((m.voice for m in self.models if m.model == model), self.models[0].voice)


# One female US English stock voice per model. Deepgram addresses a voice by
# model string, so there the voice is the model.
SKYLAR = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"       # Cartesia
RACHEL = "21m00Tcm4TlvDq8ikWAM"                       # ElevenLabs

PROVIDERS: dict[str, ProviderEntry] = {
    "fake": ProviderEntry("fake", FakeTTSAdapter, "FAKE_KEY_UNUSED", (Model("scripted", "tone"),)),
    "cartesia": ProviderEntry(
        "cartesia", CartesiaAdapter, "CARTESIA_API_KEY",
        (Model("sonic-3.6", SKYLAR), Model("sonic-3.5", SKYLAR)),
        notes="websocket with continue/cancel per context_id",
    ),
    "elevenlabs": ProviderEntry(
        "elevenlabs", ElevenLabsAdapter, "ELEVENLABS_API_KEY",
        (Model("eleven_flash_v2_5", RACHEL),),
        notes="multi-context websocket; flush per context; close_context is the cancel",
    ),
    "elevenlabs-dialogue": ProviderEntry(
        "elevenlabs-dialogue", ElevenLabsDialogueAdapter, "ELEVENLABS_API_KEY",
        (Model("eleven_v3_conversational", RACHEL),),
        notes="text-to-dialogue websocket; whole text per utterance; no streamed input, continuation or cancel",
    ),
    "deepgram": ProviderEntry(
        "deepgram", DeepgramSpeakAdapter, "DEEPGRAM_API_KEY",
        (Model("aura-2-thalia-en", "aura-2-thalia-en"),),
        notes="/v1/speak; one utterance at a time per websocket; Speak/Flush/Clear; the voice is the model",
    ),
    "deepgram-flux": ProviderEntry(
        "deepgram-flux", DeepgramFluxAdapter, "DEEPGRAM_API_KEY",
        (Model("flux-haley-en", "flux-haley-en"),),
        notes="/v2/speak; SpeechMetadata ends a turn; Interrupt is the cancel; the voice is the model",
    ),
    "openai": ProviderEntry(
        "openai", OpenAITTSAdapter, "OPENAI_API_KEY",
        (Model("gpt-4o-mini-tts", "alloy"),),
        notes="HTTP streaming response; whole text per request; no cancel short of closing the connection",
    ),
    "gemini": ProviderEntry(
        "gemini", GeminiTTSAdapter, "GEMINI_AUTHORIZATION",
        (Model("gemini-3.8-flash-tts", "Kore"), Model("gemini-3.8-flash-lite-tts", "Kore"),
         Model("gemini-3.1-flash-tts-preview", "Kore")),
        notes="HTTP streamGenerateContent; audio in parts; whole text per request",
    ),
    "inworld": ProviderEntry(
        "inworld", InworldAdapter, "INWORLD_API_KEY",
        (Model("inworld-tts-2-flash", "Brooke"),),
        notes="bidirectional websocket, contexts by contextId; raw PCM; no cancel (close_context flushes first)",
    ),
    "xai": ProviderEntry(
        "xai", XaiTTSAdapter, "XAI_API_KEY",
        (Model("grok-tts", "carina"),),
        notes="websocket, voice fixed per socket in the URL; text.delta / text.done; text.clear is the cancel; no model parameter",
    ),
    "smallest": ProviderEntry(
        "smallest", SmallestAdapter, "SMALLEST_API_KEY",
        (Model("lightning_v3.1_pro", "kelsey"),),
        notes="websocket in continuation mode (context_id, continue); ends on quiet; no cancel",
    ),
    "soniox": ProviderEntry(
        "soniox", SonioxAdapter, "SONIOX_API_KEY",
        (Model("tts-rt-v2", "Emma"),),
        notes="websocket, one stream per context; text_end ends input; cancel answered by terminated",
    ),
    "deepinfra": ProviderEntry(
        "deepinfra", DeepInfraTTSAdapter, "DEEPINFRA_API_KEY",
        (Model("Qwen/Qwen3-TTS", "Vivian"),),
        notes="OpenAI-compatible HTTP streaming, raw PCM; whole text per request; any hosted model is a lineup entry",
    ),
}

# Protocols not yet spoken. Listed so the gap is a fact, not an oversight.
PENDING = ("qwen-audio", "speechify", "rime", "minimax",
           "murf", "fish-audio", "hume", "lmnt", "azure-speech", "google-cloud-tts", "polly")


def lineup() -> list[tuple[str, Model]]:
    """Every (provider, model) a campaign runs, in registry order."""
    return [(key, model) for key, entry in PROVIDERS.items() if key != "fake" for model in entry.models]
