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

from tts_bench.adapters.azure_speech import AzureSpeechAdapter
from tts_bench.adapters.base import TTSAdapter
from tts_bench.adapters.cartesia import CartesiaAdapter
from tts_bench.adapters.deepgram import DeepgramSpeakAdapter
from tts_bench.adapters.deepgram_flux import DeepgramFluxAdapter
from tts_bench.adapters.deepinfra import DeepInfraTTSAdapter
from tts_bench.adapters.elevenlabs import ElevenLabsAdapter
from tts_bench.adapters.elevenlabs_dialogue import ElevenLabsDialogueAdapter
from tts_bench.adapters.elevenlabs_dialogue_multi import ElevenLabsDialogueMultiAdapter
from tts_bench.adapters.fake import FakeTTSAdapter
from tts_bench.adapters.fish_audio import FishAudioAdapter
from tts_bench.adapters.gemini_tts import GeminiTTSAdapter
from tts_bench.adapters.gradium import GradiumAdapter
from tts_bench.adapters.hume import HumeAdapter
from tts_bench.adapters.inworld import InworldAdapter
from tts_bench.adapters.openai_tts import OpenAITTSAdapter
from tts_bench.adapters.rime import RimeAdapter
from tts_bench.adapters.smallest import SmallestAdapter
from tts_bench.adapters.soniox import SonioxAdapter
from tts_bench.adapters.speechify import SpeechifyAdapter
from tts_bench.adapters.xai import XaiTTSAdapter


DEFAULT_RATE = 24000


@dataclass(frozen=True)
class Model:
    model: str
    voice: str
    rate: int = DEFAULT_RATE    # a model that emits only one PCM rate names it here


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

    def rate_for(self, model: str) -> int:
        """The lineup's sample rate for ``model``; the harness default for a model outside the lineup."""
        return next((m.rate for m in self.models if m.model == model), DEFAULT_RATE)

    def voice_for(self, model: str) -> str:
        """The lineup's voice for ``model``; the default model's voice for a model outside the lineup."""
        return next((m.voice for m in self.models if m.model == model), self.models[0].voice)


# One female US English stock voice per model. Deepgram addresses a voice by
# model string, so there the voice is the model.
SKYLAR = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"       # Cartesia
RACHEL = "21m00Tcm4TlvDq8ikWAM"                       # ElevenLabs
EVERLY = "vjFyp2BcCyTlUmYb"     # Gradium, its first choice of its recommended voices
ASTRA = "astra"                                      # Rime, female US English on Coda and Mist v3
SARAH = "933563129e564b19a115bedd57b7406a"              # Fish Official, female US English
DELIA = "c70cfb98867a4bfb9ae20d2401257b92"              # Fish Audio, female US English, from its Drama 3 voice cast
AVA = "5bb7de05-c8fe-426a-8fcc-ba4fc4ce9f9c"           # Hume "Ava Song", female US English, from its Voice Library
HARPER = "harper_32"                                  # Speechify, its first staff pick of female US English voices

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
    "elevenlabs-dialogue-multi": ProviderEntry(
        "elevenlabs-dialogue-multi", ElevenLabsDialogueMultiAdapter, "ELEVENLABS_API_KEY",
        (Model("eleven_v4_turbo", RACHEL), Model("eleven_v4", RACHEL)),
        notes="multi-context text-to-dialogue websocket (v4 is served only there); text in pieces per context; "
              "close_context flushes, so the cancel is measured, not assumed",
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
        (Model("inworld-tts-2-flash", "Sarah"), Model("inworld-tts-2", "Sarah")),
        notes="bidirectional websocket, contexts by contextId; raw PCM; TTS-2 in creative delivery mode; "
              "no cancel (close_context flushes first)",
    ),
    "gradium": ProviderEntry(
        "gradium", GradiumAdapter, "GRADIUM_API_KEY",
        (Model("default", EVERLY),),
        notes="websocket, contexts by client_req_id; setup/ready before t0; streamed input and continuation "
              "via text frames in one request (split on whitespace, punctuation attached); <flush> forces generation; "
              "end_of_stream ends input; no cancel (unscoped end waits for active requests); "
              "limits guide: 3000 s per session; reference free-tier cap: 1500 characters; paragraph cohort fits both",
    ),
    "rime": ProviderEntry(
        "rime", RimeAdapter, "RIME_API_KEY",
        (Model("coda", ASTRA), Model("mistv3", ASTRA)),
        notes="/ws3 JSON websocket; segment=never buffers concatenable text until finish flushes; "
              "done ends the batch; sequential contexts; no cancel (clear only drops unsubmitted text); "
              "Coda replaces the retired Arcana family",
    ),
    "fish-audio": ProviderEntry(
        "fish-audio", FishAudioAdapter, "FISH_AUDIO_API_KEY",
        (Model("s2.1-pro", SARAH), Model("drama-3-preview", DELIA)),
        notes="MessagePack websocket; start before t0; text pieces conditioned on previous audio; "
              "latency=balanced by default; stop drains input, finish ends output and closes the socket; "
              "new socket per utterance, including repeat; no cancel/clear; paid production model "
              "(s2.1-pro-free has no TTFA or DPA guarantees)",
    ),
    "hume": ProviderEntry(
        "hume", HumeAdapter, "HUME_API_KEY",
        (Model("octave-2", AVA, rate=48000),),
        notes="input-streaming websocket, one utterance per socket; close ends the stream; 48 kHz PCM only; no cancel",
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
    "speechify": ProviderEntry(
        "speechify", SpeechifyAdapter, "SPEECHIFY_API_KEY",
        (Model("simba-3.5", HARPER),),
        notes="text-input websocket, one session per utterance; model, voice, PCM rate and language ride the upgrade URL; "
              "a sentence is spoken once the next starts; input.close speaks the rest and ends the session; no cancel",
    ),
    "azure-speech": ProviderEntry(
        "azure-speech", AzureSpeechAdapter, "AZURE_SPEECH_KEY",
        (Model("MAI-Voice-2.1-Flash", "en-US-Harper"), Model("MAI-Voice-2.1", "en-US-Harper")),
        notes="HTTP SSML, raw PCM streamed back; the model is the voice name's suffix; whole text per request; "
              "region is the host (option region, default eastus)",
    ),
}

# Protocols not yet spoken. Listed so the gap is a fact, not an oversight.
PENDING = ("qwen-audio", "minimax",
           "murf", "hume", "lmnt", "google-cloud-tts", "polly")


def lineup() -> list[tuple[str, Model]]:
    """Every (provider, model) a campaign runs, in registry order."""
    return [(key, model) for key, entry in PROVIDERS.items() if key != "fake" for model in entry.models]
