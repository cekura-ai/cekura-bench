"""Gemini 3.8 Live Extended Thinking: the service bench's Gemini Live adapter with a
thinking level in the setup.

The extended-thinking model refuses a setup without one ("Thinking level must be
specified for this model"); the plain model refuses a setup that names one. Cekura's
published speech-to-speech benchmark runs this row at thinking level HIGH
(reference-agents/pipecat-s2s/bot.py GEMINI_THINKING), so that is the default here.
The service bench's adapter is subclassed, not modified.
"""

from __future__ import annotations

from typing import Any

from turn_taking_bench.author import service_bridge

GeminiLiveAdapter = service_bridge.service_module("adapters.gemini_live").GeminiLiveAdapter

DEFAULT_MODEL = "gemini-3.8-live-extended-thinking"
THINKING_LEVEL = "HIGH"


class GeminiThinkingAdapter(GeminiLiveAdapter):
    name = "gemini-live-thinking"

    def __init__(self, *, model: str = DEFAULT_MODEL, thinking_level: str = THINKING_LEVEL, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.thinking_level = thinking_level

    def _setup_payload(self) -> dict[str, Any]:
        setup = super()._setup_payload()
        setup["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        return setup


def provider_entry():
    from service.registry import PROVIDERS, ProviderEntry

    base = PROVIDERS["gemini-live"]
    return ProviderEntry(
        key="gemini-live-thinking",
        adapter=GeminiThinkingAdapter,
        default_model=DEFAULT_MODEL,
        credential_env=base.credential_env,
        default_voice=base.default_voice,
        discloses=(f"thinking level {THINKING_LEVEL} (as Cekura's published S2S benchmark)",),
    )
