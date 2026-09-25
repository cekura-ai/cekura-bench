"""DeepInfra-hosted TTS models over its OpenAI-compatible speech endpoint.

The same request and the same streamed raw PCM as OpenAI's speech API, at
24 kHz 16-bit mono, served from DeepInfra's host. One adapter covers every model
DeepInfra hosts on this endpoint; each model names its own voices, and an
unknown voice is refused with HTTP 500. Whole text per request, so streamed
input, cancel and continuation are declared exclusions, as for OpenAI.
"""

from __future__ import annotations

from typing import ClassVar

from tts_bench.adapters.openai_tts import OpenAITTSAdapter


class DeepInfraTTSAdapter(OpenAITTSAdapter):
    name: ClassVar[str] = "deepinfra-speech"
    base = "https://api.deepinfra.com"
    speech_path = "/v1/openai/audio/speech"

    def _prewarm_url(self) -> str:
        return f"{self.base}/models/{self.config.model}"
