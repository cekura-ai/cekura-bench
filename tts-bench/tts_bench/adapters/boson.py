"""Boson AI Higgs TTS 3 over its OpenAI-shaped speech endpoint.

The request is OpenAI's speech body plus Boson's fields, as Boson asked it to be
run: ``stream`` true (raw 24 kHz 16-bit mono PCM back as it is decoded),
``enable_tn`` true with ``tn_language`` "en" (Boson's own text normalisation
expands numbers, dates, codes and symbols before synthesis), and ``timestamps``
false (timestamps need a non-streamed response). Whole text per request, so
streamed input, cancel and continuation are declared exclusions, as for OpenAI.
"""

from __future__ import annotations

from typing import Any, ClassVar

from tts_bench.adapters.openai_tts import OpenAITTSAdapter


class BosonTTSAdapter(OpenAITTSAdapter):
    name: ClassVar[str] = "boson-speech"
    base = "https://api.boson.ai"
    speech_path = "/v1/audio/speech"

    def _prewarm_url(self) -> str:
        return f"{self.base}/v1/audio/voices"

    def _request_args(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.config.model, "input": text, "voice": self.config.voice,
                                   "stream": True, "response_format": "pcm",
                                   "enable_tn": self.config.option("enable_tn", "true") == "true",
                                   "tn_language": self.config.option("tn_language", "en"), "timestamps": False}
        self.log.raw({**payload, "input": f"<{len(text)} chars>"}, direction="out")
        return {"json": payload}
