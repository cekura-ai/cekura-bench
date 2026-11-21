"""AssemblyAI TTS over its OpenAI-compatible speech endpoint.

OpenAI's speech request with one change, as AssemblyAI specified it for this
benchmark: the voice is the ``voice_id`` field, and OpenAI's ``voice`` is left
out (sending both is refused with 400). Raw 24 kHz 16-bit mono PCM streams
back. ``speed`` accepts only 1, so it is never sent. Whole text per request, so
streamed input, cancel and continuation are declared exclusions, as for OpenAI.

The pre-warm is ``GET /models``, which answers 200 with or without a valid
key; a refused key surfaces as the first cell's 401 instead of before t0.

Reference: https://www.assemblyai.com/docs/tts/openai-compatibility
"""

from __future__ import annotations

from typing import Any, ClassVar

from tts_bench.adapters.openai_tts import OpenAITTSAdapter


class AssemblyAITTSAdapter(OpenAITTSAdapter):
    name: ClassVar[str] = "assemblyai-speech"
    base = "https://tts.us.assemblyai.com"
    speech_path = "/v1/openai/audio/speech"

    def _prewarm_url(self) -> str:
        return f"{self.base}/v1/openai/models"

    def _request_args(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.config.model, "voice_id": self.config.voice, "input": text,
                                   "response_format": "pcm"}
        self.log.raw({**payload, "input": f"<{len(text)} chars>"}, direction="out")
        return {"json": payload}
