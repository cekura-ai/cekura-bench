"""Fish Audio TTS over HTTP: one ``POST /v1/tts`` per utterance, raw PCM streamed back.

The model goes in the ``model`` header and the voice is ``reference_id``, as on
the websocket. ``latency`` (default ``balanced``, as on the websocket),
``language`` and ``normalize`` are adapter options, sent only when the run names
them so every request is what the run recorded. Audio is requested as ``pcm``
at the run's rate: the samples of ``wav`` without its header.

The whole text goes in one request, so streamed input, cancel and continuation
are declared exclusions, as for every HTTP adapter. The connection is warmed
before t0 with ``GET /model/<voice>``, which also checks the key and the voice.

Reference: https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech
"""

from __future__ import annotations

from typing import Any, ClassVar

from tts_bench.adapters.fish_audio import request_options
from tts_bench.adapters.openai_tts import OpenAITTSAdapter


class FishAudioHTTPAdapter(OpenAITTSAdapter):
    name: ClassVar[str] = "fish-audio-http"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 24000, 32000, 44100)
    base = "https://api.fish.audio"
    speech_path = "/v1/tts"

    def _prewarm_url(self) -> str:
        return f"{self.base}/model/{self.config.voice}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "model": self.config.model}

    def _request_args(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"text": text, "reference_id": self.config.voice, "format": "pcm",
                                   "sample_rate": self.config.sample_rate, **request_options(self.config, self.name)}
        self.log.raw({**payload, "text": f"<{len(text)} chars>"}, direction="out")
        return {"json": payload}
