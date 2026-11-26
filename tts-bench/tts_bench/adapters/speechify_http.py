"""Speechify over its HTTP stream, ``POST /v1/audio/stream``: one request per utterance.

The endpoint Speechify asked us to use for latency. The whole text goes in one
JSON request (``input``, ``voice_id``, ``model``, ``language``) and raw 16-bit
mono PCM streams back as a chunked body (``output_format`` ``pcm_<rate>``,
``Content-Type: audio/L16``). There is no streamed input and no cancel short of
closing the connection, so streamed input, cancel and continuation are declared
exclusions, as for every whole-text HTTP adapter. The request carries the
settings, so nothing per utterance is set up before t0 beyond the warm
connection.

The pre-warm is ``GET /v1/audio/models``, which needs the key: a refused key
stops the run before t0.

HTTP chunk boundaries do not respect samples (live, about one chunk in five
ends inside a 16-bit sample), so a stray byte is held for the next chunk and
the timeline only ever sees whole samples. A body that ends on a stray byte is
the synthesis's error.

The response headers name the request (``Speechify-Request-Id``), the request
rate left (``RateLimit-Remaining``), the billed characters
(``X-Speechify-Billable-Characters-Count``) and Speechify's own time to first
byte and model time (``Server-Timing``); all are kept with the synthesis.

Reference: https://docs.speechify.ai/build/api-reference/v1/audio/stream
"""

from __future__ import annotations

from typing import Any, ClassVar

from tts_bench.adapters.openai_tts import OpenAITTSAdapter


class SpeechifyHTTPAdapter(OpenAITTSAdapter):
    name: ClassVar[str] = "speechify-http"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # output_format ulaw_8000
    base = "https://api.speechify.ai"
    speech_path = "/v1/audio/stream"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stray: dict[str, bytes] = {}

    def _prewarm_url(self) -> str:
        return f"{self.base}/v1/audio/models"

    def _request_args(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"input": text, "voice_id": self.config.voice, "model": self.config.model,
                                   "language": self.config.language or "en-US",
                                   "output_format": f"pcm_{self.config.sample_rate}"}
        self.log.raw({**payload, "input": f"<{len(text)} chars>"}, direction="out")
        return {"json": payload}

    def _on_audio(self, context_id: str, pcm: bytes) -> None:
        data = self._stray.pop(context_id, b"") + pcm
        if len(data) % 2:
            self._stray[context_id], data = data[-1:], data[:-1]
        super()._on_audio(context_id, data)

    def _on_done(self, context_id: str, **meta: Any) -> None:
        if self._stray.pop(context_id, b""):
            self._on_error(context_id, "the stream ended inside a 16-bit sample")
            return
        super()._on_done(context_id, **meta)
