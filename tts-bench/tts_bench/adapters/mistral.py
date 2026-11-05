"""Mistral Voxtral text to speech over HTTP: one request per utterance, audio streamed back as server-sent events.

``POST /v1/audio/speech`` with ``stream: true`` answers with ``speech.audio.delta``
events, each carrying base64 ``audio_data``, and one ``speech.audio.done`` that
ends the utterance and reports usage. Requested as ``pcm``, the samples are
32-bit float at 24 kHz mono, not the 16-bit integers every other row records,
so they are converted here as they arrive; a chunk's float can straddle two
events, so the remainder is carried to the next one. Only ``pcm`` streams with
no container to strip.

The whole text goes in one request, so streamed input, cancel and continuation
are declared exclusions, as for OpenAI. The connection is warmed with the
model's own record before t0, so the timed part is synthesis alone.

A request the content filter rejects comes back as HTTP 403 with a message
that reads like a refused key. It is the service declining the text, not the
account being stopped, so it is recorded as a failure the voice owns and not
retried as a refusal.
"""

from __future__ import annotations

import base64
import json
from typing import Any, ClassVar

import numpy as np

from tts_bench.adapters.openai_tts import OpenAITTSAdapter

RATE = 24000
FLOAT_WIDTH = 4


def to_int16(samples: bytes) -> bytes:
    """Little-endian float32 in [-1, 1] as little-endian int16, clipped rather than wrapped."""
    floats = np.frombuffer(samples, dtype="<f4")
    return (np.clip(floats, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


class MistralTTSAdapter(OpenAITTSAdapter):
    name: ClassVar[str] = "mistral-speech"
    native_rates: ClassVar[tuple[int, ...]] = (RATE,)
    base = "https://api.mistral.ai"
    speech_path = "/v1/audio/speech"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending: dict[str, bytes] = {}

    def _prewarm_url(self) -> str:
        return f"{self.base}/v1/models/{self.config.model}"

    def _request_args(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.config.model, "voice_id": self.config.voice, "input": text,
                                   "response_format": "pcm", "stream": True}
        self.log.raw({**payload, "input": f"<{len(text)} chars>"}, direction="out")
        return {"json": payload}

    async def _request(self, context_id: str, text: str) -> None:
        assert self._session is not None
        args = self._request_args(text)
        try:
            async with self._session.post(f"{self.base}{self.speech_path}", **args) as response:
                self._on_response(context_id, response.status, response.headers)
                if response.status != 200:
                    body = await response.text()
                    self._on_error(context_id, self._http_error(response.status, body))
                    return
                # One event is a single ``data:`` line of tens of kilobytes, past a
                # line reader's limit, so the stream is split on newlines by hand.
                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        self._line(context_id, line)
                self._line(context_id, buffer)
            synthesis = self.contexts.get(context_id)
            if synthesis is not None and not synthesis.done.is_set():
                self._on_error(context_id, "stream ended without speech.audio.done")
        except Exception as exc:  # noqa: BLE001 -- anything that stops the stream is the cell's error, never a silent timeout
            self._on_error(context_id, f"http: {exc!r}")

    @staticmethod
    def _http_error(status: int, body: str) -> str:
        if status == 403 and "moderat" in body.lower():
            return f"content filter declined the text: {body[:300]}"
        return f"HTTP {status}: {body[:300]}"

    def _line(self, context_id: str, line: bytes) -> None:
        text_line = line.decode("utf-8", "replace").strip()
        if not text_line.startswith("data:"):
            return
        try:
            message = json.loads(text_line[5:].strip())
        except json.JSONDecodeError:
            self.log.raw({"unparsed_sse": text_line[:120]})
            return
        self._handle(context_id, message)

    def _handle(self, context_id: str, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "speech.audio.delta":
            audio = message.get("audio_data") or ""
            self.log.raw({**message, "audio_data": f"<{len(audio)} b64 chars>"})
            samples = self._pending.pop(context_id, b"") + base64.b64decode(audio)
            whole = len(samples) - len(samples) % FLOAT_WIDTH
            if len(samples) > whole:
                self._pending[context_id] = samples[whole:]
            self._on_audio(context_id, to_int16(samples[:whole]))
        elif kind == "speech.audio.done":
            self.log.raw(message)
            left = self._pending.pop(context_id, b"")
            self._on_done(context_id, usage=message.get("usage"),
                          **({"trailing_bytes_dropped": len(left)} if left else {}))
        else:
            self.log.raw(message)
            if kind == "error" or message.get("object") == "error" or message.get("error"):
                self._on_error(context_id, str(message.get("message") or message.get("error") or message)[:300])
