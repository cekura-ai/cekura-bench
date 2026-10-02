"""OpenAI speech over HTTP: one request per utterance, raw PCM streamed back.

The whole text goes in one request, so there is no streamed input and no cancel
short of closing the connection; both are declared exclusions. The connection is
warmed with a cheap request before t0 so the timed part is synthesis alone.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import aiohttp

from tts_bench.adapters.base import AdapterError, TTSAdapter

RATE = 24000


class OpenAITTSAdapter(TTSAdapter):
    name: ClassVar[str] = "openai-speech"
    transport: ClassVar[str] = "http"
    supports_streamed_input: ClassVar[bool] = False
    supports_cancel: ClassVar[bool] = False
    supports_continuation: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (RATE,)
    setup_excluded: ClassVar[str] = "TCP, TLS (session pre-warmed with one request)"
    base = "https://api.openai.com"
    speech_path = "/v1/audio/speech"

    def _prewarm_url(self) -> str:
        return f"{self.base}/v1/models/{self.config.model}"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._session: aiohttp.ClientSession | None = None
        self._text: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def _connect(self) -> None:
        self._session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            async with self._session.get(self._prewarm_url()) as response:
                self.log.raw({"prewarm_status": response.status}, direction="in")
                if response.status in (401, 403):
                    raise AdapterError(f"{self.name} rejected the key")
        except aiohttp.ClientError as exc:
            raise AdapterError(f"{self.name} prewarm failed: {exc}") from exc

    async def _close(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        if self._session:
            await self._session.close()

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        self._text[context_id] = self._text.get(context_id, "") + text

    async def _finish(self, context_id: str) -> None:
        self._tasks[context_id] = asyncio.create_task(self._request(context_id, self._text.get(context_id, "")))

    async def _request(self, context_id: str, text: str) -> None:
        assert self._session is not None
        payload: dict[str, Any] = {"model": self.config.model, "voice": self.config.voice, "input": text, "response_format": "pcm"}
        if self.config.speed is not None:
            payload["speed"] = self.config.speed
        self.log.raw({**payload, "input": f"<{len(text)} chars>"}, direction="out")
        try:
            async with self._session.post(f"{self.base}{self.speech_path}", json=payload) as response:
                self._on_response(context_id, response.status, response.headers)
                if response.status != 200:
                    body = await response.text()
                    self._on_error(context_id, f"HTTP {response.status}: {body[:300]}")
                    return
                async for chunk in response.content.iter_any():
                    self._on_audio(context_id, chunk)
            self._on_done(context_id)
        except Exception as exc:  # noqa: BLE001 -- anything that stops the stream is the cell's error, never a silent timeout
            self._on_error(context_id, f"http: {exc!r}")
