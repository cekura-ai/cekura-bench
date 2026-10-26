"""Gemini text-to-speech over HTTP ``streamGenerateContent`` (server-sent events).

The text is the prompt; the model is asked for audio only. Audio arrives as
base64 ``inlineData`` parts, one or a few per response, so chunk arrivals are
what they are and the playout metrics report them as such. Whole text per
request: no streamed input, no cancel, no continuation -- all declared.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, ClassVar

import aiohttp

from tts_bench.adapters.base import AdapterError, TTSAdapter

RATE = 24000


class GeminiTTSAdapter(TTSAdapter):
    name: ClassVar[str] = "gemini-tts"
    transport: ClassVar[str] = "http"
    supports_streamed_input: ClassVar[bool] = False
    supports_cancel: ClassVar[bool] = False
    supports_continuation: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (RATE,)
    setup_excluded: ClassVar[str] = "TCP, TLS (session pre-warmed with one request)"
    base = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._session: aiohttp.ClientSession | None = None
        self._text: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def _connect(self) -> None:
        self._session = aiohttp.ClientSession(headers={"x-goog-api-key": self.api_key})
        try:
            async with self._session.get(f"{self.base}/models/{self.config.model}") as response:
                self.log.raw({"prewarm_status": response.status}, direction="in")
                if response.status in (401, 403):
                    raise AdapterError("gemini rejected the key")
        except aiohttp.ClientError as exc:
            raise AdapterError(f"gemini prewarm failed: {exc}") from exc

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
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.config.voice}}},
            },
        }
        self.log.raw({"model": self.config.model, "chars": len(text)}, direction="out")
        url = f"{self.base}/models/{self.config.model}:streamGenerateContent?alt=sse"
        try:
            async with self._session.post(url, json=payload) as response:
                self._on_response(context_id, response.status, response.headers)
                if response.status != 200:
                    body = await response.text()
                    self._on_error(context_id, f"HTTP {response.status}: {body[:300]}")
                    return
                # One event can carry a whole utterance as a single ``data:`` line of
                # several hundred kilobytes, far past any line reader's limit, so the
                # stream is split on newlines by hand from whatever arrives.
                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        self._line(context_id, line)
                self._line(context_id, buffer)
            self._on_done(context_id)
        except Exception as exc:  # noqa: BLE001 -- anything that stops the stream is the cell's error, never a silent timeout
            self._on_error(context_id, f"http: {exc!r}")

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
        logged = json.loads(json.dumps(message))
        synthesis = self.contexts.get(context_id)
        for candidate in message.get("candidates", []):
            # A finish reason other than STOP with no audio is the provider
            # declining; it belongs in the row, not only in the raw log.
            if candidate.get("finishReason") and synthesis is not None:
                synthesis.meta["finish_reason"] = candidate["finishReason"]
            for part in candidate.get("content", {}).get("parts", []):
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    self._on_audio(context_id, base64.b64decode(inline["data"]))
        for candidate in logged.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if part.get("inlineData", {}).get("data"):
                    part["inlineData"]["data"] = f"<{len(part['inlineData']['data'])} b64 chars>"
        self.log.raw(logged)
        if message.get("error"):
            self._on_error(context_id, str(message["error"]))
