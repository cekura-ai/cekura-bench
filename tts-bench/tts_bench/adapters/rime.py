"""Rime Coda and Mist v3 over the recommended JSON websocket, ``/ws3``.

``segment=never`` buffers concatenable text frames until an explicit flush,
then emits base64 PCM chunks followed by ``done``. One flush per utterance
avoids treating a synthesis batch boundary as the end of the whole input.
Contexts run sequentially on one socket; contextId only tags responses.

``clear`` only discards unsubmitted text, not in-flight or queued synthesis,
so it is not a cancel. ``eos`` flushes remaining text and closes the socket;
the bench uses ``flush`` to keep the socket for the next utterance.

Reference: https://docs.rime.ai/api-reference/coda/websockets-json
Buffering and clear: https://docs.rime.ai/docs/websockets-segment
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar
from urllib.parse import urlencode

from tts_bench.adapters._ws import WebSocketAdapter


class RimeAdapter(WebSocketAdapter):
    name: ClassVar[str] = "rime"
    supports_cancel: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # audioFormat=mulaw, samplingRate=8000
    base = "wss://users-ws.rime.ai/ws3"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?" + urlencode({
            "speaker": self.config.voice, "modelId": self.config.model,
            "audioFormat": "pcm", "samplingRate": self.config.sample_rate,
            "lang": self.config.language or "en", "segment": "never",
        })

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"text": text, "contextId": context_id})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"operation": "flush"})

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        context_id = message.get("contextId") or self._active
        if kind == "chunk" and message.get("data"):
            self._on_audio(context_id, base64.b64decode(message["data"]))
        elif kind == "done":
            self._on_done(context_id)
        elif kind == "error":
            self._on_error(context_id, str(message.get("message") or message))
