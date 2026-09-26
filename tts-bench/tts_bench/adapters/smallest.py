"""Smallest.ai Lightning over its live websocket, in continuation mode.

Every frame names a ``context_id``; with one the socket stays open across
contexts (without one the server closes it after the first utterance). Text
frames carry ``continue: true`` and the input ends with ``continue: false``.
The server releases text on sentence boundaries and sends base64 PCM as
``{"status": "chunk", "data": {"audio": ...}}``.

**No message ends an utterance.** ``complete`` arrives once per released
segment, never names the context, and nothing marks the last one, so a context
ends when its stream goes quiet (``ended_by: quiet``), as for ElevenLabs. The
count of ``complete`` frames is kept in the row. ``complete_backoff_ms`` is set
to 0: by default the server holds each ``complete`` back 4 s after the last
chunk, which would only lengthen that wait.

There is no cancel. ``cancel_request`` drops text still buffered for the
context but not text already released (measured: about 15 s of a 15 s answer
delivered after it), so cancel is a declared exclusion.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

from tts_bench.adapters._ws import WebSocketAdapter


class SmallestAdapter(WebSocketAdapter):
    name: ClassVar[str] = "smallest-lightning"
    supports_cancel: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 24000, 44100)
    url = "wss://api.smallest.ai/waves/v1/tts/live"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _frame(self, context_id: str, text: str, cont: bool) -> dict[str, Any]:
        frame: dict[str, Any] = {
            "text": text, "voice_id": self.config.voice, "model": self.config.model,
            "sample_rate": self.config.sample_rate, "language": self.config.language or "en",
            "context_id": context_id, "continue": cont, "complete_backoff_ms": 0,
        }
        if self.config.speed is not None:
            frame["speed"] = self.config.speed
        return frame

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json(self._frame(context_id, text, cont=True))

    async def _finish(self, context_id: str) -> None:
        await self._send_json(self._frame(context_id, "", cont=False))

    def _on_message(self, message: dict[str, Any]) -> None:
        if not self._active:
            return
        status = message.get("status")
        if status == "chunk":
            audio = (message.get("data") or {}).get("audio")
            if audio:
                self._on_audio(self._active, base64.b64decode(audio))
        elif status == "complete":
            synthesis = self.contexts.get(self._active)
            if synthesis is not None:
                synthesis.meta["completes"] = synthesis.meta.get("completes", 0) + 1
        elif status == "error":
            self._on_error(self._active, str(message.get("message") or message))
