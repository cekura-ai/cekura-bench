"""Cartesia over its websocket: one socket, contexts by ``context_id``.

Frames on one context carry ``continue: true``; the utterance is closed with an
empty transcript and ``continue: false``. ``cancel: true`` stops a context. Audio
comes back as base64 chunks tagged with the context, and ``done`` marks the end.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

from tts_bench.adapters._ws import WebSocketAdapter

API_VERSION = "2025-04-16"


class CartesiaAdapter(WebSocketAdapter):
    name: ClassVar[str] = "cartesia"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100)
    native_mulaw_8k: ClassVar[bool] = True     # encoding pcm_mulaw at 8000
    base = "wss://api.cartesia.ai/tts/websocket"

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?api_key={self.api_key}&cartesia_version={API_VERSION}"

    def _message(self, context_id: str, text: str, cont: bool) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "transcript": text,
            "continue": cont,
            "context_id": context_id,
            "model_id": self.config.model,
            "voice": {"mode": "id", "id": self.config.voice},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": self.config.sample_rate},
            "language": self.config.language or "en",
            "add_timestamps": False,
        }
        if self.config.speed is not None:
            msg["speed"] = self.config.speed
        return msg

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json(self._message(context_id, text, cont=True))

    async def _finish(self, context_id: str) -> None:
        await self._send_json(self._message(context_id, "", cont=False))

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"context_id": context_id, "cancel": True})

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        context_id = message.get("context_id")
        if kind == "chunk":
            data = message.get("data")
            if data:
                self._on_audio(context_id, base64.b64decode(data))
            if message.get("done"):
                self._on_done(context_id)
        elif kind == "done":
            self._on_done(context_id)
        elif kind == "error":
            self._on_error(context_id, str(message.get("error") or message))
        elif kind in ("flush_done", "timestamps"):
            pass
