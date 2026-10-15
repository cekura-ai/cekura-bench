"""xAI Grok TTS over its streaming websocket.

The voice, codec and sample rate are fixed per socket in the URL. Text goes in
as ``text.delta`` frames and ``text.done`` ends the input; audio comes back as
base64 ``audio.delta`` events and ``audio.done`` ends the utterance. The socket
has no context ids, so contexts on one socket run one after another.

The cancel is ``text.clear``: the server drops what it holds and acknowledges
with ``audio.clear``, after which the socket takes the next utterance.

The API has no model parameter; the service is priced and served as one TTS
model, so the model recorded for a row is the service name.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar
from urllib.parse import urlencode

from tts_bench.adapters._ws import WebSocketAdapter


class XaiTTSAdapter(WebSocketAdapter):
    name: ClassVar[str] = "xai-tts"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # codec=mulaw
    base = "wss://api.x.ai/v1/tts"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?" + urlencode({
            "language": self.config.language or "en", "voice": self.config.voice,
            "codec": "pcm", "sample_rate": self.config.sample_rate,
        })

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"type": "text.delta", "delta": text})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"type": "text.done"})

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"type": "text.clear"})

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if not self._active:
            return
        if kind == "audio.delta" and message.get("delta"):
            self._on_audio(self._active, base64.b64decode(message["delta"]))
        elif kind == "audio.done":
            self._on_done(self._active, trace_id=message.get("trace_id"))
        elif kind == "audio.clear":
            self._on_cancel_ack(self._active)
            self._on_done(self._active, cleared=True)
        elif kind == "error":
            self._on_error(self._active, str(message.get("message") or message))
