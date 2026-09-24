"""Deepgram Speak (Aura) over its websocket.

Text is appended with ``Speak`` frames, ``Flush`` generates what is buffered and
is acknowledged with ``Flushed`` after the last audio byte, ``Clear`` drops the
buffer and stops the audio, acknowledged with ``Cleared``. The socket has no
context ids: audio belongs to the most recently opened context, so contexts on
one socket are used one after another. Audio arrives as binary frames.
"""

from __future__ import annotations

from typing import Any, ClassVar

from tts_bench.adapters._ws import WebSocketAdapter


class DeepgramSpeakAdapter(WebSocketAdapter):
    name: ClassVar[str] = "deepgram-speak"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 24000, 32000, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # encoding=mulaw
    base = "wss://api.deepgram.com/v1/speak"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None

    @property
    def url(self) -> str:  # type: ignore[override]
        # The voice *is* the model string on this API (aura-2-<name>-en).
        return f"{self.base}?model={self.config.voice}&encoding=linear16&sample_rate={self.config.sample_rate}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"type": "Speak", "text": text})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"type": "Flush"})

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"type": "Clear"})

    async def _close(self) -> None:
        try:
            if self._ws:
                await self._send_json({"type": "Close"})
        except Exception:  # noqa: BLE001
            pass
        await super()._close()

    def _on_binary(self, data: bytes) -> None:
        if self._active:
            self._on_audio(self._active, data)

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "Flushed":
            self._on_done(self._active, sequence_id=message.get("sequence_id"))
        elif kind == "Cleared":
            self._on_cancel_ack(self._active, sequence_id=message.get("sequence_id"))
            self._on_done(self._active, cleared=True)
        elif kind == "Error" or "err_code" in message or message.get("type") == "Warning":
            self._on_error(self._active, str(message))
