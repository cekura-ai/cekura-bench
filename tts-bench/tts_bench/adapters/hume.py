"""Hume Octave over its input-streaming websocket, one utterance per socket.

Text frames carry the voice on the first frame and plain text after it;
``close`` forces generation of whatever is buffered and ends the stream, so
the socket closing marks the end of the utterance. Audio arrives as base64
PCM in JSON frames (``no_binary``, ``strip_headers``). Octave 2 needs a voice
from the library, and its PCM is 48 kHz only.

There is no cancel: the protocol's only controls are ``flush`` and ``close``,
and both finish the text already sent rather than interrupting it.

Reference: https://dev.hume.ai/voice/reference/text-to-speech-tts/stream-input
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar
from urllib.parse import urlencode

from tts_bench.adapters._ws import WebSocketAdapter

RATE = 48000


class HumeAdapter(WebSocketAdapter):
    name: ClassVar[str] = "hume"
    supports_cancel: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (RATE,)
    native_mulaw_8k: ClassVar[bool] = False    # formats: mp3, pcm, wav
    setup_excluded: ClassVar[str] = "TCP, TLS and websocket upgrade; new socket per utterance"
    base = "wss://api.hume.ai/v0/tts/stream/input"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?" + urlencode({
            "version": self.config.model.removeprefix("octave-"), "format_type": "pcm", "no_binary": "true",
            "strip_headers": "true", "instant_mode": "true",
        })

    def _headers(self) -> dict[str, str]:
        return {"X-Hume-Api-Key": self.api_key}

    async def _open_context(self, context_id: str) -> None:
        if self._active is not None:
            await self._close()
            await self._connect()
        self._active = context_id

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        frame: dict[str, Any] = {"text": text}
        if first:
            frame["voice"] = {"id": self.config.voice}
        await self._send_json(frame)

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"close": True})

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "audio" and message.get("audio"):
            self._on_audio(self._active, base64.b64decode(message["audio"]))
        elif kind == "error" or message.get("error"):
            self._on_error(self._active, str(message.get("message") or message.get("error") or message))

    async def _receive_loop(self) -> None:
        await super()._receive_loop()
        # The server closes the socket cleanly once the stream is done, which the base loop does not report.
        self._on_closed("closed by server")

    def _on_closed(self, reason: str) -> None:
        if self._active is not None and self._active in self.contexts:
            self._on_done(self._active)
        super()._on_closed(reason)
