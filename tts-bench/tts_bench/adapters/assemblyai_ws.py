"""AssemblyAI Streaming TTS over its websocket, one utterance at a time per socket.

The session is configured entirely in the URL (model, voice, rate, encoding)
and opens with a ``Begin`` frame before anything is read from us, so it is
waited for before t0. Text is appended with ``Generate`` frames, joined exactly
as sent; ``Flush`` ends the utterance and is answered by ``FlushDone`` once its
last audio byte is on the socket. ``Cancel`` stops the synthesis in flight and
drops unspoken text, answered by ``Cancelled``. Audio arrives as base64 PCM in
``Audio`` frames tagged with a ``flush_id``; the socket has no context ids, so
contexts on one socket are used one after another.

Word timings are switched off: they are an extra frame after ``FlushDone`` that
the bench does not read. ``Terminate`` ends the session cleanly before close.

Reference: https://www.assemblyai.com/docs/tts/message-frames
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, ClassVar
from urllib.parse import urlencode

from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import AdapterError


class AssemblyAIStreamingAdapter(WebSocketAdapter):
    name: ClassVar[str] = "assemblyai-streaming"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # encoding=pcm_mulaw at 8000
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, Begin frame"
    base = "wss://tts.us.assemblyai.com/v1/ws"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None
        self._begin: asyncio.Future | None = None

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?" + urlencode({
            "model_id": self.config.model, "voice_id": self.config.voice, "sample_rate": self.config.sample_rate,
            "encoding": "pcm_s16le", "word_boundaries": "false",
        })

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self.api_key}

    async def _connect(self) -> None:
        self._begin = asyncio.get_running_loop().create_future()
        await super()._connect()
        # The server answers the handshake with Begin, or with an Error and a close: either way, before t0.
        try:
            begin = await asyncio.wait_for(asyncio.shield(self._begin), timeout=20)
        except asyncio.TimeoutError:
            begin = {"error": "no Begin frame within 20 s"}
        if "error" in begin:
            # The cell never opens, so nothing else would stop the receive loop before the cell's log closes.
            await super()._close()
            if self._receiver:
                await asyncio.gather(self._receiver, return_exceptions=True)
            raise AdapterError(f"{self.name} refused the session: {begin['error']}")

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"type": "Generate", "text": text})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"type": "Flush"})

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"type": "Cancel"})

    async def _close(self) -> None:
        try:
            if self._ws:
                await self._send_json({"type": "Terminate"})
        except Exception:  # noqa: BLE001 -- the socket may already be gone; closing is all that is left
            pass
        await super()._close()

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "Begin":
            if self._begin and not self._begin.done():
                self._begin.set_result(message)
        elif kind == "Audio" and self._active:
            self._on_audio(self._active, base64.b64decode(message.get("audio", "")))
        elif kind == "FlushDone" and self._active:
            self._on_done(self._active, flush_id=message.get("flush_id"), audio_duration_ms=message.get("audio_duration_ms"))
        elif kind == "Cancelled" and self._active:
            self._on_cancel_ack(self._active, audio_duration_seconds=message.get("audio_duration_seconds"))
            self._on_done(self._active, cancelled=True)
        elif kind == "Error":
            # The error text carries the close code, so a refused account (1008 unauthorized, 3009 rate_limited)
            # reads as a refusal rather than a failure of the service.
            text = f"{message.get('error_code')} {message.get('error_type')}: {message.get('error')}"
            if self._begin and not self._begin.done():
                self._begin.set_result({"error": text})
            self._on_error(self._active, text)

    def _on_closed(self, reason: str) -> None:
        if self._begin and not self._begin.done():
            self._begin.set_result({"error": f"closed before Begin: {reason}"})
        super()._on_closed(reason)
