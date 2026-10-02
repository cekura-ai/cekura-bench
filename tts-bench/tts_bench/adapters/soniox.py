"""Soniox real-time TTS over its websocket, one stream per context.

A context is a ``stream_id``. Its configuration frame (key, model, voice,
language, format) is sent before t0; text goes in as ``{"text", "text_end":
false}`` frames and ``text_end: true`` ends the input. Audio comes back as
base64 ``audio`` frames tagged with the stream; the last carries ``audio_end:
true`` and ``terminated`` closes the stream. The cancel is ``{"stream_id",
"cancel": true}``, answered by ``terminated`` with no further audio.

A stream that receives no text for a few seconds before ``text_end`` is
terminated by the server (``request_timeout``); the probes never pause that
long between frames.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

from tts_bench.adapters._ws import WebSocketAdapter


class SonioxAdapter(WebSocketAdapter):
    name: ClassVar[str] = "soniox-tts"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 24000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # audio_format pcm_mulaw at 8000
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, per-stream configuration frame"
    url = "wss://tts-rt.soniox.com/tts-websocket"

    async def _open_context(self, context_id: str) -> None:
        payload: dict[str, Any] = {
            "api_key": self.api_key, "stream_id": context_id, "model": self.config.model,
            "language": self.config.language or "en", "voice": self.config.voice,
            "audio_format": "pcm_s16le", "sample_rate": self.config.sample_rate,
        }
        if self.config.speed is not None:
            payload["speed"] = self.config.speed
        await self._send_json(payload)

    def _logged(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "api_key": "<redacted>"} if "api_key" in payload else payload

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"stream_id": context_id, "text": text, "text_end": False})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"stream_id": context_id, "text": "", "text_end": True})

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"stream_id": context_id, "cancel": True})

    def _on_message(self, message: dict[str, Any]) -> None:
        context_id = message.get("stream_id")
        if message.get("error_code") or message.get("error_message"):
            self._on_error(context_id, f"{message.get('error_type') or message.get('error_code')}: {message.get('error_message')}")
            return
        audio = message.get("audio")
        if audio and context_id:
            self._on_audio(context_id, base64.b64decode(audio))
        if message.get("terminated") and context_id:
            synthesis = self.contexts.get(context_id)
            if synthesis is not None and synthesis.t_cancel is not None:
                self._on_cancel_ack(context_id)
            self._on_done(context_id)
