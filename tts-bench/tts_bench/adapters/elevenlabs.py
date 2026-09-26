"""ElevenLabs over the multi-context input-streaming websocket.

One socket, many contexts. Text is appended per context, ``flush`` forces
generation of what is buffered, and ``close_context`` stops a context and drops
what it still had queued -- which is the cancel. Audio frames carry a
``contextId`` so several contexts can be in flight.

``auto_mode`` (option ``auto_mode=true|false``, default false) is recorded in the
configuration because it changes the number: on, the service starts generating
at sentence punctuation on its own; off, it generates on the flush. Measured on
this endpoint, off-plus-flush was both faster to first audio and stable, and it
leaves the context open so a cancel can still reach it.

``finish`` sends only the flush. Closing the context would give a completion
signal but also makes a later cancel impossible, which is the production order
of operations (flush at the end of the text, close on interruption). The end
of a context is therefore detected by the stream going quiet.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import TTSConfig


class ElevenLabsAdapter(WebSocketAdapter):
    name: ClassVar[str] = "elevenlabs"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100)
    native_mulaw_8k: ClassVar[bool] = True     # output_format=ulaw_8000
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, per-context init frame"
    base = "wss://api.elevenlabs.io"

    @property
    def auto_mode(self) -> bool:
        return (self.config.option("auto_mode", "false") or "false").lower() == "true"

    @property
    def url(self) -> str:  # type: ignore[override]
        c = self.config
        return (
            f"{self.base}/v1/text-to-speech/{c.voice}/multi-stream-input"
            f"?model_id={c.model}&output_format=pcm_{c.sample_rate}&auto_mode={str(self.auto_mode).lower()}"
            f"&inactivity_timeout=60"
        )

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    async def _open_context(self, context_id: str) -> None:
        payload: dict[str, Any] = {"text": " ", "context_id": context_id}
        if self.config.speed is not None:
            payload["voice_settings"] = {"speed": self.config.speed}
        await self._send_json(payload)

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"text": text, "context_id": context_id})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"context_id": context_id, "flush": True})

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"context_id": context_id, "close_context": True})

    async def _close(self) -> None:
        try:
            if self._ws:
                await self._send_json({"close_socket": True})
        except Exception:  # noqa: BLE001
            pass
        await super()._close()

    def _on_message(self, message: dict[str, Any]) -> None:
        context_id = message.get("contextId") or message.get("context_id")
        if message.get("error") or (message.get("code") and message.get("message")):
            self._on_error(context_id, f"{message.get('error') or message.get('code')}: {message.get('message')}")
            return
        audio = message.get("audio")
        if audio:
            self._on_audio(context_id, base64.b64decode(audio))
        if message.get("isFinal") or message.get("is_final"):
            synthesis = self.contexts.get(context_id)
            if synthesis is not None and synthesis.t_cancel is not None:
                self._on_cancel_ack(context_id)
            self._on_done(context_id)
