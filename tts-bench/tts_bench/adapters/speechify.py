"""Speechify over its text-input websocket, one utterance per session.

``GET /v1/audio/stream/ws`` takes the synthesis settings as query parameters,
so the upgrade fixes the model, voice, PCM rate and language before t0. Text
goes in as ``input.text`` pieces and ``input.close`` speaks whatever is left
and ends the session with ``speech.done``. Audio comes back as base64 PCM in
``speech.chunk`` events.

The server speaks a sentence once the next one starts, and sentences that
complete while one is speaking join the next segment; each segment is a
separate synthesis. There is no cancel: the only controls are ``input.flush``
and ``input.close``, and both speak the text already sent rather than
interrupting it.

A session holds one of the account's concurrent requests, and every session,
and every segment after a session's first, draws on one per-account request
rate. An empty bucket refuses an upgrade with 429 but makes a running session
wait, so each context records the upgrade's ``RateLimit-Remaining`` to show
whether a wait could have landed inside a latency.

Reference: https://docs.speechify.ai/build/api-reference/v1/audio/stream/ws
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar
from urllib.parse import urlencode

from tts_bench.adapters._ws import WebSocketAdapter


class SpeechifyAdapter(WebSocketAdapter):
    name: ClassVar[str] = "speechify"
    supports_cancel: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # output_format ulaw_8000
    setup_excluded: ClassVar[str] = "TCP, TLS and websocket upgrade carrying the settings; new socket per utterance"
    base = "wss://api.speechify.ai/v1/audio/stream/ws"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None
        self._upgrade: dict[str, Any] = {}

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?" + urlencode({
            "voice_id": self.config.voice, "model": self.config.model,
            "output_format": f"pcm_{self.config.sample_rate}", "language": self.config.language or "en-US",
        })

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _connect(self) -> None:
        await super()._connect()
        headers = self._ws.response.headers if self._ws is not None and self._ws.response is not None else {}
        self._upgrade = {"request_id": headers.get("Speechify-Request-Id"),
                         "ratelimit_remaining": headers.get("RateLimit-Remaining")}

    async def _open_context(self, context_id: str) -> None:
        if self._active is not None:
            await self._close()
            await self._connect()
        self._active = context_id
        self.contexts[context_id].meta.update(self._upgrade)

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"type": "input.text", "text": text})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"type": "input.close"})

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "speech.chunk" and message.get("audio"):
            self._on_audio(self._active, base64.b64decode(message["audio"]))
        elif kind == "speech.done":
            self._on_done(self._active, reason=message.get("reason"),
                          billable_characters=message.get("billable_characters_count"),
                          server_duration_ms=message.get("audio_duration_ms"))
        elif kind == "speech.error":
            error = message.get("error") or {}
            self._on_error(self._active, f"{error.get('code')}: {error.get('message')}")
