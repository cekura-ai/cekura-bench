"""ElevenLabs over the text-to-dialogue input-streaming websocket (``eleven_v3_conversational``).

The v3 conversational model is not served on the per-voice endpoints; it speaks
this one. The socket registers its voice once, before t0. The utterance goes in
as one ``inputs`` frame carrying ``flush: true``. The server ends each utterance
with ``is_final_audio_for_turn`` and keeps the socket open, so contexts on one
socket run one after another.

Whole text per utterance, like the HTTP providers, because the endpoint cannot
take text that arrives in pieces as one utterance:

* once about eight words are buffered without a flush, the server starts
  generating by itself and closes the socket with ``invalid_argument`` (1008);
* a flush with no text is rejected the same way;
* a flush on every frame makes each frame its own turn, with its own
  completion signal and a pause between turns.

So streamed input and continuation are declared exclusions. There is no cancel
either: ``close_socket`` finishes the buffered text and delivers the rest of the
audio.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

from tts_bench.adapters._ws import WebSocketAdapter


class ElevenLabsDialogueAdapter(WebSocketAdapter):
    name: ClassVar[str] = "elevenlabs-dialogue"
    supports_streamed_input: ClassVar[bool] = False
    supports_cancel: ClassVar[bool] = False
    supports_continuation: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (16000, 22050, 24000, 44100)
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, voice registration frame"
    base = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None
        self._text: dict[str, str] = {}

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?model_id={self.config.model}&output_format=pcm_{self.config.sample_rate}"

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    async def _connect(self) -> None:
        await super()._connect()
        await self._send_json({"voices": [self.config.voice]})

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        self._text[context_id] = self._text.get(context_id, "") + text

    async def _finish(self, context_id: str) -> None:
        text = self._text.pop(context_id, "")
        await self._send_json({"inputs": [{"text": text, "voice_id": self.config.voice}], "flush": True})

    async def _close(self) -> None:
        try:
            if self._ws:
                await self._send_json({"close_socket": True})
        except Exception:  # noqa: BLE001
            pass
        await super()._close()

    def _on_message(self, message: dict[str, Any]) -> None:
        context_id = self._active
        if message.get("error"):
            self._on_error(context_id, f"{message.get('error')}: {message.get('message')}")
            return
        audio = message.get("audio")
        if audio and context_id:
            self._on_audio(context_id, base64.b64decode(audio))
        if (message.get("is_final_audio_for_turn") or message.get("is_final")) and context_id:
            self._on_done(context_id)
