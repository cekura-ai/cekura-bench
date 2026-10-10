"""ElevenLabs over the multi-context text-to-dialogue websocket (``eleven_v4``, ``eleven_v4_turbo``).

The v4 models are served only on the text-to-dialogue endpoints; the per-voice
``stream-input`` and ``multi-stream-input`` sockets reject them at the upgrade.
This is the multi-context variant: one socket, a context per utterance keyed by
``context_id``, each registering its voice with its first frame (before t0) and
ending with its own ``is_final_audio_for_turn``, so several can be in flight.

Unlike ``eleven_v3_conversational`` on the single-context socket, v4 takes text
that arrives in pieces as one utterance: ``inputs`` frames without a flush are
buffered, the server starts speaking once it has enough, and the closing
``flush`` generates the rest. So streamed input and continuation are measured.

``close_context`` is the only per-context stop the protocol has. The documented
behaviour is to flush what the context holds and then close it, so the cancel
probe measures how much audio still arrives after it rather than assuming it
stops; the ElevenLabs flash adapter's ``close_context`` is measured the same way.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

from tts_bench.adapters._ws import WebSocketAdapter


class ElevenLabsDialogueMultiAdapter(WebSocketAdapter):
    name: ClassVar[str] = "elevenlabs-dialogue-multi"
    native_rates: ClassVar[tuple[int, ...]] = (16000, 22050, 24000, 44100)
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, per-context voice registration frame"
    base = "wss://api.elevenlabs.io/v1/text-to-dialogue/multi-stream-input"

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?model_id={self.config.model}&output_format=pcm_{self.config.sample_rate}"

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    async def _open_context(self, context_id: str) -> None:
        await self._send_json({"context_id": context_id, "voices": [self.config.voice]})

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"context_id": context_id, "inputs": [{"text": text, "voice_id": self.config.voice}]})

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
        context_id = message.get("context_id") or message.get("contextId")
        if message.get("error") or (message.get("code") and message.get("message")):
            self._on_error(context_id, f"{message.get('error') or message.get('code')}: {message.get('message')}")
            return
        audio = message.get("audio")
        if audio and context_id:
            self._on_audio(context_id, base64.b64decode(audio))
        if (message.get("is_final_audio_for_turn") or message.get("is_final") or message.get("isFinal")) and context_id:
            synthesis = self.contexts.get(context_id)
            if synthesis is not None and synthesis.t_cancel is not None:
                self._on_cancel_ack(context_id)
            self._on_done(context_id)
