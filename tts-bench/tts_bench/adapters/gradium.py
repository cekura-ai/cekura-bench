"""Gradium TTS over its streaming websocket, contexts by ``client_req_id``.

Each context sends setup and waits for ``ready`` before t0. Text frames are one
utterance, with a space inserted between messages; the probes split on words
and sentences, keeping punctuation attached. ``end_of_stream`` ends the input
and the server's matching event follows the last base64 audio chunk.

``close_ws_on_eos: false`` keeps the socket for the next synthesis. The protocol
has no cancel; even an unscoped end waits for active requests to finish.
"""

from __future__ import annotations

import asyncio
import base64
import re
from typing import Any, ClassVar

from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import AdapterError

CREATE_TIMEOUT_S = 10.0


class GradiumAdapter(WebSocketAdapter):
    name: ClassVar[str] = "gradium"
    supports_cancel: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 24000, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # output_format ulaw_8000
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, context setup frame and ready acknowledgement"
    url = "wss://api.gradium.ai/api/speech/tts"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._created: dict[str, asyncio.Event] = {}

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    async def _open_context(self, context_id: str) -> None:
        created = self._created[context_id] = asyncio.Event()
        await self._send_json({
            "type": "setup", "model_name": self.config.model, "voice_id": self.config.voice,
            "output_format": "pcm" if self.config.sample_rate == 48000 else f"pcm_{self.config.sample_rate}",
            "close_ws_on_eos": False, "client_req_id": context_id,
        })
        try:
            await asyncio.wait_for(created.wait(), CREATE_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise AdapterError(f"gradium did not acknowledge context {context_id}") from exc
        synthesis = self.contexts[context_id]
        if synthesis.error:
            raise AdapterError(f"gradium refused context {context_id}: {synthesis.error}")

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"type": "text", "text": text, "client_req_id": context_id})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"type": "end_of_stream", "client_req_id": context_id})

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        context_id = message.get("client_req_id")
        if kind == "ready" and context_id in self._created:
            self.contexts[context_id].meta["ready"] = message
            self._created[context_id].set()
        elif kind == "audio" and message.get("audio"):
            self._on_audio(context_id, base64.b64decode(message["audio"]))
        elif kind == "end_of_stream":
            self._on_done(context_id)
        elif kind == "error":
            error = str(message.get("message") or message)
            # Code 1008 also covers malformed input; only account errors are refusals.
            if re.search(r"api.?key|authenticat|subscription", error, re.IGNORECASE):
                error = f"unauthorized: {error}"
            self._on_error(context_id, error)
            for key, created in self._created.items():
                if context_id is None or key == context_id:
                    created.set()

    def _on_closed(self, reason: str) -> None:
        super()._on_closed(reason)
        for context_id, created in self._created.items():
            if not created.is_set():
                self._on_error(context_id, f"closed before ready: {reason}")
                created.set()
