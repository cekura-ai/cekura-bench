"""Fish Audio realtime TTS over a MessagePack websocket, one utterance per socket.

``start`` configures raw PCM before t0, ``text`` appends text pieces, and
``stop`` drains the input. Binary MessagePack ``audio`` events contain PCM;
``finish`` with reason ``stop`` ends the synthesis and closes the socket.
The next context reconnects before t0, including the warm repeat probe.

``flush`` forces buffered text to generate without ending input. Previous
audio conditions subsequent chunks for voice consistency. There is no
documented cancel/clear operation; ``stop`` finishes synthesis rather than
interrupting it. Option ``latency`` defaults to ``balanced`` for voice agents.

Reference: https://docs.fish.audio/api-reference/endpoint/websocket/tts-live
PCM rates: https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech
"""

from __future__ import annotations

from typing import Any, ClassVar

import msgpack

from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import AdapterError


class FishAudioAdapter(WebSocketAdapter):
    name: ClassVar[str] = "fish-audio"
    supports_cancel: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 24000, 32000, 44100)
    native_mulaw_8k: ClassVar[bool] = False    # documented formats: wav, pcm, mp3, opus
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade and start configuration frame; new socket per utterance"
    url = "wss://api.fish.audio/v1/tts/live"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "model": self.config.model}

    async def _open_context(self, context_id: str) -> None:
        if self._active is not None:
            await self._close()
            await self._connect()
        self._active = context_id
        await self._send_json({"event": "start", "request": {
            "text": "", "reference_id": self.config.voice, "format": "pcm",
            "sample_rate": self.config.sample_rate, "latency": self.config.option("latency", "balanced"),
            "condition_on_previous_chunks": True,
        }})

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise AdapterError("fish-audio not connected")
        self.log.raw(self._logged(payload), direction="out")
        await self._ws.send(msgpack.packb(payload, use_bin_type=True))

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"event": "text", "text": text})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"event": "stop"})

    def _on_binary(self, data: bytes) -> None:
        message = msgpack.unpackb(data, raw=False)
        self.log.raw({key: f"<{len(value)} bytes>" if isinstance(value, bytes) else value
                      for key, value in message.items()})
        self._on_message(message)

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("event")
        if kind == "audio" and message.get("audio"):
            self._on_audio(self._active, message["audio"])
        elif kind == "finish":
            if message.get("reason") == "stop":
                self._on_done(self._active)
            else:
                self._on_error(self._active, str(message.get("message") or message))
