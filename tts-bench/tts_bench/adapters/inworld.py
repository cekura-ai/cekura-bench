"""Inworld TTS over its bidirectional streaming websocket.

One socket, contexts by ``contextId``. A context is created with a ``create``
frame naming voice, model and output format, and the server acknowledges it with
``contextCreated``; the adapter waits for that acknowledgement before t0, so
context setup is excluded like every other protocol's setup frame. Text goes in
with ``send_text`` frames, ``flush_context`` ends the input, and
``flushCompleted`` follows the last audio of the flush.

Audio is requested as raw ``PCM`` (s16le, no header). ``LINEAR16`` would wrap
every chunk in its own WAV header; a header that still arrives is stripped so
the timeline holds samples only.

There is no cancel. ``close_context`` is documented as a flush followed by a
close: whatever the context was already asked to say is still generated and
delivered (measured: about 14 s of a 15 s answer after a close sent 300 ms into
the audio), so cancel is a declared exclusion rather than a measured one.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, ClassVar

from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import AdapterError

CREATE_TIMEOUT_S = 10.0


def pcm_from_chunk(chunk: bytes) -> bytes:
    """Raw PCM from one chunk: the WAV header dropped, a chunk without one passed through."""
    if chunk[:4] != b"RIFF":
        return chunk
    data = chunk.find(b"data")
    return chunk[data + 8:] if data != -1 else chunk


class InworldAdapter(WebSocketAdapter):
    name: ClassVar[str] = "inworld"
    supports_cancel: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # audioEncoding MULAW, 8 kHz
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, context create frame and its acknowledgement"
    url = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._created: dict[str, asyncio.Event] = {}

    def _headers(self) -> dict[str, str]:
        # The portal credential is already the base64 Basic token; re-encoding it fails auth.
        return {"Authorization": f"Basic {self.api_key}"}

    async def _open_context(self, context_id: str) -> None:
        created = self._created[context_id] = asyncio.Event()
        await self._send_json({
            "create": {
                "voiceId": self.config.voice, "modelId": self.config.model,
                "audioConfig": {"audioEncoding": "PCM", "sampleRateHertz": self.config.sample_rate},
            },
            "contextId": context_id,
        })
        try:
            await asyncio.wait_for(created.wait(), CREATE_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise AdapterError(f"inworld did not acknowledge context {context_id}") from exc
        synthesis = self.contexts[context_id]
        if synthesis.error:
            raise AdapterError(f"inworld refused context {context_id}: {synthesis.error}")

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"send_text": {"text": text}, "contextId": context_id})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"flush_context": {}, "contextId": context_id})

    def _on_message(self, message: dict[str, Any]) -> None:
        result = message.get("result") or {}
        context_id = result.get("contextId") or message.get("contextId")
        error = message.get("error")
        status = result.get("status") or {}
        if error or status.get("code"):
            self._on_error(context_id, str(error or status))
            if context_id in self._created:
                self._created[context_id].set()
            return
        if "contextCreated" in result and context_id in self._created:
            self._created[context_id].set()
        audio = (result.get("audioChunk") or {}).get("audioContent")
        if audio:
            self._on_audio(context_id, pcm_from_chunk(base64.b64decode(audio)))
        if "flushCompleted" in result:
            self._on_done(context_id)
        if "contextClosed" in result:
            self._on_done(context_id, closed=True)
