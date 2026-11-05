"""Mistral Voxtral TTS over its realtime websocket, one generation at a time per socket.

The server opens with ``session.created`` and its defaults; ``session.update``
sets the voice, raw ``pcm`` output and the streaming mode, and is answered by
``session.updated`` with the settings the session applies. Both happen before
t0, and the echo is checked, so a voice or format the server did not apply is a
refused session, never a measurement. The echo also names the served build
(``voxtral-mini-tts-3-0`` resolves to a build such as ``-3-0-13``), which every
synthesis records.

Text goes in as ``input_text.append`` deltas, joined exactly as sent;
``input_text.flush`` ends the generation and is answered by
``speech.audio.done`` once its last audio has been sent. Audio can arrive
before the flush once the server has enough text. The socket has no context
ids, so contexts run one after another, and the next generation starts with
the next append. There is no cancel message: ``input_text.end`` drains what
was sent and closes the socket.

``pcm`` arrives as little-endian 32-bit float at 24 kHz mono, not the 16-bit
integers every other row records, so it is converted as it arrives; a float
can straddle two deltas, so the remainder is carried to the next one.

An unknown voice is not refused at ``session.update``: the server closes the
socket with 1008 once text arrives. A rejected key is HTTP 401 at the upgrade.

Reference: https://docs.mistral.ai/studio/audio/text-to-speech/speak-in-realtime
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, ClassVar
from urllib.parse import urlencode

import numpy as np

from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import AdapterError

FLOAT_WIDTH = 4
HANDSHAKE_TIMEOUT_S = 20.0


def to_int16(samples: bytes) -> bytes:
    """Little-endian float32 in [-1, 1] as little-endian int16, clipped rather than wrapped."""
    floats = np.frombuffer(samples, dtype="<f4")
    return (np.clip(floats, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


class MistralRealtimeAdapter(WebSocketAdapter):
    name: ClassVar[str] = "mistral-realtime"
    supports_cancel: ClassVar[bool] = False
    native_rates: ClassVar[tuple[int, ...]] = (24000,)
    native_mulaw_8k: ClassVar[bool] = False    # pcm, wav, mp3, flac, opus; all 24 kHz
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, session.created and session.update"
    AUDIO_KEYS: ClassVar[frozenset[str]] = WebSocketAdapter.AUDIO_KEYS | {"audio_data"}
    base = "wss://api.mistral.ai/v1/audio/speech/realtime"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active: str | None = None
        self._pending = b""
        self._handshake: asyncio.Future | None = None
        self.session: dict[str, Any] = {}

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}?" + urlencode({"model": self.config.model})

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _settings(self) -> dict[str, str]:
        """What session.update asks for, and what session.updated must echo."""
        return {"voice": self.config.voice, "response_format": "pcm",
                "streaming_mode": self.config.option("streaming_mode", "responsive")}

    async def _await_handshake(self, stage: str) -> dict[str, Any]:
        assert self._handshake is not None
        try:
            return await asyncio.wait_for(asyncio.shield(self._handshake), timeout=HANDSHAKE_TIMEOUT_S)
        except asyncio.TimeoutError:
            return {"error": f"no {stage} within {HANDSHAKE_TIMEOUT_S:g} s"}

    async def _refuse(self, reason: str) -> None:
        # The cell never opens, so nothing else would stop the receive loop before the cell's log closes.
        await super()._close()
        if self._receiver:
            await asyncio.gather(self._receiver, return_exceptions=True)
        raise AdapterError(f"{self.name} refused the session: {reason}")

    async def _connect(self) -> None:
        loop = asyncio.get_running_loop()
        self._handshake = loop.create_future()
        await super()._connect()
        created = await self._await_handshake("session.created")
        if "error" in created:
            await self._refuse(created["error"])
        self._handshake = loop.create_future()
        await self._send_json({"type": "session.update", "session": self._settings()})
        updated = await self._await_handshake("session.updated")
        if "error" in updated:
            await self._refuse(updated["error"])
        session = updated.get("session") or {}
        wrong = {k: session.get(k) for k, v in self._settings().items() if session.get(k) != v}
        if not str(session.get("model", "")).startswith(self.config.model):
            wrong["model"] = session.get("model")
        if wrong:
            await self._refuse(f"session.updated did not apply the requested settings: {wrong}")
        self.session = session

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id
        self._pending = b""
        self.contexts[context_id].meta["session"] = {
            k: self.session.get(k) for k in ("model", "voice", "streaming_mode", "response_format", "expressiveness")}

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"type": "input_text.append", "text": text})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"type": "input_text.flush"})

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind in ("session.created", "session.updated"):
            if self._handshake and not self._handshake.done():
                self._handshake.set_result(message)
        elif kind == "speech.audio.delta" and self._active:
            samples = self._pending + base64.b64decode(message.get("audio_data") or "")
            whole = len(samples) - len(samples) % FLOAT_WIDTH
            self._pending = samples[whole:]
            self._on_audio(self._active, to_int16(samples[:whole]))
        elif kind == "speech.audio.done" and self._active:
            left, self._pending = self._pending, b""
            self._on_done(self._active, usage=message.get("usage"),
                          **({"trailing_bytes_dropped": len(left)} if left else {}))
        elif kind == "error":
            error = message.get("error") or {}
            text = f"{error.get('code')}: {error.get('message')}" if isinstance(error, dict) else str(error)
            if self._handshake and not self._handshake.done():
                self._handshake.set_result({"error": text})
            self._on_error(self._active, text[:500])

    def _on_closed(self, reason: str) -> None:
        if self._handshake and not self._handshake.done():
            self._handshake.set_result({"error": f"closed: {reason}"})
        # A close before speech.audio.done (an unknown voice is a 1008 once text arrives, a failed
        # generation a 1011) cut the recording short: it is the generation's error, never its end.
        for context_id, synthesis in self.contexts.items():
            if not synthesis.done.is_set():
                self._on_error(context_id, f"closed before speech.audio.done: {reason}")
