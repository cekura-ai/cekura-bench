"""Bland TTS over its realtime websocket, one turn at a time.

``init`` fixes the voice and the PCM format for the socket and is answered by
``ready`` before t0. The voice decides the model (a BTTS_V3 voice renders on
V3), so there is no model parameter. Text goes in as ``speak`` deltas tagged
with a ``context_id`` and ``end_of_turn`` flushes the rest.

Audio comes back as raw binary frames with no tag, so a frame belongs to the
turn in flight: a new ``context_id`` preempts the active turn, and the probes
run contexts one after another. ``utterance_end`` follows the last frame and
says why the turn ended; ``cancel`` is answered by one with reason
``cancelled``.

Each socket holds one of the organisation's speech concurrency slots from
``init`` to ``close``, so ``close`` is sent and its ``done`` awaited before the
socket goes; an exhausted pool is answered with ``rate_limited``.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import websockets

from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import AdapterError

READY_TIMEOUT_S = 10.0
CLOSE_TIMEOUT_S = 2.0


class BlandAdapter(WebSocketAdapter):
    name: ClassVar[str] = "bland-tts"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 24000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = True     # encoding mulaw, fixed at 8000
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, init frame and ready acknowledgement"
    url = "wss://api.bland.ai/v2/tts/ws"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ready = asyncio.Event()
        self._settled = asyncio.Event()
        self._session: dict[str, Any] = {}
        self._refused: str | None = None
        self._active: str | None = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _connect(self) -> None:
        await super()._connect()
        try:
            await self._send_json({"type": "init", "voice": self.config.voice,
                                   "audio": {"encoding": "pcm_s16le", "sample_rate": self.config.sample_rate}})
            try:
                await asyncio.wait_for(self._ready.wait(), READY_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                raise AdapterError("bland-tts did not answer init with ready") from exc
            if self._refused:
                raise AdapterError(f"bland-tts refused init: {self._refused}")
            served = (self._session.get("encoding"), self._session.get("sample_rate"))
            if served != ("pcm_s16le", self.config.sample_rate):
                raise AdapterError(f"bland-tts is serving {served}, not pcm_s16le at {self.config.sample_rate}")
        except BaseException:
            await super()._close()          # connect() failed, so no close() will follow
            raise

    async def _close(self) -> None:
        if self._ws is not None and self._ready.is_set() and not self._refused:
            try:
                await self._send_json({"type": "close"})
                await asyncio.wait_for(self._settled.wait(), CLOSE_TIMEOUT_S)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                pass
        await super()._close()

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id
        self.contexts[context_id].meta["session_id"] = self._session.get("session_id")

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"type": "speak", "context_id": context_id, "text": text})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"type": "end_of_turn", "context_id": context_id})

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"type": "cancel", "context_id": context_id})

    def _on_binary(self, data: bytes) -> None:
        if self._active:
            self._on_audio(self._active, data)

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        context_id = message.get("context_id")
        if kind == "ready":
            self._session = message
            self._ready.set()
        elif kind == "utterance_end" and context_id:
            reason = message.get("reason")
            if reason == "failed":
                self._on_error(context_id, f"utterance failed: {message}")
                return
            if reason == "cancelled":
                self._on_cancel_ack(context_id)
            self._on_done(context_id, reason=reason, server_frames=message.get("frames"),
                          server_duration_ms=message.get("duration_ms"))
        elif kind == "error":
            code = str(message.get("code") or "")
            error = f"{code}: {message.get('message')}"
            if code.startswith("AUTH_"):
                error = f"unauthorized: {error}"
            if not self._ready.is_set():
                self._refused = error
                self._ready.set()
            self._on_error(context_id, error)
        elif kind == "done":
            self._settled.set()

    def _on_closed(self, reason: str) -> None:
        super()._on_closed(reason)
        if not self._ready.is_set():
            self._refused = f"closed before ready: {reason}"
            self._ready.set()
        self._settled.set()
