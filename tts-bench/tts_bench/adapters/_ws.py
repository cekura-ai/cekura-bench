"""Shared websocket plumbing: connect, a receive loop, raw-frame logging."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets

from tts_bench.adapters.base import AdapterError, TTSAdapter


class WebSocketAdapter(TTSAdapter):
    url: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ws: websockets.ClientConnection | None = None
        self._receiver: asyncio.Task | None = None

    def _headers(self) -> dict[str, str]:
        return {}

    async def _connect(self) -> None:
        try:
            self._ws = await websockets.connect(self.url, additional_headers=self._headers(), max_size=None, open_timeout=20)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"{self.name} connect failed: {exc}") from exc
        self._receiver = asyncio.create_task(self._receive_loop())

    async def _close(self) -> None:
        if self._receiver:
            self._receiver.cancel()
        if self._ws:
            await self._ws.close()

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise AdapterError(f"{self.name} not connected")
        self.log.raw(payload, direction="out")
        await self._ws.send(json.dumps(payload))

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    self._on_binary(raw)
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    self.log.raw({"unparsed": raw[:200]})
                    continue
                self.log.raw(self._without_audio(message))
                self._on_message(message)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            self.log.raw({"closed": str(exc)})
            self._on_closed(str(exc))
        except Exception as exc:  # noqa: BLE001
            self._on_error(None, f"receive loop: {exc!r}")

    @staticmethod
    def _without_audio(message: dict[str, Any]) -> dict[str, Any]:
        out = dict(message)
        for key in ("audio", "data"):
            if isinstance(out.get(key), str) and len(out[key]) > 64:
                out[key] = f"<{len(out[key])} b64 chars>"
        return out

    def _on_binary(self, data: bytes) -> None:
        pass

    def _on_message(self, message: dict[str, Any]) -> None:
        pass

    def _on_closed(self, reason: str) -> None:
        for synthesis in self.contexts.values():
            if not synthesis.done.is_set():
                synthesis.meta.setdefault("closed", reason)
                synthesis.done.set()
