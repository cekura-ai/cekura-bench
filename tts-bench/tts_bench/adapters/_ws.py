"""Shared websocket plumbing: connect, a receive loop, raw-frame logging."""

from __future__ import annotations

import asyncio
import json
from typing import Any, ClassVar

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
        self.log.raw(self._logged(payload), direction="out")
        await self._ws.send(json.dumps(payload))

    def _logged(self, payload: dict[str, Any]) -> dict[str, Any]:
        """What the raw log records of an outgoing frame; a protocol that sends a credential in a frame redacts it here."""
        return payload

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

    AUDIO_KEYS: ClassVar[frozenset[str]] = frozenset({"audio", "data", "delta", "audioContent"})

    @classmethod
    def _without_audio(cls, message: Any) -> Any:
        """The frame with every audio payload replaced by its size, however deeply the protocol nests it."""
        if isinstance(message, dict):
            return {key: (f"<{len(value)} b64 chars>" if key in cls.AUDIO_KEYS and isinstance(value, str) and len(value) > 64
                          else cls._without_audio(value)) for key, value in message.items()}
        if isinstance(message, list):
            return [cls._without_audio(value) for value in message]
        return message

    def _on_binary(self, data: bytes) -> None:
        pass

    def _on_message(self, message: dict[str, Any]) -> None:
        pass

    def _on_closed(self, reason: str) -> None:
        for synthesis in self.contexts.values():
            if not synthesis.done.is_set():
                synthesis.meta.setdefault("closed", reason)
                synthesis.done.set()
