"""Eleven Realtime v1 Max and Mini on ElevenLabs' OpenAI-compatible socket.

The endpoint takes ``xi-api-key`` and selects the model from ``session.update``,
not the URL. Audio is 24 kHz PCM16 in both directions. The OpenAI adapter's
event mapping already accepts missing status details, null usage and unknown
events without Pipecat's strict parsing; no framework sits in this path.
Delivery tags in completed agent transcripts are not spoken and are removed.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import websockets

from service import events as ev
from service.adapters.base import AdapterError
from service.adapters.openai_realtime import (
    CONNECT_TIMEOUT_S,
    OpenAIRealtimeAdapter,
    _AGENT_TEXT_DONE,
)

URL = "wss://api.elevenlabs.io/v1/convai/realtime"
DEFAULT_MODEL = "eleven_realtime_v1_max"
_TAG = re.compile(r"\[[a-z][a-z \-]{2,30}\] ?")


class ElevenLabsRealtimeAdapter(OpenAIRealtimeAdapter):
    name = "elevenlabs-realtime"
    url = URL

    def __init__(self, *, model: str = DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._audio_carry = b""

    def _session_payload(self) -> dict[str, Any]:
        session = super()._session_payload()
        session["model"] = self.model
        return session

    async def connect(self) -> None:
        try:
            self._ws = await websockets.connect(
                self.url,
                additional_headers={"xi-api-key": self._api_key},
                open_timeout=CONNECT_TIMEOUT_S,
                max_size=None,
            )
            first = json.loads(await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT_S))
            self.log.raw(first)
            if first.get("type") != "session.created":
                raise AdapterError(f"{self.name} refused the session: {first}")
            self.session_id = first["session"].get("id")
            self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model)
            self.session_sent = self._session_payload()
            await self._send_json({"type": "session.update", "session": self.session_sent})
            await asyncio.wait_for(self._acknowledge_session(), CONNECT_TIMEOUT_S)
            self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
            if self.config.first_message:
                await self._send_json({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": self.config.first_message}],
                    },
                })
        except Exception as exc:  # noqa: BLE001 -- connect failures are unscored
            await self.close()
            if isinstance(exc, AdapterError):
                raise
            raise AdapterError(f"{self.name} connect/session failed: {exc}") from exc

    async def _acknowledge_session(self) -> None:
        assert self._ws is not None
        while not self._configured.is_set():
            payload = json.loads(await self._ws.recv())
            self._handle(payload)
            if payload.get("type") == "error":
                raise AdapterError(f"{self.name} refused session.update: {payload.get('error')}")

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        # Eleven closes the session on a half-sample append. Preserve alignment
        # across chunk boundaries rather than padding or discarding caller audio.
        pcm = self._audio_carry + pcm
        whole = len(pcm) - len(pcm) % 2
        self._audio_carry = pcm[whole:]
        if whole:
            await super()._send_audio_chunk(pcm[:whole])

    def _handle(self, payload: dict[str, Any]) -> None:
        if payload.get("type") in _AGENT_TEXT_DONE:
            # Keep the wire frame intact for audit; normalise only the transcript.
            self.log.raw(payload)
            text = _TAG.sub("", payload.get("transcript") or payload.get("text") or "")
            self.agent_text.append(text)
            self.log.emit(ev.AGENT_TRANSCRIPT, text=text)
        else:
            super()._handle(payload)

    async def close(self) -> None:
        self._audio_carry = b""
        await super().close()
