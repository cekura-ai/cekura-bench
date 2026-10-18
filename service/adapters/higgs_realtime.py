"""Boson AI Higgs Realtime over ``/v1/realtime``.

Higgs Realtime speaks OpenAI's GA Realtime protocol, so this adapter inherits
the event handling and overrides only what differs: Boson's endpoint and model,
the session naming the model, Boson's own input transcription model, and the
handshake: Boson sends nothing until the client speaks, then answers the first
``session.update`` with a ``session.created`` that already carries the applied
settings (no ``session.updated`` follows), so the update goes first and that
reply is the acknowledgement.
Turn detection is the provider's default (``server_vad``) unless the config
asks for ``semantic_vad``, which Higgs also offers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets

from service import events as ev
from service.adapters.base import AdapterError
from service.adapters.openai_realtime import CONNECT_TIMEOUT_S, OpenAIRealtimeAdapter

URL = "wss://api.boson.ai/v1/realtime"
TRANSCRIPTION_MODEL = "higgs-stt-3.1"


class HiggsRealtimeAdapter(OpenAIRealtimeAdapter):
    name = "higgs-realtime"
    input_rate = 24000
    output_rate = 24000
    supports_manual_commit = True
    supports_text_modality = True

    url = URL

    def __init__(self, *, model: str = "higgs-realtime", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)

    async def connect(self) -> None:
        try:
            self._ws = await websockets.connect(
                self.url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                open_timeout=CONNECT_TIMEOUT_S,
                max_size=None,
            )
        except Exception as exc:  # noqa: BLE001 -- the reason is what we want to publish
            raise AdapterError(f"{self.name} connect failed: {exc}") from exc

        self.session_sent = self._session_payload()
        await self._send_json({"type": "session.update", "session": self.session_sent})
        try:
            first = json.loads(await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT_S))
        except asyncio.TimeoutError as exc:
            raise AdapterError(f"{self.name} never acknowledged session.update") from exc
        self.log.raw(first)
        if first.get("type") not in ("session.created", "session.updated"):
            raise AdapterError(f"{self.name} refused the session: {first}")
        self.session_id = first["session"].get("id")
        self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model)
        self.session_ack = first.get("session")
        self._configured.set()
        self.log.emit(ev.SESSION_CONFIGURED, turn_detection=self.config.turn_detection.label)
        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        if self.config.first_message:
            await self._send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": self.config.first_message}],
                    },
                }
            )

    def _session_payload(self) -> dict[str, Any]:
        session = super()._session_payload()
        session["model"] = self.model
        transcription = session.get("audio", {}).get("input", {}).get("transcription")
        if transcription is not None:
            transcription["model"] = TRANSCRIPTION_MODEL
        return session
