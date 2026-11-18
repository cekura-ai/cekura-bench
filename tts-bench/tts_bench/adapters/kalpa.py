"""Kalpa TTS over its streaming websocket, one utterance at a time per socket.

The voice is fixed per socket in the URL (``/v1/tts/{voice_id}/stream``).
``initializeConnection`` carries the key, the model, the PCM rate and
``generation_config``, and is answered by ``sessionCreated`` with what the
session applies. Both happen before t0, and the echo is checked, so a model,
voice or rate the server did not apply is a refused session, never a
measurement.

``generation_config`` turns on server-side segmentation, which is how Kalpa's
own voice-agent integrations (LiveKit, Pipecat, Bolna) drive the socket: once
the buffered text crosses the next ``chunk_length_schedule`` threshold
(Kalpa's default ``[50, 80, 120, 150]`` characters) at a sentence boundary,
that part starts rendering while the rest is still arriving. Without it the
server generates only on flush. The option ``chunk_length_schedule`` overrides
the schedule, ``off`` omits the object, and every context records which.

Text goes in as ``sendText`` frames, joined exactly as sent; a final
``sendText`` with ``flush`` ends the utterance. The server opens one response
per utterance (``responseCreated``, possibly before the flush), streams base64
PCM in ``responseAudio`` and ends it with ``responseDone`` (``completed``).
A bare ``cancelResponse`` is the barge-in: the response ends with
``responseDone`` (``cancelled``) and the cancel is acknowledged by
``cancelled``. A ``cancelled`` response nobody cancelled, or a socket that
closes before ``responseDone``, cut the recording short: it is the synthesis's
error, never its end.

An unknown voice, a rejected key or a used-up allowance is a fatal ``error``
frame and a 44xx close before ``sessionCreated``.

Reference: https://docs.kalpalabs.ai/reference.md (WSS /v1/tts/{voice_id}/stream)
"""

from __future__ import annotations

import asyncio
import base64
import re
from typing import Any, ClassVar

import websockets

from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import AdapterError

HANDSHAKE_TIMEOUT_S = 20.0
DEFAULT_SCHEDULE = "50,80,120,150"   # Kalpa's documented default, sent explicitly so a change on their side shows
SAMPLE_WIDTH = 2

# Close codes and error types that mean the account, not the service, stopped the session; worded so
# is_refusal() reads them. ``overloaded`` (the model at capacity) is the service's own failure, so it is not here.
REFUSAL_CLOSES = {4401: "unauthorized", 4402: "allowance exhausted", 4403: "forbidden", 4429: "rate limit"}
REFUSAL_ERRORS = {"authentication_error": "unauthorized", "allowance_exhausted": "allowance exhausted",
                  "account_suspended": "forbidden", "rate_limit_exceeded": "rate limit"}


def parse_schedule(value: str) -> list[int] | None:
    """``"50,80,120,150"`` as the integer list Kalpa takes; ``"off"`` as no generation_config at all."""
    if value == "off":
        return None
    if not re.fullmatch(r"\d+(?:,\d+){0,9}", value) or not all(50 <= int(v) <= 2000 for v in value.split(",")):
        raise ValueError(f"kalpa chunk_length_schedule must be 1-10 comma-separated integers in 50-2000, or 'off', not {value!r}")
    return [int(v) for v in value.split(",")]


class KalpaAdapter(WebSocketAdapter):
    name: ClassVar[str] = "kalpa"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = False    # pcm_s16le only
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, initializeConnection and sessionCreated"
    AUDIO_KEYS: ClassVar[frozenset[str]] = WebSocketAdapter.AUDIO_KEYS | {"pcm_b64"}
    base = "wss://api.kalpalabs.ai/v1/tts"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.schedule = parse_schedule(self.config.option("chunk_length_schedule", DEFAULT_SCHEDULE))
        self._active: str | None = None
        self._responses: dict[str, str] = {}     # response_id -> context
        self._pending = b""
        self._handshake: asyncio.Future | None = None
        self.session: dict[str, Any] = {}

    @property
    def url(self) -> str:  # type: ignore[override]
        return f"{self.base}/{self.config.voice}/stream"

    def _init_frame(self) -> dict[str, Any]:
        frame: dict[str, Any] = {"type": "initializeConnection", "api_key": self.api_key, "model": self.config.model,
                                 "output_format": "pcm_s16le", "sample_rate": self.config.sample_rate}
        if self.schedule is not None:
            frame["generation_config"] = {"chunk_length_schedule": self.schedule}
        return frame

    def _logged(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "api_key": "<redacted>"} if "api_key" in payload else payload

    async def _send_json(self, payload: dict[str, Any]) -> None:
        try:
            await super()._send_json(payload)
        except websockets.ConnectionClosed as exc:
            # The server closed the socket (a fatal error closes it with 1011) and the probe moved on to its
            # next frame or next context: that synthesis failed with the close, the harness did not.
            self._on_error(self._active, f"socket closed before {payload.get('type')} was sent: {self._close_reason(str(exc))}")

    def _close_reason(self, reason: str) -> str:
        code = self._ws.close_code if self._ws is not None else None
        return f"{REFUSAL_CLOSES[code]} ({code}): {reason}" if code in REFUSAL_CLOSES else reason

    async def _refuse(self, reason: str) -> None:
        # The cell never opens, so nothing else would stop the receive loop before the cell's log closes.
        await super()._close()
        if self._receiver:
            await asyncio.gather(self._receiver, return_exceptions=True)
        raise AdapterError(f"{self.name} refused the session: {reason}")

    async def _connect(self) -> None:
        self._handshake = asyncio.get_running_loop().create_future()
        await super()._connect()
        await self._send_json(self._init_frame())
        try:
            created = await asyncio.wait_for(asyncio.shield(self._handshake), timeout=HANDSHAKE_TIMEOUT_S)
        except asyncio.TimeoutError:
            created = {"error": f"no sessionCreated within {HANDSHAKE_TIMEOUT_S:g} s"}
        if "error" in created:
            await self._refuse(created["error"])
        expected = {"model": self.config.model, "voice_id": self.config.voice, "sample_rate": self.config.sample_rate,
                    "output_format": "pcm_s16le"}
        wrong = {k: created.get(k) for k, v in expected.items() if created.get(k) != v}
        if wrong:
            await self._refuse(f"sessionCreated did not apply the requested settings: {wrong}")
        self.session = created

    async def _open_context(self, context_id: str) -> None:
        self._active = context_id
        self._pending = b""
        self.contexts[context_id].meta["session"] = {
            k: self.session.get(k) for k in ("session_id", "model", "voice_id", "audio_quality")}
        self.contexts[context_id].meta["chunk_length_schedule"] = self.schedule or "off"

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"type": "sendText", "text": text})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"type": "sendText", "text": "", "flush": True})

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"type": "cancelResponse"})

    def _context_for(self, message: dict[str, Any]) -> str | None:
        return self._responses.get(message.get("response_id") or "", self._active)

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "sessionCreated":
            if self._handshake and not self._handshake.done():
                self._handshake.set_result(message)
        elif kind == "responseCreated" and self._active:
            self._responses[message.get("response_id") or ""] = self._active
            self.contexts[self._active].meta["response_id"] = message.get("response_id")
        elif kind == "responseAudio":
            context_id = self._context_for(message)
            samples = self._pending + base64.b64decode(message.get("pcm_b64") or "")
            whole = len(samples) - len(samples) % SAMPLE_WIDTH
            self._pending = samples[whole:]
            if context_id:
                self._on_audio(context_id, samples[:whole])
                self.contexts[context_id].meta["last_frame"] = message.get("frame")
        elif kind == "responseDone":
            self._on_response_done(message)
        elif kind == "cancelled" and self._active:
            self._on_cancel_ack(self._active)
        elif kind == "error":
            error = message.get("error") or {}
            text = f"{error.get('type')}: {error.get('message')}" if isinstance(error, dict) else str(error)
            if isinstance(error, dict) and error.get("type") in REFUSAL_ERRORS:
                text = f"{REFUSAL_ERRORS[error['type']]} ({text})"
            if self._handshake and not self._handshake.done():
                self._handshake.set_result({"error": text})
                return
            self._on_error(self._context_for(message), text[:500])

    def _on_response_done(self, message: dict[str, Any]) -> None:
        context_id = self._context_for(message)
        if context_id not in self.contexts:
            return
        synthesis = self.contexts[context_id]
        status = message.get("status")
        left, self._pending = self._pending, b""
        meta = {"status": status, "usage": message.get("usage"), "server_meta": message.get("meta"),
                "response_text_chars": len(message.get("text") or ""),
                **({"trailing_bytes_dropped": len(left)} if left else {})}
        if status == "completed" or (status == "cancelled" and synthesis.t_cancel is not None):
            self._on_done(context_id, **meta)
            return
        synthesis.meta.update(meta)
        self._on_error(context_id, f"responseDone with status {status!r} and no cancel sent")

    async def _receive_loop(self) -> None:
        await super()._receive_loop()
        # A clean close (1000) ends the loop without the exception the base reports.
        self._on_closed("closed by server")

    def _on_closed(self, reason: str) -> None:
        reason = self._close_reason(reason)
        if self._handshake and not self._handshake.done():
            self._handshake.set_result({"error": f"closed before sessionCreated: {reason}"})
        # A close before responseDone (a fatal error is a 1011) cut the recording short: it is the
        # synthesis's error, never its end. After a cancel nothing more was owed.
        for context_id, synthesis in self.contexts.items():
            if synthesis.done.is_set():
                continue
            if synthesis.t_cancel is not None:
                synthesis.meta.setdefault("closed", reason)
                synthesis.done.set()
            else:
                self._on_error(context_id, f"closed before responseDone: {reason}")
