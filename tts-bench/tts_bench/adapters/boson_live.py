"""Boson AI Higgs TTS over its speech websocket, one turn at a time per socket.

``wss://api.boson.ai/v1/audio/speech/live`` takes the API key as a Bearer
header. A ``start`` frame opens the session with its speech settings and is
sent before t0. The server sends nothing back for an accepted ``start``, so
there is no acknowledgement to wait for; a refused one is an ``error`` and a
close. Measured 2026-11-26: text sent at once after ``start`` and text sent a
second later reached first audio in the same time, so not waiting costs the
turn nothing. The settings are the ones Boson asked the bench to run: model,
the ``default`` voice, Boson's text normalisation on, and language ``eng``
(ISO 639-3, used by the normalisation). Word timestamps are left off.

``text`` frames append to the turn, joined exactly as sent; ``flush`` ends it
and is answered by ``flushed`` once the turn's last ``audio`` event is out.
The socket has no context ids, so contexts run one after another and the next
turn starts with the next ``text``. ``cancel`` drops the text not yet spoken
and stops the audio in progress; ``cancelled`` acknowledges it, and every
audio event before it belongs to the cancelled turn.

``audio`` events carry base64 16-bit mono PCM at 24 kHz. An event can end
inside a sample, so the odd byte is carried to the next event; one left at
``flushed`` is recorded and dropped, and one left at ``cancelled`` is dropped.

The server paces delivery: a turn's audio comes as fast as it is generated
until the client holds one second more than it could have played, then at
playback speed in 40 ms events. Delivery speed is therefore close to
realtime on any turn longer than a second, and the run takes about as long
as its audio.

Errors are a flat ``{"type": "error", "code", "message"}``. The codes that
name the account (key, rate, concurrency and open-session limits, quota,
billing) are refusals: the cell is void and a resume runs it again. Any
other error, and a close before ``flushed`` that nobody cancelled, is a
failure of the turn. An unknown voice is not refused at ``start``: it fails
the first turn as ``internal_error`` with close 1011, so it fails every cell
of a run rather than producing a measurement.

Options: ``normalize`` (``true``/``false``, default ``true``), ``language``
(ISO 639-3, default ``eng``).

Reference: https://docs.boson.ai/api-reference/tts/speech-live
Pacing: https://docs.boson.ai/models/higgs-tts/streaming#audio-pacing
"""

from __future__ import annotations

import base64
import re
from typing import Any, ClassVar

from tts_bench import events as ev
from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import AdapterError
from tts_bench.common.audio import SAMPLE_WIDTH

# Boson's error codes that name the account rather than the service, each with
# the words that make it read as a refusal (``base.is_refusal``).
ACCOUNT_REFUSALS = {
    "invalid_api_key": "invalid API key",
    "rate_limit_exceeded": "rate limit",
    "concurrency_limit_exceeded": "concurrency limit",
    "session_limit_exceeded": "too many concurrent sessions",
    "insufficient_quota": "quota",
    "monthly_cap_reached": "quota (monthly cap)",
    "contract_ended": "billing (contract ended)",
    "no_billing_account": "billing",
}


class BosonLiveAdapter(WebSocketAdapter):
    name: ClassVar[str] = "boson-live"
    native_rates: ClassVar[tuple[int, ...]] = (24000,)
    native_mulaw_8k: ClassVar[bool] = False    # pcm only
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, start frame (not acknowledged by the server)"
    base = "wss://api.boson.ai/v1/audio/speech/live"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        normalize = self.config.option("normalize", "true")
        if normalize not in ("true", "false"):
            raise ValueError(f"boson-live normalize must be 'true' or 'false', not {normalize!r}")
        self.language = self.config.option("language", "eng")
        if not re.fullmatch(r"[a-z]{3}", self.language or ""):
            raise ValueError(f"boson-live language must be an ISO 639-3 code such as 'eng', not {self.language!r}")
        self.request = {"model": self.config.model, "voice": self.config.voice,
                        "normalize": normalize == "true", "language": self.language}
        self._active: str | None = None
        self._carry = b""
        self._session_error: str | None = None

    @property
    def url(self) -> str:  # type: ignore[override]
        return self.base

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _connect(self) -> None:
        await super()._connect()
        await self._send_json({"type": "start", "request": self.request})

    async def _send_json(self, payload: dict[str, Any]) -> None:
        try:
            await super()._send_json(payload)
        except AdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 -- the server closed the socket: say why, if it did
            raise AdapterError(f"{self.name} refused the session: {self._session_error or exc}") from exc

    async def _open_context(self, context_id: str) -> None:
        if self._session_error:
            raise AdapterError(f"{self.name} refused the session: {self._session_error}")
        self._active = context_id
        self._carry = b""
        self.contexts[context_id].meta["request"] = dict(self.request)

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        await self._send_json({"type": "text", "text": text})

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"type": "flush"})

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"type": "cancel"})

    def _turn(self) -> str | None:
        """The turn still in progress, if any: an event after a turn ended belongs to no turn."""
        synthesis = self.contexts.get(self._active) if self._active else None
        return self._active if synthesis is not None and not synthesis.done.is_set() else None

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        turn = self._turn()
        if kind == "audio" and turn:
            pcm = self._carry + base64.b64decode(message.get("data") or "")
            whole = len(pcm) - len(pcm) % SAMPLE_WIDTH
            self._carry = pcm[whole:]
            self._on_audio(turn, pcm[:whole])
        elif kind == "flushed" and turn:
            left, self._carry = self._carry, b""
            self._on_done(turn, **({"trailing_bytes_dropped": len(left)} if left else {}))
        elif kind == "cancelled" and turn:
            self._carry = b""
            self._on_cancel_ack(turn)
            self._on_done(turn, cancelled=True)
        elif kind == "error":
            code = str(message.get("code"))
            text = f"{code}: {message.get('message')}"[:500]
            if code in ACCOUNT_REFUSALS:
                text += f" [account refusal: {ACCOUNT_REFUSALS[code]}]"
            if turn:
                self._on_error(turn, text)
            else:
                # No turn to fail: the session is refused, and the next context says so before t0.
                self._session_error = self._session_error or text
                self.log.emit(ev.PROVIDER_ERROR, context_id=None, message=text)

    def _on_closed(self, reason: str) -> None:
        self._session_error = self._session_error or f"closed: {reason}"
        turn = self._turn()
        if turn and self.contexts[turn].t_cancel is None:
            # A close before flushed (a 1011 server error, a refusal's 1008) cut the turn short: its error, not its end.
            self._on_error(turn, f"closed before flushed: {reason}")
        super()._on_closed(reason)
