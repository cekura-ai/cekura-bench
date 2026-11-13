"""Reson8 end-of-turn detection (``/v1/speech-to-text/turns``), model rumble-2.

Reson8's turns endpoint is a streaming STT with built-in end-of-turn
prediction. It hears one speaker channel and emits:

* ``turn_start``          -> ``interruption`` (speech-start semantics, exactly
  as ``cascade_flux`` maps Flux's ``StartOfTurn``: the model hears one channel
  and has no notion of who holds the floor).
* ``turn_end_candidate``  -> recorded, not mapped. Sent when the turn-end
  probability crosses ``eager_turn_probability`` (interim) and again, final,
  at ``final_turn_probability`` right before ``turn_end``.
* ``turn_end``            -> ``eot``. It confirms the previous candidate as
  the final end of the turn, so it is the decision a voice agent acts on.

Track 1 (end of turn) only. Reson8 is STT-only, so Track 2 (barge-in and
backchannel, which needs a speaking agent) does not apply.

Commit time, real-time pacing, resampling (48 kHz -> 16 kHz) and the
background event loop are ``cascade_flux``'s, inherited unchanged: an event
is committed at the audio time already *sent* when its message arrives, so
network and server latency count against the model, and the adapter refuses
to run unless frames arrive at wall-clock pace (``--realtime``).

Host pacing slips are recorded exactly as in ``cascade_flux`` (inherited):
``diagnostics()["pacing"]``, with ``pacing_void`` past ``VOID_SLIP_MS``.

Protocol (https://docs.reson8.dev/api/speech-to-text/turns/, checked
2026-11-06; the global endpoint serving rumble-2 checked by a handshake
probe on 2026-11-07): ``Authorization: ApiKey <key>``; query parameters ``model``,
``encoding``, ``sample_rate``, ``channels``, ``language``,
``eager_turn_probability``, ``final_turn_probability``, ``include_timestamps``,
``filler_mode``; binary PCM frames. There is no ready message: a successful
HTTP upgrade is readiness (as in stt-bench's Reson8 realtime protocol).
Handshake rejections carry the reason in ``X-Error-Message``; after
connecting, errors arrive as close codes (1011, 4000). At the end of the audio
the adapter does NOT send ``flush_request``: that would force a ``turn_end``
the model did not decide. It waits ``flush_grace_s`` for trailing events and
closes.

The API key is ``RESON_API_KEY`` (stt-bench's ``reson8`` credential) from the
process environment or ``env_file`` (default ``turn-taking-bench/.env``); it is never
logged or recorded.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from turn_taking_bench.adapters import cascade_flux
from turn_taking_bench.adapters.cascade_flux import AiohttpSocket, CascadeFlux, Connector, Decimator, Socket
from turn_taking_bench.model import FRAME_S

MODEL_NAME = "reson8_turns"
API_KEY_NAME = "RESON_API_KEY"

DEFAULT_CONFIG: dict[str, Any] = {
    # Reson8's US/global deployment, which serves only rumble-2.
    "endpoint": "wss://api.global.reson8.dev/v1/speech-to-text/turns",
    "model": "rumble-2",
    "encoding": "pcm_s16le",
    "sample_rate": 16000,
    "channels": 1,
    "language": "en",  # docs: recommended; the benchmark audio is English
    # Turn detection: Reson8's documented defaults, sent explicitly so the run
    # records exactly what the server was asked for. Configurable per run.
    "eager_turn_probability": 0.5,
    "final_turn_probability": 0.92,
    "filler_mode": "natural",  # documented default
    "include_timestamps": True,  # recorded in diagnostics only
    "ready_timeout_s": 15.0,
    "close_timeout_s": 5.0,
    "flush_grace_s": 1.0,
    "max_ahead_s": 0.5,
    "env_file": None,  # default: turn-taking-bench/.env
}

TURN_EVENT_KIND = {"turn_end": "eot", "turn_start": "interruption"}
QUERY_KEYS = ("model", "encoding", "sample_rate", "channels", "language", "eager_turn_probability",
              "final_turn_probability", "filler_mode", "include_timestamps")


def read_api_key(env_file: str | os.PathLike[str] | None) -> str:
    """RESON_API_KEY: process environment first, then a dotenv file."""
    value = os.environ.get(API_KEY_NAME, "").strip()
    if value:
        return value
    path = Path(env_file) if env_file else cascade_flux.TURN_TAKING_BENCH_ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].strip()
            name, sep, raw = line.partition("=")
            if sep and name.strip() == API_KEY_NAME:
                raw = raw.strip()
                if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
                    raw = raw[1:-1]
                if raw.strip():
                    return raw.strip()
    raise RuntimeError(f"{API_KEY_NAME} not set (environment or {path})")


def query_url(config: dict[str, Any]) -> str:
    query: dict[str, Any] = {}
    for key in QUERY_KEYS:
        value = config.get(key)
        if value is None:
            continue
        query[key] = ("true" if value else "false") if isinstance(value, bool) else value
    return config["endpoint"] + "?" + urlencode(query)


class Reson8Socket(AiohttpSocket):
    @property
    def close_code(self) -> int | None:
        return self._ws.close_code


async def aiohttp_connect(url: str, headers: dict[str, str], timeout_s: float) -> Socket:
    import aiohttp

    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=timeout_s))
    try:
        ws = await session.ws_connect(url, headers=headers, heartbeat=None, max_msg_size=8 * 1024 * 1024)
    except aiohttp.WSServerHandshakeError as exc:
        await session.close()
        reason = (exc.headers or {}).get("X-Error-Message", "") if exc.headers else ""
        # Status and server reason only; never the request headers.
        raise RuntimeError(f"Reson8 handshake failed: HTTP {exc.status} {reason or exc.message}") from None
    except BaseException:
        await session.close()
        raise
    return Reson8Socket(session, ws)


class Reson8Turns(CascadeFlux):
    """One Reson8 turns session per speaker channel. ``push``/``flush`` and the
    background loop are ``CascadeFlux``'s; only the session setup and the wire
    protocol differ."""

    def __init__(self, config: dict[str, Any], connector: Connector | None = None) -> None:
        super().__init__({**DEFAULT_CONFIG, **config}, connector=connector or aiohttp_connect)
        self.config = {**DEFAULT_CONFIG, **config}

    def reset(self, sample_rate: int) -> None:
        self._shutdown()
        send_rate = int(self.config["sample_rate"])
        if sample_rate % send_rate != 0:
            raise ValueError(f"reson8_turns cannot stream {sample_rate} Hz audio (needs a multiple of {send_rate})")
        factor = sample_rate // send_rate
        self._decimate = Decimator(factor)
        self._frame_samples = int(FRAME_S * sample_rate)
        if self._key is None:
            self._key = read_api_key(self.config.get("env_file"))
        url = query_url(self.config)
        self._messages = []
        self._channel = {"input_sample_rate": sample_rate, "sent_sample_rate": send_rate,
                         "decimation_factor": factor, "url": url}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="reson8-turns", daemon=True)
        self._thread.start()
        self._session = _Reson8Session(self._loop, self._messages)
        ready = float(self.config["ready_timeout_s"])
        self._run(self._session.open(self._connector, url, {"Authorization": f"ApiKey {self._key}"}, ready),
                  timeout=ready + 5)
        self._first_push = None
        self._reset_pacing()

    def push(self, frame, t_end_s):
        try:
            return super().push(frame, t_end_s)
        except RuntimeError as exc:
            raise RuntimeError(str(exc).replace("cascade_flux", MODEL_NAME)) from None

    def diagnostics(self) -> dict[str, Any]:
        return {"reson8": dict(self._channel), "pacing": self._pacing(), "messages": list(self._messages)}


class _Reson8Session(cascade_flux._Session):
    async def open(self, connector: Connector, url: str, headers: dict[str, str], ready_timeout: float) -> None:
        # No ready message exists: the successful upgrade is readiness.
        self.socket = await asyncio.wait_for(connector(url, headers, ready_timeout), ready_timeout)
        self.t0 = time.monotonic()
        self._tasks = [asyncio.create_task(self._receive()), asyncio.create_task(self._send())]

    async def _receive(self) -> None:
        try:
            async for raw in self.socket:
                received = time.monotonic() - self.t0
                sent = self.audio_sent_s
                try:
                    message = json.loads(raw)
                except ValueError:
                    message = {"unparsed": str(raw)[:500]}
                self.messages.append({"received_s": round(received, 4), "audio_sent_s": sent, "message": message})
                mapped = TURN_EVENT_KIND.get(message.get("type")) if isinstance(message, dict) else None
                if mapped is not None:
                    self.pending.append((mapped, sent))
            if not self._closed and self.error is None:
                code = getattr(self.socket, "close_code", None)
                self.error = f"Reson8 closed the stream before the audio ended (close code {code})"
        except Exception as exc:
            if not self._closed:
                self.error = self.error or f"receive failed: {type(exc).__name__}: {exc}"

    async def finish(self, grace_s: float, close_timeout_s: float) -> None:
        receiver, sender = self._tasks
        await self.outbox.put(None)
        await sender
        await asyncio.sleep(grace_s)  # trailing turn_end for the last audio; no forced flush
        self._closed = True
        try:
            await asyncio.wait_for(self.socket.close(), close_timeout_s)
            await asyncio.wait_for(asyncio.shield(receiver), close_timeout_s)
        except Exception:
            pass


# ---- registry hooks -------------------------------------------------------------


def build(config: dict[str, Any]) -> Reson8Turns:
    return Reson8Turns(config)


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "provider": "reson8",
        "request": {k: config.get(k) for k in ("endpoint", *QUERY_KEYS)},
        "event_mapping": {"turn_end": "eot", "turn_start": "interruption (speech-start semantics)",
                          "turn_end_candidate": "recorded, not mapped"},
        "commit_time": "audio already sent to Reson8 when the message was received",
        "end_of_audio": "no flush_request (would force a turn_end); grace period, then close",
        "requires_realtime": True,
        "tracks": "Track 1 only (STT; no speaking agent for Track 2)",
    }
    try:
        from importlib import metadata

        info["transport"] = f"aiohttp {metadata.version('aiohttp')}"
    except Exception:
        info["transport"] = "aiohttp"
    return info
