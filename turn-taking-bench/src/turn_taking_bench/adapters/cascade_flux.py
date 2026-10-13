"""Cascade reference: Deepgram Flux end-of-turn as the cascade's turn decision.

The published S2S benchmark lists an unranked cascade baseline
"Flux -> GPT-4.1 -> ElevenLabs Flash". In a cascade the LLM and the TTS only
run *after* the STT says the user's turn is over, so the cascade's turn-taking
behaviour is exactly Deepgram Flux's turn detection. This adapter streams one
speaker channel to Flux in real time and maps its ``TurnInfo`` events:

* ``EndOfTurn``   -> ``eot``
* ``StartOfTurn`` -> ``interruption``. Flux has no notion of "the other
  speaker holds the floor" (it hears one channel), so this is speech-start
  semantics: every turn start is reported, the same way Sesame scores OpenAI
  server VAD's ``speech_started`` as an interruption.
* ``Update`` / ``EagerEndOfTurn`` / ``TurnResumed``: recorded, not mapped
  (eager EOT is off by default, as in a plain cascade).

Commit time. An event's commit time is the audio time already *sent* to Flux
when the message is received (``audio_sent_s``), so it can never exceed the
audio the model has heard; the runner's causality check holds by construction.
Network and server latency therefore count against the model, as they would in
a live cascade.

Real time only. Flux is a live service: frames must reach it at wall-clock
pace or its timing (and EOT timeouts) mean nothing. The runner's ``--realtime``
flag is not visible to a model, so the adapter checks the push cadence itself
and refuses (raises) once audio runs ahead of wall-clock time.

Protocol and config. The wire protocol is stt-bench's Flux protocol
(``stt-bench/src/stt_bench/provider_protocol.py``: ``wss://api.deepgram.com/v2/listen``,
``Authorization: Token``, linear16 binary frames, ``TurnInfo`` messages,
``CloseStream`` to end) and the model config is stt-bench's
``config/models/deepgram-flux-en.json`` (``flux-general-en``, 16 kHz, 20 ms
frames). It is reimplemented here rather than imported: stt-bench is a separate
uv project whose pins conflict with turn-taking-bench's (numpy 2.2 vs 2.4), it needs
``websockets`` (not in turn-taking-bench's lock), and its Flux connection hard-codes
``eot_threshold=1.0, eot_timeout_ms=60000`` to *disable* automatic end of turn
for STT timing, which is the opposite of what a turn-taking cascade uses. Here
Flux's documented turn-detection defaults are sent explicitly and recorded.
Transport is aiohttp, already in turn-taking-bench's lock (via fsspec[http]).

The API key is ``DEEPGRAM_API_KEY`` from the process environment or the
``env_file`` (default ``turn-taking-bench/.env``); it is never logged or recorded.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np

from turn_taking_bench.model import FRAME_S, Event

MODEL_NAME = "cascade_flux"
TURN_TAKING_BENCH_ROOT = Path(__file__).resolve().parents[3]
STT_BENCH_FLUX_CONFIG = TURN_TAKING_BENCH_ROOT.parent / "stt-bench" / "config" / "models" / "deepgram-flux-en.json"

# Flux's supported linear16 rates; anything else must be resampled to 16 kHz.
FLUX_SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)

DEFAULT_CONFIG: dict[str, Any] = {
    # From stt-bench config/models/deepgram-flux-en.json (provenance checks they still match).
    "endpoint": "wss://api.deepgram.com/v2/listen",
    "model": "flux-general-en",
    "encoding": "linear16",
    "sample_rate": 16000,
    "ready_timeout_s": 15.0,
    "close_timeout_s": 5.0,
    # Flux turn detection: the documented defaults, sent explicitly so the run
    # records exactly what the server was asked for. None = parameter not sent.
    "eot_threshold": 0.7,
    "eot_timeout_ms": 5000,
    "eager_eot_threshold": None,
    # After the last frame, how long to keep listening for trailing events
    # (committed at the end of the audio, which is all the model has heard).
    "flush_grace_s": 1.0,
    # Refuse when audio runs this far ahead of wall-clock time (no --realtime).
    "max_ahead_s": 0.5,
    "env_file": None,  # default: turn-taking-bench/.env
}

TURN_EVENT_KIND = {"EndOfTurn": "eot", "StartOfTurn": "interruption"}


# ---- helpers ------------------------------------------------------------------


def read_api_key(env_file: str | os.PathLike[str] | None) -> str:
    """DEEPGRAM_API_KEY: process environment first, then a dotenv file."""
    value = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if value:
        return value
    path = Path(env_file) if env_file else TURN_TAKING_BENCH_ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].strip()
            name, sep, raw = line.partition("=")
            if sep and name.strip() == "DEEPGRAM_API_KEY":
                raw = raw.strip()
                if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
                    raw = raw[1:-1]
                if raw.strip():
                    return raw.strip()
    raise RuntimeError(f"DEEPGRAM_API_KEY not set (environment or {path})")


def query_url(config: dict[str, Any], sample_rate: int) -> str:
    query: dict[str, Any] = {
        "model": config["model"],
        "encoding": config["encoding"],
        "sample_rate": sample_rate,
        "eot_threshold": config["eot_threshold"],
        "eot_timeout_ms": config["eot_timeout_ms"],
    }
    if config.get("eager_eot_threshold") is not None:
        query["eager_eot_threshold"] = config["eager_eot_threshold"]
    query = {k: v for k, v in query.items() if v is not None}
    return config["endpoint"] + "?" + urlencode(query)


class Decimator:
    """Causal integer-factor downsampler: windowed-sinc low-pass FIR, then keep
    every ``factor``-th sample. Carries filter history and phase across frames.
    Group delay is (taps - 1) / 2 input samples (1 ms at 48 kHz, factor 3)."""

    def __init__(self, factor: int, taps_per_factor: int = 32) -> None:
        self.factor = int(factor)
        if self.factor == 1:
            self.taps = np.array([1.0])
        else:
            n = taps_per_factor * self.factor + 1
            cutoff = 0.45 / self.factor  # cycles per input sample, just under Nyquist out
            m = np.arange(n) - (n - 1) / 2
            taps = 2 * cutoff * np.sinc(2 * cutoff * m) * np.hamming(n)
            self.taps = taps / taps.sum()
        self.history = np.zeros(len(self.taps) - 1)
        self.phase = 0  # index (mod factor) of the next input sample

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        buffer = np.concatenate([self.history, frame.astype(np.float64)])
        filtered = np.convolve(buffer, self.taps, mode="valid")  # one output per input sample
        if len(self.taps) > 1:
            self.history = buffer[len(buffer) - (len(self.taps) - 1):]
        start = (-self.phase) % self.factor
        out = filtered[start :: self.factor]
        self.phase = (self.phase + len(frame)) % self.factor
        return out


def to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


# ---- transport ----------------------------------------------------------------


class Socket:
    """What the adapter needs from a websocket; tests substitute a fake."""

    async def send_bytes(self, data: bytes) -> None: ...
    async def send_str(self, data: str) -> None: ...
    async def close(self) -> None: ...
    def __aiter__(self): ...  # yields text messages until the socket closes


Connector = Callable[[str, dict[str, str], float], Awaitable[Socket]]


class AiohttpSocket:
    def __init__(self, session: Any, ws: Any) -> None:
        self._session, self._ws = session, ws

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)

    async def send_str(self, data: str) -> None:
        await self._ws.send_str(data)

    async def close(self) -> None:
        try:
            await self._ws.close()
        finally:
            await self._session.close()

    async def __aiter__(self):
        import aiohttp

        async for message in self._ws:
            if message.type == aiohttp.WSMsgType.TEXT:
                yield message.data
            elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                break


async def aiohttp_connect(url: str, headers: dict[str, str], timeout_s: float) -> Socket:
    import aiohttp

    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=timeout_s))
    try:
        ws = await session.ws_connect(url, headers=headers, heartbeat=None, max_msg_size=8 * 1024 * 1024)
    except aiohttp.WSServerHandshakeError as exc:
        await session.close()
        # Status and reason only; never the request headers.
        raise RuntimeError(f"Flux handshake failed: HTTP {exc.status} {exc.message}") from None
    except BaseException:
        await session.close()
        raise
    return AiohttpSocket(session, ws)


# ---- the model ----------------------------------------------------------------


class CascadeFlux:
    """One Flux streaming session per speaker channel, run on a background
    event loop; ``push`` hands frames over and collects the mapped events."""

    def __init__(self, config: dict[str, Any], connector: Connector | None = None) -> None:
        self.config = {**DEFAULT_CONFIG, **config}
        self._connector = connector or aiohttp_connect
        self._key: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: _Session | None = None
        self._messages: list[dict[str, Any]] = []
        self._channel: dict[str, Any] = {}

    # TurnModel ----------------------------------------------------------------

    def reset(self, sample_rate: int) -> None:
        self._shutdown()
        if sample_rate in FLUX_SAMPLE_RATES and sample_rate == self.config["sample_rate"]:
            factor, send_rate = 1, sample_rate
        elif sample_rate % self.config["sample_rate"] == 0:
            factor, send_rate = sample_rate // self.config["sample_rate"], self.config["sample_rate"]
        elif sample_rate in FLUX_SAMPLE_RATES:
            factor, send_rate = 1, sample_rate
        else:
            raise ValueError(f"cascade_flux cannot stream {sample_rate} Hz audio to Flux")
        self._decimate = Decimator(factor)
        self._frame_samples = int(FRAME_S * sample_rate)
        if self._key is None:
            self._key = read_api_key(self.config.get("env_file"))
        url = query_url(self.config, send_rate)
        self._messages = []
        self._channel = {"input_sample_rate": sample_rate, "sent_sample_rate": send_rate,
                         "decimation_factor": factor, "url": url}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="cascade-flux", daemon=True)
        self._thread.start()
        self._session = _Session(self._loop, self._messages)
        self._run(self._session.open(self._connector, url, {"Authorization": f"Token {self._key}"},
                                     float(self.config["ready_timeout_s"])),
                  timeout=float(self.config["ready_timeout_s"]) + 5)
        self._first_push: tuple[float, float] | None = None

    def push(self, frame: np.ndarray, t_end_s: float) -> list[Event]:
        session = self._require_session()
        if len(frame) != self._frame_samples:
            raise ValueError(f"cascade_flux expects {self._frame_samples}-sample frames, got {len(frame)}")
        now = time.monotonic()
        if self._first_push is None:
            self._first_push = (now, t_end_s)
        else:
            wall0, audio0 = self._first_push
            ahead = (t_end_s - audio0) - (now - wall0)
            if ahead > float(self.config["max_ahead_s"]):
                self._shutdown()
                raise RuntimeError(
                    f"cascade_flux requires real-time pacing (run with --realtime): audio is "
                    f"{ahead:.2f}s ahead of wall-clock time"
                )
        pcm = to_pcm16(self._decimate(frame))
        self._loop.call_soon_threadsafe(session.outbox.put_nowait, (pcm, t_end_s))
        return self._collect(t_end_s)

    def flush(self, t_end_s: float) -> list[Event]:
        session = self._require_session()
        try:
            self._run(session.finish(float(self.config["flush_grace_s"]), float(self.config["close_timeout_s"])),
                      timeout=float(self.config["flush_grace_s"]) + float(self.config["close_timeout_s"]) + 30)
            return self._collect(t_end_s)
        finally:
            self._shutdown()

    def diagnostics(self) -> dict[str, Any]:
        """Raw Flux messages with receipt time and audio already sent, plus the
        session parameters; ``events.jsonl`` stores this next to the events."""
        return {"flux": dict(self._channel), "messages": list(self._messages)}

    # internals ----------------------------------------------------------------

    def _require_session(self) -> _Session:
        if self._session is None:
            raise RuntimeError("reset(sample_rate) must be called before push()/flush()")
        return self._session

    def _run(self, coroutine: Awaitable[Any], timeout: float) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except BaseException:
            future.cancel()
            self._shutdown()
            raise

    def _collect(self, t_end_s: float) -> list[Event]:
        session = self._require_session()
        if session.error is not None:
            error = session.error
            self._shutdown()
            raise RuntimeError(self._redact(error))
        events: list[Event] = []
        while session.pending:
            kind, commit = session.pending.pop(0)
            events.append(Event(kind, min(commit, t_end_s)))
        return events

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "[REDACTED]") if self._key else text

    def _shutdown(self) -> None:
        session, loop, thread = self._session, self._loop, self._thread
        self._session = self._loop = self._thread = None
        if loop is None:
            return
        if session is not None:
            try:
                asyncio.run_coroutine_threadsafe(session.abort(), loop).result(timeout=10)
            except Exception:
                pass
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=10)
        if not loop.is_running():
            loop.close()


class _Session:
    """Background half of one channel: sender + receiver on the event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, messages: list[dict[str, Any]]) -> None:
        self.loop = loop
        self.messages = messages
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.pending: list[tuple[str, float]] = []  # appended on the loop thread, popped by push()
        self.error: str | None = None
        self.audio_sent_s = 0.0
        self.socket: Socket | None = None
        self.t0 = time.monotonic()
        self._tasks: list[asyncio.Task] = []
        self._closed = False

    async def open(self, connector: Connector, url: str, headers: dict[str, str], ready_timeout: float) -> None:
        self.socket = await asyncio.wait_for(connector(url, headers, ready_timeout), ready_timeout)
        self.t0 = time.monotonic()
        connected = asyncio.Event()
        self._connected = connected
        self._tasks = [asyncio.create_task(self._receive()), asyncio.create_task(self._send())]
        try:
            await asyncio.wait_for(connected.wait(), ready_timeout)
        except TimeoutError:
            if self.error is None:
                self.error = "Flux did not send Connected"
        if self.error is not None:
            raise RuntimeError(self.error)

    async def _send(self) -> None:
        try:
            while True:
                item = await self.outbox.get()
                if item is None:
                    return
                pcm, t_end_s = item
                await self.socket.send_bytes(pcm)
                self.audio_sent_s = t_end_s
        except Exception as exc:
            self.error = self.error or f"send failed: {type(exc).__name__}: {exc}"

    async def _receive(self) -> None:
        try:
            async for raw in self.socket:
                received = time.monotonic() - self.t0
                sent = self.audio_sent_s  # audio Flux could have heard by now
                try:
                    message = json.loads(raw)
                except ValueError:
                    message = {"unparsed": str(raw)[:500]}
                self.messages.append({"received_s": round(received, 4), "audio_sent_s": sent, "message": message})
                kind = message.get("type")
                if kind == "Connected":
                    self._connected.set()
                elif kind == "TurnInfo":
                    mapped = TURN_EVENT_KIND.get(message.get("event"))
                    if mapped is not None:
                        self.pending.append((mapped, sent))
                elif kind in ("Error", "FatalError"):
                    self.error = f"Flux error: {message.get('code')} {message.get('description')}"
                    self._connected.set()
                    return
            if not self._closed and self.error is None:
                self.error = "Flux closed the stream before the audio ended"
        except Exception as exc:
            if not self._closed:
                self.error = self.error or f"receive failed: {type(exc).__name__}: {exc}"
        finally:
            self._connected.set()

    async def finish(self, grace_s: float, close_timeout_s: float) -> None:
        receiver, sender = self._tasks
        await self.outbox.put(None)
        await sender
        await asyncio.sleep(grace_s)  # trailing TurnInfo for the last audio
        self._closed = True
        try:
            await self.socket.send_str(json.dumps({"type": "CloseStream"}))
            await asyncio.wait_for(asyncio.shield(receiver), close_timeout_s)
        except Exception:
            pass

    async def abort(self) -> None:
        self._closed = True
        for task in self._tasks:
            task.cancel()
        if self.socket is not None:
            try:
                await asyncio.wait_for(self.socket.close(), 5)
            except Exception:
                pass


# ---- registry hooks -------------------------------------------------------------


def build(config: dict[str, Any]) -> CascadeFlux:
    return CascadeFlux(config)


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    """What run.json records: the Flux request parameters, the event mapping and
    whether the model config still matches stt-bench's Flux config."""
    info: dict[str, Any] = {
        "provider": "deepgram",
        "cascade": "Flux -> GPT-4.1 -> ElevenLabs Flash (turn decision = Flux EndOfTurn)",
        "request": {k: config.get(k) for k in ("endpoint", "model", "encoding", "sample_rate",
                                                 "eot_threshold", "eot_timeout_ms", "eager_eot_threshold")},
        "event_mapping": {"EndOfTurn": "eot", "StartOfTurn": "interruption (speech-start semantics)"},
        "commit_time": "audio already sent to Flux when the message was received",
        "requires_realtime": True,
        "stt_bench_config": None,
    }
    try:
        from importlib import metadata

        info["transport"] = f"aiohttp {metadata.version('aiohttp')}"
    except Exception:
        info["transport"] = "aiohttp"
    if STT_BENCH_FLUX_CONFIG.is_file():
        reference = json.loads(STT_BENCH_FLUX_CONFIG.read_text(encoding="utf-8"))
        same = {k: reference.get(k) == config.get(k) for k in ("endpoint", "model", "sample_rate")}
        same["encoding"] = reference.get("encoding") == "pcm_s16le" and config.get("encoding") == "linear16"
        info["stt_bench_config"] = {"path": "stt-bench/config/models/deepgram-flux-en.json",
                                    "model_id": reference.get("model_id"), "matches": same}
    return info
