"""The adapter contract: one class per wire protocol, one record per synthesis.

An adapter owns a connection to a provider and any number of *contexts* on it.
A context is one utterance being synthesised: text goes in through ``send`` (as
many frames as the caller likes), ``finish`` declares the text complete, and
audio comes back into the context's ``Synthesis`` record with the arrival time
of every chunk. ``cancel`` asks the provider to stop the context early.

Two conventions, stated once and published with the results:

* **t0 is the first text frame.** Connecting, authenticating and any session
  setup happen in ``connect`` before t0, so the round-trip term of every latency
  is synthesis, not handshake. What each protocol excludes is recorded in the
  adapter's ``setup_excluded`` and printed in the row.
* **Arrival is the anchor.** A chunk that arrives at ``t`` is playable from
  ``t``; sample ``k`` in it is heard at ``t + k/rate``. The realtime-player view
  (``AudioTimeline``) then says when a listener would actually have heard each
  sample, which is what playout margin and cancel latency are measured on.
"""

from __future__ import annotations

import abc
import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from tts_bench.common.audio import AudioTimeline
from tts_bench.common.events import Clock, EventLog

from tts_bench import events as ev


class AdapterError(RuntimeError):
    """The provider refused or the protocol broke; the cell is void, not scored."""


# An error that says the account, not the service, stopped the request: a key,
# a balance, a quota or a rate limit. Such a cell measured nothing about the
# voice, so it is void and a resume runs it again; any other provider error on
# a request the account was allowed to make is a failure of the service.
_REFUSAL = re.compile(
    r"\b(?:HTTP )?(?:401|402|403|429)\b|unauthori[sz]ed|forbidden|invalid.{0,12}(?:api.?key|token|credential)"
    r"|quota|rate.?limit|too many (?:requests|concurrent)|concurren\w* (?:limit|requests? exceeded)"
    r"|balance|credits?\b|insufficient|billing|payment|exhausted|permission denied",
    re.IGNORECASE,
)


def is_refusal(message: str | None) -> bool:
    return bool(message) and _REFUSAL.search(message) is not None


@dataclass(frozen=True)
class TTSConfig:
    model: str
    voice: str
    sample_rate: int = 24000
    language: str | None = None
    speed: float | None = None
    # Protocol switches that change the number (an auto-generation mode, a
    # buffer setting). Recorded in every cell; a row without them is not
    # reproducible. Keys are adapter-specific and documented on the adapter.
    options: tuple[tuple[str, str], ...] = ()

    def option(self, key: str, default: str | None = None) -> str | None:
        return dict(self.options).get(key, default)

    @property
    def label(self) -> str:
        opts = "".join(f",{k}={v}" for k, v in self.options)
        return f"{self.model}/{self.voice}@{self.sample_rate}{opts}"

    def as_json(self) -> dict[str, Any]:
        return {"model": self.model, "voice": self.voice, "sample_rate": self.sample_rate, "language": self.language,
                "speed": self.speed, "options": dict(self.options)}


@dataclass
class Synthesis:
    """Everything one context produced, with when it happened."""

    context_id: str
    rate: int
    timeline: AudioTimeline
    pcm: bytearray = field(default_factory=bytearray)
    t_open: float | None = None
    t_first_text: float | None = None       # t0
    t_input_done: float | None = None
    t_first_chunk: float | None = None
    t_last_chunk: float | None = None
    t_done: float | None = None             # provider's completion signal
    t_cancel: float | None = None
    t_cancel_ack: float | None = None
    chars_sent: int = 0
    text_frames: list[tuple[float, int]] = field(default_factory=list)   # (t, chars) per frame
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def t0(self) -> float | None:
        return self.t_first_text

    def as_json(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "rate": self.rate,
            "chars_sent": self.chars_sent,
            "text_frames": [[round(t, 6), n] for t, n in self.text_frames],
            "t_open": self.t_open,
            "t_first_text": self.t_first_text,
            "t_input_done": self.t_input_done,
            "t_first_chunk": self.t_first_chunk,
            "t_last_chunk": self.t_last_chunk,
            "t_done": self.t_done,
            "t_cancel": self.t_cancel,
            "t_cancel_ack": self.t_cancel_ack,
            "n_chunks": len(self.timeline.chunks),
            "audio_s": round(self.timeline.duration_s, 4),
            "meta": self.meta,
            "error": self.error,
            "timeline": self.timeline.as_json(),
        }


class TTSAdapter(abc.ABC):
    """One provider wire protocol."""

    name: ClassVar[str] = "abstract"
    transport: ClassVar[str] = "websocket"           # or "http"
    supports_streamed_input: ClassVar[bool] = True   # text may arrive in frames before finish()
    supports_cancel: ClassVar[bool] = True           # a context can be stopped early
    supports_continuation: ClassVar[bool] = True     # frames are one utterance, prosody carried across
    native_rates: ClassVar[tuple[int, ...]] = (24000,)
    native_mulaw_8k: ClassVar[bool] = False          # emits 8 kHz mu-law itself (capability metadata)
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade"  # what happens before t0

    def __init__(self, config: TTSConfig, api_key: str, log: EventLog, clock: Clock) -> None:
        self.config = config
        self.api_key = api_key
        self.log = log
        self.clock = clock
        self.contexts: dict[str, Synthesis] = {}
        self._connected = False

    # -- capability declaration -------------------------------------------

    @classmethod
    def unsupported_reason(cls, config: TTSConfig, needs: tuple[str, ...] = ()) -> str | None:
        """Why this adapter cannot run ``config`` for a probe that ``needs`` these features.

        Declared up front so a configuration the protocol has no equivalent for
        is published as an exclusion with its reason, never dialled, and never
        mistaken for a failure of the service.
        """
        for need in needs:
            if not getattr(cls, f"supports_{need}", False):
                return f"{cls.name} has no {need.replace('_', ' ')}"
        if config.sample_rate not in cls.native_rates:
            return f"{cls.name} does not emit {config.sample_rate} Hz PCM"
        return None

    def capabilities(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "streamed_input": self.supports_streamed_input,
            "cancel": self.supports_cancel,
            "continuation": self.supports_continuation,
            "native_rates": list(self.native_rates),
            "native_mulaw_8k": self.native_mulaw_8k,
            "setup_excluded_from_t0": self.setup_excluded,
        }

    # -- lifecycle -----------------------------------------------------------

    async def __aenter__(self) -> "TTSAdapter":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        """Connect, authenticate, warm. Everything here is before t0 by definition."""
        started = self.clock.now()
        await self._connect()
        self._connected = True
        self.log.emit(ev.SESSION_OPEN, provider=self.name, setup_ms=round((self.clock.now() - started) * 1000, 1),
                      excluded_from_t0=self.setup_excluded)

    async def close(self) -> None:
        if self._connected:
            try:
                await self._close()
            finally:
                self._connected = False
                self.log.emit(ev.SESSION_CLOSED, provider=self.name)

    # -- the context API (what probes call) ---------------------------------

    async def open_context(self, context_id: str) -> Synthesis:
        """A context to synthesise into. Any per-context setup frame goes here, before t0."""
        synthesis = Synthesis(context_id=context_id, rate=self.config.sample_rate,
                              timeline=AudioTimeline(self.config.sample_rate))
        synthesis.t_open = self.clock.now()
        self.contexts[context_id] = synthesis
        await self._open_context(context_id)
        self.log.emit(ev.CONTEXT_OPEN, at=synthesis.t_open, context_id=context_id)
        return synthesis

    async def send(self, context_id: str, text: str) -> None:
        """One text frame. The first one on a context is t0."""
        synthesis = self.contexts[context_id]
        at = self.clock.now()
        if synthesis.t_first_text is None:
            synthesis.t_first_text = at
        synthesis.chars_sent += len(text)
        synthesis.text_frames.append((at, len(text)))
        await self._send_text(context_id, text, first=len(synthesis.text_frames) == 1)
        self.log.emit(ev.TEXT_SENT, at=at, context_id=context_id, chars=len(text), cumulative=synthesis.chars_sent)

    async def finish(self, context_id: str) -> None:
        """The text is complete; the provider may flush whatever it is holding."""
        synthesis = self.contexts[context_id]
        synthesis.t_input_done = self.clock.now()
        await self._finish(context_id)
        self.log.emit(ev.INPUT_DONE, at=synthesis.t_input_done, context_id=context_id)

    async def cancel(self, context_id: str) -> None:
        synthesis = self.contexts[context_id]
        synthesis.t_cancel = self.clock.now()
        await self._cancel(context_id)
        self.log.emit(ev.CANCEL_SENT, at=synthesis.t_cancel, context_id=context_id)

    async def wait(self, context_id: str, timeout_s: float, quiet_s: float = 1.5) -> Synthesis:
        """Until the provider says the context is done, or it has gone quiet.

        A provider with no completion signal (or one that never sends it after a
        cancel) is done when no chunk has arrived for ``quiet_s``. The record
        says which of the two ended the wait.
        """
        synthesis = self.contexts[context_id]
        deadline = self.clock.now() + timeout_s
        while True:
            remaining = deadline - self.clock.now()
            if remaining <= 0:
                synthesis.meta["ended_by"] = "timeout"
                return synthesis
            try:
                await asyncio.wait_for(synthesis.done.wait(), timeout=min(remaining, 0.05))
                synthesis.meta.setdefault("ended_by", "provider")
                return synthesis
            except asyncio.TimeoutError:
                pass
            if synthesis.error:
                synthesis.meta["ended_by"] = "error"
                return synthesis
            last = synthesis.t_last_chunk
            anchor = last if last is not None else (synthesis.t_cancel or synthesis.t_input_done)
            if anchor is not None and self.clock.now() - anchor > quiet_s and (last is not None or synthesis.t_cancel):
                synthesis.meta["ended_by"] = "quiet"
                return synthesis

    # -- what subclasses report back ----------------------------------------

    def _on_audio(self, context_id: str, pcm: bytes) -> None:
        """A non-empty audio chunk arrived. Stamped first, stored second."""
        if not pcm:
            return
        at = self.clock.now()
        synthesis = self.contexts.get(context_id)
        if synthesis is None:
            return
        synthesis.timeline.record_pcm(pcm, at)
        synthesis.pcm.extend(pcm)
        synthesis.t_last_chunk = at
        if synthesis.t_first_chunk is None:
            synthesis.t_first_chunk = at
            self.log.emit(ev.AUDIO_FIRST, at=at, context_id=context_id, bytes=len(pcm),
                          roundtrip_ms=None if synthesis.t0 is None else round((at - synthesis.t0) * 1000, 1))

    def _on_done(self, context_id: str, **meta: Any) -> None:
        synthesis = self.contexts.get(context_id)
        if synthesis is None or synthesis.done.is_set():
            return
        synthesis.t_done = self.clock.now()
        synthesis.meta.update(meta)
        synthesis.done.set()
        self.log.emit(ev.AUDIO_DONE, at=synthesis.t_done, context_id=context_id, **meta)

    def _on_cancel_ack(self, context_id: str, **meta: Any) -> None:
        synthesis = self.contexts.get(context_id)
        if synthesis is None:
            return
        synthesis.t_cancel_ack = self.clock.now()
        self.log.emit(ev.CANCEL_ACK, at=synthesis.t_cancel_ack, context_id=context_id, **meta)

    # Headers that identify a session or carry a credential stay out of the record.
    _PRIVATE_HEADERS = frozenset({"set-cookie", "cookie", "authorization", "x-api-key", "x-goog-api-key"})

    def _on_response(self, context_id: str, status: int, headers: Any) -> None:
        """An HTTP response's status line arrived: when, and what the server said about the request.

        The gap from t0 to the headers is the server's time before it streamed
        anything; request ids, processing times and usage headers are what a
        cost or a server-side latency is later read from.
        """
        at = self.clock.now()
        kept = {k.lower(): v for k, v in headers.items() if k.lower() not in self._PRIVATE_HEADERS}
        synthesis = self.contexts.get(context_id)
        if synthesis is not None:
            synthesis.meta["http_status"] = status
            synthesis.meta["t_headers"] = at
            synthesis.meta["headers"] = kept
        self.log.raw({"http_status": status, "headers": kept}, direction="in")

    def _on_error(self, context_id: str | None, message: str) -> None:
        self.log.emit(ev.PROVIDER_ERROR, context_id=context_id, message=message)
        targets = [self.contexts[context_id]] if context_id in self.contexts else list(self.contexts.values())
        for synthesis in targets:
            synthesis.error = synthesis.error or message
            synthesis.done.set()

    # -- protocol ------------------------------------------------------------

    @abc.abstractmethod
    async def _connect(self) -> None: ...

    @abc.abstractmethod
    async def _close(self) -> None: ...

    async def _open_context(self, context_id: str) -> None:
        """Per-context setup, if the protocol has any. Excluded from t0 by construction."""

    @abc.abstractmethod
    async def _send_text(self, context_id: str, text: str, first: bool) -> None: ...

    @abc.abstractmethod
    async def _finish(self, context_id: str) -> None: ...

    async def _cancel(self, context_id: str) -> None:
        raise AdapterError(f"{self.name} has no cancel")
