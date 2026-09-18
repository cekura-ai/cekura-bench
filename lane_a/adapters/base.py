"""What a Lane A adapter is, and what it deliberately is not.

An adapter connects to one realtime wire protocol and does four things: send
realtime-paced PCM, receive audio and events, timestamp both, and normalize the
events onto the vocabulary in ``lane_a.events``. It does not play audio, does not
resample on the fly, does not retry, and does not integrate with any
observability system -- the JSONL log *is* the output. That is why a benchmark
adapter is a few hundred lines where a production one is a few thousand.

Adapters are written per **wire protocol**, not per model: one OpenAI Realtime
adapter serves every ``gpt-realtime-*``. They are validated against Pipecat's
service classes on a shared probe -- if the provider's own transcripts and event
ordering agree, ours is faithful -- but Pipecat is never in the measured path,
because its per-provider integration maturity would be read as a provider
difference.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from lane_a import events as ev
from lane_a.audio import AudioTimeline, SAMPLE_WIDTH
from mock_tools.spec import ToolSpec

__all__ = ["AdapterError", "RealtimeAdapter", "SessionConfig", "ToolSpec", "TurnDetection", "parse_arguments"]


@dataclass(frozen=True)
class TurnDetection:
    """How the turn boundary gets decided. Each is a separately published config.

    ``manual`` is available only because the caller audio is authored here: we know
    the exact sample at which speech ends, so committing there yields a
    generation-latency floor with zero endpointing error. Native minus manual is
    then that provider's endpointing cost, measured rather than inferred from its
    configuration. Both numbers are published; the difference never replaces them.
    """

    mode: str = "server_vad"  # server_vad | semantic_vad | manual
    silence_duration_ms: int | None = None
    threshold: float | None = None
    prefix_padding_ms: int | None = None

    @property
    def is_manual(self) -> bool:
        return self.mode == "manual"

    @property
    def label(self) -> str:
        if self.is_manual:
            return "manual"
        parts = [self.mode]
        if self.silence_duration_ms is not None:
            parts.append(f"{self.silence_duration_ms}ms")
        return "-".join(parts)


@dataclass
class SessionConfig:
    instructions: str = ""
    voice: str | None = None
    tools: tuple[ToolSpec, ...] = ()
    turn_detection: TurnDetection = field(default_factory=TurnDetection)
    modality: str = "audio"  # "text" runs the identical scenario with no speech at all
    transcribe_input: bool = True
    temperature: float | None = None


class AdapterError(RuntimeError):
    """The provider refused the session. Distinct from a scored failure."""


class RealtimeAdapter(ABC):
    name: ClassVar[str] = "adapter"
    input_rate: ClassVar[int] = 24000
    output_rate: ClassVar[int] = 24000
    supports_manual_commit: ClassVar[bool] = True
    supports_text_modality: ClassVar[bool] = True

    def __init__(self, *, model: str, api_key: str, log: ev.EventLog, config: SessionConfig | None = None) -> None:
        self.model = model
        self.config = config or SessionConfig()
        self._api_key = api_key
        self.log = log
        self.caller_timeline = AudioTimeline(self.input_rate)
        self.agent_timeline = AudioTimeline(self.output_rate)
        self.agent_pcm = bytearray()
        self.caller_pcm = bytearray()
        self.agent_text: list[str] = []
        self.caller_text: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        # What the session actually is, as opposed to what we asked for. The
        # request is ours and the acknowledgement is the provider's, and where
        # they differ the difference is the finding -- a provider that silently
        # clamps a silence duration would otherwise be published under the
        # configuration we believed we set.
        self.session_id: str | None = None
        self.session_sent: dict[str, Any] | None = None
        self.session_ack: dict[str, Any] | None = None
        self.closed = asyncio.Event()
        self._speaking = False
        self._receiver: asyncio.Task | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Open the socket and apply ``self.config``. Raise AdapterError if refused."""

    @abstractmethod
    async def _send_json(self, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self) -> "RealtimeAdapter":
        await self.connect()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    # ── sending ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def _send_audio_chunk(self, pcm: bytes) -> None:
        """Wire-format one chunk. Timestamping is handled by ``send_audio``."""

    async def send_audio(self, pcm: bytes, t_send: float | None = None) -> None:
        """Hand one chunk to the provider and record when its first sample left.

        ``t_send`` is taken before the write, not after: for a realtime-paced
        stream that is the instant the audio would have been spoken, and the
        socket write is buffered. The caller's pacer reports slip separately, so
        a blocked write shows up as slip rather than as silently late audio.
        """
        if not pcm:
            return
        at = self.log.clock.now() if t_send is None else t_send
        self.caller_timeline.record_pcm(pcm, at)
        self.caller_pcm.extend(pcm)
        await self._send_audio_chunk(pcm)

    async def commit(self) -> None:
        """Declare the caller's turn over at exactly this sample. Manual mode only."""
        if not self.supports_manual_commit:
            raise AdapterError(f"{self.name} has no manual commit")
        self.log.emit(ev.CALLER_COMMIT, sample=self.caller_timeline.n_samples)
        await self._commit()

    @abstractmethod
    async def _commit(self) -> None: ...

    async def send_text(self, text: str) -> None:
        """The text control arm's turn. Timestamped, delivered, then recorded.

        Normalized here rather than in each adapter so the caller side of a text
        run is visible in the log: without it the arm that exists to attribute a
        failure to the speech pathway would ship artifacts showing only one half
        of the conversation.
        """
        if not self.supports_text_modality:
            raise AdapterError(f"{self.name} has no text modality")
        at = self.log.clock.now()
        await self._send_text(text)
        self.log.emit(ev.CALLER_TEXT, at=at, text=text)

    async def _send_text(self, text: str) -> None:
        raise AdapterError(f"{self.name} declares text support but does not implement it")

    async def send_tool_result(self, call_id: str, output: Any) -> None:
        at = self.log.clock.now()
        self.tool_results.append({"call_id": call_id, "output": output})
        await self._send_tool_result(call_id, output)
        self.log.emit(ev.TOOL_RESULT, at=at, call_id=call_id, output=output)

    @abstractmethod
    async def _send_tool_result(self, call_id: str, output: Any) -> None: ...

    # ── receiving ────────────────────────────────────────────────────────────

    def _on_agent_audio(self, pcm: bytes) -> None:
        """Record inbound audio at its arrival instant. Called by the receive loop."""
        at = self.log.clock.now()
        chunk = self.agent_timeline.record_pcm(pcm, at)
        self.agent_pcm.extend(pcm)
        if not self._speaking:
            self._speaking = True
            self.log.emit(
                ev.AGENT_AUDIO_START,
                sample=chunk.first_sample,
                chunk_ms=round(1000 * chunk.n_samples / self.output_rate, 2),
            )

    def _on_agent_audio_done(self, reason: str = "provider") -> None:
        if self._speaking:
            self._speaking = False
            self.log.emit(ev.AGENT_AUDIO_END, sample=self.agent_timeline.n_samples, reason=reason)

    @property
    def agent_speaking(self) -> bool:
        return self._speaking

    def state(self) -> dict[str, Any]:
        """Everything the adapter knows about its own session, for the record.

        The adapter describes itself rather than letting the runner reach in one
        field at a time: new adapter state then reaches the published cell on its
        own, instead of only when someone remembers to edit the runner too --
        which is the failure the whole record exists to prevent.
        """
        return {
            "adapter": self.name,
            "model": self.model,
            "turn_detection": self.config.turn_detection.label,
            "modality": self.config.modality,
            "session": {
                "provider_session_id": self.session_id,
                "sent": self.session_sent,
                "acknowledged": self.session_ack,
            },
            "audio": {
                "input_rate": self.input_rate,
                "output_rate": self.output_rate,
                "caller_samples": self.caller_timeline.n_samples,
                "agent_samples": self.agent_timeline.n_samples,
                "caller_ms": round(self.caller_timeline.duration_s * 1000.0, 1),
                "agent_ms": round(self.agent_timeline.duration_s * 1000.0, 1),
                "caller_chunks": len(self.caller_timeline.chunks),
                "agent_chunks": len(self.agent_timeline.chunks),
            },
            "transcripts": {"agent": list(self.agent_text), "caller_asr": list(self.caller_text)},
            "tools": {"calls": list(self.tool_calls), "results": list(self.tool_results)},
        }


def parse_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Tool arguments as a dict. A model that emits malformed JSON is a finding."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__unparsed__": raw}
    return parsed if isinstance(parsed, dict) else {"__value__": parsed}
