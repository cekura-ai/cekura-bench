"""The normalized event stream. This is the benchmark's output.

No observability backend, no database: a run produces one JSONL file of events
plus the audio it sent and received, and every published number is a function of
those files. That is what makes a result auditable by someone who does not trust
us -- they can recompute the metric from the artifacts, or disagree with our
detector and run their own over the same audio.

Times are seconds since the run's monotonic origin. One wall-clock anchor is
written at the top so a run can be lined up against provider-side logs; nothing
downstream ever does arithmetic on wall clock, which is not monotonic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Event kinds. Adapters normalize onto these; anything provider-specific also
# lands in the raw log, so a kind we have not modelled is never simply lost.
SESSION_OPEN = "session.open"
SESSION_CONFIGURED = "session.configured"
SESSION_CLOSED = "session.closed"
SESSION_ERROR = "session.error"

CALLER_AUDIO_START = "caller.audio.start"      # first sample of an utterance left us
CALLER_AUDIO_END = "caller.audio.end"          # last sample of an utterance left us
CALLER_COMMIT = "caller.commit"                # manual turn boundary declared
CALLER_TRANSCRIPT = "caller.transcript"        # provider's ASR of what we sent
CALLER_TEXT = "caller.text"                    # the text arm's turn, sent as words not audio
CALLER_SLIP = "caller.pacing.slip"             # we fell behind realtime; the run is suspect

AGENT_AUDIO_START = "agent.audio.start"        # first audio chunk of a response arrived
AGENT_AUDIO_END = "agent.audio.end"            # provider said the response audio is done
AGENT_TRANSCRIPT = "agent.transcript"
AGENT_THOUGHT = "agent.thought"                # a reasoning step the provider chose to expose
AGENT_INTERRUPTED = "agent.interrupted"

VAD_SPEECH_START = "vad.speech.start"          # the provider's own endpointer fired
VAD_SPEECH_END = "vad.speech.end"

STEP_START = "step.start"
STEP_END = "step.end"

TOOL_CALL = "tool.call"
TOOL_RESULT = "tool.result"
RESPONSE_DONE = "response.done"


class Clock:
    """Monotonic run clock with a single wall-clock anchor."""

    def __init__(self) -> None:
        self._origin = time.monotonic()
        self.wall_origin = time.time()

    def now(self) -> float:
        return time.monotonic() - self._origin


@dataclass(frozen=True)
class Event:
    t: float
    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {"t": round(self.t, 6), "kind": self.kind, **self.data}


class EventLog:
    """Append-only, in memory and on disk at once."""

    def __init__(self, clock: Clock, path: str | Path | None = None, raw_path: str | Path | None = None) -> None:
        self.clock = clock
        self.events: list[Event] = []
        self.raw_frames = 0
        self._file = self._open(path)
        self._raw = self._open(raw_path)
        if self._file:
            self._file.write(json.dumps({"wall_origin": clock.wall_origin}) + "\n")
            self._file.flush()

    @staticmethod
    def _open(path: str | Path | None):
        if path is None:
            return None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return open(path, "w", encoding="utf-8")

    def emit(self, kind: str, at: float | None = None, **data: Any) -> Event:
        """``at`` timestamps the event before the work it describes.

        Writing a line to disk is cheap but not free, and it must not land inside
        an interval the benchmark is measuring. Taking the instant first and
        writing afterwards keeps the record honest without putting I/O in the
        path.
        """
        event = Event(self.clock.now() if at is None else at, kind, data)
        self.events.append(event)
        if self._file:
            self._file.write(json.dumps(event.as_json()) + "\n")
            self._file.flush()
        return event

    def raw(self, payload: Any, direction: str = "in") -> None:
        """Every provider frame verbatim, for the audit trail.

        Flushed per frame like the normalized log. A run killed part-way must
        leave the frames it had already seen on disk: the alternative is
        rerunning a provider call to recover evidence that was already gathered.
        """
        self.raw_frames += 1
        if self._raw:
            self._raw.write(json.dumps({"t": round(self.clock.now(), 6), "dir": direction, "payload": payload}) + "\n")
            self._raw.flush()

    def of_kind(self, *kinds: str) -> Iterator[Event]:
        return (event for event in self.events if event.kind in kinds)

    def first(self, *kinds: str) -> Event | None:
        return next(self.of_kind(*kinds), None)

    def last(self, *kinds: str) -> Event | None:
        return next(reversed([e for e in self.events if e.kind in kinds]), None)

    def close(self) -> None:
        for handle in (self._file, self._raw):
            if handle:
                handle.close()
