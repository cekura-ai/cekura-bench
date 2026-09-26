"""The per-cell event log. Together with the audio, this is the benchmark's output.

Every published number is a function of these files and the audio, so a result can
be recomputed by someone who does not trust us. Times are seconds since the run's
monotonic origin; one wall-clock anchor is written at the top so a run can be lined
up against provider-side logs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


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
