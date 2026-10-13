"""The streaming adapter interface every turn-taking model implements.

A model sees one speaker channel as a sequence of fixed 20 ms frames, in order,
and never sees future audio: the runner (``turn_taking_bench.runner``) pushes frame
``i`` only after frames ``0..i-1`` and tells the model the audio time reached
(``t_end_s``). Every event a model emits carries a *commit time* — the time by
which all audio the decision depended on has been heard, TurnBench's causality
rule (docs/SUBMISSION_FORMAT.md in TurnBench). The runner rejects any event
whose commit time lies past the audio pushed so far, so a model built on this
interface is causal by construction and the submission is causal by checking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

EventKind = Literal["eot", "interruption"]
EVENT_KINDS: tuple[EventKind, ...] = ("eot", "interruption")

# Frame the runner feeds, in seconds. It equals TurnBench's rms_vad WINDOW_S so the
# streamed baseline can reproduce the reference predictions sample for sample.
FRAME_S = 0.02


@dataclass(frozen=True)
class Event:
    """One committed decision on a speaker channel.

    ``eot``: this speaker's turn ended. ``interruption``: this speaker took the
    floor while the other held it. ``commit_time_s`` is on the conversation
    clock, in seconds, and must not exceed the audio time the model has heard.
    """

    kind: EventKind
    commit_time_s: float


class TurnModel(Protocol):
    """A per-speaker-channel streaming turn-taking model.

    One instance handles one channel of one conversation at a time; ``reset``
    starts a new channel. ``push`` receives exactly one frame of
    ``int(FRAME_S * sample_rate)`` float32 samples in ``[-1, 1]`` and the audio
    time at the end of that frame. ``flush`` is called once after the last frame
    so a model holding pending decisions can commit them (at or before
    ``t_end_s``). Both return the events decided during that call, in commit
    order. A network model that streams in real time implements the same three
    calls; the runner's ``--realtime`` pacing makes the push cadence wall-clock.
    """

    def reset(self, sample_rate: int) -> None: ...

    def push(self, frame: np.ndarray, t_end_s: float) -> list[Event]: ...

    def flush(self, t_end_s: float) -> list[Event]: ...


ModelFactory = Callable[[dict[str, Any]], TurnModel]
