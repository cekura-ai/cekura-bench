"""RMS energy VAD, Sesame's reference baseline, as a streaming adapter.

A port of ``baselines/rms_vad/predict.py`` from TurnBench into the
``TurnModel`` interface. The rule is unchanged: a 20 ms window is "speaking"
when its RMS exceeds the threshold; a flip to speaking is an interruption, a
flip to silent is an EOT, each committed at the end of the window that revealed
the flip. Sesame computes every window up front from the whole channel; here
each window arrives as one frame, and the decision for window ``i`` uses only
windows ``0..i``, which is exactly the information the offline rule uses.

With the runner's frame equal to ``WINDOW_S`` the predictions are identical to
the reference baseline (``turn-bench parity`` checks that).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from turn_bench.model import Event

# Sesame's constants (baselines/rms_vad/predict.py), verbatim.
WINDOW_S = 0.02
RMS_THRESHOLD = 0.01


class RmsVad:
    """Energy VAD: onset -> interruption, offset -> EOT, per 20 ms window."""

    def __init__(self, rms_threshold: float = RMS_THRESHOLD) -> None:
        self.rms_threshold = float(rms_threshold)
        self._samples_per_window: int | None = None
        self._speaking: bool | None = None  # previous window's state

    def reset(self, sample_rate: int) -> None:
        # Same expression as the reference, so the window length matches it.
        self._samples_per_window = int(WINDOW_S * sample_rate)
        # The reference loop starts at window 1 (the first window has nothing
        # to flip from), so the first pushed frame only seeds the state.
        self._speaking = None

    def push(self, frame: np.ndarray, t_end_s: float) -> list[Event]:
        if self._samples_per_window is None:
            raise RuntimeError("reset(sample_rate) must be called before push()")
        if len(frame) != self._samples_per_window:
            raise ValueError(
                f"rms_vad expects frames of {self._samples_per_window} samples "
                f"({WINDOW_S * 1000:.0f} ms), got {len(frame)}"
            )
        # Reference: np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        speaking = rms > self.rms_threshold
        events: list[Event] = []
        if self._speaking is not None and speaking != self._speaking:
            # The flip is known once this window has been heard: commit at its end.
            events.append(Event("interruption" if speaking else "eot", t_end_s))
        self._speaking = speaking
        return events

    def flush(self, t_end_s: float) -> list[Event]:
        return []


def build(config: dict[str, Any]) -> RmsVad:
    return RmsVad(rms_threshold=config.get("rms_threshold", RMS_THRESHOLD))


DEFAULT_CONFIG: dict[str, Any] = {"rms_threshold": RMS_THRESHOLD, "window_s": WINDOW_S}
