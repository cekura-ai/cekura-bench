"""Recompute a published latency from a cell's artifacts alone.

This module exists to make the central claim checkable rather than merely
asserted. It imports nothing from the runner, opens no socket, and reads only the
four files a cell ships: the agent audio, the caller and agent timelines, and the
event log. If it disagrees with the published number, the published number is
wrong.

    python -m lane_a.recompute data/lane-a/<run>/response_latency-open.book/<config>/<voice>/r1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from lane_a.audio import AudioTimeline, read_wav
from lane_a.detector import speech_bounds


def recompute_latency(cell_dir: str | Path) -> dict[str, float | None]:
    """Authored speech-end to first audible agent sample, from files only."""
    directory = Path(cell_dir)
    timelines = json.loads((directory / "timelines.json").read_text())
    caller = AudioTimeline.from_json(timelines["caller"])
    agent = AudioTimeline.from_json(timelines["agent"])

    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines() if line.strip()]
    ends = [e for e in events if e.get("kind") == "caller.audio.end" and "speech_end_sample" in e]
    if not ends:
        return {"latency_ms": None}
    caller_end_t = caller.time_of_sample(max(ends[0]["speech_end_sample"] - 1, 0))
    if caller_end_t is None:
        return {"latency_ms": None}

    pcm, rate = read_wav(directory / "agent.wav")
    chunk = next((c for c in agent.chunks if c.t_wall > caller_end_t), None)
    if chunk is None:
        return {"latency_ms": None}
    bounds = speech_bounds(pcm[chunk.first_sample * 2 :], rate)
    if not bounds.found:
        return {"latency_ms": None}
    onset_sample = chunk.first_sample + int(round(max(bounds.start_ms, 0.0) * rate / 1000.0))
    onset_t = agent.time_of_sample(onset_sample)
    if onset_t is None:
        return {"latency_ms": None}
    return {
        "latency_ms": round((onset_t - caller_end_t) * 1000.0, 1),
        "caller_speech_end_t": round(caller_end_t, 4),
        "agent_onset_t": round(onset_t, 4),
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path, recompute_latency(path))
