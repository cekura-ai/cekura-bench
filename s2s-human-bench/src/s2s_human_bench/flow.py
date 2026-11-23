"""What the caller hears, measured from the transcript's word timings.

Every call is checked for four things a caller notices, using only speech timings (tool-call
timestamps are not reliable across export versions, so they are never used here):

* **Waits**: how long the agent took to start replying after the caller finished. A wait over a
  threshold is a flow problem, weighted by how long it ran (``waits`` in the settings).
* **Talking over the caller**: the agent starts a real reply (not a backchannel) while the caller
  is still mid-turn and overlaps them for at least ``talkOverSeconds``.
* **Jumping into a pause**: the agent replies after the caller stopped, but the caller carries on
  within ``pauseJumpSeconds`` of the agent starting, while it is still talking.
* **Slow stops**: the caller cuts in (not a backchannel) while the agent is talking; how long the
  agent keeps talking after that, weighted as for waits (``stops``).

``clean`` is the share of turns with no flow problem: 100 × (1 − problems ÷ opportunities), where
every reply, talk-over, pause jump and interruption is an opportunity and problems are weighted.
Settings live in the config (``scoring.flow``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Flow:
    replies: int = 0
    waits: list[float] = field(default_factory=list)  # seconds from the caller finishing to the agent starting, per reply
    talk_over: list[float] = field(default_factory=list)  # overlap seconds, per time the agent talked over the caller
    pause_jumps: int = 0
    stops: list[float] = field(default_factory=list)  # seconds the agent kept talking after the caller cut in
    problems: float = 0.0
    opportunities: int = 0

    @property
    def clean(self) -> float | None:
        """Share of turns with no flow problem, 0-100; None when the call had no turn to judge."""
        return None if not self.opportunities else 100 * max(0.0, 1 - self.problems / self.opportunities)


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _turns(transcript: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Speech turns of one side, from word timings where present, in time order."""
    out = []
    for t in transcript:
        if not (t.get("role") or "").startswith(prefix) or not (t.get("content") or "").strip():
            continue
        words = [
            (s, e, (w.get("text") or "")) for w in t.get("words") or [] if (s := _num(w.get("start"))) is not None and (e := _num(w.get("end"))) is not None
        ]
        if words:
            start, end = words[0][0], words[-1][1]
        else:
            start, end = _num(t.get("start_time")), _num(t.get("end_time"))
        if start is None or end is None:
            continue
        text = [w for _, _, w in words] if words else (t.get("content") or "").split()
        out.append({"s": start, "e": end, "words": words, "text": [x.strip(".,!?;:'\"").lower() for x in text]})
    return sorted(out, key=lambda x: x["s"])


def _backchannel(turn: dict[str, Any], settings: dict[str, Any]) -> bool:
    """A short acknowledgement ("mhm", "okay", "yeah, sure"): at most ``backchannelMaxWords`` words, or a few words all
    from the backchannel list."""
    words = [w for w in turn["text"] if w]
    if len(words) <= settings["backchannelMaxWords"]:
        return True
    return len(words) <= settings["backchannelListMaxWords"] and all(w in settings["backchannelWords"] for w in words)


def _weight(value: float, thresholds: list[list[float]]) -> float:
    """The weight of the highest threshold ``value`` reaches; 0 below the lowest."""
    w = 0.0
    for limit, weight in thresholds:
        if value >= limit:
            w = weight
    return w


def measure(transcript: list[dict[str, Any]], agent_prefix: str, caller_prefix: str, settings: dict[str, Any]) -> Flow:
    agent = _turns(transcript, agent_prefix)
    caller = _turns(transcript, caller_prefix)
    flow = Flow()
    continued: set[int] = set()  # caller turns that carried on after the agent jumped into a pause
    # Replies: the agent's first real turn after a real caller turn, when the caller did not speak again first.
    for i, c in enumerate(caller):
        if _backchannel(c, settings):
            continue
        a = next((x for x in agent if x["s"] > c["s"] + 0.2 and not _backchannel(x, settings)), None)
        if a is None or a["s"] < c["e"] - 0.2:
            continue  # no reply, or it started inside the caller's turn (counted as talking over, below)
        again = next((x for x in caller[i + 1 :] if x["s"] > c["e"]), None)
        if again is not None and again["s"] < a["s"]:
            continue  # the caller spoke again first: a pause or a hold, not a wait
        wait = max(0.0, a["s"] - c["e"])
        flow.replies += 1
        flow.waits.append(round(wait, 2))
        flow.problems += _weight(wait, settings["waits"])
        if again is not None and not _backchannel(again, settings) and a["s"] <= again["s"] <= min(a["e"], a["s"] + settings["pauseJumpSeconds"]):
            flow.pause_jumps += 1
            flow.problems += 1
            continued.add(id(again))
    # Talking over the caller: a real agent turn that starts while a caller turn is under way.
    for a in agent:
        if _backchannel(a, settings):
            continue
        c = next((x for x in caller if x["s"] + 0.2 < a["s"] < x["e"] - 0.2), None)
        if c is not None and (overlap := min(c["e"], a["e"]) - a["s"]) >= settings["talkOverSeconds"]:
            flow.talk_over.append(round(overlap, 2))
            flow.problems += 1
    # Interruptions: a real caller turn that starts while the agent is talking; how long the agent goes on. A caller
    # carrying on after the agent jumped into their pause is already counted above.
    for c in caller:
        if _backchannel(c, settings) or id(c) in continued:
            continue
        a = next((x for x in agent if x["s"] + 0.2 < c["s"] < x["e"] - 0.3), None)
        if a is None:
            continue
        if a["words"]:  # the last agent word that started after the caller cut in
            after = [e for s, e, _ in a["words"] if s >= c["s"]]
            stop = after[-1] - c["s"] if after else 0.0
        else:
            stop = a["e"] - c["s"]
        flow.stops.append(round(max(0.0, stop), 2))
        flow.problems += _weight(stop, settings["stops"])
    flow.opportunities = flow.replies + len(flow.talk_over) + flow.pause_jumps + len(flow.stops)
    return flow
