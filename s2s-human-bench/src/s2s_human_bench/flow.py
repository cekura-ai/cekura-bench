"""What the caller hears, measured from the transcript's word timings.

Every call is checked for what a caller notices, using only speech timings (tool-call timestamps
are not reliable across export versions, so they are never used here). Bracketed stage directions
in the caller's lines ("[silence]", hold music, background voices) are not the caller speaking.
Timings can be off by a second, so thresholds sit above where listeners start to notice, and an
event never counts more than one problem:

* **Waits**: how long the agent took to make any sound after the caller's audio ended, an
  acknowledgement included. Kept for the typical reply time; the problem is the silence below.
* **Silences**: every gap on the call's timeline that the agent ends while the caller waits, before
  its reply or after a short "okay" or "one moment". Weighted by length (``waits`` in the settings).
  A gap after the caller asked to hold (``holdPhrases``), or after the agent asked a question, is
  the caller's turn to fill, not a problem.
* **Talking over the caller** and **jumping into a pause**: recorded, weighted by
  ``talkOverWeight`` and ``pauseJumpWeight``. Listening found most timing-detected overlaps were not
  real, so the config weights them 0 and leaves this to the judged turn-taking skills.
* **Slow stops**: the caller cuts in with at least ``stopMinWords`` words, and not just to check the
  line (``checkInPhrases``), while the agent is talking; how long the agent keeps talking, across
  transcript turns, until it pauses for ``stopPauseSeconds``. Weighted by length (``stops``).

``clean`` is the share of turns with no flow problem: 100 × (1 − problems ÷ opportunities), where
every agent turn and every interruption is an opportunity and problems are weighted.
Settings live in the config (``scoring.flow``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Flow:
    replies: int = 0
    waits: list[float] = field(default_factory=list)  # seconds from the caller finishing to the agent's first sound, per reply
    silences: list[float] = field(default_factory=list)  # gaps of at least the lowest wait threshold that the agent ended while the caller waited
    talk_over: list[float] = field(default_factory=list)  # overlap seconds, per time the agent talked over the caller
    pause_jumps: int = 0
    stops: list[float] = field(default_factory=list)  # seconds the agent kept talking after the caller cut in
    problems: float = 0.0
    opportunities: int = 0
    # (kind, seconds into the call, measured value): "wait" and "silence" where the silence began, "talk_over" and
    # "pause_jump" where the agent started, "stop" where the caller cut in; for clips and spot checks.
    events: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def clean(self) -> float | None:
        """Share of turns with no flow problem, 0-100; None when the call had no turn to judge."""
        return None if not self.opportunities else 100 * max(0.0, 1 - self.problems / self.opportunities)


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def clip_key(text: str) -> str:
    """A caller line or clip text reduced for matching: lower case, no stage directions, punctuation or extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", re.sub(r"\[[^\]]*\]", " ", text.lower()))).strip()


def _spoken(words: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """The words actually spoken: bracketed stage directions ("[silence]", "[noise:hold-music]", "[colleague in the
    background: ...]") are left out, whether they come as one token or several."""
    out, depth = [], 0
    for s, e, w in words:
        opens, closes = w.count("["), w.count("]")
        if depth == 0 and opens == 0:
            out.append((s, e, w))
        depth = max(0, depth + opens - closes)
    return out


def _turns(transcript: list[dict[str, Any]], prefix: str, settings: dict[str, Any], clip_seconds: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Speech segments of one side, in time order. A line splits wherever its spoken words pause for ``splitGapSeconds``
    (a hold or a scripted silence inside one line is not talking). A line with nothing spoken outside brackets (hold
    music, background voices) becomes a ``noise`` segment: that side is busy, but not talking. A recorded caller clip
    often runs past its transcript line (a trailing pause or background noise); with the clip lengths
    (``clip_seconds``, by clip text), the rest of the clip is a ``noise`` segment too."""
    out = []
    for t in transcript:
        content = (t.get("content") or "").strip()
        if not (t.get("role") or "").startswith(prefix) or not content:
            continue
        words = [
            (s, e, (w.get("text") or "")) for w in t.get("words") or [] if (s := _num(w.get("start"))) is not None and (e := _num(w.get("end"))) is not None
        ]
        start, end = (words[0][0], words[-1][1]) if words else (_num(t.get("start_time")), _num(t.get("end_time")))
        if start is None or end is None:
            continue
        clip = (clip_seconds or {}).get(clip_key(content))
        if clip and start + clip > end + 0.05:  # the rest of the clip: the caller's side is busy, then the caller waits
            out.append({"s": end, "e": start + clip, "words": [], "text": [], "question": False, "noise": True, "tail": True})
        spoken = _spoken(words) if words else _spoken([(start, end, w) for w in content.split()])
        if not spoken:
            out.append({"s": start, "e": end, "words": [], "text": [], "question": False, "noise": True})
            continue
        if not words:  # no word timings: one segment over the line
            spoken = [(start, end, " ".join(w for _, _, w in spoken))]
        segments, current = [], [spoken[0]]
        for prev, nxt in zip(spoken, spoken[1:], strict=False):
            if nxt[0] - prev[1] >= settings["splitGapSeconds"]:
                segments.append(current)
                current = []
            current.append(nxt)
        segments.append(current)
        for i, seg in enumerate(segments):
            text = [x.strip(".,!?;:'\"").lower() for _, _, w in seg for x in w.split()]
            last = i == len(segments) - 1
            out.append(
                {
                    "s": seg[0][0],
                    "e": seg[-1][1],
                    "busy": max(seg[-1][1], start + clip) if last and clip else seg[-1][1],  # when the caller's audio ends
                    "words": seg,
                    "text": text,
                    "question": last and content.endswith("?"),
                    "noise": False,
                }
            )
    return sorted(out, key=lambda x: x["s"])


def _backchannel(turn: dict[str, Any], settings: dict[str, Any]) -> bool:
    """A short acknowledgement ("mhm", "okay", "yeah, sure"): at most ``backchannelMaxWords`` words, or a few words all
    from the backchannel list."""
    words = [w for w in turn["text"] if w]
    if len(words) <= settings["backchannelMaxWords"]:
        return True
    return len(words) <= settings["backchannelListMaxWords"] and all(w in settings["backchannelWords"] for w in words)


def _asks_to_hold(turn: dict[str, Any], settings: dict[str, Any]) -> bool:
    text = " ".join(turn["text"])
    return any(phrase in text for phrase in settings["holdPhrases"])


def _weight(value: float, thresholds: list[list[float]]) -> float:
    """The weight of the highest threshold ``value`` reaches; 0 below the lowest."""
    w = 0.0
    for limit, weight in thresholds:
        if value >= limit:
            w = weight
    return w


def measure(
    transcript: list[dict[str, Any]], agent_prefix: str, caller_prefix: str, settings: dict[str, Any], clip_seconds: dict[str, float] | None = None
) -> Flow:
    agent = [x for x in _turns(transcript, agent_prefix, settings) if not x["noise"]]
    caller_all = _turns(transcript, caller_prefix, settings, clip_seconds)
    caller = [x for x in caller_all if not x["noise"]]  # the caller talking; noise segments only keep the line busy
    flow = Flow()
    continued: set[int] = set()  # caller turns that carried on after the agent jumped into a pause
    # Replies: the agent's first sound after a real caller turn ended, an acknowledgement included, when the caller did
    # not speak again first.
    for i, c in enumerate(caller):
        if _backchannel(c, settings):
            continue
        a = next((x for x in agent if x["s"] >= c["e"] - 0.2), None)
        if a is None:
            continue
        again = next((x for x in caller[i + 1 :] if x["s"] > c["e"]), None)
        if again is not None and again["s"] < a["s"]:
            continue  # the caller spoke again first: a pause or a hold, not a wait
        wait = max(0.0, a["s"] - c["busy"])  # from the end of the caller's audio, which can run past their last word
        flow.replies += 1
        flow.waits.append(round(wait, 2))
        flow.events.append(("wait", round(c["busy"], 2), round(wait, 2)))
        real = a if not _backchannel(a, settings) else None
        jumped = real is not None and again is not None and real["s"] <= again["s"] <= min(real["e"], real["s"] + settings["pauseJumpSeconds"])
        if jumped and not _backchannel(again, settings):
            flow.pause_jumps += 1
            flow.events.append(("pause_jump", round(a["s"], 2), round(wait, 2)))
            flow.problems += settings["pauseJumpWeight"]
            continued.add(id(again))
    # Silences while the caller waits: gaps on the merged timeline that the agent ends, unless the caller asked to hold
    # or the agent had just asked a question (then the caller was the one expected to speak).
    timeline = sorted([(x["s"], x["e"], "agent", x) for x in agent] + [(x["s"], x["e"], "caller", x) for x in caller_all], key=lambda t: t[0])
    end: float | None = None
    before: tuple[str, dict[str, Any]] | None = None  # the turn that ended last, and whose it was
    last_caller: dict[str, Any] | None = None
    for s, e, side, turn in timeline:
        gap = None if end is None else s - end
        waiting = (
            before is not None
            and not (before[1]["noise"] and not before[1].get("tail"))  # hold music or background noise: the caller's side was busy
            and not (before[0] == "agent" and before[1]["question"])
            and not (last_caller and _asks_to_hold(last_caller, settings))
        )
        if side == "agent" and gap is not None and gap >= settings["waits"][0][0] and waiting:
            flow.silences.append(round(gap, 2))
            flow.events.append(("silence", round(end, 2), round(gap, 2)))
            flow.problems += _weight(gap, settings["waits"])
        if side == "caller" and not turn["noise"] and not _backchannel(turn, settings):
            last_caller = turn
        if end is None or e >= end:
            end, before = e, (side, turn)
    # Talking over the caller: a real agent turn that starts while a caller turn is under way.
    for a in agent:
        if _backchannel(a, settings):
            continue
        c = next((x for x in caller if x["s"] + 0.2 < a["s"] < x["e"] - 0.2), None)
        if c is not None and (overlap := min(c["e"], a["e"]) - a["s"]) >= settings["talkOverSeconds"]:
            flow.talk_over.append(round(overlap, 2))
            flow.events.append(("talk_over", round(a["s"], 2), round(overlap, 2)))
            flow.problems += settings["talkOverWeight"]
    # Interruptions: a real caller turn that starts while the agent is talking; how long the agent's speech runs on,
    # across turns, until it pauses. A caller carrying on after the agent jumped into their pause is counted above.
    words = sorted((s, e) for x in agent for s, e, _ in x["words"])
    for c in caller:
        said = " ".join(c["text"])
        if _backchannel(c, settings) or id(c) in continued or len([w for w in c["text"] if w]) < settings["stopMinWords"]:
            continue  # a short "wait" is not an interruption to stop for
        if any(phrase in said for phrase in settings["checkInPhrases"]):
            continue  # "are you still there?" while the agent answers: the agent is right to keep talking
        if not any(x["s"] + 0.2 < c["s"] < x["e"] - 0.3 for x in agent):
            continue
        stop_at = c["s"]
        for s, e in words:
            if e <= c["s"]:
                continue
            if s > stop_at + settings["stopPauseSeconds"]:
                break
            stop_at = max(stop_at, e)
        stop = max(0.0, stop_at - c["s"])
        flow.stops.append(round(stop, 2))
        flow.events.append(("stop", round(c["s"], 2), round(stop, 2)))
        flow.problems += _weight(stop, settings["stops"])
    flow.opportunities = len(agent) + len(flow.stops)
    return flow
