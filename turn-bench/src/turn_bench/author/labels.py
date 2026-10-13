"""Sesame annotation tracks from the timeline, and the gold events the script plants.

Annotation events are ``{start_s, end_s, label, text}`` per speaker, exactly the
dataset's ``speaker_{n}_annotation_{a,b,c}`` columns. The three tracks are
identical — they are derived from the script, not from three annotators — and
the dataset metadata says so. Labels are Sesame's fine labels (``turnbench.gold.LABEL_MAP``):

    turn item        every segment            "Normal Turn"
    barge_in item    first segment            "Floor-taking Competitive Interruption"
                     later segments (if any)  "Normal Turn"
    backchannel item every segment            "Acknowledgement Backchannel"

``planted_events`` states what ``turnbench.gold`` must derive from those tracks,
written from the script's own floor logic rather than Sesame's code:

    EOT positive  a floor-claiming segment (turn / barge-in) ends and the other
                  speaker speaks next, is already speaking, or nobody speaks
                  again — including the agent's segment a barge-in cut short;
    EOT negative  a floor-claiming segment ends and the same speaker speaks
                  next: the pause until they resume, cut at the first of their
                  own backchannels or the other speaker's barge-ins inside it;
    INT positive  a barge-in onset, on the barge-in's channel;
    INT negative  a backchannel's extent, on its channel.

Note the consequence for a caller who barges in and then continues ("one
quick question" ... pause ... the question): the pause after the barge-in is a
mid-turn pause (EOT negative), and the floor only passes when the agent
replies. ``check`` compares these planted events with what Sesame's gold
builder derives, time by time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from turnbench.data import ANNOTATORS, Conversation
from turnbench.gold import AnchorEvent, ConversationEvents, Interval

from turn_bench.author.compose import Timeline

LABEL_TURN = "Normal Turn"
LABEL_BARGE_IN = "Floor-taking Competitive Interruption"
LABEL_BACKCHANNEL = "Acknowledgement Backchannel"
FLOOR_KINDS = ("turn", "barge_in")

Annotation = dict[str, Any]  # {"start_s", "end_s", "label", "text"}


def annotation_track(timeline: Timeline, channel: int) -> list[Annotation]:
    """One speaker's annotation events, sorted by start."""
    rate = timeline.sample_rate
    events: list[Annotation] = []
    for placed in timeline.items:
        if placed.channel != channel:
            continue
        for index, segment in enumerate(placed.segments):
            if placed.item.kind == "backchannel":
                label = LABEL_BACKCHANNEL
            elif placed.item.kind == "barge_in" and index == 0:
                label = LABEL_BARGE_IN
            else:
                label = LABEL_TURN
            events.append({"start_s": segment.start / rate, "end_s": segment.end / rate, "label": label, "text": segment.text})
    return sorted(events, key=lambda e: e["start_s"])


def annotation_tracks(timeline: Timeline) -> dict[int, list[Annotation]]:
    return {channel: annotation_track(timeline, channel) for channel in (1, 2)}


def as_conversation(conversation_id: str, tracks: dict[int, list[Annotation]], duration_s: float) -> Conversation:
    """A Sesame ``Conversation`` (no audio) with the three identical tracks, for
    running ``turnbench.gold.events_for_conversation`` in memory."""
    return Conversation(
        conversation_id=conversation_id,
        duration_s=duration_s,
        annotations={
            (channel, annotator): [(e["start_s"], e["end_s"], e["label"], e["text"]) for e in tracks[channel]]
            for channel in (1, 2)
            for annotator in ANNOTATORS
        },
        audio_bytes={},
    )


# ---- planted gold -------------------------------------------------------------------


@dataclass
class Planted:
    eot_positive: list[AnchorEvent] = field(default_factory=list)
    eot_negative: list[Interval] = field(default_factory=list)
    int_positive: list[AnchorEvent] = field(default_factory=list)
    int_negative: list[Interval] = field(default_factory=list)

    def counts(self, speaker: int | None = None) -> dict[str, int]:
        def n(events) -> int:
            return sum(1 for e in events if speaker is None or e.speaker == speaker)

        return {"eot_pos": n(self.eot_positive), "eot_neg": n(self.eot_negative), "int_pos": n(self.int_positive), "int_neg": n(self.int_negative)}


def planted_events(timeline: Timeline) -> Planted:
    rate = timeline.sample_rate
    floor: dict[int, list[tuple[int, int]]] = {1: [], 2: []}  # (start, end) samples per channel
    backchannels: dict[int, list[tuple[int, int]]] = {1: [], 2: []}
    barge_onsets: dict[int, list[int]] = {1: [], 2: []}
    for placed in timeline.items:
        if placed.item.kind in FLOOR_KINDS:
            floor[placed.channel].extend((s.start, s.end) for s in placed.segments)
        else:
            backchannels[placed.channel].extend((s.start, s.end) for s in placed.segments)
        if placed.item.kind == "barge_in":
            barge_onsets[placed.channel].append(placed.segments[0].start)
    for channel in floor:
        floor[channel].sort()

    planted = Planted()
    for speaker in (1, 2):
        other = 2 if speaker == 1 else 1
        own = floor[speaker]
        for start, end in own:
            next_self = min((s for s, _ in own if s > end), default=None)
            next_other = min((s for s, _ in floor[other] if s > end), default=None)
            other_speaking = any(s < end < e for s, e in floor[other])
            if next_self is None or (next_other is not None and next_other < next_self) or other_speaking:
                planted.eot_positive.append(AnchorEvent(speaker, end / rate))
            else:
                cut = [next_self]
                cut += [s for s, _ in backchannels[speaker] if end < s < next_self]
                cut += [s for s in barge_onsets[other] if end < s < next_self]
                pause_end = min(cut)
                if pause_end > end:
                    planted.eot_negative.append(Interval(speaker, end / rate, pause_end / rate))
        planted.int_positive.extend(AnchorEvent(speaker, s / rate) for s in barge_onsets[speaker])
        planted.int_negative.extend(Interval(speaker, s / rate, e / rate) for s, e in backchannels[speaker])
    return planted


# ---- planted vs derived ----------------------------------------------------------------

TIME_TOLERANCE_S = 0.001  # the gold rounds medians to 4 decimals; sample times are exact


def _anchor_key(event: AnchorEvent) -> tuple[int, float]:
    return event.speaker, round(event.time_s, 3)


def _interval_key(interval: Interval) -> tuple[int, float, float]:
    return interval.speaker, round(interval.start, 3), round(interval.end, 3)


def compare(planted: Planted, derived: ConversationEvents) -> list[str]:
    """Differences between the planted events and Sesame's derived gold; empty when identical."""
    problems: list[str] = []
    pairs = [
        ("eot_positive", planted.eot_positive, derived.eot_positive_events, _anchor_key),
        ("eot_negative", planted.eot_negative, derived.eot_negative_spans, _interval_key),
        ("int_positive", planted.int_positive, derived.int_positive_events, _anchor_key),
        ("int_negative", planted.int_negative, derived.int_negative_spans, _interval_key),
    ]
    for name, ours, theirs, key in pairs:
        ours_set = {key(e) for e in ours}
        theirs_set = {key(e) for e in theirs}
        for missing in sorted(ours_set - theirs_set):
            problems.append(f"{name}: planted {missing} not derived")
        for extra in sorted(theirs_set - ours_set):
            problems.append(f"{name}: derived {extra} not planted")
    if derived.eot_excluded or derived.int_excluded:
        problems.append(f"excluded intervals derived: eot={derived.eot_excluded} int={derived.int_excluded}")
    return problems
