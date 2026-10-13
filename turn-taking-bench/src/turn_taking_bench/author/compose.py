"""Two speakers on one clock: exact sample offsets, barge-in cuts, ground truth.

Output is two mono float32 channels of equal length at ``OUTPUT_RATE`` (48 kHz,
the rate of Sesame's released FLAC channels, so the harness streams authored
and released conversations identically). Every placement is an integer sample
on that clock:

* a ``turn`` item's speech onset lands its gap (``spec.gap_for``: ``hold_s``,
  ``gap_s``, the seeded draw, or the default) after its anchor's speech end
  (the first item at ``timing.start_s``);
* a ``backchannel`` / ``barge_in`` onset lands ``offset_s`` after the onset of
  the agent item it overlaps, or ``offset_from_end_s`` before that item's
  (uncut) speech end;
* a barge-in cuts the agent ``stop_delay_s`` after the barge-in onset: a linear
  fade of ``fade_s`` ends at the cut sample, everything after it is silence,
  and the agent's segments that start at or after the cut are dropped. The cut
  must land before the agent's speech end (a "barge-in" after the agent has
  finished is a scripting error, not an overlap). A backchannel changes nothing
  on the agent channel.

"Speech onset/end" are the clip's detected boundaries (``bounds``), so a
render's leading and trailing silence does not move the timeline; it is kept
in the channel (it is silence) unless it would run before sample 0.

Clips rendered at another rate are resampled with the service bench's Kaiser
polyphase FIR (``service.audio.resample``): zero-phase, so boundaries scale
exactly (24 kHz frame grid -> 48 kHz: ×2).

The result carries, per item, the placed segments with their text — the ground
truth the labels are written from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from turn_taking_bench.author import service_bridge
from turn_taking_bench.author.bounds import Bounds
from turn_taking_bench.author.render import Render
from turn_taking_bench.author.spec import ConversationSpec, Item

OUTPUT_RATE = 48000


class ComposeError(ValueError):
    """The script cannot be placed (overlap on one channel, a cut that lands
    before the agent's onset, segment count mismatch)."""


@dataclass(frozen=True)
class PlacedSegment:
    start: int  # output samples
    end: int
    text: str

    def start_s(self, rate: int) -> float:
        return self.start / rate

    def end_s(self, rate: int) -> float:
        return self.end / rate


@dataclass
class PlacedItem:
    item: Item
    channel: int
    audio: np.ndarray  # float32 at the output rate, positioned at `offset`
    offset: int  # output sample of audio[0]
    segments: list[PlacedSegment]
    render_sha256: str
    cut_at: int | None = None  # output sample where a barge-in cut this item
    dropped_segments: list[PlacedSegment] = field(default_factory=list)
    uncut_end: int | None = None  # the speech end this item had before a barge-in cut it

    @property
    def speech_start(self) -> int:
        return self.segments[0].start

    @property
    def speech_end(self) -> int:
        return self.segments[-1].end

    @property
    def audio_end(self) -> int:
        return self.offset + len(self.audio)


@dataclass
class Timeline:
    spec: ConversationSpec
    sample_rate: int
    items: list[PlacedItem]
    channels: dict[int, np.ndarray]  # may be released after packaging; the length is kept
    length: int = 0  # samples per channel

    @property
    def duration_samples(self) -> int:
        return self.length

    @property
    def duration_s(self) -> float:
        return self.duration_samples / self.sample_rate

    def item(self, item_id: str) -> PlacedItem:
        for placed in self.items:
            if placed.item.id == item_id:
                return placed
        raise KeyError(item_id)

    def ground_truth(self) -> dict[str, Any]:
        """The JSON record of what was placed where (seconds and samples)."""
        rate = self.sample_rate
        return {
            "sample_rate": rate,
            "duration_s": self.duration_s,
            "items": [
                {
                    "id": p.item.id,
                    "speaker": p.item.speaker,
                    "channel": p.channel,
                    "kind": p.item.kind,
                    "text": p.item.text,
                    "speed": p.item.speed,
                    "gap_s": self.spec.gap_for(p.item) if p.item.kind == "turn" and p is not self.items[0] else None,
                    "offset_s": None if p.item.kind == "turn" else (p.speech_start - self.item(p.item.into).speech_start) / rate,
                    "offset_from_end_s": None if p.item.kind == "turn" else ((self.item(p.item.into).uncut_end or self.item(p.item.into).speech_end) - p.speech_start) / rate,
                    "render_sha256": p.render_sha256,
                    "onset_s": p.speech_start / rate,
                    "end_s": p.speech_end / rate,
                    "cut_at_s": None if p.cut_at is None else p.cut_at / rate,
                    "uncut_end_s": None if p.uncut_end is None else p.uncut_end / rate,
                    "segments": [
                        {"start": s.start, "end": s.end, "start_s": s.start / rate, "end_s": s.end / rate, "text": s.text}
                        for s in p.segments
                    ],
                    "dropped_segments": [
                        {"start": s.start, "end": s.end, "text": s.text} for s in p.dropped_segments
                    ],
                }
                for p in self.items
            ],
        }


def to_output_rate(samples: np.ndarray, sample_rate: int, output_rate: int) -> np.ndarray:
    if sample_rate == output_rate:
        return np.asarray(samples, dtype=np.float32)
    audio = service_bridge.audio()
    pcm = audio.to_pcm(np.asarray(samples, dtype=np.float64) * 32767.0)
    resampled = audio.to_array(audio.resample(pcm, sample_rate, output_rate))
    return (resampled.astype(np.float32) / 32768.0).astype(np.float32)


def _samples(seconds: float, rate: int) -> int:
    return int(round(seconds * rate))


def _cut(placed: PlacedItem, cut_at: int, fade_s: float, rate: int) -> None:
    """Silence `placed` from output sample `cut_at` on, fading out before it."""
    local = cut_at - placed.offset
    if cut_at >= placed.speech_end:
        raise ComposeError(
            f"item {placed.item.id!r}: the barge-in cut at {cut_at / rate:.2f}s lands after its speech "
            f"ends at {placed.speech_end / rate:.2f}s; the barge-in would not overlap the agent"
        )
    if local <= 0:
        raise ComposeError(f"item {placed.item.id!r}: cut at {cut_at} lands before its audio starts")
    placed.uncut_end = placed.speech_end
    fade = min(_samples(fade_s, rate), local)
    if fade:
        placed.audio[local - fade : local] *= np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
    placed.audio[local:] = 0.0
    placed.cut_at = cut_at
    kept: list[PlacedSegment] = []
    for segment in placed.segments:
        if segment.start >= cut_at:
            placed.dropped_segments.append(segment)
        elif segment.end > cut_at:
            kept.append(PlacedSegment(segment.start, cut_at, segment.text))
        else:
            kept.append(segment)
    if not kept:
        raise ComposeError(f"item {placed.item.id!r}: cut at {cut_at} leaves no speech")
    placed.segments = kept


def compose(
    spec: ConversationSpec,
    renders: dict[str, tuple[Render, Bounds]],
    *,
    output_rate: int = OUTPUT_RATE,
) -> Timeline:
    """Place every item of `spec` using its render and detected bounds."""
    placed_items: list[PlacedItem] = []
    by_id: dict[str, PlacedItem] = {}
    for item in spec.items:
        render, bounds = renders[item.id]
        texts = item.segments
        if len(bounds.segments) != len(texts):
            raise ComposeError(
                f"spec {spec.id}, item {item.id!r}: {len(texts)} scripted segment(s) but the render has "
                f"{len(bounds.segments)} detected run(s) of speech "
                f"({[(round(s.start_s, 2), round(s.end_s, 2)) for s in bounds.segments]}); "
                "a break may have rendered too short or a sentence pause too long"
            )
        audio = to_output_rate(render.samples, render.sample_rate, output_rate).copy()
        local = [(_samples(s.start_s, output_rate), _samples(s.end_s, output_rate)) for s in bounds.segments]

        if item.kind == "turn":
            if placed_items:
                anchor = by_id[item.after] if item.after else max(placed_items, key=lambda p: p.speech_end)
                onset = anchor.speech_end + _samples(spec.gap_for(item), output_rate)
            else:
                onset = _samples(spec.timing.start_s, output_rate)
        elif item.offset_from_end_s is not None:
            onset = by_id[item.into].speech_end - _samples(item.offset_from_end_s, output_rate)
            if onset <= by_id[item.into].speech_start:
                raise ComposeError(f"spec {spec.id}, item {item.id!r}: offset_from_end_s {item.offset_from_end_s} reaches before the onset of {item.into!r}")
        else:
            onset = by_id[item.into].speech_start + _samples(item.offset_s, output_rate)
        if item.kind == "backchannel" and onset >= by_id[item.into].speech_end:
            # A backchannel that starts after the agent has finished is not an overlap:
            # it would be labelled as one while sounding like a short turn. (A barge-in
            # placed too late fails in ``_cut`` the same way.)
            raise ComposeError(
                f"spec {spec.id}, item {item.id!r}: the backchannel onset at {onset / output_rate:.2f}s lands after "
                f"{item.into!r} ends at {by_id[item.into].speech_end / output_rate:.2f}s; it would not overlap the agent"
            )

        offset = onset - local[0][0]
        if offset < 0:  # leading silence longer than the timeline has room for
            audio = audio[-offset:]
            local = [(s + offset, e + offset) for s, e in local]
            offset = 0
        segments = [PlacedSegment(offset + s, offset + e, text) for (s, e), text in zip(local, texts)]
        placed = PlacedItem(item=item, channel=item.channel, audio=audio, offset=offset, segments=segments, render_sha256=render.sha256)

        if item.kind == "barge_in":
            target = by_id[item.into]
            _cut(target, onset + _samples(item.stop_delay_s, output_rate), item.fade_s, output_rate)

        for other in placed_items:
            if other.channel == placed.channel and other.speech_start < placed.speech_end and placed.speech_start < other.speech_end:
                raise ComposeError(f"spec {spec.id}: items {other.item.id!r} and {item.id!r} overlap on the same channel")
        placed_items.append(placed)
        by_id[item.id] = placed

    speech_end = max(p.speech_end for p in placed_items)
    length = max(speech_end + _samples(spec.timing.tail_s, output_rate), max(p.audio_end for p in placed_items))
    channels = {1: np.zeros(length, dtype=np.float32), 2: np.zeros(length, dtype=np.float32)}
    for placed in placed_items:
        channels[placed.channel][placed.offset : placed.audio_end] += placed.audio
    for channel in channels.values():
        np.clip(channel, -1.0, 1.0, out=channel)
    return Timeline(spec=spec, sample_rate=output_rate, items=placed_items, channels=channels, length=length)
