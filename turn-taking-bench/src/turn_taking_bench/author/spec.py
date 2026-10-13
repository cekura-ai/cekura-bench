"""Conversation specs: the versioned JSON script an authored conversation is built from.

Schema (``schema_version`` 2)::

    {
      "schema_version": 2,
      "id": "TB-01",                       # scenario id; goes into the dataset metadata
      "conversation_id": "901",            # numeric string: Sesame sorts ids with int()
      "title": "Pause ladder",
      "family": "pauses",                  # free tags for stratified reporting
      "subtype": "clause_pause",           # finer tag within the family (optional)
      "difficulty": "medium",
      "tags": ["eot", "mid-turn-pause"],
      "pairings": [                        # one conversation per caller/agent voice pairing
        {"caller": "f-us", "agent": "m-us"},   # labels from render.VOICES
        {"caller": "m-us", "agent": "f-gb"},
        {"caller": "f-gb", "agent": "f-us"}
      ],
      "timing": {                          # all optional
        "start_s": 0.5,                    # first speech onset on the timeline
        "tail_s": 1.0,                     # silence after the last speech
        "default_gap_s": 0.6,              # gap for items that give none...
        "gap_range_s": [0.5, 1.0],         # ...or, when given, a seeded draw from this range
        "gap_seed": 2026
      },
      "items": [ ...ordered items... ]
    }

``pairings`` expands the file into one conversation per pairing, numbered in
list order from 1: pairing ``i`` gets ``conversation_id + str(i)`` (``"901"`` ->
``"9011"``, ``"9012"``, ``"9013"``), the same ``id``, and ``pairing`` set to
``"<caller>/<agent>"``. A single conversation is written with ``"voices":
{"caller": ..., "agent": ...}`` instead (exactly one of the two).

Gaps between turns: an item with no ``gap_s`` takes ``timing.gap_range_s`` if
given — a uniform draw seeded by ``gap_seed``, the conversation id and the
item id, so every build of a spec places identically, each pairing differs,
and no single constant separates turns — else ``timing.default_gap_s``.

An item is one rendered utterance of one speaker::

    {"id": "c1", "speaker": "caller", "text": "Hi. <break time=\\"1s\\"/> I need help.",
     "gap_s": 0.0}                                    # kind "turn" (default)
    {"id": "a1", "speaker": "agent", "text": "...", "after": "c1", "gap_s": 0.7}
    {"id": "c2", "speaker": "caller", "text": "Um, <break time=\\"1s\\"/> so ...",
     "hold_s": 3.5, "speed": 0.8, "min_pause_s": 0.3}
    {"id": "b1", "speaker": "caller", "kind": "backchannel", "text": "Mm-hmm.",
     "into": "a1", "offset_s": 1.5}
    {"id": "x1", "speaker": "caller", "kind": "barge_in", "text": "Sorry, stop.",
     "into": "a1", "offset_s": 1.5, "stop_delay_s": 0.4, "fade_s": 0.02}
    {"id": "x2", "speaker": "caller", "kind": "barge_in", "text": "Wait.",
     "into": "a2", "offset_from_end_s": 1.2}                 # placed from the agent's speech END

* ``speaker``: ``caller`` is Sesame's speaker_1 channel, ``agent`` speaker_2.
* ``text`` may contain pause markers. The SSML form ``<break time="1.5s"/>``
  (also ``time="300ms"``) is what ElevenLabs renders; the shorthand ``<1.5s>``
  is accepted and normalised to it. Each marker splits the utterance into
  segments; the composer labels every segment and the gap between them is a
  mid-turn pause.
* ``speed``: the TTS speaking rate for this line (default 1.0; ElevenLabs'
  range is 0.7-1.2). ``min_pause_s``: the shortest detected length a scripted
  break of this line may have and still be a labelled pause (default
  ``bounds.MIN_PAUSE_S``); lower it for lines that script sub-0.6 s clause
  pauses on purpose — the ambiguity rule against natural gaps still applies.
  ``splice_breaks``: render each segment as its own TTS call and join them
  with exactly the scripted silence (for a line where the TTS stretches a
  filler across the break instead of pausing); the pause report marks these.
* ``kind``:
  - ``turn`` — placed ``gap_s`` after the speech end of ``after`` (an item id;
    default: whichever item placed so far ends last). The first item starts at
    ``timing.start_s``. ``hold_s`` instead of ``gap_s`` is scripted dead air: the
    turn starts exactly ``hold_s`` after the anchor (no drawn gap), with nothing
    on either channel in between.
  - ``backchannel`` — starts ``offset_s`` after the speech onset of ``into`` (an
    agent item); the agent continues uninterrupted.
  - ``barge_in`` — starts ``offset_s`` after the onset of ``into``; the agent's
    audio is cut ``stop_delay_s`` later (``fade_s`` linear fade ending at the cut)
    and the agent's remaining segments are dropped.
  Either kind may give ``offset_from_end_s`` instead of ``offset_s``: the onset
  is that many seconds *before* the speech end of ``into`` as rendered (so "in
  the last second of the reply" is exact for every voice, whose renders differ
  in length). ``offset_s`` and ``offset_from_end_s`` are mutually exclusive.

Items are processed in list order, so an item can only reference an earlier one.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
SPEAKERS = {"caller": 1, "agent": 2}
KINDS = ("turn", "barge_in", "backchannel")

DEFAULT_START_S = 0.5
DEFAULT_TAIL_S = 1.0
DEFAULT_GAP_S = 0.6
DEFAULT_STOP_DELAY_S = 0.4
DEFAULT_FADE_S = 0.02
SPEED_RANGE = (0.7, 1.2)  # render.SPEED_MIN / SPEED_MAX (kept here so spec needs no renderer)

_SSML_BREAK = re.compile(r'<break\s+time="(?P<value>\d+(?:\.\d+)?)(?P<unit>s|ms)"\s*/>')
_SHORT_BREAK = re.compile(r"<(?P<value>\d+(?:\.\d+)?)(?P<unit>s|ms)>")
_CONVERSATION_ID = re.compile(r"^\d+$")


class SpecError(ValueError):
    """A spec that cannot be built; the message names the spec and the field."""


def normalise_breaks(text: str) -> str:
    """Rewrite ``<1.5s>`` shorthand to the SSML ``<break time="1.5s"/>`` form."""
    return _SHORT_BREAK.sub(lambda m: f'<break time="{m.group("value")}{m.group("unit")}"/>', text)


def split_breaks(text: str) -> tuple[list[str], list[float]]:
    """``text`` -> (segment texts, break durations in seconds between them).

    ``len(breaks) == len(segments) - 1``. Segment texts are stripped; an empty
    segment (two adjacent markers, or a marker at either end) is an error
    because it would be a pause without speech on either side.
    """
    text = normalise_breaks(text)
    segments: list[str] = []
    breaks: list[float] = []
    cursor = 0
    for match in _SSML_BREAK.finditer(text):
        segments.append(text[cursor : match.start()].strip())
        value = float(match.group("value"))
        breaks.append(value / 1000.0 if match.group("unit") == "ms" else value)
        cursor = match.end()
    segments.append(text[cursor:].strip())
    if any(not segment for segment in segments):
        raise SpecError(f"break marker without speech on both sides in {text!r}")
    return segments, breaks


@dataclass(frozen=True)
class Item:
    id: str
    speaker: str  # "caller" | "agent"
    text: str  # with SSML break markers
    kind: str = "turn"
    after: str | None = None  # turn: anchor item id (None = latest end so far)
    gap_s: float | None = None  # turn: silence after the anchor's speech end
    hold_s: float | None = None  # turn: scripted dead air before the line, replaces the gap
    into: str | None = None  # barge_in / backchannel: the agent item overlapped
    offset_s: float = 0.0  # barge_in / backchannel: seconds after `into`'s onset
    offset_from_end_s: float | None = None  # barge_in / backchannel: seconds before `into`'s speech end instead
    stop_delay_s: float = DEFAULT_STOP_DELAY_S  # barge_in: agent keeps talking this long
    fade_s: float = DEFAULT_FADE_S  # barge_in: fade length ending at the cut
    speed: float = 1.0  # TTS speaking rate for this line
    min_pause_s: float | None = None  # labelled-pause minimum for this line's breaks (None = bounds default)
    splice_breaks: bool = False  # render segments separately, join with exact scripted silence

    @property
    def channel(self) -> int:
        return SPEAKERS[self.speaker]

    @property
    def segments(self) -> list[str]:
        return split_breaks(self.text)[0]

    @property
    def breaks(self) -> list[float]:
        return split_breaks(self.text)[1]


@dataclass(frozen=True)
class Timing:
    start_s: float = DEFAULT_START_S
    tail_s: float = DEFAULT_TAIL_S
    default_gap_s: float = DEFAULT_GAP_S
    gap_range_s: tuple[float, float] | None = None
    gap_seed: int = 0


@dataclass(frozen=True)
class ConversationSpec:
    id: str
    conversation_id: str
    title: str
    voices: dict[str, str]  # speaker -> voice label
    items: list[Item]
    family: str = ""
    subtype: str = ""
    difficulty: str = ""
    tags: list[str] = field(default_factory=list)
    timing: Timing = Timing()
    pairing: str | None = None  # "<caller>/<agent>" when expanded from `pairings`
    source_path: str | None = None
    source_sha256: str | None = None

    def item(self, item_id: str) -> Item:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def gap_for(self, item: Item) -> float:
        """The silence placed before a turn item: its ``hold_s`` or ``gap_s``, else
        the seeded draw from ``timing.gap_range_s``, else ``timing.default_gap_s``."""
        if item.hold_s is not None:
            return item.hold_s
        if item.gap_s is not None:
            return item.gap_s
        if self.timing.gap_range_s is not None:
            low, high = self.timing.gap_range_s
            rng = random.Random(f"{self.timing.gap_seed}:{self.conversation_id}:{item.id}")
            return round(rng.uniform(low, high), 3)
        return self.timing.default_gap_s


def _item_from_json(raw: dict[str, Any], spec_id: str, seen: dict[str, Item]) -> Item:
    def fail(message: str) -> SpecError:
        return SpecError(f"spec {spec_id}, item {raw.get('id')!r}: {message}")

    allowed = {"id", "speaker", "text", "kind", "after", "gap_s", "hold_s", "into", "offset_s", "offset_from_end_s", "stop_delay_s", "fade_s", "speed", "min_pause_s", "splice_breaks"}
    unknown = set(raw) - allowed
    if unknown:
        raise fail(f"unknown fields {sorted(unknown)}")
    for key in ("id", "speaker", "text"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise fail(f"{key!r} must be a non-empty string")
    if raw["id"] in seen:
        raise fail("duplicate item id")
    if raw["speaker"] not in SPEAKERS:
        raise fail(f"speaker must be one of {sorted(SPEAKERS)}")
    kind = raw.get("kind", "turn")
    if kind not in KINDS:
        raise fail(f"kind must be one of {KINDS}")
    split_breaks(raw["text"])  # validates the markers
    item = Item(
        id=raw["id"],
        speaker=raw["speaker"],
        text=normalise_breaks(raw["text"].strip()),
        kind=kind,
        after=raw.get("after"),
        gap_s=raw.get("gap_s"),
        hold_s=raw.get("hold_s"),
        into=raw.get("into"),
        offset_s=float(raw.get("offset_s", 0.0)),
        offset_from_end_s=None if raw.get("offset_from_end_s") is None else float(raw["offset_from_end_s"]),
        stop_delay_s=float(raw.get("stop_delay_s", DEFAULT_STOP_DELAY_S)),
        fade_s=float(raw.get("fade_s", DEFAULT_FADE_S)),
        speed=float(raw.get("speed", 1.0)),
        min_pause_s=raw.get("min_pause_s"),
        splice_breaks=bool(raw.get("splice_breaks", False)),
    )
    if not isinstance(raw.get("splice_breaks", False), bool):
        raise fail("splice_breaks must be a boolean")
    if not (SPEED_RANGE[0] <= item.speed <= SPEED_RANGE[1]):
        raise fail(f"speed must be within {SPEED_RANGE}")
    if item.min_pause_s is not None and item.min_pause_s <= 0:
        raise fail("min_pause_s must be > 0")
    if kind == "turn":
        if item.into is not None or item.offset_from_end_s is not None:
            raise fail("'into' / 'offset_from_end_s' are only for barge_in / backchannel items")
        if item.after is not None and item.after not in seen:
            raise fail(f"'after' must name an earlier item, got {item.after!r}")
        if item.gap_s is not None and item.gap_s < 0:
            raise fail("gap_s must be >= 0")
        if item.hold_s is not None and (item.hold_s <= 0 or item.gap_s is not None):
            raise fail("hold_s must be > 0 and replaces gap_s")
    else:
        if item.after is not None or item.gap_s is not None or item.hold_s is not None:
            raise fail("'after'/'gap_s'/'hold_s' are only for turn items")
        if item.into not in seen:
            raise fail(f"'into' must name an earlier item, got {item.into!r}")
        if seen[item.into].speaker == item.speaker:
            raise fail("'into' must name the other speaker's item")
        if seen[item.into].kind != "turn":
            raise fail("'into' must name a turn item")
        if item.offset_s < 0:
            raise fail("offset_s must be >= 0")
        if item.offset_from_end_s is not None and ("offset_s" in raw or item.offset_from_end_s <= 0):
            raise fail("offset_from_end_s must be > 0 and replaces offset_s")
        if kind == "barge_in" and (item.stop_delay_s < 0 or item.fade_s < 0):
            raise fail("stop_delay_s and fade_s must be >= 0")
    return item


def _voices_from_json(raw: dict[str, Any], spec_id: str) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != set(SPEAKERS):
        raise SpecError(f"spec {spec_id}: voices must name exactly {sorted(SPEAKERS)}")
    return {speaker: str(label) for speaker, label in raw.items()}


def _timing_from_json(raw: dict[str, Any], spec_id: str) -> Timing:
    gap_range = raw.get("gap_range_s")
    if gap_range is not None:
        if not (isinstance(gap_range, (list, tuple)) and len(gap_range) == 2 and 0 <= float(gap_range[0]) <= float(gap_range[1])):
            raise SpecError(f"spec {spec_id}: timing.gap_range_s must be [low, high] with 0 <= low <= high")
        gap_range = (float(gap_range[0]), float(gap_range[1]))
    return Timing(
        start_s=float(raw.get("start_s", DEFAULT_START_S)),
        tail_s=float(raw.get("tail_s", DEFAULT_TAIL_S)),
        default_gap_s=float(raw.get("default_gap_s", DEFAULT_GAP_S)),
        gap_range_s=gap_range,
        gap_seed=int(raw.get("gap_seed", 0)),
    )


def specs_from_json(raw: dict[str, Any], *, source_path: str | None = None, source_sha256: str | None = None) -> list[ConversationSpec]:
    """Every conversation a spec file describes: one per ``pairings`` entry, or
    the single ``voices`` conversation (see the module docstring)."""
    spec_id = raw.get("id", "<no id>")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise SpecError(f"spec {spec_id}: schema_version {raw.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    for key in ("id", "conversation_id", "title", "items"):
        if key not in raw:
            raise SpecError(f"spec {spec_id}: missing {key!r}")
    if ("voices" in raw) == ("pairings" in raw):
        raise SpecError(f"spec {spec_id}: give exactly one of 'voices' (one conversation) or 'pairings' (one per pairing)")
    if not isinstance(raw["conversation_id"], str) or not _CONVERSATION_ID.match(raw["conversation_id"]):
        raise SpecError(f"spec {spec_id}: conversation_id must be a string of digits (Sesame sorts ids numerically)")
    timing = _timing_from_json(raw.get("timing", {}), spec_id)
    seen: dict[str, Item] = {}
    items: list[Item] = []
    for raw_item in raw["items"]:
        item = _item_from_json(raw_item, spec_id, seen)
        seen[item.id] = item
        items.append(item)
    if not items:
        raise SpecError(f"spec {spec_id}: no items")
    if "voices" in raw:
        expansions = [(raw["conversation_id"], _voices_from_json(raw["voices"], spec_id), None)]
    else:
        pairings = raw["pairings"]
        if not isinstance(pairings, list) or not pairings:
            raise SpecError(f"spec {spec_id}: pairings must be a non-empty list")
        expansions = []
        for index, pairing in enumerate(pairings, start=1):
            voices = _voices_from_json(pairing, spec_id)
            expansions.append((raw["conversation_id"] + str(index), voices, f"{voices['caller']}/{voices['agent']}"))
        if len({pairing for _, _, pairing in expansions}) != len(expansions):
            raise SpecError(f"spec {spec_id}: duplicate pairing")
    return [
        ConversationSpec(
            id=raw["id"],
            conversation_id=conversation_id,
            title=raw["title"],
            voices=voices,
            items=items,
            family=raw.get("family", ""),
            subtype=raw.get("subtype", ""),
            difficulty=raw.get("difficulty", ""),
            tags=list(raw.get("tags", [])),
            timing=timing,
            pairing=pairing,
            source_path=source_path,
            source_sha256=source_sha256,
        )
        for conversation_id, voices, pairing in expansions
    ]


def spec_from_json(raw: dict[str, Any], **kwargs: Any) -> ConversationSpec:
    """A spec that describes exactly one conversation (the ``voices`` form)."""
    specs = specs_from_json(raw, **kwargs)
    if len(specs) != 1:
        raise SpecError(f"spec {raw.get('id')}: describes {len(specs)} conversations, expected one (use specs_from_json)")
    return specs[0]


def load_spec(path: Path) -> list[ConversationSpec]:
    import hashlib

    data = path.read_bytes()
    return specs_from_json(
        json.loads(data.decode("utf-8")),
        source_path=path.name,
        source_sha256=hashlib.sha256(data).hexdigest(),
    )


def load_specs(directory: Path) -> list[ConversationSpec]:
    """Every conversation of every ``*.json`` under ``directory`` (sorted by
    name, then pairing order); conversation ids must be unique."""
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise SpecError(f"no *.json specs in {directory}")
    specs = [spec for path in paths for spec in load_spec(path)]
    ids = [spec.conversation_id for spec in specs]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        raise SpecError(f"duplicate conversation_id across specs: {duplicates}")
    return specs
