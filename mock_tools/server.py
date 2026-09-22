"""Serves the public mock-tool contract to whatever is being measured.

Shared on purpose. The service bench talks to a provider websocket directly, the text arm
talks to the same model with no audio, and the agent bench reference agent talks over
a phone line -- and all three must be answered by the *same* tool implementation,
or a difference between lanes could be our two servers disagreeing rather than
anything about the agents.

The contract in ``agent-definitions/`` is tool schemas plus input-to-output
lookup tables. It is published, so a third party can run the same scenarios
against their own agent and be scored the same way -- but the repository ships no
server for it, and scoring has until now happened inside a platform. This module
is that missing piece, and it deliberately serves every lane: the direct provider
websocket, the text control arm, and the reference agent.

That matters more than it sounds. A control needs a treatment that differs from
it in exactly one thing. Text arm to service-bench voice differs by *the speech pathway* alone;
service bench to agent bench differs by *framework and telephony* alone. If tools only existed in the
lane that also introduced a framework and a phone line, a pass-in-text /
fail-in-voice result could be blamed on any of the three, which is precisely the
ambiguity the control exists to remove.

State: v1 is **stateless**, because the published tables are stateless. Booking
then cancelling then verifying needs a store, and that is a contract extension
rather than something to fake here.
"""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mock_tools.spec import ToolSpec

DEFINITIONS_ROOT = Path(__file__).resolve().parent.parent / "agent-definitions"

# Why a call resolved the way it did is the question every tool-accuracy
# investigation starts from, so the answer is logged rather than reconstructed
# afterwards. Standard library logging keeps this module free of a dependency:
# the agent under test bridges it into its own log, and nothing is emitted
# unless the host configures a handler.
log = logging.getLogger("mock_tools")


# One slot, several spellings. The contract asks for ``YYYY-MM-DDTHH:MM:SS`` and
# services oblige to different degrees: a space where the T belongs, the seconds
# left off, fractional seconds, a trailing Z. All name the same slot, and scoring
# them as different appointments would rank a service on its ISO spelling.
#
# A real UTC offset is deliberately not canonicalised: a shifted time is a
# different instant and must never match quietly. ``Z`` and ``+00:00`` are read as
# notation, because nothing in these contracts carries a timezone for them to be
# relative to.
def _timestamp(value: str) -> str | None:
    """The one spelling of a slot, or None when this is not a slot.

    A dated slot always carries its separators here, and requiring one keeps the
    compact form the parser also accepts from reading a bare number as a date.
    """
    if "-" not in value:
        return None
    try:
        moment = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if moment.utcoffset():
        return None
    return moment.replace(tzinfo=None, microsecond=0).isoformat()


def _normalize(value: Any) -> Any:
    """Compare arguments the way the contract means them, not byte for byte.

    Digit strings are compared as digits, and everything else case- and
    punctuation-insensitively, so a slot, a phone number or an enum written two
    ways names one record. ``original_medicare`` and "Original Medicare" are the
    same category; the wrong category still misses, which is the part worth
    measuring. Formatting is a habit that differs between services, and scoring
    it would tilt the column it feeds.
    """
    if isinstance(value, str):
        stripped = value.strip()
        slot = _timestamp(stripped)
        if slot is not None:
            return slot
        digits = "".join(ch for ch in stripped if ch.isdigit())
        if digits and len(digits) >= 7 and not any(ch.isalpha() for ch in stripped):
            # A leading country code is a dialling detail, not a different
            # number. The tables store ten digits, and a model that says the
            # same number in full is saying the same number.
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]
            return digits
        # Anything that is not a letter or a digit separates words; what the
        # words are is what distinguishes one record from another.
        return " ".join("".join(
            ch if ch.isalnum() else " " for ch in stripped
        ).lower().split())
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        # The arrays these contracts declare are sets -- the product types a
        # caller consented to, the fields collected for a handoff -- so order is
        # not part of the record. A narrower or wider list still misses.
        return sorted((_normalize(item) for item in value), key=repr)
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in sorted(value.items())}
    return value


# How close a call has to be, over the fields it and a row have between them,
# before the nearest row answers it. Speech recognition bends names and the odd
# digit, and a table lookup that treats a bent name as an unknown record turns a
# transcription error into a task failure -- which is a different measurement.
FUZZY_THRESHOLD = 30.0

# Enum values that decline to answer rather than naming a category. Spelled out
# because the difference is a matter of meaning: "unknown" withholds a fact,
# where "spouse" asserts one and can be wrong.
DECLINES = frozenset({"unknown", "not_sure", "not_applicable", "none", "other"})


def _declined(field_name: str, value: Any, abstentions: frozenset) -> bool:
    """Whether this argument declines to answer. Only a plain string can."""
    return isinstance(value, str) and (field_name, value) in abstentions


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    matched: bool
    output: Any
    # How the row was found: "exact", "fuzzy", "none", or "unknown" for a tool
    # the contract does not declare. ``matched`` stays exact-only on purpose, so
    # the figures built from it keep meaning "asked for precisely this record".
    resolution: str = "none"

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "matched": self.matched,
            "resolution": self.resolution,
            "output": self.output,
        }


def _similarity(left: Any, right: Any) -> float:
    """How alike two argument values are, 0 to 100."""
    if _normalize(left) == _normalize(right):
        return 100.0
    return difflib.SequenceMatcher(
        None, str(left).strip().lower(), str(right).strip().lower()
    ).ratio() * 100.0


@dataclass
class MockToolServer:
    """One suite's tools, answered from its published lookup table."""

    suite: str
    root: Path = DEFINITIONS_ROOT
    calls: list[ToolCallRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        directory = self.root / self.suite
        self._definitions = json.loads((directory / "tool-definitions.json").read_text())
        self._mocks = {m["name"]: m for m in json.loads((directory / "mock-tools.json").read_text())}
        self.system_prompt = (directory / "system-prompt.txt").read_text().strip()
        self.first_message = (directory / "first-message.txt").read_text().strip()
        self._abstentions = self._build_abstentions()

    # -- what the model is told ------------------------------------------

    def tool_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            ToolSpec(name=d["name"], description=d.get("description", ""), parameters=d.get("parameters", {}))
            for d in self._definitions
        )

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(d["name"] for d in self._definitions)

    # -- answering --------------------------------------------------------

    def _build_abstentions(self) -> dict[str, frozenset[str]]:
        """Per tool, the (field, value) pairs that say nothing about the record.

        An optional field's enum may offer a value meaning the agent has nothing
        to report, while the tool's prose says to omit an argument it has no value
        for. Both say the same thing, so both must score alike, and which one an
        agent reaches for is a habit of that agent rather than a fact about the
        call.

        Two conditions, and both are needed. The value must be one that declines
        to answer rather than asserting something -- naming a category the caller
        does not fit is a claim, and a wrong claim must still miss. And no record
        may use it: where a record does, the contract treats it as a real answer
        and it stays compared, as an ``unknown`` Part B status does, which is a
        thing callers say and the tables keep a record for.

        Required fields are excluded throughout: there the agent is asked to
        commit, and declining is itself an answer.
        """
        abstentions: dict[str, frozenset[str]] = {}
        for definition in self._definitions:
            name = definition["name"]
            parameters = definition.get("parameters") or {}
            required = set(parameters.get("required") or ())
            rows = self._mocks.get(name, {}).get("mock_data", [])
            pairs = set()
            for field_name, schema in (parameters.get("properties") or {}).items():
                if field_name in required:
                    continue
                recorded = {
                    json.dumps(row["input"][field_name], sort_keys=True)
                    for row in rows
                    if field_name in (row.get("input") or {})
                }
                for choice in schema.get("enum") or ():
                    if choice in DECLINES and json.dumps(choice, sort_keys=True) not in recorded:
                        pairs.add((field_name, choice))
            abstentions[name] = frozenset(pairs)
        return abstentions

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Answer one tool call from the table, in two stages.

        The tables are records, not assertions. A row lists the fields that were
        present when that record was captured, and several of those fields are
        optional in the tool's own schema -- so requiring a call to repeat every
        one of them makes a row unreachable for any agent that follows the
        schema. Stage one therefore compares only the fields the call and the
        row have in common, and the row sharing the most of them wins, so a
        specific record still beats a general one.

        Stage two exists because speech is lossy. A misheard surname or a dropped
        digit is a transcription error, and answering it with nothing turns it
        into a task failure -- a different thing, measured in the same column. So
        when no row matches outright, the nearest row above ``FUZZY_THRESHOLD``
        answers instead. The tables are built for this: they carry seeded rows
        for the ordinary ways a call goes wrong, a missing postcode or an unknown
        number, whose outputs tell the agent what is missing and how to recover.
        Reaching those rows is the point of this stage. Without it an agent has
        nowhere to go but to tell the caller the system failed, and the
        conversation being scored becomes a conversation about our harness.

        Only when both stages come up empty is the answer a miss, which is a
        legitimate answer and not an error: the contract's own wording says a
        no-match means no record was found.

        The two stages are the resolution the scoring side already performs
        against the same tables. Serving more strictly than the score is read
        would fail agents for answers the score accepts.
        """
        mock = self._mocks.get(name)
        if mock is None:
            log.warning("tool %s is not declared by the %s contract", name, self.suite)
            return self._record(name, arguments, {"error": f"unknown tool {name}"}, "unknown")

        freetext = set(mock.get("freetext_params") or ())
        abstained = self._abstentions.get(name, frozenset())
        # Free text cannot be compared: no two agents write the same sentence,
        # and a row never stores one. An argument the model left empty is an
        # argument it did not send.
        wanted = {
            key: value
            for key, value in (arguments or {}).items()
            if value is not None and key not in freetext and not _declined(key, value, abstained)
        }
        rows = [
            (row, {k: v for k, v in (row.get("input") or {}).items() if k not in freetext})
            for row in mock.get("mock_data", [])
        ]

        best, best_shared = None, 0
        for row, stored in rows:
            shared = set(stored) & set(wanted)
            if not shared or len(shared) <= best_shared:
                continue
            if all(_normalize(wanted[key]) == _normalize(stored[key]) for key in shared):
                best, best_shared = row, len(shared)
        if best is not None:
            log.info("tool %s matched a record on %d field(s)", name, best_shared)
            return self._record(name, arguments, best.get("output"), "exact")

        nearest, best_score = None, -1.0
        for row, stored in rows:
            fields = set(stored) | set(wanted)
            if not fields:
                continue
            # A field only one side names scores zero rather than being skipped,
            # so a row that answers half the call cannot outrank one that
            # answers all of it.
            score = sum(
                _similarity(wanted[key], stored[key]) if key in wanted and key in stored else 0.0
                for key in fields
            ) / len(fields)
            if score > best_score:
                nearest, best_score = row, score
        if nearest is not None and best_score >= FUZZY_THRESHOLD:
            self._explain(name, wanted, rows, nearest, best_score)
            return self._record(name, arguments, nearest.get("output"), "fuzzy")

        self._explain(name, wanted, rows, nearest, best_score)
        return self._record(name, arguments, {"result": "no_match"}, "none")

    def _explain(self, name, wanted, rows, nearest, score) -> None:
        """Name the fields that kept a call off the record it came closest to.

        Only ever called when a call did not match outright, so the comparison
        it repeats costs nothing on the path that did.
        """
        if nearest is None:
            log.info("tool %s matched no record; the contract has none to compare", name)
            return
        stored = next((s for row, s in rows if row is nearest), {})
        disagreed = {
            key: (wanted[key], stored[key])
            for key in set(stored) & set(wanted)
            if _normalize(wanted[key]) != _normalize(stored[key])
        }
        log.info(
            "tool %s did not match outright (nearest record scored %.1f); "
            "%d field(s) shared and equal, disagreed on %s",
            name, score, len(set(stored) & set(wanted)) - len(disagreed),
            ", ".join(f"{k}={s!r} sent as {w!r}" for k, (w, s) in sorted(disagreed.items())) or "nothing",
        )
        for key in sorted(set(wanted) - set(stored)):
            log.debug("tool %s sent %s=%r, which that record does not carry", name, key, wanted[key])

    def _record(self, name: str, arguments: dict[str, Any], output: Any, resolution: str) -> Any:
        record = ToolCallRecord(name, arguments, resolution == "exact", output, resolution)
        self.calls.append(record)
        return record.output

    def reset(self) -> None:
        self.calls.clear()
