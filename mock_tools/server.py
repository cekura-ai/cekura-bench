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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mock_tools.spec import ToolSpec

DEFINITIONS_ROOT = Path(__file__).resolve().parent.parent / "agent-definitions"


def _normalize(value: Any) -> Any:
    """Compare arguments the way the contract means them, not byte for byte.

    A model that answers "(415) 555-0123" for a field documented as ten digits
    has followed the instruction; one that answers "415-555-0123" has too. Digit
    strings are compared as digits and everything else case-insensitively, so the
    score reflects whether the right record was requested rather than whose
    formatter ran.

    Separators go the same way, and for the same reason. Several of these fields
    are declared as enumerations whose values are written ``original_medicare``,
    while the caller says "Original Medicare" -- so a model that echoes the
    caller and a model that writes the token have chosen the *same category* and
    differ only in punctuation. Treating those as different records would score
    a provider on its formatting habits rather than on whether it understood the
    caller, and providers differ in that habit, so the column would tilt.
    Choosing the wrong category still misses, which is the part worth measuring.
    """
    if isinstance(value, str):
        stripped = value.strip()
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
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in sorted(value.items())}
    return value


# How close a call has to be, over the fields it and a row have between them,
# before the nearest row answers it. Speech recognition bends names and the odd
# digit, and a table lookup that treats a bent name as an unknown record turns a
# transcription error into a task failure -- which is a different measurement.
FUZZY_THRESHOLD = 30.0


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
            return self._record(name, arguments, {"error": f"unknown tool {name}"}, "unknown")

        freetext = set(mock.get("freetext_params") or ())
        # Free text cannot be compared: no two agents write the same sentence,
        # and a row never stores one. An argument the model left empty is an
        # argument it did not send.
        wanted = {
            key: value
            for key, value in (arguments or {}).items()
            if value is not None and key not in freetext
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
            return self._record(name, arguments, nearest.get("output"), "fuzzy")

        return self._record(name, arguments, {"result": "no_match"}, "none")

    def _record(self, name: str, arguments: dict[str, Any], output: Any, resolution: str) -> Any:
        record = ToolCallRecord(name, arguments, resolution == "exact", output, resolution)
        self.calls.append(record)
        return record.output

    def reset(self) -> None:
        self.calls.clear()
