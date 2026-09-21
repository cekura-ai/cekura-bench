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
    """
    if isinstance(value, str):
        stripped = value.strip()
        digits = "".join(ch for ch in stripped if ch.isdigit())
        if digits and len(digits) >= 7 and not any(ch.isalpha() for ch in stripped):
            return digits
        return stripped.lower()
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in sorted(value.items())}
    return value


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    matched: bool
    output: Any

    def as_json(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "matched": self.matched, "output": self.output}


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
        """Answer one tool call from the table. Unknown inputs get an explicit miss.

        A miss is a legitimate answer, not an error: the contract's own
        description says a no-match means no record was found. Inventing a record
        for an unrecognised argument would let a model that asked for the wrong
        thing look like one that asked for the right thing.
        """
        mock = self._mocks.get(name)
        if mock is None:
            record = ToolCallRecord(name, arguments, False, {"error": f"unknown tool {name}"})
            self.calls.append(record)
            return record.output

        wanted = _normalize(arguments)
        best, best_keys = None, -1
        for row in mock.get("mock_data", []):
            expected = _normalize(row.get("input", {}))
            if not expected or not all(wanted.get(key) == value for key, value in expected.items()):
                continue
            # The most specific row wins, not the first one listed. One table
            # here holds a row whose inputs are a strict subset of another's, so
            # first-match returns the general answer to a question that named
            # the particular one -- and the table's order, which nothing
            # guarantees, would decide a scored result.
            if len(expected) > best_keys:
                best, best_keys = row, len(expected)

        if best is not None:
            record = ToolCallRecord(name, arguments, True, best.get("output"))
            self.calls.append(record)
            return record.output

        record = ToolCallRecord(name, arguments, False, {"result": "no_match"})
        self.calls.append(record)
        return record.output

    def reset(self) -> None:
        self.calls.clear()
