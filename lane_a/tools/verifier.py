"""Scores a tool-call trace against what the scenario expected.

Deterministic on purpose. A judge model scoring whether an agent "handled the
booking well" imports the judge's opinion into the ranking; comparing the trace
of tool calls against a declared expectation does not. What the agent *said* is
recorded and published, but it is not what decides the cell.

Three things are scored separately rather than folded into one number, because
they fail for different reasons and a single score hides which:

* **presence**  -- was every required call made, and nothing forbidden?
* **arguments** -- did the required calls carry the right values?
* **order**     -- did they happen in the sequence the contract requires?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from lane_a.tools.server import _normalize


@dataclass(frozen=True)
class ExpectedCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)   # subset; unlisted keys are free
    optional: bool = False


@dataclass(frozen=True)
class TraceVerdict:
    presence: bool
    arguments: bool
    order: bool
    missing: tuple[str, ...]
    forbidden_used: tuple[str, ...]
    wrong_arguments: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.presence and self.arguments and self.order

    def as_json(self) -> dict[str, Any]:
        return {
            "tool_trace_pass": self.passed,
            "presence": self.presence,
            "arguments": self.arguments,
            "order": self.order,
            "missing": list(self.missing),
            "forbidden_used": list(self.forbidden_used),
            "wrong_arguments": list(self.wrong_arguments),
        }


def verify_trace(
    observed: Sequence[dict[str, Any]],
    expected: Sequence[ExpectedCall],
    forbidden: Sequence[str] = (),
) -> TraceVerdict:
    """Compare an observed trace to the expectation.

    Order is checked as a subsequence, not as equality: an agent that looks a
    patient up twice, or checks availability again after the caller changes their
    mind, has not violated the contract. An agent that books before it checks has.
    """
    observed_names = [call.get("name", "") for call in observed]
    missing: list[str] = []
    wrong: list[str] = []
    order_positions: list[int] = []

    for want in expected:
        candidates = [
            index
            for index, call in enumerate(observed)
            if call.get("name") == want.name
        ]
        if not candidates:
            if not want.optional:
                missing.append(want.name)
            continue

        matching = [
            index
            for index in candidates
            if _arguments_match(observed[index].get("arguments", {}), want.arguments)
        ]
        if not matching and want.arguments:
            wrong.append(want.name)
            order_positions.append(candidates[0])
        else:
            order_positions.append((matching or candidates)[0])

    forbidden_used = tuple(sorted({name for name in observed_names if name in set(forbidden)}))
    return TraceVerdict(
        presence=not missing and not forbidden_used,
        arguments=not wrong,
        order=order_positions == sorted(order_positions),
        missing=tuple(missing),
        forbidden_used=forbidden_used,
        wrong_arguments=tuple(wrong),
    )


def _arguments_match(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    if not expected:
        return True
    normalized = _normalize(observed)
    return all(normalized.get(key) == value for key, value in _normalize(expected).items())
