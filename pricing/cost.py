"""Turn a run's recorded consumption into money, or say why it cannot.

This runs after the fact, over the ``usage`` the reference agent stamped on the
run. Nothing here touches a call, which is the point: a price is a judgement
about a vendor page on a date and will need correcting, and correcting it must
never mean re-running the calls.

A row is priced only from what was measured. Where a rate is missing, or the
provider reported no usage, the answer is an exclusion carrying its reason
rather than a zero -- a zero in a cost column reads as "free", and free and
"we could not measure it" are opposite findings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TABLE = Path(__file__).resolve().parent / "prices.json"

MILLION = 1_000_000


@dataclass(frozen=True)
class Price:
    """What a call cost, and how much that figure can be trusted.

    ``usd`` is None whenever the figure would be a guess. ``verified`` says
    whether the rate behind it has been checked against the vendor since it was
    written down; an unverified price may be computed and inspected but must not
    be published. ``lines`` is the figure by component, so a row with a second
    model behind the first shows both. ``unmetered`` names what the vendor bills
    and the figure leaves out, which a reader needs next to the number.
    """

    usd: float | None
    verified: bool
    basis: str | None
    reason: str | None = None
    lines: dict[str, float] = field(default_factory=dict)
    unmetered: tuple[str, ...] = ()

    @property
    def publishable(self) -> bool:
        return self.usd is not None and self.verified


def load_table(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or TABLE).read_text())


def _entry(table: dict[str, Any], row: str) -> dict[str, Any] | None:
    for section in ("rows", "cascade_text_models"):
        found = (table.get(section) or {}).get(row)
        if found is not None:
            return found
    return None


def lanes(usage: dict[str, Any], reasoning_in_output: bool = True) -> dict[str, int]:
    """Split a call's recorded tokens into the lanes vendors price separately.

    The record keeps the framework's fields, in which the audio and cached
    counts sit *inside* the prompt and output totals, so each lane is a
    difference. Reasoning is inside the output total on some vendors and beside
    it on others, and the row says which.
    """
    def n(key: str) -> int:
        return int(usage.get(key) or 0)

    cached, cached_audio = n("cache_read_input_tokens"), n("cache_read_input_audio_tokens")
    audio_in, audio_out = n("input_audio_tokens"), n("output_audio_tokens")
    split = {
        "text_input": n("prompt_tokens") - audio_in - (cached - cached_audio),
        "text_input_cached": cached - cached_audio,
        "audio_input": audio_in - cached_audio,
        "audio_input_cached": cached_audio,
        "text_output": n("completion_tokens") - audio_out
        + (0 if reasoning_in_output else n("reasoning_tokens")),
        "audio_output": audio_out,
    }
    return split


def price_call(row: str, usage: dict[str, Any], table: dict[str, Any] | None = None) -> Price:
    """Price one call of one row from the usage recorded on its run.

    ``row`` is the provider key the run was launched with, and ``usage`` is the
    dictionary the agent stamped on the record.
    """
    table = table if table is not None else load_table()
    entry = _entry(table, row)
    if entry is None:
        return Price(None, False, None, f"no price entry for {row!r}")

    components = entry.get("components") or []
    if not any(c.get("rates") for c in components):
        return Price(None, False, None, f"no rate recorded for {row!r}")
    verified = bool(entry.get("verified"))
    basis = "+".join(c["basis"] for c in components)
    unmetered = tuple(entry.get("unmetered") or ())

    def refuse(reason: str) -> Price:
        return Price(None, verified, basis, reason, unmetered=unmetered)

    if not usage.get("usage_reports") and not any(c["basis"] == "per_minute" for c in components):
        return refuse("the provider reported no usage for this call")

    split = lanes(usage, bool(entry.get("reasoning_in_output", True)))
    if any(count < 0 for count in split.values()):
        # A lane below zero means the parts do not add up to the totals, so the
        # record does not mean what this arithmetic assumes it means.
        return refuse(f"the recorded usage does not add up: {split}")

    lines: dict[str, float] = {}
    for component in components:
        name, rates = component["name"], component.get("rates") or {}
        missing = [key for key in component.get("requires") or () if not usage.get(key)]
        if missing:
            # Without these the lanes cannot be told apart, and pricing the
            # total at the text rate would make the call look cheap.
            return refuse(f"{name}: the record lacks {', '.join(missing)}")

        if component["basis"] == "per_minute":
            seconds = usage.get(component.get("measure", "call_seconds"))
            if not seconds:
                return refuse(f"{name}: the call's {component.get('measure', 'call_seconds')} was not recorded")
            lines[name] = rates["minute"] * seconds / 60
        elif component["basis"] == "per_million_tokens":
            # A lane the vendor bills and the table omits would otherwise be
            # silently free, so an unpriced lane with a count is a refusal.
            unpriced = [lane for lane, count in split.items() if count and lane not in rates]
            if unpriced:
                return refuse(f"{name}: tokens recorded in unpriced lane(s) {', '.join(unpriced)}")
            lines[name] = sum(rates[lane] * split[lane] / MILLION for lane in rates)
        else:
            return refuse(f"{name}: unknown pricing basis {component['basis']!r}")

    return Price(sum(lines.values()), verified, basis, lines=lines, unmetered=unmetered)
