"""Turn a run's recorded consumption into money, or say why it cannot.

This runs after the fact, over the ``usage`` the reference agent stamped on the
run. Nothing here touches a call, which is the point: a price is a judgement
about a vendor page on a date and will need correcting, and correcting it must
never mean running 2,700 calls again.

A row is priced only from what was measured. Where a rate is missing, or the
provider reported no usage, the answer is an exclusion carrying its reason
rather than a zero -- a zero in a cost column reads as "free", and free and
"we could not measure it" are opposite findings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    be published.
    """

    usd: float | None
    verified: bool
    basis: str | None
    reason: str | None = None

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


def price_call(row: str, usage: dict[str, Any], table: dict[str, Any] | None = None) -> Price:
    """Price one call of one row from the usage recorded on its run.

    ``row`` is the provider key the run was launched with, and ``usage`` is the
    dictionary the agent stamped on the record.
    """
    table = table if table is not None else load_table()
    entry = _entry(table, row)
    if entry is None:
        return Price(None, False, None, f"no price entry for {row!r}")

    basis, rates = entry.get("basis"), entry.get("rates") or {}
    if not rates:
        return Price(None, False, basis, f"no rate recorded for {row!r}")
    verified = bool(entry.get("verified"))

    if basis == "per_minute":
        seconds = usage.get("call_seconds")
        if not seconds:
            return Price(None, verified, basis, "the call's length was not recorded")
        return Price(rates["minute"] * seconds / 60, verified, basis)

    if basis == "per_million_tokens":
        # Only the lanes the table actually prices. A lane the vendor bills for
        # and the table omits would otherwise be silently free.
        counted = {lane: usage.get(lane) or 0 for lane in rates}
        if not any(counted.values()):
            return Price(
                None, verified, basis,
                "the provider reported no usage for the lanes this row is priced on",
            )
        return Price(
            sum(rates[lane] * count / MILLION for lane, count in counted.items()),
            verified, basis,
        )

    return Price(None, verified, basis, f"unknown pricing basis {basis!r}")
