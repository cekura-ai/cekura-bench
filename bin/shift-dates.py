#!/usr/bin/env python
"""Move every calendar date inside a window forward by a number of days.

    python bin/shift-dates.py --window 2026-07-01..2026-07-31 --days 154 PATH...
    python bin/shift-dates.py --window 2026-07-01..2026-07-31 --days 154 --check PATH...
    echo "July the eighth" | python bin/shift-dates.py --window ... --days 154 --stdin

The scenarios, the mock tables and the agent prompt all name the same few days,
and an agent that reads the real clock refuses a day that has passed. Shifting
them by whole weeks keeps every weekday where it was, so nothing else about a
scenario changes. Only dates inside the window move: a date of birth in a
patient record is a date too, and it stays.

Three spellings are rewritten, in place and in their own form:

* ISO ``2026-07-08`` and ``2026-07-08T09:00:00``;
* a month with a day, ``July 8``, ``July 8th``, ``July eighth``, ``July the
  eighth``, ``the eighth of July``, ``July 8, 2026``;
* a bare ordinal, ``the eighth``, on a line that also names a month inside the
  window, because "no, I mean the eighth" means the eighth of that month.

Clip ids such as ``task.date.july9`` are left alone: they are names, not dates.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
    "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    "twentieth", "twenty-first", "twenty-second", "twenty-third",
    "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh",
    "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first",
]
MONTH_RE = "|".join(MONTHS)
ORDINAL_RE = "|".join(sorted(ORDINALS, key=len, reverse=True))
DAY_RE = rf"(?P<day>\d{{1,2}}(?:st|nd|rd|th)?|{ORDINAL_RE})"

ISO = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?P<time>T\d{2}:\d{2}:\d{2})?\b")
# "July the eighth", "July eighth", "July 8", "July 8th", optionally ", 2026"
MONTH_DAY = re.compile(
    rf"\b(?P<month>{MONTH_RE})\s+(?P<the>the\s+)?{DAY_RE}(?P<year>,\s+\d{{4}})?\b", re.IGNORECASE
)
# "the eighth of July", "8th of July"
DAY_OF_MONTH = re.compile(rf"\b{DAY_RE}\s+of\s+(?P<month>{MONTH_RE})\b", re.IGNORECASE)
BARE_ORDINAL = re.compile(rf"\bthe\s+(?P<day>{ORDINAL_RE})\b", re.IGNORECASE)


def _day_number(token: str) -> int:
    lowered = token.lower()
    if lowered in ORDINALS:
        return ORDINALS.index(lowered) + 1
    return int(re.sub(r"\D", "", token))


def _day_like(token: str, day: int) -> str:
    """The shifted day in the spelling the original used."""
    if token.lower() in ORDINALS:
        word = ORDINALS[day - 1]
        return word.capitalize() if token[0].isupper() else word
    if re.search(r"(st|nd|rd|th)$", token, re.IGNORECASE):
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{day}{suffix}"
    return str(day)


def _month_like(token: str, month: int) -> str:
    name = MONTHS[month - 1]
    return name if token[0].isupper() else name.lower()


class Shifter:
    def __init__(self, start: date, end: date, days: int, year: int | None = None):
        self.start, self.end, self.delta = start, end, timedelta(days=days)
        # A spoken date without a year belongs to the window's year.
        self.year = year or start.year
        self.moved = 0

    def _inside(self, moment: date) -> bool:
        return self.start <= moment <= self.end

    def _iso(self, match: re.Match) -> str:
        try:
            moment = date(int(match["y"]), int(match["m"]), int(match["d"]))
        except ValueError:
            return match.group(0)
        if not self._inside(moment):
            return match.group(0)
        self.moved += 1
        return (moment + self.delta).isoformat() + (match["time"] or "")

    def _month_day(self, match: re.Match) -> str:
        month = MONTHS.index(match["month"].capitalize()) + 1
        year = int(match["year"].strip(", ")) if match["year"] else self.year
        try:
            moment = date(year, month, _day_number(match["day"]))
        except ValueError:
            return match.group(0)
        if not self._inside(moment):
            return match.group(0)
        self.moved += 1
        moved = moment + self.delta
        out = _month_like(match["month"], moved.month) + " "
        out += (match["the"] or "") + _day_like(match["day"], moved.day)
        if match["year"]:
            out += f", {moved.year}"
        return out

    def _day_of_month(self, match: re.Match) -> str:
        month = MONTHS.index(match["month"].capitalize()) + 1
        try:
            moment = date(self.year, month, _day_number(match["day"]))
        except ValueError:
            return match.group(0)
        if not self._inside(moment):
            return match.group(0)
        self.moved += 1
        moved = moment + self.delta
        return f"{_day_like(match['day'], moved.day)} of {_month_like(match['month'], moved.month)}"

    def _bare_ordinal(self, month: int):
        def replace(match: re.Match) -> str:
            try:
                moment = date(self.year, month, _day_number(match["day"]))
            except ValueError:
                return match.group(0)
            if not self._inside(moment):
                return match.group(0)
            self.moved += 1
            moved = moment + self.delta
            return match.group(0)[: match.start("day") - match.start()] + _day_like(match["day"], moved.day)
        return replace

    def line(self, text: str) -> str:
        # A bare "the eighth" is resolved against the month this line names
        # before that month is rewritten, or it would be read in the new month.
        named = [m["month"].capitalize() for m in MONTH_DAY.finditer(text)]
        named += [m["month"].capitalize() for m in DAY_OF_MONTH.finditer(text)]
        months = {MONTHS.index(name) + 1 for name in named}
        month = months.pop() if len(months) == 1 else None
        text = ISO.sub(self._iso, text)
        # Dated phrases are set aside while the bare ordinals move, or a month
        # already rewritten would have its day moved a second time.
        held: list[str] = []

        def hold(replacer):
            def keep(match: re.Match) -> str:
                held.append(replacer(match))
                return f"\x00{len(held) - 1}\x00"
            return keep

        text = MONTH_DAY.sub(hold(self._month_day), text)
        text = DAY_OF_MONTH.sub(hold(self._day_of_month), text)
        if month is not None:
            text = BARE_ORDINAL.sub(self._bare_ordinal(month), text)
        return re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], text)

    def text(self, text: str) -> str:
        return "".join(self.line(line) for line in text.splitlines(keepends=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", required=True, help="first..last date that moves, ISO, inclusive")
    parser.add_argument("--days", type=int, required=True, help="days to add; a multiple of 7 keeps weekdays")
    parser.add_argument("--check", action="store_true", help="report what would move, write nothing")
    parser.add_argument("--stdin", action="store_true", help="shift text from stdin to stdout")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    first, last = (date.fromisoformat(part) for part in args.window.split(".."))

    if args.stdin:
        shifter = Shifter(first, last, args.days)
        sys.stdout.write(shifter.text(sys.stdin.read()))
        return 0

    for path in args.paths:
        shifter = Shifter(first, last, args.days)
        original = path.read_text(encoding="utf-8")
        shifted = shifter.text(original)
        if shifted != original and not args.check:
            path.write_text(shifted, encoding="utf-8")
        print(f"{path}: {shifter.moved} date(s) {'would move' if args.check else 'moved'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
