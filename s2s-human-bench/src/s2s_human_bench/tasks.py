"""Task completion from a call's tool calls.

With a task file (``--tasks``), a call's Task Outcome is pass or fail on its scenario's main task
instead of the share of expected-outcome lines met. The file names each scenario's tools and the
values the caller gave, so it is private and stays out of this repository.

A scenario's task is a list of requirements. A requirement is met when the agent called one of its
tools with arguments that match one of that tool's accepted rows. In a row, a plain key must be in
the call with that value (``"*"``: any value); a key starting with ``?`` is optional, but must
match when the agent passes it. The task passes when:

* every requirement is met, each by its own call;
* no tool in ``must_not_run`` was called (the caller took the action back before anything was sent);
* each tool in ``then_undo`` was called after the action (the caller asked to undo it);
* when no undo is expected, the action was not undone afterwards by a tool in ``undo_tools``.

Values are compared tolerantly: case, accents, punctuation, common street abbreviations, number
words ("level three"), phone and account number formatting, date and time formats and number
formatting do not matter.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

_ABBREV = {
    "street": "st",
    "avenue": "ave",
    "apartment": "apt",
    "road": "rd",
    "drive": "dr",
    "court": "ct",
    "suite": "ste",
    "boulevard": "blvd",
    "lane": "ln",
    "place": "pl",
}
_MONTHS = {m: i for i, m in enumerate("january february march april may june july august september october november december".split(), 1)}
_STOP = {"the", "a", "an", "of", "on", "at", "in", "and", "to", "for", "by"}
_NUMBER_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
_ON_FILE = "on file"
_ANY = "*"
_UNSET = {"", "none", "null", "n/a", "unknown"}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


class TaskFileError(SystemExit):
    """The task file is missing a scenario or is malformed."""


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8")).get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise TaskFileError(f"{path}: no tasks")
    for code, spec in tasks.items():
        if not spec.get("requirements"):
            raise TaskFileError(f"{path}: {code} has no requirements")
    return tasks


def tool_calls(transcript: list[dict[str, Any]]) -> list[tuple[float, str, dict[str, Any]]]:
    """(start time, tool name, arguments) for every tool call the agent made, in time order."""
    calls = []
    for i, turn in enumerate(transcript):
        data = turn.get("data") or {}
        if turn.get("role") != "Function Call" or not data.get("name"):
            continue
        args = data.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        calls.append((float(turn.get("start_time") or i), data["name"], args if isinstance(args, dict) else {}))
    return sorted(calls, key=lambda c: c[0])


def _plain(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9]+", " ", text).split()
    return " ".join(_ABBREV.get(w, _NUMBER_WORDS.get(w, w)) for w in words)


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value))


def _dates(text: str, year: int | None = None) -> set[str]:
    found = {f"{y}-{int(m):02d}-{int(d):02d}" for y, m, d in re.findall(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)}
    for month, day, yr in re.findall(r"([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?", text.lower()):
        if month in _MONTHS and (yr or year):
            found.add(f"{yr or year}-{_MONTHS[month]:02d}-{int(day):02d}")
    return found


def _times(text: str) -> set[str]:
    found = set()
    for h1, m1, h2, m2, ampm in re.findall(r"(?<![\d-])(\d{1,2})(?::(\d{2}))?\s*(?:-|–|to)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", text.lower()):
        for h, m in ((h1, m1), (h2, m2)):  # a range such as "5-7 pm": both ends take the am/pm
            found.add(f"{int(h) % 12 + (12 if ampm.startswith('p') else 0):02d}:{m or '00'}")
    for h, m, ampm in re.findall(r"(?<![\d-])(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", text.lower()):
        hour = int(h) % 12 + (12 if ampm.startswith("p") else 0)
        found.add(f"{hour:02d}:{m or '00'}")
    for h, m in re.findall(r"(?<![\d])(\d{1,2}):(\d{2})(?!\s*(?:am|pm|a\.m|p\.m))", text.lower()):
        found.add(f"{int(h):02d}:{m}")
    return found


def _number(value: Any) -> float | None:
    text = str(value).replace(",", "").replace("$", "").strip()
    return float(text) if re.fullmatch(r"-?\d+(\.\d+)?", text) else None


def _unset(value: Any) -> bool:
    return value is None or value in ([], {}) or (isinstance(value, str) and value.strip().lower() in _UNSET)


def value_matches(expected: Any, actual: Any) -> bool:
    """Whether the agent's argument ``actual`` gives the value ``expected``."""
    if _unset(actual):
        return _ON_FILE in _plain(expected)
    if isinstance(expected, list):
        pool = actual if isinstance(actual, list) else [actual]
        return all(any(value_matches(e, a) for a in pool) or any(_plain(e) in _plain(a) for a in pool) for e in expected)
    exp = str(expected).strip()
    if exp.lower() in ("true", "false"):
        return str(actual).strip().lower() in (("true", "yes", "1") if exp.lower() == "true" else ("false", "no", "0", "none"))
    num = _number(exp)
    if num is not None and len(_digits(exp)) < 7:
        got = _number(actual)
        return got is not None and abs(got - num) < 0.005
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", exp)]
    dates = _dates(exp)
    if dates:
        got = str(actual)
        times = _times(exp)
        got_dates = _dates(got, years[0] if years else None)
        if not dates <= got_dates:
            # Every call runs on a pinned clock, so a weekday ("Monday after 5 pm") names one date, and
            # a bare time ("1:30 pm") is on the day the record it belongs to (an appointment) fixes.
            named = {w for w in _WEEKDAYS if w in got.lower()}
            wanted = {date.fromisoformat(d).strftime("%A").lower() for d in dates}
            if got_dates or (named and named != wanted) or (not named and not (times and min(times) in _times(got))):
                return False
        return not times or min(times) in _times(got)  # a window must start when the caller asked
    times = _times(exp)
    if times and not re.search(r"[a-z]{4,}", exp.lower().replace("am", "").replace("pm", "")):
        return min(times) in _times(str(actual))
    digits, alnum = _digits(exp), re.sub(r"[^A-Za-z0-9]", "", exp)
    if len(digits) >= 7 and len(digits) >= 0.6 * len(alnum):  # phone, account or routing number
        return _digits(actual)[-10:] == digits[-10:]
    if alnum and " " not in exp and any(c.isdigit() for c in alnum):  # an id such as A-8812 or S3
        got = re.sub(r"[^A-Za-z0-9]", "", str(actual)).upper()
        return got == alnum.upper() or (len(alnum) >= 4 and alnum.upper() in got)
    want, got = _plain(exp).split(), set(_plain(actual).split())
    words = [w for w in want if w not in _STOP]
    if not words:
        return True
    numbers = [w for w in words if w.isdigit()]
    return all(n in got for n in numbers) and sum(w in got for w in words) >= 0.7 * len(words)


def _leaves(value: Any) -> list[Any]:
    """The scalar values inside nested arguments (a list of plain values counts as one value)."""
    if isinstance(value, dict):
        return [leaf for v in value.values() for leaf in _leaves(v)]
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        return [leaf for v in value for leaf in _leaves(v)]
    return [value]


def row_matches(row: dict[str, Any], args: dict[str, Any]) -> bool:
    """Every value of the row is in the call. A value is looked for under its own argument name;
    when the call has no such argument (it was renamed in the deployed tool definition), under
    any argument the row does not name. Optional (``?``) values only count when passed."""
    names = {key.lstrip("?") for key in row}
    others = [leaf for k, v in args.items() if k not in names for leaf in _leaves(v)]

    def found(key: str, value: Any) -> bool:
        given = args.get(key.lstrip("?"))
        if _unset(given):
            if key.startswith("?"):
                return True
            return value != _ANY and (value_matches(value, None) or any(value_matches(value, leaf) for leaf in others))
        return value == _ANY or any(value_matches(value, leaf) for leaf in [given, *_leaves(given)])

    return all(value is None or found(key, value) for key, value in row.items())


def _separate(options: list[list[int]], used: frozenset[int] = frozenset()) -> bool:
    """Whether each requirement can be met by a different call (two rides need two bookings)."""
    return not options or any(_separate(options[1:], used | {i}) for i in options[0] if i not in used)


def task_verdict(spec: dict[str, Any], transcript: list[dict[str, Any]]) -> tuple[bool, str]:
    """(passed, short reason) for one call against its scenario's task."""
    calls = tool_calls(transcript)
    names = {name for _, name, _ in calls}
    for tool in spec.get("must_not_run", []):
        if tool in names:
            return False, f"{tool} ran after the caller took it back"
    options: list[list[int]] = []
    for requirement in spec["requirements"]:
        tools = [alt["tool"] for alt in requirement["any_of"]]
        hits = sorted(
            {
                i
                for alt in requirement["any_of"]
                for i, (_, name, args) in enumerate(calls)
                if name == alt["tool"] and any(row_matches(row, args) for row in alt["accepted"])
            }
        )
        if not hits:
            tried = [name for _, name, _ in calls if name in tools]
            return False, f"{tools[0]} {'called with other values' if tried else 'not called'}"
        options.append(hits)
    if not _separate(options):
        return False, f"{calls[options[-1][0]][1]} called once where {len(options)} separate calls were needed"
    first_action = min(calls[i][0] for hits in options for i in hits)
    last_match: dict[str, float] = {}
    for hits in options:
        for i in hits:
            t, name, _ = calls[i]
            last_match[name] = max(last_match.get(name, t), t)
    for undo in spec.get("then_undo", []):
        if not any(name == undo and t > first_action for t, name, _ in calls):
            return False, f"{undo} not called after the caller took it back"
    if not spec.get("then_undo"):
        for tool, last in last_match.items():
            for undo in spec.get("undo_tools", {}).get(tool, []):
                if any(name == undo and t > last for t, name, _ in calls):
                    return False, f"{tool} undone by {undo}"
    return True, "done"
