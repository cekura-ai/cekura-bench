"""Tool calls scored against each scenario's expected calls, twice: as written, and as stored.

The platform's Tool Call Accuracy asks whether every call was *written* exactly
as the scenario's contract says: one argument too many, a name in lower case or
a phone number with dashes fails the call. That is the right question for an
integration that validates its payloads, and it is the board's number. It is
not the only question a reader has. A model that writes "ellen parker" into a
record keyed "Ellen Parker" has stored the right person; a model that writes
"Alan Parker" has stored the wrong one, and the strict score cannot tell those
apart. So every run is scored a second time, asking whether each call would
*store the same record*, and the two are published side by side. The gap
between them is how much of a row's failure is formatting.

Input is the same export ``agent.report`` reads, one JSON object per run, plus
each suite's expected calls in ``agent-definitions/<suite>/expected-tool-calls.json``:

    {"<scenario_id>": [{"name": ..., "arguments": {...}, "optional": false}, ...], ...}

An empty list means the scenario expects no tool calls; its runs are not scored,
and are counted as such. A scenario missing from the file is refused rather than
skipped, so a run cannot drop out of a row unnoticed.

Rules, stated once.

**Strict** -- the platform's rule:

* The tool name matches exactly, and so does every argument the expected call
  lists. Nothing else may be sent.
* An expected value may be a marker instead of a literal: ``<freetext>`` is any
  non-empty string, ``<optional>`` may be left out and takes any value when
  sent, ``<string_array>`` is any list of strings, and ``{"$one_of": [...]}`` is
  any one of the listed values.
* An expected call marked ``optional`` may be left out; when made, it must match.
* Order does not matter. Hanging up and transferring are call control, not
  tool use, and cancelled calls never reached a tool; neither is scored.
* A run passes when every required expected call is matched and every scored
  call it made is one of them.

**Equivalent** -- the same record, by rules read from the tool schemas and
applied to every row alike, never tuned to any row's results:

* A name (``*_name``) compares without regard to case, spacing or punctuation.
  A different spelling is a different person and still fails.
* A phone number (``phone``, ``*_phone``) compares as its digits, with a leading
  US country code dropped.
* A list compares as a set.
* An extra argument passes only when it adds nothing to the record: it is null,
  it is the schema's own ``unknown`` choice, or it repeats another argument of
  the same call (a beneficiary named as the caller who is the beneficiary).
* Everything else is as strict: a wrong choice, an invented or unchained id, a
  missing argument, a missing call and an unmatched call all fail.

    python -m agent.tool_score runs.jsonl [--definitions agent-definitions] [--out tool-scores.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.report import CALL_CONTROL, ReportError, load_runs

DEFINITIONS = Path(__file__).resolve().parent.parent / "agent-definitions"
MODES = ("strict", "equivalent")

FREETEXT, OPTIONAL, STRING_ARRAY, ONE_OF = "<freetext>", "<optional>", "<string_array>", "$one_of"
UNKNOWN_CHOICE = "unknown"

# Defect names, one per way an expected call can go unmet. A call can carry
# several; a run's tally counts each once per expected call it affects.
EXTRA_ARGUMENT = "extra_argument"
MISSING_ARGUMENT = "missing_argument"
NAME_VALUE = "name_value"
OTHER_VALUE = "other_value"
MISSING_CALL = "missing_call"
UNMATCHED_CALL = "unmatched_call"


def is_name(key: str) -> bool:
    return key.endswith("_name")


def is_phone(key: str) -> bool:
    return key == "phone" or key.endswith("_phone")


def _name(value: Any) -> Any:
    return " ".join(re.sub(r"[^\w\s]", " ", value).casefold().split()) if isinstance(value, str) else value


def _phone(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    digits = re.sub(r"\D", "", value)
    return digits[1:] if len(digits) == 11 and digits.startswith("1") else digits


def _canonical(key: str, value: Any, mode: str) -> Any:
    if mode == "strict":
        return value
    if is_name(key):
        return _name(value)
    if is_phone(key):
        return _phone(value)
    if isinstance(value, list):
        return sorted(json.dumps(_canonical(key, item, mode), sort_keys=True) for item in value)
    return value


def value_matches(key: str, expected: Any, actual: Any, mode: str) -> bool:
    if expected == FREETEXT:
        return isinstance(actual, str) and bool(actual.strip())
    if expected == OPTIONAL:
        return True
    if expected == STRING_ARRAY:
        return isinstance(actual, list) and all(isinstance(item, str) for item in actual)
    if isinstance(expected, dict) and set(expected) == {ONE_OF}:
        return any(value_matches(key, choice, actual, mode) for choice in expected[ONE_OF])
    return _canonical(key, expected, mode) == _canonical(key, actual, mode)


def adds_nothing(key: str, value: Any, arguments: dict[str, Any], schema: dict[str, Any]) -> bool:
    """An extra argument the stored record would not notice."""
    if value is None:
        return True
    if value == UNKNOWN_CHOICE and UNKNOWN_CHOICE in ((schema.get(key) or {}).get("enum") or ()):
        return True
    return any(
        other != key and _canonical(key, value, "equivalent") == _canonical(key, other_value, "equivalent")
        for other, other_value in arguments.items()
    )


def defects(expected: dict[str, Any], call: dict[str, Any], mode: str, schema: dict[str, Any]) -> Counter:
    """What keeps ``call`` from being ``expected``, by defect. Empty means a match."""
    want = expected.get("arguments") or {}
    got = call.get("arguments") or {}
    found = Counter()
    for key, value in want.items():
        if key not in got:
            if value != OPTIONAL:
                found[MISSING_ARGUMENT] += 1
        elif not value_matches(key, value, got[key], mode):
            found[NAME_VALUE if is_name(key) else OTHER_VALUE] += 1
    for key, value in got.items():
        if key in want:
            continue
        if mode == "equivalent" and adds_nothing(key, value, got, schema):
            continue
        found[EXTRA_ARGUMENT] += 1
    return found


def _pair(expected: list[dict[str, Any]], calls: list[dict[str, Any]], mode: str, schema: dict[str, Any]):
    """Pair one tool's calls with its expected calls, matching as many as possible.

    Among pairings that match the same number, the one with the fewest defects
    wins, so the diagnosis describes the nearest expected call rather than an
    arbitrary one. Exhaustive over used-expected sets: a scenario expects a
    handful of calls per tool, so this stays small.
    """
    costs = [[defects(e, c, mode, schema) for e in expected] for c in calls]
    # state: mask of expected entries used -> (unmatched, defect total, assignment)
    best: dict[int, tuple[int, int, tuple[int | None, ...]]] = {0: (0, 0, ())}
    for i in range(len(calls)):
        step: dict[int, tuple[int, int, tuple[int | None, ...]]] = {}
        for mask, (bad, total, chosen) in best.items():
            options = [(None, mask, bad + 1, total)]
            for j in range(len(expected)):
                if not mask & (1 << j):
                    cost = costs[i][j]
                    options.append((j, mask | (1 << j), bad + (1 if cost else 0), total + sum(cost.values())))
            for j, new_mask, new_bad, new_total in options:
                candidate = (new_bad, new_total, chosen + (j,))
                if new_mask not in step or candidate[:2] < step[new_mask][:2]:
                    step[new_mask] = candidate
        best = step

    def finish(mask: int, state: tuple[int, int, tuple[int | None, ...]]):
        missing = sum(1 for j, e in enumerate(expected) if not mask & (1 << j) and not e.get("optional"))
        return state[0] + missing, state[1], state[2]

    mask, state = min(best.items(), key=lambda item: finish(*item)[:2])
    return mask, state[2], costs


@dataclass
class Score:
    """One run in one mode."""

    passed: bool = True
    expected_calls: int = 0
    matched_calls: int = 0
    defects: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "expected_calls": self.expected_calls, "matched_calls": self.matched_calls,
                "defects": dict(self.defects)}


def score_run(expected: list[dict[str, Any]], calls: Sequence[dict[str, Any]], mode: str,
              schemas: dict[str, dict[str, Any]]) -> Score:
    scored = [c for c in calls if not c.get("cancelled") and c.get("name") not in CALL_CONTROL]
    score = Score(expected_calls=len(expected))
    by_name: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for entry in expected:
        by_name[entry["name"]][0].append(entry)
    for call in scored:
        by_name[call.get("name")][1].append(call)
    for name, (want, got) in by_name.items():
        schema = (schemas.get(name) or {}).get("properties") or {}
        mask, assignment, costs = _pair(want, got, mode, schema)
        for i, j in enumerate(assignment):
            if j is None:
                score.defects[UNMATCHED_CALL] += 1
            elif costs[i][j]:
                score.defects.update({kind: 1 for kind in costs[i][j]})
            else:
                score.matched_calls += 1
        for j, entry in enumerate(want):
            if not mask & (1 << j) and not entry.get("optional"):
                score.defects[MISSING_CALL] += 1
    # Every unmet required call and every unmatched call left a defect, so a
    # run with none has met the contract.
    score.passed = not score.defects
    return score


def load_expected(suite: str, definitions: Path = DEFINITIONS) -> dict[str, list[dict[str, Any]]]:
    path = definitions / suite / "expected-tool-calls.json"
    if not path.exists():
        raise ReportError(f"{suite}: no expected calls at {path.name}; a suite without a contract cannot be scored")
    return {str(key): value for key, value in json.loads(path.read_text()).items()}


def load_schemas(suite: str, definitions: Path = DEFINITIONS) -> dict[str, dict[str, Any]]:
    tools = json.loads((definitions / suite / "tool-definitions.json").read_text())
    return {tool["name"]: tool.get("parameters") or {} for tool in tools}


def summarize(runs: Sequence[dict[str, Any]], definitions: Path = DEFINITIONS) -> dict[str, Any]:
    """Both scores for every row, a row being one configuration on one suite, as in ``agent.report``."""
    from agent.report import _row, _suite

    contracts: dict[str, tuple[dict, dict]] = {}
    rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        rows[_row(run["custom_metadata"]), _suite(run["custom_metadata"])].append(run)

    out = []
    for (row, suite), members in sorted(rows.items()):
        if suite not in contracts:
            contracts[suite] = (load_expected(suite, definitions), load_schemas(suite, definitions))
        expected, schemas = contracts[suite]
        tally = {mode: {"passed_runs": 0, "expected_calls": 0, "matched_calls": 0, "defects": Counter()} for mode in MODES}
        scored = unscored = 0
        failing: dict[str, list] = {mode: [] for mode in MODES}
        for run in members:
            key = str(run.get("scenario_id"))
            if key not in expected:
                raise ReportError(f"{row} ({suite}): scenario {key} has no entry in expected-tool-calls.json")
            if not expected[key]:
                unscored += 1
                continue
            scored += 1
            for mode in MODES:
                score = score_run(expected[key], run["custom_metadata"].get("tool_calls") or [], mode, schemas)
                bucket = tally[mode]
                bucket["passed_runs"] += score.passed
                bucket["expected_calls"] += score.expected_calls
                bucket["matched_calls"] += score.matched_calls
                bucket["defects"].update(score.defects)
                if not score.passed:
                    failing[mode].append(run.get("run_id"))
        out.append({
            "row": row, "suite": suite, "runs": len(members), "scored_runs": scored,
            "runs_without_expected_calls": unscored,
            **{mode: {**{k: v for k, v in tally[mode].items() if k != "defects"},
                      "defects": dict(tally[mode]["defects"]),
                      "failing_run_ids": sorted(failing[mode], key=str)} for mode in MODES},
        })
    return {"schema": "agent_bench_tool_scores_v1", "rows": out}


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exports", nargs="+", help="JSONL files, one run per line")
    parser.add_argument("--definitions", type=Path, default=DEFINITIONS, help="agent-definitions directory")
    parser.add_argument("--out", help="write here instead of stdout")
    args = parser.parse_args(argv)
    try:
        report = summarize(load_runs(args.exports), args.definitions)
    except ReportError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
