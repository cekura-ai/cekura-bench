"""Self-contained mock backend for the canonical benchmark agent (Ava).

Loads the SAME mock data the other platforms use (`definitions/<persona>/mock-tools.json`)
and resolves each tool call to its canonical output, so this self-hosted
LiveKit worker IS the identical fake backend — full cross-platform parity,
no external service. (Identical to pipecat-agent/mock_backend.py.)

Resolution mirrors Cekura's real mock-serve logic (test_framework/mock_tool_utils.py):
  1. Flexible exact match on the keys common to both the stored input and the call
     args, with numeric-/case-normalization (`"750" == "750.00"`, case-insensitive
     strings), and freetext params (e.g. book_appointment.reason) stripped first.
  2. If nothing matches exactly, a fuzzy fallback over the union of keys, scored with
     stdlib difflib.SequenceMatcher (~thefuzz.ratio); the closest entry is returned
     if its score >= FUZZY_MATCH_THRESHOLD (30/100).
Only when both stages miss do we return {"error": "no_match"}. The fallbacks fire
ONLY after exact match fails, so the designated no-match/error seed outcomes
(9995550000, 5005550911, appt_0000, appt_9119, ...) still exact-match first — no
risk to currently-passing runs. Uses only the stdlib (no thefuzz dependency).

2026-07-30: refactored from a module-singleton (fixed ./mock-tools.json) into
MockBackend(path) so the worker can load one backend per test persona.
"""
import json
import difflib
from pathlib import Path

_FUZZY_THRESHOLD = 30.0  # mirrors test_framework/mock_tool_utils.FUZZY_MATCH_THRESHOLD


def _norm(v):
    """Numeric-looking strings -> float; other strings -> lowercased; else as-is."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v.lower()
    return v


def _flex_eq(a, b):
    return _norm(a) == _norm(b)


def _ratio(a, b):
    if _norm(a) == _norm(b):
        return 100.0
    return difflib.SequenceMatcher(None, str(a), str(b)).ratio() * 100.0


class MockBackend:
    def __init__(self, mock_path: str | Path):
        with open(mock_path) as f:
            raw = json.load(f)
        self._tools = {t["name"]: t.get("mock_data", []) for t in raw}
        self._freetext = {t["name"]: set(t.get("freetext_params", [])) for t in raw}

    def tool_names(self):
        return list(self._tools.keys())

    def resolve(self, tool_name: str, args: dict) -> dict:
        """Resolve a tool call the way Cekura's mock endpoint does: flexible match on
        shared keys, then a fuzzy fallback (threshold 30). Returns no_match only if
        both miss."""
        entries = self._tools.get(tool_name, [])
        freetext = self._freetext.get(tool_name, set())
        call = {k: v for k, v in (args or {}).items() if v is not None and k not in freetext}

        # 1. Flexible match on keys common to both (empty-common-keys never matches).
        for e in entries:
            stored = {k: v for k, v in e.get("input", {}).items() if k not in freetext}
            common = set(stored) & set(call)
            if common and all(_flex_eq(stored[k], call[k]) for k in common):
                return e.get("output", {})

        # 2. Fuzzy fallback over the union of keys (missing key on either side scores 0).
        best, best_score = None, -1.0
        for e in entries:
            stored = {k: v for k, v in e.get("input", {}).items() if k not in freetext}
            keys = set(stored) | set(call)
            if not keys:
                continue
            score = sum(_ratio(stored[k], call[k]) if k in stored and k in call else 0.0
                        for k in keys) / len(keys)
            if score > best_score:
                best, best_score = e, score
        if best is not None and best_score >= _FUZZY_THRESHOLD:
            return best.get("output", {})

        return {"error": "no_match", "message": f"No mock entry for {tool_name} with input {call}"}
