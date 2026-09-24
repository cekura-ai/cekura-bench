"""Tool Call Accuracy (same record): a Cekura custom-code metric body.

The platform runs this file with ``data`` bound to the run and reads ``_result``
(0-5, or None when not scored) and ``_explanation``. It is the platform's Tool
Call Accuracy with one question changed: not "was every call written exactly as
the contract says" but "would every call store the same record". The rules are
``agent.tool_score``'s equivalent mode, so a score of 5 here is that mode's pass.

Scoring is Tool Call Accuracy's: expected calls come from the scenario's
``generated_mock_tool_entries``, actual calls from the transcript's Function
Call entries, and the score is 5 x F1 over calls. A missing call, a wrong call,
a repeated call and a call nobody expected all cost the same as there. A
scenario that expects no tool call is not scored unless the agent called one.

What same-record forgives, read from the tool schemas and applied to every row:

* A name (``*_name``) compares without regard to case, spacing or punctuation.
  A different spelling is a different person and still fails.
* A phone number (``phone``, ``*_phone``) compares as its digits, with a leading
  US country code dropped.
* A list compares as a set.
* An argument the contract does not list passes only when it adds nothing to the
  record: it is null, it is the schema's own ``unknown`` choice, or it repeats
  another argument of the same call.

The explanation names tools and arguments, never their values.
"""


def _evaluate(data):
    import json
    import re

    CONTROL_TOOLS = {
        "endcall", "transfercall", "transfertoagent", "transfertonumber",
        "hangup", "endsession", "dtmf", "senddtmf", "playkeypadtouchtone",
        "skipturn", "voicemaildetection",
    }
    FREETEXT = "<freetext>"
    OPTIONAL = "<optional>"
    STRING_ARRAY = "<string_array>"
    ONE_OF = "$one_of"
    ABSENT = "<absent>"
    UNKNOWN = "unknown"
    # Arguments whose schema offers "unknown" as a choice, from the suites' tool definitions.
    UNKNOWN_CHOICE = {
        "save_medicare_qualification": {
            "age_band", "has_medicare_part_a", "has_medicare_part_b", "current_coverage", "product_interest",
        },
        "route_medicare_call": {"product_interest", "availability_context"},
    }

    def name_key(value):
        name = str(value or "").strip().lower().removeprefix("c_")
        return re.sub(r"[^a-z0-9_]", "", name)

    def is_control(value):
        return name_key(value).replace("_", "") in CONTROL_TOOLS

    def arguments_of(entry):
        raw = (entry.get("data") or {}).get("arguments", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                return {}
        return raw if isinstance(raw, dict) else {}

    def canonical(key, value):
        if isinstance(value, str) and key.endswith("_name"):
            return " ".join(re.sub(r"[^\w\s]", " ", value).casefold().split())
        if isinstance(value, str) and (key == "phone" or key.endswith("_phone")):
            digits = re.sub(r"\D", "", value)
            return digits[1:] if len(digits) == 11 and digits.startswith("1") else digits
        if isinstance(value, list):
            return sorted(json.dumps(canonical(key, item), sort_keys=True, default=str) for item in value)
        return value

    def one_of_values(expected_value):
        if isinstance(expected_value, dict) and set(expected_value) == {ONE_OF} and isinstance(expected_value[ONE_OF], list):
            return expected_value[ONE_OF]
        return None

    def field_matches(expected_value, actual, key):
        options = one_of_values(expected_value)
        if options is not None:
            return key in actual and any(canonical(key, actual[key]) == canonical(key, option) for option in options)
        if expected_value == FREETEXT:
            return key in actual and isinstance(actual[key], str) and bool(actual[key].strip())
        if expected_value in (OPTIONAL, ABSENT):
            # An absent key that was sent anyway is judged below as an extra argument.
            return True
        if expected_value == STRING_ARRAY:
            return key in actual and isinstance(actual[key], list) and all(isinstance(item, str) for item in actual[key])
        return key in actual and canonical(key, actual[key]) == canonical(key, expected_value)

    def adds_nothing(tool, key, value, actual):
        if value is None:
            return True
        if value == UNKNOWN and key in UNKNOWN_CHOICE.get(tool, ()):
            return True
        return any(other != key and canonical(key, value) == canonical(key, other_value) for other, other_value in actual.items())

    def defects(expected, actual_call):
        """Argument names keeping the call from storing the expected record, by kind."""
        want, got = expected["input"], actual_call["input"]
        found = {"missing": [], "value": [], "extra": []}
        for key, value in want.items():
            if value in (OPTIONAL, ABSENT):
                continue
            if key not in got:
                found["missing"].append(key)
            elif not field_matches(value, got, key):
                found["value"].append(key)
        allowed = {key for key, value in want.items() if value != ABSENT}
        for key in sorted(set(got) - allowed):
            if not adds_nothing(expected["key"], key, got[key], got):
                found["extra"].append(key)
        return found

    def matches(expected, actual_call):
        return actual_call["key"] == expected["key"] and not any(defects(expected, actual_call).values())

    expected_entries = []
    for entry in data.get("generated_mock_tool_entries") or []:
        tool_name = entry.get("tool_name")
        expected_input = (entry.get("new_entry") or {}).get("input", {})
        if tool_name and isinstance(expected_input, dict) and not is_control(tool_name):
            expected_entries.append({
                "tool_name": str(tool_name),
                "key": name_key(tool_name),
                "input": expected_input,
                "optional": entry.get("optional") is True,
            })

    actual_calls = []
    for entry in data.get("transcript_json") or []:
        if entry.get("role") != "Function Call":
            continue
        payload = entry.get("data") or {}
        tool_name = payload.get("name") or entry.get("content") or ""
        if tool_name and not is_control(tool_name):
            actual_calls.append({"tool_name": str(tool_name), "key": name_key(tool_name), "input": arguments_of(entry)})

    if not expected_entries:
        if actual_calls:
            return 0.0, "No expected tool calls, but the agent called: " + ", ".join(sorted({c["tool_name"] for c in actual_calls}))
        return None, "No expected tool calls for this scenario"

    # The most expected calls matched, required ones first. A looser comparison can
    # then never pair a call away from the entry it was the only match for.
    candidates = [
        [j for j, actual in enumerate(actual_calls) if matches(expected, actual)] for expected in expected_entries
    ]
    owner = {}

    def assign(i, seen):
        for j in candidates[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    order = sorted(range(len(expected_entries)), key=lambda i: expected_entries[i]["optional"])
    for i in order:
        assign(i, set())
    matched = {i: j for j, i in owner.items()}

    true_positive = sum(1 for i in matched if not expected_entries[i]["optional"])
    optional_matched = len(matched) - true_positive
    false_negative = sum(1 for i, e in enumerate(expected_entries) if i not in matched and not e["optional"])
    unmatched_actual = [j for j in range(len(actual_calls)) if j not in owner]
    false_positive = len(unmatched_actual)
    denominator = 2 * true_positive + false_positive + false_negative
    score = round(2 * true_positive / denominator * 5, 2) if denominator else 5.0

    required = sum(1 for e in expected_entries if not e["optional"])
    lines = [
        "Same record. Expected calls: " + str(len(expected_entries))
        + " (required: " + str(required) + ", optional: " + str(len(expected_entries) - required) + ")"
        + " | required matched: " + str(true_positive)
        + " | optional matched: " + str(optional_matched)
        + " | missing/wrong required: " + str(false_negative)
        + " | actual calls not matched: " + str(false_positive)
    ]

    # Pair each unmatched entry with its closest same-name call, for the explanation only.
    spare = list(unmatched_actual)
    for i, expected in enumerate(expected_entries):
        if i in matched:
            lines.append("  " + expected["tool_name"] + ": same record ✓" + (" (optional)" if expected["optional"] else ""))
            continue
        same_name = [j for j in spare if actual_calls[j]["key"] == expected["key"]]
        if not same_name:
            lines.append("  " + expected["tool_name"] + (": optional call not made ✓" if expected["optional"] else ": not called ✗"))
            continue
        j = min(same_name, key=lambda k: sum(len(v) for v in defects(expected, actual_calls[k]).values()))
        spare.remove(j)
        found = defects(expected, actual_calls[j])
        detail = "; ".join(kind + ": " + ", ".join(keys) for kind, keys in found.items() if keys)
        lines.append("  " + expected["tool_name"] + ": closest call differs ✗ (" + detail + ")")

    expected_keys = {e["key"] for e in expected_entries}
    for j in spare:
        label = "additional call" if actual_calls[j]["key"] in expected_keys else "unexpected tool call"
        lines.append("  " + actual_calls[j]["tool_name"] + ": " + label + " ✗")

    return score, "\n".join(lines)


_result, _explanation = _evaluate(data)  # noqa: F821 -- the platform binds ``data``
