"""Entity capture (skill 1.2) as first-attempt accuracy, from the call's own evidence.

With an entity file (``--entities``), the skill it names is scored here instead of by its
platform metric. The file gives, per scenario, the detail the caller spells out (a plate, an
account's last four, a surname, a date) and how to recognise it, so it is private and stays out
of this repository, like the task file.

A call passes when the agent's first use of the detail is right; there is no read-back
requirement. The first use is whichever comes first after the caller gives the detail:

* a tool call carrying it, in an argument whose name matches the scenario's ``keys``;
* an agent line that says it.

No use at all is "not reached", and so is a detail that reached the agent from a tool result
before the caller gave it.

A line is read from the agent's own text when the run carries it (``agent_text`` in the run
record, written by the reference agent: ``{"from": "model" | "transcription", "turns": [...]}``)
and from the call transcript otherwise. The agent's
own text is what the model meant, so it decides over the transcript, which can mishear a letter
or turn a stutter into a wrong digit. Some services only give a transcription of their audio
(``from: "transcription"``); that is a second reading rather than the model's text, so it
decides nothing alone: when it and the transcript disagree, a later tool value settles it, and
without one the call is not reached. When only the transcript is there and it differs from the
detail by letters transcribers often swap (M/N, B/P/D), the call is not reached either.

Entity file::

    {"skill": "1.2", "entities": {"<code>": {"kind": "code", "true": "ab1234", "keys": "plate",
                                             "cue": "plate"}}}

``kind`` is one of ``digits`` (``true``: the digits; ``cue``: the words that introduce them), ``amount`` (``true``: the amount;
``right``/``wrong``: patterns for spoken forms), ``code`` (letters and digits, lower case; ``cue``:
the words that introduce it), ``name`` (``true``: the surname, ``first``: the first name),
``address`` (``true``: house number and street, e.g. ``"12 elm"``) and ``date`` (``true``:
``[month, day]`` or ``[month, day, year]``; ``cue``, or ``right``/``wrong`` patterns). ``tools``
limits the tool calls read to those whose name matches.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from s2s_human_bench.tasks import _MONTHS, _dates, _digits, tool_calls


class EntityFileError(SystemExit):
    """The entity file is missing or malformed."""


@dataclass
class EntityFile:
    skill: str  # the skill the file scores instead of its platform metric
    details: dict[str, dict[str, Any]]  # scenario code -> the detail the caller gives


KINDS = ("digits", "amount", "code", "name", "address", "date")

_ONES = {
    **dict.fromkeys(("zero", "oh", "o", "cero"), 0),
    **{w: i for i, w in enumerate("one two three four five six seven eight nine".split(), 1)},
    **{w: i for i, w in enumerate("uno dos tres cuatro cinco seis siete ocho nueve".split(), 1)},
    "una": 1,
}
_TEENS = {
    **{w: i for i, w in enumerate("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), 10)},
    **{w: i for i, w in enumerate("diez once doce trece catorce quince".split(), 10)},
}
_TENS = {
    **{w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)},
    **{w: 10 * i for i, w in enumerate("veinte treinta cuarenta cincuenta sesenta".split(), 2)},
}
_ORD = {
    w: i
    for i, w in enumerate(
        "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth"
        " sixteenth seventeenth eighteenth nineteenth twentieth".split(),
        1,
    )
}
_ORDINAL = re.compile(r"\b(" + "|".join(_ORD) + r")\b")
_MONTH_NAMES = "|".join(_MONTHS)
_MONTH_DAY = re.compile(rf"({_MONTH_NAMES})\s+(\d{{1,2}})|(\d{{1,2}})\s*(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_NAMES})")
# "B as in Bravo" -> "b"
_NATO = re.compile(r"\b([a-z])\s+(?:as in|like|de|for)\s+[a-z]+\b")
# Short words that are not letters of a spoken code.
_STOP = set(
    "as is the and a i an it in on to of my so or at by be we you yes no que es el la de su con una"
    " por id got did get for now can are was his her its our has had let me him all not but see too say new old one two six ten oh zip car way".split()
)
# A question about the detail rather than a use of it.
_CLARIFY = re.compile(
    r"sounded like|didn.t catch|did not catch|missed (one|a|the|part)|could you (repeat|say|spell)|can you (repeat|spell|say)|"
    r"say (it|that) again|no (escuché|entendí|alcancé)|repet|only caught|i only (got|heard)|could you confirm the",
    re.I,
)
# Letters a transcriber often swaps.
_CONFUSABLE = (set("mn"), set("bpdvtge"), set("fs"), set("cz"))
_NOT_A_SURNAME = {"and", "can", "could", "thanks", "please", "what", "your", "is"}


def load_entities(path: Path, skill_ids: set[str]) -> EntityFile:
    """The skill the file scores (one of ``skill_ids``) and its per-scenario details."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    skill, entities = doc.get("skill"), doc.get("entities")
    if not isinstance(skill, str) or not isinstance(entities, dict) or not entities:
        raise EntityFileError(f"{path}: needs a skill id and a non-empty entities map")
    if skill not in skill_ids:
        raise EntityFileError(f"{path}: skill {skill!r} is not in the config")
    for code, spec in entities.items():
        if spec.get("kind") not in KINDS or "true" not in spec or not spec.get("keys"):
            raise EntityFileError(f"{path}: {code} needs kind (one of {', '.join(KINDS)}), true and keys")
        if spec["kind"] in ("code", "digits") and not spec.get("cue"):
            raise EntityFileError(f"{path}: {code} needs a cue, the words that introduce the {spec['kind']}")
        if spec["kind"] == "name" and not spec.get("first"):
            raise EntityFileError(f"{path}: {code} is a name and needs the first name")
        if spec["kind"] == "date" and not (spec.get("cue") or spec.get("right")):
            raise EntityFileError(f"{path}: {code} is a date and needs a cue or right/wrong patterns")
    return EntityFile(skill, entities)


def _alnum(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _tens_or_teens(toks: list[str], i: int) -> tuple[str | None, int]:
    """'forty one' / 'forty' / 'fifteen' at ``toks[i]`` as digits, and the index after it; else (None, i + 1)."""
    w = toks[i]
    if w in _TENS:
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if _ONES.get(nxt, 0) > 0 and nxt not in ("oh", "o"):
            return str(_TENS[w] + _ONES[nxt]), i + 2
        return str(_TENS[w]), i + 1
    if w in _TEENS:
        return str(_TEENS[w]), i + 1
    return None, i + 1


def words_to_digits(text: str) -> str:
    """Spoken numbers as digits, other words as gaps: 'three oh zero five' -> '3005'."""
    toks = re.findall(r"[a-záéíóúñ]+|\d+", re.sub(r"\bdouble\s+(\w+)", r"\1 \1", text.lower().replace("-", " ")))
    out, i = [], 0
    while i < len(toks):
        w = toks[i]
        number, nxt = _tens_or_teens(toks, i)
        if w.isdigit():
            out.append(w)
        elif number is not None:
            out.append(number)
        elif w in _ONES and not (w == "o" and not out):
            out.append(str(_ONES[w]))
        elif w in ("hundred", "cientos"):
            out.append("00")
        else:
            out.append(" ")
        i = nxt
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _inline_digits(text: str) -> str:
    """Number words as digits, other words kept: 'forty-one Elm Court' -> '41 elm court'."""
    toks = re.findall(r"[a-záéíóúñ]+|\d+|\S", text.lower().replace("-", " "))
    out, i = [], 0
    while i < len(toks):
        w = toks[i]
        number, nxt = _tens_or_teens(toks, i)
        if number is not None:
            out.append(number)
        elif w in _ONES and w not in ("o", "oh", "una", "uno"):
            out.append(str(_ONES[w]))
        else:
            out.append(w)
        i = nxt
    return " ".join(out)


def _letters(text: str) -> str:
    """Spelled letters joined: 'N, Y, M, A, N' -> 'nyman'."""
    t = _NATO.sub(r"\1", text.lower())
    t = re.sub(r"\b([a-z])\b[\s,.\-]*(?=\b[a-z]\b)", r"\1", t)
    return re.sub(r"[^a-z0-9]", "", t)


def _code_flat(text: str) -> str:
    """Letters and digits as spoken, words between them as '|': 'B as in Bravo, P, D, 6271' -> 'bpd6271'."""
    out = []
    for w in re.findall(r"[a-záéíóúñ]+|\d+", _NATO.sub(r"\1", text.lower()).replace("-", " ")):
        if w.isdigit():
            out.append(w)
        elif w in _ONES and w != "o":
            out.append(str(_ONES[w]))
        elif w in _TEENS:
            out.append(str(_TEENS[w]))
        elif len(w) == 1 or (len(w) <= 3 and w not in _STOP and w.isalpha()):
            out.append(w)
        else:
            out.append("|")
    return "".join(out).replace("o", "0")


def _tool_value_ok(spec: dict[str, Any], value: Any) -> bool:
    s, kind, true = str(value), spec["kind"], spec["true"]
    if kind == "digits":
        return str(true) in _digits(s)
    if kind == "amount":
        return re.sub(r"[^\d.]", "", s).split(".")[0] == str(true)
    if kind == "date":  # a month and day with no year is read as the right year
        year = true[2] if len(true) == 3 else 2000
        return any(d.endswith(f"-{true[0]:02d}-{true[1]:02d}") and (len(true) < 3 or d.startswith(str(year))) for d in _dates(s, year))
    return _alnum(true) in _alnum(s)


def _digits_attempt(spec: dict[str, Any], low: str) -> tuple[bool, str] | None:
    true = str(spec["true"])
    if true in re.sub(r"\s", "", words_to_digits(low)):
        return True, true
    if spec.get("wrong") and (m := re.search(spec["wrong"], low)):
        return False, m.group(0)
    for m in re.finditer(spec["cue"], low):
        tail = re.sub(r"\s", "", words_to_digits(low[m.end() : m.end() + 24]))
        if mm := re.search(rf"\d{{{max(3, len(true) - 1)},}}", tail):
            return true in mm.group(0), mm.group(0)
    return None


def _amount_attempt(spec: dict[str, Any], low: str) -> tuple[bool, str] | None:
    true = str(spec["true"])
    if re.search(rf"\b{true}\b", words_to_digits(low)) or (spec.get("right") and re.search(spec["right"], low)):
        return True, true
    if spec.get("wrong") and (m := re.search(spec["wrong"], low)):
        return False, m.group(0)
    return None


def _code_attempt(spec: dict[str, Any], low: str) -> tuple[bool, str] | None:
    true, alnum = _alnum(spec["true"]), _alnum(low)
    flat = _code_flat(low)
    if true in flat or true in alnum:
        return True, true
    letters, digits = re.match(r"([a-z]*)(\d*)$", true).groups()
    joined = flat.replace("|", "")
    if letters and digits and digits in joined and re.search(r"\b" + r"\W*".join(letters) + r"\b", _NATO.sub(r"\1", low)):
        return True, true
    if _CLARIFY.search(low) and not re.search(r"\bi heard\b", low):
        return None
    if digits and digits in joined:
        m = re.search(r"([a-z0-9]{0,4})" + digits, joined)
        got = m.group(0) if m else digits
        if not got.endswith(true):
            return False, got
    if not re.search(spec["cue"], low):
        return None
    near = true[:2] + r"[a-z0-9]{3,}"
    if m := (re.search(near, flat) or re.search(near, alnum)):
        return False, m.group(0)
    return None


def _address_attempt(spec: dict[str, Any], low: str) -> tuple[bool, str] | None:
    number, street = str(spec["true"]).split(" ", 1)
    stem = _alnum(street)[:4]
    flat = _alnum(_inline_digits(low))
    if _alnum(spec["true"]) in flat:
        return True, spec["true"]
    if m := re.search(rf"(\d+){stem}\w*", flat):
        return m.group(1) == number, m.group(0)
    return None


def _name_attempt(spec: dict[str, Any], low: str) -> tuple[bool, str] | None:
    true, first = _alnum(spec["true"]), spec["first"].lower()
    if true in _alnum(low) or true in _letters(low):
        return True, true
    # A wrong attempt: the first name followed by another surname, or a spelled run of four or more letters.
    m = re.search(re.escape(first) + r"\s+([a-záéíóú]{3,})", low)
    if m and m.group(1) not in _NOT_A_SURNAME:
        return False, m.group(0)
    spelled = re.findall(r"(?:\b[a-z]\b[\s,.\-]+){3,}\b[a-z]\b", _NATO.sub(r"\1", low))
    if spelled:
        s = re.sub(r"[^a-z]", "", spelled[-1])
        if len(s) >= 4 and true not in s and s not in first and difflib.SequenceMatcher(None, s, true).ratio() >= 0.5:
            return False, s
    return None


def _date_attempt(spec: dict[str, Any], low: str) -> tuple[bool, str] | None:
    if spec.get("right"):
        if m := re.search(spec["right"], low):
            return True, m.group(0)
        if spec.get("wrong") and (m := re.search(spec["wrong"], low)):
            return False, m.group(0)
        return None
    true = list(spec["true"])
    if not re.search(spec["cue"], low):
        return None
    low = _ORDINAL.sub(lambda m: str(_ORD[m.group(1)]), low)
    if m := re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", low):
        return list(map(int, m.groups()))[: len(true)] == true, m.group(0)
    spoken = words_to_digits(low)
    hit = None
    if m := _MONTH_DAY.search(low + " " + spoken):
        hit = [_MONTHS[m.group(1) or m.group(4)], int(m.group(2) or m.group(3))]
    elif len(true) == 2 and (m := re.search(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)?\b|\b(\d{1,2})(?:st|nd|rd|th)\b", low)):
        hit = [true[0], int(m.group(1) or m.group(2))]  # 'the thirteenth': a day alone
    if hit is None:
        return None
    ok = hit == true[:2]
    years = re.findall(r"(?:19|20)\d\d", spoken.replace(" ", ""))
    if ok and len(true) == 3 and years:  # a year said must be the right one
        ok = str(true[2]) in years
    return ok, str(hit)


_ATTEMPTS = {
    "digits": _digits_attempt,
    "amount": _amount_attempt,
    "code": _code_attempt,
    "address": _address_attempt,
    "name": _name_attempt,
    "date": _date_attempt,
}


def spoken_attempt(spec: dict[str, Any], text: str) -> tuple[bool, str] | None:
    """None when the line does not use the detail; else whether it is right, and what was said."""
    return _ATTEMPTS[spec["kind"]](spec, text.lower())


def _swapped_letters(spec: dict[str, Any], got: str) -> bool:
    """Whether ``got`` differs from the detail only by letters a transcriber often swaps."""
    true, g = _alnum(spec["true"]), _alnum(got)
    for i in range(len(g) - len(true) + 1):
        diffs = [(a, b) for a, b in zip(true, g[i : i + len(true)], strict=True) if a != b]
        if diffs and all(a.isalpha() and b.isalpha() and any(a in c and b in c for c in _CONFUSABLE) for a, b in diffs):
            return True
    return False


def _tool_use(spec: dict[str, Any], entry: dict[str, Any]) -> tuple[bool, str] | None:
    name = str((entry.get("data") or {}).get("name"))
    if spec.get("tools") and not re.search(spec["tools"], name):
        return None
    parsed = tool_calls([entry])  # the task checker's reading of the arguments, cut-off ones included
    for key, value in (parsed[0][2] if parsed else {}).items():
        if re.search(spec["keys"], key) and str(value).strip():
            return _tool_value_ok(spec, value), f"{name}.{key}={value}"
    return None


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _own_text_attempt(spec: dict[str, Any], line: str, turns: list[str], turn_words: list[set[str]]) -> tuple[bool, str] | None:
    """The detail as the agent's own text has it, for the turn that best matches a transcript line."""
    if not turns:
        return None
    words = _words(line)
    i = max(range(len(turns)), key=lambda k: len(words & turn_words[k]))  # the first best match
    if len(words & turn_words[i]) < max(3, 0.4 * len(words)):
        return None
    # A service may log one reply in pieces: read the neighbours too.
    return spoken_attempt(spec, turns[i]) or spoken_attempt(spec, " ".join(turns[max(0, i - 1) : i + 2]))


def _role(entry: dict[str, Any], agent_prefix: str, caller_prefix: str) -> str | None:
    role = str(entry.get("role") or "")
    if role == "Function Call Result":
        return "result"
    if role == "Function Call":
        return "tool"
    return "agent" if role.startswith(agent_prefix) else "caller" if role.startswith(caller_prefix) else None


def entity_verdict(
    spec: dict[str, Any],
    transcript: list[dict[str, Any]],
    own_text: dict[str, Any] | None = None,
    agent_prefix: str = "Main",
    caller_prefix: str = "Testing",
) -> tuple[bool | None, str]:
    """The call's verdict on the detail (None: not reached) and how it was decided."""
    events = sorted(
        ((float(e.get("start_time") or 0), role, e) for e in transcript if (role := _role(e, agent_prefix, caller_prefix))),
        key=lambda x: x[0],
    )
    gave = next((t for t, role, e in events if role == "caller" and (said := spoken_attempt(spec, e.get("content") or "")) and said[0]), None)
    own = own_text or {}
    turns = [str(t) for t in own.get("turns") or []]
    turn_words = [_words(t) for t in turns]
    from_model = own.get("from", "model") == "model"
    for i, (t, role, e) in enumerate(events):
        if gave is not None and t < gave:
            continue
        text = e.get("content") or ""
        if role == "result" and gave is None and spec["kind"] != "name" and _tool_value_ok(spec, text):
            return None, "not reached: the detail came back from a tool"
        if role == "tool" and (used := _tool_use(spec, e)):
            return used[0], f"tool: {used[1]}"
        if role != "agent":
            continue
        heard = spoken_attempt(spec, text)
        if heard is None:
            continue
        doubtful = heard[0] is False and spec["kind"] in ("code", "name") and _swapped_letters(spec, heard[1])
        if heard[0] is False and not doubtful and _CLARIFY.search(text):
            continue
        said = _own_text_attempt(spec, text, turns, turn_words)
        if said is not None and from_model:
            return said[0], f"agent text: {said[1]}"
        if said is not None and said[0] != heard[0]:  # two readings of the audio that disagree
            later = next((u for _, r, x in events[i + 1 :] if r == "tool" and (u := _tool_use(spec, x))), None)
            if later:
                return later[0], f"readings disagree; later tool: {later[1]}"
            return None, f"not reached: readings disagree ({heard[1]} / {said[1]})"
        if doubtful and said is None:
            return None, f"not reached: transcript has {heard[1]}, letters often misheard"
        return heard[0], f"transcript: {heard[1]}"
    return None, "not reached: no use of the detail"


def agent_text(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """The agent's own text from an exported call's metadata (``call_additional_metadata.json``), when the run record has it."""
    doc = metadata or {}
    for record in ((doc.get("provider_call_details") or {}).get("custom_metadata"), doc.get("metadata")):
        if isinstance(record, dict) and isinstance(record.get("agent_text"), dict):
            return record["agent_text"]
    return None
