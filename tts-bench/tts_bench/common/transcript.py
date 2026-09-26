"""Token normalisation and edit distance, the fallback scorer's two primitives.

Letters and digits are split into separate tokens and spoken digits are collapsed
to digits, so "CW5001", "C W 5001" and "c w five zero zero one" compare equal while
a different digit does not.
"""

from __future__ import annotations

import re


_ONES = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_MULTIPLIERS = {"double": 2, "triple": 3}


def words(text: str) -> list[str]:
    """Letter runs and digit runs as separate tokens: ``CW5001`` is ``cw`` then ``5001``."""
    return re.findall(r"[a-z]+|[0-9]+", text.lower())


def normalize(text: str) -> list[str]:
    """Words with spoken digits collapsed to digits, so both sides are comparable.

    ``triple five`` becomes ``5 5 5``; a literal ``555`` becomes the same three
    tokens. Everything else is left alone: the point is to stop formatting from
    deciding the score, not to guess at meaning.
    """
    out: list[str] = []
    pending = 1
    for word in words(text):
        if word in _MULTIPLIERS:
            pending = _MULTIPLIERS[word]
            continue
        if word in _ONES:
            out.extend([_ONES[word]] * pending)
        elif word.isdigit():
            out.extend(list(word))
        elif len(word) == 1 and out and out[-1].isalpha() and len(out[-1]) < 4 and not out[-1].isdigit():
            out[-1] += word  # letters spelled out, "c w", join as one code the way "CW" is written
        else:
            out.append(word)
        pending = 1
    return out


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]
