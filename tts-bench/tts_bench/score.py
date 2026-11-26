"""Offline round-trip scoring: transcribe every synthesised utterance and compare.

Kept separate from the run on purpose. The provider call is the expensive,
unrepeatable part; the transcription instrument is cheap and replaceable, so
changing it must never require re-running the provider. This pass reads the
audio a run wrote, sends it to one or more ASR instruments, and writes
``scores.jsonl`` beside ``cells.jsonl``. The report picks it up if present.

Scoring rule: written shorthand (times, addresses, symbols) is put into words
the same way on both sides, then the hypothesis is compared with the spoken
reference, the text as sent and any accepted reading, choosing among them word
by word: each is a correct reading. Digit sequences are compared separately
because a task depends on them and a word error rate hides one wrong digit.

The instrument is not the subject. Two instruments are run where two are
available, and a row on which they disagree by more than the declared margin
is flagged, not averaged. Each hard part gets one verdict: a term must be heard
right by every transcriber, any other kind by their majority, and where the two
split, a third transcriber (the arbiter) hears that recording and decides.
Word errors are counted the same way: a word is wrong when most transcribers
hear it wrong, so one transcriber's mishearing does not count against the model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path
from collections import defaultdict
from typing import Any, Sequence

import aiohttp

from tts_bench import corpus
from tts_bench.common.transcript import edit_distance, normalize as _fallback_normalize
from tts_bench.store import append_line, cell_dir, latest_cells, read_jsonl, write_manifest

DISAGREE_WER = 0.10   # two instruments further apart than this on one row => row flagged


# ── normalisation ─────────────────────────────────────────────────────────────

def _repair_tokens(tokens: list[str]) -> list[str]:
    """Letter/digit runs split, spelled letters joined: ``cw5001``, ``c w 5001`` and ``CW 5001`` agree.

    Applied identically to both sides after the pinned normaliser. Whether an
    identifier was read as a word or letter by letter is a pronunciation
    question; this keeps it out of the word count so the digits and letters
    themselves decide the score.
    """
    out: list[str] = []
    spelled: list[bool] = []          # parallel to out: was this token built from single letters
    for token in tokens:
        for part in re.findall(r"[a-z]+|[0-9]+(?:[.,][0-9]+)*|[^a-z0-9\s]+", token):
            single = len(part) == 1 and part.isalpha()
            if single and out and spelled[-1]:
                out[-1] += part           # "c" then "w" -> "cw"; never onto an ordinary word
            else:
                out.append(part)
                spelled.append(single)
    return out


def _normalizer():
    """The pinned English text normaliser, or the built-in tokeniser if the package is absent (recorded either way)."""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        norm = EnglishTextNormalizer()
        return ("whisper-normalizer", _pkg_version("whisper-normalizer"), lambda text: _repair_tokens(norm(text).split()))
    except Exception:  # noqa: BLE001
        return ("tts_bench.common.transcript", "builtin", _fallback_normalize)


NORMALIZER_NAME, NORMALIZER_VERSION, normalize = _normalizer()


def _words(text: str) -> list[str]:
    return normalize(_spoken_form(text))


def score_pair(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = _words(reference), _words(hypothesis)
    errors = edit_distance(ref, hyp)
    return {"errors": errors, "reference_words": len(ref), "wer": round(errors / max(len(ref), 1), 4),
            "normalized_reference": " ".join(ref), "normalized_hypothesis": " ".join(hyp)}


def _reading_graph(references: Sequence[list[str]]) -> tuple[list[str], dict[int, list[tuple[list[str], int]]]]:
    """The first reference as a chain of words, and every other reference's differing stretches as detours over it.

    A detour from word ``i`` to word ``j`` replaces ``base[i:j]`` with the other
    reading's words, so a transcript can follow the written form in one place and
    the spoken form in another ("Chapter four covers Henry VIII").
    """
    import difflib

    base = references[0]
    detours: dict[int, list[tuple[list[str], int]]] = defaultdict(list)
    for other in references[1:]:
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, base, other, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            words = other[j1:j2]
            if i1 == i2:                     # pure insertion: anchor it on a neighbouring base word
                if i1 > 0:
                    i1, words = i1 - 1, [base[i1 - 1], *words]
                else:
                    i2, words = i2 + 1, [*words, base[0]] if base else words
            detours[i1].append((words, i2))
    return base, detours


def _graph_align(base: list[str], detours: dict[int, list[tuple[list[str], int]]], hyp: list[str]) -> tuple[int, list[int], dict[int, int]]:
    """Least word edits turning the transcript into a path through the reading graph, and where they fell.

    Returns the edit count, the base words a transcript got wrong (by index) and
    the extra transcript words before each base word. An edit inside a detour is
    put on the base words that detour replaces, in order, so transcribers that
    took different readings of the same stretch can still be compared word by word.
    """
    inf = 1 << 30
    width = len(hyp) + 1
    order: list[tuple] = []
    edges: dict[tuple, list[tuple[tuple, str, tuple]]] = defaultdict(list)    # node -> (next node, word, where)
    for i in range(len(base) + 1):
        order.append(("b", i))
        if i < len(base):
            edges[("b", i)].append((("b", i + 1), base[i], ("w", i)))
        for n, (words, j) in enumerate(detours.get(i, ())):
            where = ("d", i, j)
            prev = ("b", i)
            for t, word in enumerate(words):
                node = ("b", j) if t == len(words) - 1 else ("d", i, n, t)
                if node[0] == "d":
                    order.append(node)
                edges[prev].append((node, word, where))
                prev = node
            if not words:
                edges[prev].append((("b", j), "", where))
    cost = {node: [inf] * width for node in order}
    back: dict[tuple, list] = {node: [None] * width for node in order}
    cost[("b", 0)][0] = 0
    for node in order:
        row, bp = cost[node], back[node]
        for k in range(1, width):              # an extra transcript word
            if row[k - 1] + 1 < row[k]:
                row[k], bp[k] = row[k - 1] + 1, (node, k - 1, "ins", node)
        for nxt, word, where in edges[node]:
            nrow, nbp = cost[nxt], back[nxt]
            for k in range(width):
                if word == "":                   # an empty detour: nothing to read
                    if row[k] < nrow[k]:
                        nrow[k], nbp[k] = row[k], (node, k, "skip", where)
                    continue
                if row[k] + 1 < nrow[k]:         # a reference word the transcript lacks
                    nrow[k], nbp[k] = row[k] + 1, (node, k, "del", where)
                if k and row[k - 1] + (word != hyp[k - 1]) < nrow[k]:
                    nrow[k], nbp[k] = row[k - 1] + (word != hyp[k - 1]), (node, k - 1, "ok" if word == hyp[k - 1] else "sub", where)
    end = ("b", len(base))
    distance = cost[end][len(hyp)]
    wrong: set[int] = set()
    inserted: dict[int, int] = defaultdict(int)
    in_detour: dict[tuple[int, int], int] = defaultdict(int)
    node, k = end, len(hyp)
    while back[node][k] is not None:
        prev, pk, op, where = back[node][k]
        if op in ("del", "sub"):
            if where[0] == "w":
                wrong.add(where[1])
            else:
                in_detour[where[1:]] += 1
        elif op == "ins":
            if node[0] == "b":
                inserted[node[1]] += 1
            else:
                in_detour[node[1:3] if len(node) == 4 else node[1:]] += 1
        node, k = prev, pk
    for (i, j), errors in in_detour.items():
        span = list(range(i, j))
        wrong.update(span[:errors])
        if errors > len(span):
            inserted[j] += errors - len(span)
    return distance, sorted(wrong), dict(inserted)


def _graph_distance(base: list[str], detours: dict[int, list[tuple[list[str], int]]], hyp: list[str]) -> int:
    """Least word edits turning the transcript into any path through the reading graph."""
    return _graph_align(base, detours, hyp)[0]


def score_readings(references: Sequence[str], hypothesis: str) -> dict[str, Any]:
    """Word errors against the closest reading, choosing among the references word by word.

    The first reference is the spoken one and sets the word count; the text as
    sent and any accepted alternative readings join it as detours. ``wrong``
    lists the spoken reference's words the transcript got wrong and
    ``inserted`` its extra words by position, so transcribers can be compared word by word.
    """
    refs = [_words(r) for r in references]
    hyp = _words(hypothesis)
    base, detours = _reading_graph(refs)
    errors, wrong, inserted = _graph_align(base, detours, hyp)
    return {"errors": errors, "reference_words": len(base), "wer": round(errors / max(len(base), 1), 4),
            "normalized_reference": " ".join(base), "normalized_hypothesis": " ".join(hyp),
            "wrong": wrong, "inserted": {str(k): v for k, v in sorted(inserted.items())}}


def _canonical_amounts(text: str) -> str:
    """``$12,480.00`` is read as twelve thousand four hundred eighty; ``$0.75`` as seventy five cents.

    The zeros a normaliser drops from a spoken amount are not digits a listener
    was meant to hear, so they are dropped from the expectation as well.
    """
    text = re.sub(r"(\d)\.00\b", r"\1", text)
    return re.sub(r"\b0\.(\d\d)\b", r"\1", text)


_DIGIT_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def _spoken_digits(text: str) -> str:
    """Every digit a listener heard, in order: written digits, digit words, and "eight hundred" as 800.

    The WER normaliser folds numbers (``0199`` becomes ``199``), which is right
    for a sentence and wrong for a code, where every digit and its position is
    the content.
    """
    out: list[str] = []
    previous_digit = False
    for token in re.findall(r"[0-9]|[a-z]+", text.lower()):
        if token.isdigit():
            out.append(token)
            previous_digit = True
        elif token in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[token])
            previous_digit = True
        elif token == "hundred" and previous_digit:
            out.append("00")
        elif token in ("double", "triple"):
            continue
        else:
            previous_digit = False
    return "".join(out)


_WORD_FOR_DIGIT = {digit: word for word, digit in _DIGIT_WORDS.items() if digit != "0"}
_TLD = r"(?:com|org|net|edu|gov|io|co|us|ai)"


def _spell_address(m: re.Match[str]) -> str:
    """``billing.support@example.com/a/b-c`` as it is read: billing dot support at example dot com slash a slash b dash c."""
    text = m.group(0).replace("@", " at ")
    return re.sub(r"[./-]", lambda c: {".": " dot ", "/": " slash ", "-": " dash "}[c.group(0)], text)


_WRITTEN_FORMS = (
    (re.compile(r"\(\s*(\d+)\s*\)"), r"\1"),                       # (555) 020-8816: the normaliser drops a bracketed group
    (re.compile(r"\bo'?clock\b", re.IGNORECASE), ""),              # eight o'clock is eight; the normaliser glues it into 80
    (re.compile(r"\b(\d{1,2}):00\b"), r"\1"),
    # 2.15am or "6.40 this morning": a transcriber's way of writing a time; an amount or a rate ($12.50, 3.75%) is left alone
    (re.compile(r"(?<![$\d.,])\b([1-9]|1\d|2[0-3])\.([0-5]\d)(?!\s*(?:%|per\s?cent|dollars?|cents?|[.,]?\d))", re.IGNORECASE), r"\1:\2"),
    (re.compile(r"(\d+)\s*'\s*(\d+)\s*(?:''|\"|″|”)"), r"\1 foot \2"),
    (re.compile(r"(?<![\w-])[-−](?=\d)"), "minus "),
    (re.compile(r"(?<=\w)\+(?=\w)"), " plus "),
    (re.compile(r"\bco[\s-]pay", re.IGNORECASE), "copay"),
    (re.compile(r"\bninth\b", re.IGNORECASE), "nine th"),           # the normaliser makes every ordinal "9 th" except this one
    (re.compile(rf"[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?:/[\w/.-]*\w)?"), _spell_address),
    (re.compile(rf"\b[\w.+-]*[.][\w.+-]*(?=\s+at\s+[\w-]+(?:\.[\w-]+)*\.{_TLD}\b)"), _spell_address),
    (re.compile(rf"\b[\w-]+(?:\.[\w-]+)*\.{_TLD}\b(?:/[\w/.-]*\w)?"), _spell_address),
    (re.compile(r"°\s*F\b", re.IGNORECASE), " degrees fahrenheit"),
    (re.compile(r"°\s*C\b", re.IGNORECASE), " degrees celsius"),
    (re.compile(r"°"), " degrees"),
    (re.compile(r"(?<![\d.,:/-])(\d{1,2})\s*[-–]\s*(\d{1,2})(?![\d.,:/-])"), r"\1 to \2"),
    (re.compile(r"(?<![\d.,:/-])0([1-9])(?![\d.,:/])"), lambda m: "oh " + _WORD_FOR_DIGIT[m.group(1)]),
)


def _spoken_form(text: str) -> str:
    """How written shorthand is said: ``36°F``, ``the 2-4 window``, ``class of 09``, ``2.15am``, an email or a web address.

    The normaliser reads these as ``36 f``, ``2 4``, ``9``, ``2.15`` and
    ``billing support example com``, which no spoken reading matches, so a
    correct reading would fail on how the transcriber chose to write it.
    Applied alike to the reference, the accepted readings and every transcript,
    for the word error rate and the hard parts.
    """
    for pattern, replacement in _WRITTEN_FORMS:
        text = pattern.sub(replacement, text)
    return text


def score_spans(spans: Sequence[dict[str, Any]], hypothesis: str) -> list[dict[str, Any]]:
    """Each labelled span: did one of its accepted readings come through, after the same normaliser as WER.

    A span marked for a listener is carried through unscored, so its count is
    visible beside the scored ones rather than silently dropped.
    """
    heard = " " + " ".join(normalize(_spoken_form(hypothesis))) + " "
    out = []
    for span in spans:
        row = {"text": span["text"], "category": span["category"], "check": span.get("check", "transcript")}
        if row["check"] == "transcript":
            forms = [" ".join(normalize(_spoken_form(form))) for form in span.get("accept", ())]
            row["pass"] = any(form and f" {form} " in heard for form in forms)
            digits = "".join(re.findall(r"[0-9]", span["text"]))
            if span["category"] == "sequence" and digits and not row["pass"]:
                row["pass"] = digits in _spoken_digits(hypothesis)
        out.append(row)
    return out


def score_transcript(text: str, spoken_reference: str, hypothesis: str,
                     spans: Sequence[dict[str, Any]] = (), readings: Sequence[str] = ()) -> dict[str, Any]:
    """``best`` counts errors against the closest of the spoken reference, the text as sent and any accepted
    reading, chosen word by word; ``against_text`` and ``against_spoken`` keep each whole reference on its own."""
    against_text = score_pair(text, hypothesis)
    against_spoken = score_pair(spoken_reference, hypothesis)
    best = score_readings([spoken_reference, text, *readings], hypothesis)
    # Digits come from the written text, in order; the hypothesis is read after
    # the same normaliser has turned its number words into digits. An item with
    # no digit in its text has no digit check.
    ref_digits = "".join(re.findall(r"[0-9]", _canonical_amounts(text)))
    hyp_digits = "".join(re.findall(r"[0-9]", " ".join(normalize(hypothesis))))
    return {
        "best": best,
        "against_text": against_text,
        "against_spoken": against_spoken,
        "digits_expected": ref_digits,
        "digits_heard": hyp_digits,
        "digits_match": bool(ref_digits) and ref_digits == hyp_digits,
        "spans": score_spans(spans, hypothesis),
        "hypothesis": hypothesis,
    }


# ── instruments ───────────────────────────────────────────────────────────────

class Instrument:
    name = "abstract"
    model = ""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def transcribe(self, session: aiohttp.ClientSession, wav_path: Path) -> str:
        raise NotImplementedError


class DeepgramInstrument(Instrument):
    """Pre-recorded transcription on a pinned model; the same reference model our published STT work standardises on."""

    name = "deepgram"
    model = "nova-3"

    async def transcribe(self, session: aiohttp.ClientSession, wav_path: Path) -> str:
        url = f"https://api.deepgram.com/v1/listen?model={self.model}&language=en&smart_format=false&punctuate=false"
        content_type = "audio/flac" if wav_path.suffix == ".flac" else "audio/wav"
        async with session.post(url, data=wav_path.read_bytes(),
                                headers={"Authorization": f"Token {self.api_key}", "Content-Type": content_type}) as response:
            if response.status != 200:
                raise RuntimeError(f"deepgram HTTP {response.status}: {(await response.text())[:200]}")
            payload = await response.json()
        return payload["results"]["channels"][0]["alternatives"][0]["transcript"]


class OpenAIWhisperInstrument(Instrument):
    name = "openai-whisper"
    model = "whisper-1"

    async def transcribe(self, session: aiohttp.ClientSession, wav_path: Path) -> str:
        form = aiohttp.FormData()
        form.add_field("model", self.model)
        form.add_field("response_format", "json")
        content_type = "audio/flac" if wav_path.suffix == ".flac" else "audio/wav"
        form.add_field("file", wav_path.read_bytes(), filename=wav_path.name, content_type=content_type)
        async with session.post("https://api.openai.com/v1/audio/transcriptions", data=form,
                                headers={"Authorization": f"Bearer {self.api_key}"}) as response:
            if response.status != 200:
                raise RuntimeError(f"openai HTTP {response.status}: {(await response.text())[:200]}")
            payload = await response.json()
        return payload.get("text", "")


INSTRUMENTS = {"deepgram": (DeepgramInstrument, "DEEPGRAM_API_KEY"), "openai-whisper": (OpenAIWhisperInstrument, "OPENAI_API_KEY")}


def _pcm16k(path: Path) -> bytes:
    """Mono 16-bit PCM at 16 kHz, low-passed before resampling so nothing above 8 kHz folds back."""
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        if rate > 16000:
            cutoff = 0.45 * 16000 / rate
            n = np.arange(-63, 64)
            taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(len(n))
            audio = np.convolve(audio, taps / taps.sum(), mode="same")
        audio = np.interp(np.arange(0, len(audio) * 16000 / rate) * rate / 16000, np.arange(len(audio)), audio)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


class ZoomScribeInstrument(Instrument):
    """Zoom Scribe Pro live mode, used only to settle a hard part the two transcribers disagree on.

    The recording is streamed in 20 ms frames at real-time pace, the way the
    live endpoint expects; Zoom's own voice activity detection segments it and
    each segment's final transcript is joined in order. The model is requested
    inside ``config`` and the server's echo in ``session.updated`` is checked.
    """

    name = "zoom"
    model = "zoom-scribe-en-pro-v2.1"
    url = "wss://api.zoom.us/v2/aiservices/scribe/live"
    frame = 640                                  # 20 ms of 16 kHz 16-bit audio

    async def transcribe(self, session: aiohttp.ClientSession, wav_path: Path) -> str:
        from websockets.asyncio.client import connect

        pcm = _pcm16k(wav_path) + b"\0" * self.frame * 15        # 300 ms of silence closes the last segment
        async with connect(self.url, subprotocols=["live-asr"], additional_headers={"Authorization": f"Bearer {self.api_key}"},
                           open_timeout=15, close_timeout=5) as ws:
            await ws.send(json.dumps({"type": "session.update", "language": "en-US", "audio": {"format": "pcm16"},
                                      "config": {"model": self.model}}))
            order: list[str] = []
            finals: dict[str, str] = {}
            ready = asyncio.Event()

            async def read() -> None:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    message = json.loads(raw)
                    kind = message.get("type")
                    if kind == "session.updated":
                        if message.get("model") != self.model:
                            raise RuntimeError(f"zoom served {message.get('model')!r}, not {self.model}")
                        ready.set()
                    elif kind in ("input_audio_buffer.speech_started", "transcription.completed") and message.get("item_id"):
                        if message["item_id"] not in order:
                            order.append(message["item_id"])
                        if kind == "transcription.completed":
                            finals[message["item_id"]] = message.get("transcript", "")
                    elif kind == "error":
                        raise RuntimeError(f"zoom error: {str(message.get('error') or message)[:200]}")
                    elif kind == "session.closed":
                        return

            reader = asyncio.create_task(read())
            await asyncio.wait({reader, asyncio.create_task(ready.wait())}, timeout=15, return_when=asyncio.FIRST_COMPLETED)
            if reader.done():
                reader.result()
            if not ready.is_set():
                reader.cancel()
                raise RuntimeError("zoom session.updated not received")
            for start in range(0, len(pcm), self.frame):
                await ws.send(pcm[start:start + self.frame])
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "session.close"}))
            await asyncio.wait_for(reader, 15)
        return " ".join(" ".join(finals.get(item, "") for item in order).split())


ARBITERS = {"zoom": (ZoomScribeInstrument, "ZOOM_API_KEY")}

# A misread drug or medical name is often written as the real word by a transcriber
# whose language model knows it, so one transcriber hearing it wrong is the evidence.
UNANIMOUS = frozenset({"term"})


def hard_part_verdicts(instruments: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One verdict per hard part from every transcriber's reading of it.

    A term passes only if every transcriber heard it right. Any other kind
    passes or fails when the transcribers agree; when they split, the majority
    of three decides, and with two it is unresolved (``pass: None``) until the
    arbiter has transcribed the recording.
    """
    readings = {name: [s for s in r["spans"] if s.get("check", "transcript") == "transcript"]
                for name, r in instruments.items() if r.get("spans") is not None and "best" in r}
    if not readings:
        return []
    out = []
    for k, span in enumerate(next(iter(readings.values()))):
        votes = {name: spans[k]["pass"] for name, spans in readings.items() if k < len(spans)}
        heard = sum(votes.values())
        if span["category"] in UNANIMOUS or heard in (0, len(votes)):
            verdict: bool | None = heard == len(votes)
        elif len(votes) >= 3:
            verdict = 2 * heard > len(votes)
        else:
            verdict = None
        out.append({"text": span["text"], "category": span["category"], "pass": verdict, "votes": votes})
    return out


def word_verdict(instruments: dict[str, dict[str, Any]], names: frozenset[int] = frozenset()) -> dict[str, Any] | None:
    """Word errors the transcribers agree on: a reference word is wrong, or an extra word was said, when most of them hear it so.

    One transcriber mishearing a word the others heard right is the
    transcriber's error, not the model's. A name (``names``, positions in the
    spoken reference) is wrong only when every transcriber missed it:
    transcribers garble unusual names that were said right, while writing one
    exactly is hard to do by chance. With two transcribers that disagree, or
    that both missed a name, the count is unresolved (``errors: None``) until
    the arbiter has heard the recording; ``mean`` keeps the two transcribers'
    average meanwhile.
    """
    heard = [r["best"] for r in instruments.values() if "wrong" in (r.get("best") or {})]
    if not heard:
        return None
    words = heard[0]["reference_words"]
    mean = sum(h["errors"] for h in heard) / len(heard)
    marks = [(set(h["wrong"]), {int(k): v for k, v in h["inserted"].items()}) for h in heard]
    if len(heard) == 2 and (marks[0] != marks[1] or names & marks[0][0]):
        return {"errors": None, "reference_words": words, "transcribers": 2, "mean": mean}
    need = len(heard) // 2 + 1                   # most of them: 1 of 1, 2 of 2, 2 of 3
    wrong = sum(1 for i in range(words) if sum(i in w for w, _ in marks) >= (len(heard) if i in names else need))
    gaps = {g for _, ins in marks for g in ins}
    extra = sum(sorted((ins.get(g, 0) for _, ins in marks), reverse=True)[need - 1] for g in gaps)
    return {"errors": wrong + extra, "reference_words": words, "transcribers": len(heard), "mean": mean}


async def arbitrate_run(run_dir: Path, arbiter: Instrument, concurrency: int = 10) -> dict[str, int]:
    """Transcribe, with the arbiter, every recording whose transcribers split on a hard part or a word; then rescore.

    Only one-shot recordings, the ones hard parts are reported from, and only
    those are sent, so the arbiter's cost follows the disagreements, not the
    run. Its transcripts are journalled like the others and kept in the scores,
    so a later rescore keeps them.
    """
    rescore_run(run_dir)
    path = run_dir / "scores-all.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    pending = [r for r in rows if r["probe"] == "one_shot" and arbiter.name not in r["instruments"]
               and (any(v["pass"] is None for v in r.get("hardParts", ())) or (r.get("words") or {}).get("errors", 0) is None)]
    semaphore = asyncio.Semaphore(concurrency)
    counts = {"split": len(pending), "transcribed": 0, "failed": 0, "no_audio": 0}
    journal = open(run_dir / JOURNAL, "a", encoding="utf-8")

    async def one(session: aiohttp.ClientSession, row: dict[str, Any]) -> None:
        audio = run_dir / row["audio"]
        if not audio.exists():               # recorded as WAV, kept as FLAC once the run is archived
            audio = audio.with_suffix(".flac")
        if not audio.exists():
            counts["no_audio"] += 1
            return
        async with semaphore:
            entry: dict[str, Any] = {"cell_id": row["cell_id"], "context": row["context"], "instrument": arbiter.name, "model": arbiter.model}
            for attempt in range(2):
                try:
                    entry["hypothesis"] = await arbiter.transcribe(session, audio)
                    entry.pop("error", None)
                    break
                except Exception as exc:  # noqa: BLE001 -- the arbiter failed, not the provider
                    entry["error"] = repr(exc)[:300]
            append_line(journal, entry)
            if "hypothesis" in entry:
                row["instruments"][arbiter.name] = {"model": arbiter.model, "hypothesis": entry["hypothesis"]}
                counts["transcribed"] += 1
            else:
                counts["failed"] += 1

    try:
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*(one(session, r) for r in pending))
    finally:
        journal.close()
    _write_scores(run_dir, rows)
    rescore_run(run_dir)
    return counts


# ── the pass ──────────────────────────────────────────────────────────────────

def rescore_run(run_dir: Path) -> int:
    """Recompute every score from the transcripts already on disk; no instrument is called.

    The transcript is the expensive, external part; the comparison is ours. A
    change to the normaliser or the digit rule must be applicable to every run
    ever scored without paying for the audio to be transcribed again, or
    even having the audio at hand.
    """
    path = run_dir / "scores-all.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    texts = {cell["cell_id"]: cell for _, cell in _scorable_cells(run_dir)}
    for row in rows:
        target = texts.get(row["cell_id"])
        if target is None:
            continue
        for name, score in row["instruments"].items():
            if "hypothesis" in score:
                row["instruments"][name] = {"model": score["model"], **score_transcript(target["text"], target["spoken_reference"], score["hypothesis"], target["spans"], target["readings"])}
        wers = [r["best"]["wer"] for name, r in row["instruments"].items() if "best" in r and name not in ARBITERS]
        row["normalizer"] = {"name": NORMALIZER_NAME, "version": NORMALIZER_VERSION}
        row["instruments_disagree"] = len(wers) >= 2 and (max(wers) - min(wers)) > DISAGREE_WER
        row["hardParts"] = hard_part_verdicts(row["instruments"])
        row["words"] = word_verdict(row["instruments"], name_positions(target["spoken_reference"], target["names"]))
    _write_scores(run_dir, rows)
    return len(rows)


def _write_scores(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_cell: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: (r["cell_id"], r["context"] != "main", r["context"])):
        by_cell.setdefault(row["cell_id"], row)
    with open(run_dir / "scores.jsonl", "w", encoding="utf-8") as handle:
        for row in by_cell.values():
            handle.write(json.dumps(row) + "\n")
    with open(run_dir / "scores-all.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    write_manifest(run_dir)              # the run changed on purpose; the manifest follows it


def name_positions(spoken_reference: str, names: Sequence[str]) -> frozenset[int]:
    """The spoken reference's word positions that belong to a name, after the same normaliser as WER."""
    tokens = {token for name in names for token in _words(name)}
    return frozenset(i for i, word in enumerate(_words(spoken_reference)) if word in tokens)


def _with_current_readings(item_id: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A run's spans plus any reading the corpus has accepted since the run was recorded.

    Accepted readings are a scoring rule like the normaliser, so a rescore
    applies the current list to every run; a span whose text changed is left as recorded.
    """
    item = corpus.by_id().get(item_id)
    current = {span.text: span.accept for span in (item.spans if item else ())}
    return [{**span, "accept": list(dict.fromkeys([*span.get("accept", ()), *current.get(span["text"], ())]))}
            for span in spans]


def _scorable_cells(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every cell with text to score against, from the run's records alone; no audio is read."""
    out = []
    corpus_path = run_dir / "corpus.json"
    spans = {} if not corpus_path.exists() else {
        item["id"]: _with_current_readings(item["id"], item.get("spans", []))
        for item in json.loads(corpus_path.read_text())["items"]}
    for cell in latest_cells(run_dir):
        if cell.get("void") or cell.get("probe") == "cancel":
            continue                         # a cancelled utterance is partial by design; its transcript scores nothing
        directory = cell_dir(run_dir, cell)
        record = json.loads((directory / "cell.json").read_text())
        item = corpus.by_id().get(cell["item"])
        out.append((directory, {"cell_id": cell["artifacts"]["slug"], "text": record["text"]["sent"],
                                "spoken_reference": record["text"]["spoken_reference"], "cohort": cell["cohort"],
                                "probe": cell["probe"], "spans": spans.get(cell["item"], []),
                                # accepted readings are a scoring rule, so the current list applies to every run
                                "readings": list(item.readings) if item and item.text == record["text"]["sent"] else [],
                                "names": list(item.names) if item and item.text == record["text"]["sent"] else []}))
    return out


def _targets(run_dir: Path) -> list[dict[str, Any]]:
    """Every (cell, audio file) pair that has text to score against."""
    out = []
    for directory, cell in _scorable_cells(run_dir):
        # Audio is WAV as recorded, or FLAC once the run is archived; the samples are the same.
        for wav in sorted([*directory.glob("audio-*.wav"), *directory.glob("audio-*.flac")]):
            out.append({**cell, "context": wav.stem.split("-", 1)[1], "wav": wav})
    return out


JOURNAL = "transcripts.jsonl"


def _journal_from_scores(run_dir: Path, journal_path: Path) -> None:
    with open(journal_path, "w", encoding="utf-8") as journal:
        for row in read_jsonl(run_dir / "scores-all.jsonl"):
            for name, score in row["instruments"].items():
                entry = {"cell_id": row["cell_id"], "context": row["context"], "instrument": name, "model": score["model"]}
                entry.update({"hypothesis": score["hypothesis"]} if "hypothesis" in score else {"error": score.get("error", "")})
                journal.write(json.dumps(entry) + "\n")


async def score_run(run_dir: Path, instruments: Sequence[Instrument], concurrency: int = 4) -> int:
    """Transcribe every recording once per instrument, journalling each transcript as it arrives.

    A transcript is paid for, so it is appended to ``transcripts.jsonl`` the
    moment it returns; a scoring pass that dies keeps every transcript it got,
    and the next pass asks only for the ones still missing. An instrument
    error is journalled too and retried by the next pass.
    """
    targets = _targets(run_dir)
    journal_path = run_dir / JOURNAL
    if not journal_path.exists() and (run_dir / "scores-all.jsonl").exists():
        _journal_from_scores(run_dir, journal_path)      # a run scored before the journal existed keeps what it paid for
    have = {(r["cell_id"], r["context"], r["instrument"]) for r in read_jsonl(journal_path) if "hypothesis" in r}
    semaphore = asyncio.Semaphore(concurrency)
    journal = open(journal_path, "a", encoding="utf-8")

    async def one(session: aiohttp.ClientSession, target: dict[str, Any]) -> None:
        async with semaphore:
            for instrument in instruments:
                if (target["cell_id"], target["context"], instrument.name) in have:
                    continue
                entry: dict[str, Any] = {"cell_id": target["cell_id"], "context": target["context"],
                                         "instrument": instrument.name, "model": instrument.model}
                try:
                    entry["hypothesis"] = await instrument.transcribe(session, target["wav"])
                except Exception as exc:  # noqa: BLE001 -- the instrument failed, the provider did not
                    entry["error"] = repr(exc)
                append_line(journal, entry)

    try:
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*(one(session, t) for t in targets))
    finally:
        journal.close()

    latest: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for entry in read_jsonl(journal_path):          # the last entry per instrument wins
        latest.setdefault((entry["cell_id"], entry["context"]), {})[entry["instrument"]] = entry
    rows: list[dict[str, Any]] = []
    for target in targets:
        results: dict[str, Any] = {}
        for name, entry in latest.get((target["cell_id"], target["context"]), {}).items():
            if "hypothesis" in entry:
                results[name] = {"model": entry["model"], **score_transcript(target["text"], target["spoken_reference"], entry["hypothesis"], target["spans"], target["readings"])}
            else:
                results[name] = {"model": entry["model"], "error": entry["error"]}
        wers = [r["best"]["wer"] for name, r in results.items() if "best" in r and name not in ARBITERS]
        rows.append({
            "cell_id": target["cell_id"], "context": target["context"], "cohort": target["cohort"], "probe": target["probe"],
            "audio": str(target["wav"].relative_to(run_dir)),
            "normalizer": {"name": NORMALIZER_NAME, "version": NORMALIZER_VERSION},
            "instruments": results,
            "instruments_disagree": len(wers) >= 2 and (max(wers) - min(wers)) > DISAGREE_WER,
            "hardParts": hard_part_verdicts(results),
            "words": word_verdict(results, name_positions(target["spoken_reference"], target["names"])),
        })
    _write_scores(run_dir, rows)   # scores.jsonl holds one row per cell (the main context); scores-all.jsonl everything
    return len(rows)


async def score_floor(directory: Path, instruments: Sequence[Instrument], concurrency: int = 4) -> dict[str, Any]:
    """The instrument's own error on human speech with verified transcripts.

    Every round-trip WER published for a TTS service has this number under it:
    an instrument that gets 3% of a human reader wrong cannot certify a service
    at 1%. Written beside the recordings as ``floor-scores.json`` and quoted in
    the methodology.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    semaphore = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []

    async def one(session: aiohttp.ClientSession, item: dict[str, Any]) -> None:
        async with semaphore:
            results: dict[str, Any] = {}
            for instrument in instruments:
                try:
                    hypothesis = await instrument.transcribe(session, directory / item["audio"])
                    results[instrument.name] = {"model": instrument.model, **score_pair(item["text"], hypothesis), "hypothesis": hypothesis}
                except Exception as exc:  # noqa: BLE001
                    results[instrument.name] = {"model": instrument.model, "error": repr(exc)}
            rows.append({"id": item["id"], "instruments": results})

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(one(session, item) for item in manifest["items"]))
    pooled: dict[str, Any] = {}
    for instrument in instruments:
        scored = [r["instruments"][instrument.name] for r in rows if "wer" in r["instruments"][instrument.name]]
        words = sum(s["reference_words"] for s in scored)
        errors = sum(s["errors"] for s in scored)
        pooled[instrument.name] = {"model": instrument.model, "recordings": len(scored), "reference_words": words,
                                   "pooled_wer": None if not words else round(errors / words, 4),
                                   "failed": len(rows) - len(scored)}
    out = {"source": manifest.get("source"), "normalizer": {"name": NORMALIZER_NAME, "version": NORMALIZER_VERSION},
           "pooled": pooled, "rows": sorted(rows, key=lambda r: r["id"])}
    (directory / "floor-scores.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Transcribe a TTS run's audio and score it against the corpus.")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--floor", help="an instrument-floor directory (human speech + manifest) to score instead of runs")
    parser.add_argument("--rescore", action="store_true", help="recompute from stored transcripts; calls no instrument")
    parser.add_argument("--instrument", action="append", choices=sorted(INSTRUMENTS), help="default: every one with a key")
    parser.add_argument("--env", help="dotenv file holding instrument credentials")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--arbitrate", action="store_true",
                        help="then send each recording whose transcribers split on a hard part or a word to Zoom Scribe (ZOOM_API_KEY)")
    args = parser.parse_args(argv)

    values: dict[str, str] = {}
    if args.env:
        from dotenv import dotenv_values

        values = {k: v for k, v in dotenv_values(args.env).items() if v}
    arbiter = None
    if args.arbitrate:
        cls, env = ARBITERS["zoom"]
        key = os.environ.get(env) or values.get(env)
        if not key:
            print(f"{env} is not set for --arbitrate", file=sys.stderr)
            return 2
        arbiter = cls(key)

    def settle(run: str) -> None:
        if arbiter:
            counts = asyncio.run(arbitrate_run(Path(run), arbiter, max(args.concurrency, 10)))
            print(f"{run}: {counts['split']} recordings split on a hard part or a word; {counts['transcribed']} sent to "
                  f"{arbiter.name}:{arbiter.model}, {counts['failed']} failed, {counts['no_audio']} without audio")

    if args.rescore:
        for run in args.runs:
            print(f"{run}: {rescore_run(Path(run))} rows rescored with {NORMALIZER_NAME} {NORMALIZER_VERSION}")
            settle(run)
        return 0
    chosen = args.instrument or list(INSTRUMENTS)
    instruments = []
    for name in chosen:
        cls, env = INSTRUMENTS[name]
        key = os.environ.get(env) or values.get(env)
        if key:
            instruments.append(cls(key))
        elif args.instrument:
            print(f"{env} is not set for instrument {name}", file=sys.stderr)
            return 2
    if not instruments:
        print("no instrument has a credential", file=sys.stderr)
        return 2
    print(f"instruments: {[f'{i.name}:{i.model}' for i in instruments]} · normaliser {NORMALIZER_NAME} {NORMALIZER_VERSION}")
    if args.floor:
        floor = asyncio.run(score_floor(Path(args.floor), instruments, args.concurrency))
        for name, summary in floor["pooled"].items():
            print(f"  floor {name}:{summary['model']} pooled WER {summary['pooled_wer']} over {summary['reference_words']} words "
                  f"({summary['recordings']} recordings, {summary['failed']} failed)")
    for run in args.runs:
        n = asyncio.run(score_run(Path(run), instruments, args.concurrency))
        print(f"{run}: {n} recordings scored -> scores.jsonl")
        settle(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
