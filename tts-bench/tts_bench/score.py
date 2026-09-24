"""Offline round-trip scoring: transcribe every synthesised utterance and compare.

Kept separate from the run on purpose. The provider call is the expensive,
unrepeatable part; the transcription instrument is cheap and replaceable, so
changing it must never require re-running the provider. This pass reads the
audio a run wrote, sends it to one or more ASR instruments, and writes
``scores.jsonl`` beside ``cells.jsonl``. The report picks it up if present.

Scoring rule: the hypothesis is compared with both the text as sent and its
``spoken_reference`` after the same normaliser, and the lower error is kept.
Either rendering is a correct reading. Digit sequences are compared separately
because a task depends on them and a word error rate hides one wrong digit.

The instrument is not the subject. Two instruments are run where two are
available, and a row on which they disagree by more than the declared margin
is flagged, not averaged: that disagreement is the instrument's error, and the
honest treatment is to withhold the row rather than pick the friendlier one.
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
from typing import Any, Sequence

import aiohttp

from tts_bench.common.transcript import edit_distance, normalize as _fallback_normalize

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


def score_pair(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = normalize(reference), normalize(hypothesis)
    errors = edit_distance(ref, hyp)
    return {"errors": errors, "reference_words": len(ref), "wer": round(errors / max(len(ref), 1), 4),
            "normalized_reference": " ".join(ref), "normalized_hypothesis": " ".join(hyp)}


def _canonical_amounts(text: str) -> str:
    """``$12,480.00`` is read as twelve thousand four hundred eighty; ``$0.75`` as seventy five cents.

    The zeros a normaliser drops from a spoken amount are not digits a listener
    was meant to hear, so they are dropped from the expectation as well.
    """
    text = re.sub(r"(\d)\.00\b", r"\1", text)
    return re.sub(r"\b0\.(\d\d)\b", r"\1", text)


def score_transcript(text: str, spoken_reference: str, hypothesis: str) -> dict[str, Any]:
    against_text = score_pair(text, hypothesis)
    against_spoken = score_pair(spoken_reference, hypothesis)
    best = min((against_text, against_spoken), key=lambda s: s["wer"])
    # Digits come from the written text, in order; the hypothesis is read after
    # the same normaliser has turned its number words into digits. An item with
    # no digit in its text has no digit check.
    ref_digits = "".join(re.findall(r"[0-9]", _canonical_amounts(text)))
    hyp_digits = "".join(re.findall(r"[0-9]", " ".join(normalize(hypothesis))))
    return {
        "best": best,
        "best_reference": "text" if best is against_text else "spoken_reference",
        "against_text": against_text,
        "against_spoken": against_spoken,
        "digits_expected": ref_digits,
        "digits_heard": hyp_digits,
        "digits_match": bool(ref_digits) and ref_digits == hyp_digits,
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


# ── the pass ──────────────────────────────────────────────────────────────────

def rescore_run(run_dir: Path) -> int:
    """Recompute every score from the transcripts already on disk; no instrument is called.

    The transcript is the expensive, external part; the comparison is ours. A
    change to the normaliser or the digit rule must be applicable to every run
    ever scored without paying for the audio to be transcribed again.
    """
    path = run_dir / "scores-all.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    texts = {t["cell_id"]: t for t in _targets(run_dir)}
    for row in rows:
        target = texts.get(row["cell_id"])
        if target is None:
            continue
        for name, score in row["instruments"].items():
            if "hypothesis" in score:
                row["instruments"][name] = {"model": score["model"], **score_transcript(target["text"], target["spoken_reference"], score["hypothesis"])}
        wers = [r["best"]["wer"] for r in row["instruments"].values() if "best" in r]
        row["normalizer"] = {"name": NORMALIZER_NAME, "version": NORMALIZER_VERSION}
        row["instruments_disagree"] = len(wers) >= 2 and (max(wers) - min(wers)) > DISAGREE_WER
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


def _targets(run_dir: Path) -> list[dict[str, Any]]:
    """Every (cell, audio file) pair that has text to score against."""
    out = []
    for line in (run_dir / "cells.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        cell = json.loads(line)
        if cell.get("void") or cell.get("probe") == "cancel":
            continue                         # a cancelled utterance is partial by design; its transcript scores nothing
        directory = Path(cell["artifacts"]["dir"])
        record = json.loads((directory / "cell.json").read_text())
        for wav in sorted(directory.glob("audio-*.wav")):
            out.append({"cell_id": cell["artifacts"]["slug"], "context": wav.stem.split("-", 1)[1], "wav": wav,
                        "text": record["text"]["sent"], "spoken_reference": record["text"]["spoken_reference"],
                        "cohort": cell["cohort"], "probe": cell["probe"]})
    return out


async def score_run(run_dir: Path, instruments: Sequence[Instrument], concurrency: int = 4) -> int:
    targets = _targets(run_dir)
    semaphore = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []

    async def one(session: aiohttp.ClientSession, target: dict[str, Any]) -> None:
        async with semaphore:
            results: dict[str, Any] = {}
            for instrument in instruments:
                try:
                    hypothesis = await instrument.transcribe(session, target["wav"])
                    results[instrument.name] = {"model": instrument.model, **score_transcript(target["text"], target["spoken_reference"], hypothesis)}
                except Exception as exc:  # noqa: BLE001 -- the instrument failed, the provider did not
                    results[instrument.name] = {"model": instrument.model, "error": repr(exc)}
            wers = [r["best"]["wer"] for r in results.values() if "best" in r]
            rows.append({
                "cell_id": target["cell_id"], "context": target["context"], "cohort": target["cohort"], "probe": target["probe"],
                "audio": str(target["wav"].relative_to(run_dir)),
                "normalizer": {"name": NORMALIZER_NAME, "version": NORMALIZER_VERSION},
                "instruments": results,
                "instruments_disagree": len(wers) >= 2 and (max(wers) - min(wers)) > DISAGREE_WER,
            })

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(one(session, t) for t in targets))

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
    args = parser.parse_args(argv)

    if args.rescore:
        for run in args.runs:
            print(f"{run}: {rescore_run(Path(run))} rows rescored with {NORMALIZER_NAME} {NORMALIZER_VERSION}")
        return 0
    values: dict[str, str] = {}
    if args.env:
        from dotenv import dotenv_values

        values = {k: v for k, v in dotenv_values(args.env).items() if v}
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
