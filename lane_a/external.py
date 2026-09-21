"""Grounding the harness against audio someone else authored, with known answers.

Everything else in Lane A is measured with audio we wrote, which is what makes
the caller-side boundary exact -- and also means a fault in our own audio path
would be invisible, because the only thing checking it is us. A bug that fed the
provider audio at the wrong rate, or truncated it, or resampled it badly, would
not fail any test here: it would quietly look like a provider that reasons less
well than it does.

So the instrument is checked against a public set with ground-truth answers:

    Big Bench Audio -- 1,000 spoken reasoning questions adapted from BIG-bench
    Hard, MIT licensed, audio included.
    https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio

This is **never a published ranking of ours**. It is a single-turn reasoning
quiz with no interaction in it, and it is the most widely circulated audio set in
the field, so it is also the most likely to have been trained on.

Its value here is that the answers are known and guessing is cheap to price.
Three of the four categories are binary and the fourth is a count, so an
instrument that has destroyed the audio scores near half on the first three and
near nothing on the last. A working path scores far above that, and the distance
between those outcomes is tens of points -- which is what makes the check
trustworthy at a sample size a credential can afford, and why a result near the
floor means the fault is ours to find before any ranking is published.

Answers are closed-form -- ``valid``/``invalid``, ``Yes``/``No``, or a count --
so grading is exact match on an extracted token rather than a model's opinion.
Using a judge here would mean checking one instrument with another.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lane_a.caller import Clip
from lane_a.probes import ProbeContext, ProbeResult

DATASET = "ArtificialAnalysis/big_bench_audio"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
LICENSE = "MIT"
TARGET_RATE = 24000

# The reply must be gradable without a judge, so the model is asked for the
# answer and nothing else. Stated here because it is part of the configuration:
# a chattier instruction would lower the score without anything being wrong.
INSTRUCTIONS = (
    "You are answering a spoken reasoning question. "
    "Give only the final answer, in as few words as possible, and say nothing else."
)

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


@dataclass(frozen=True)
class Question:
    id: int
    category: str
    official_answer: str
    file_name: str


def _download(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            target.write_bytes(response.read())
    return target


def metadata(cache: Path) -> list[Question]:
    path = _download(f"{BASE_URL}/metadata.jsonl", cache / "metadata.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [Question(r["id"], r["category"], str(r["official_answer"]), r["file_name"]) for r in rows]


def sample(questions: list[Question], per_category: int, seed: int = 11) -> list[Question]:
    """A stratified draw, seeded, so a validation run is repeatable.

    Stratified because the four categories are different tasks: an unstratified
    draw could land mostly on one and turn a category-specific problem into an
    apparent harness fault, or hide one.
    """
    rng = random.Random(seed)
    chosen: list[Question] = []
    for category in sorted({q.category for q in questions}):
        pool = sorted((q for q in questions if q.category == category), key=lambda q: q.id)
        chosen.extend(rng.sample(pool, min(per_category, len(pool))))
    return sorted(chosen, key=lambda q: q.id)


def load_clip(question: Question, cache: Path, rate: int = TARGET_RATE) -> Clip:
    """One question as PCM at the provider's rate.

    ffmpeg decodes the mp3 and resamples in one pass rather than going through
    our own resampler: this audio is the reference, and running it through a
    component under test would leave that component checking itself.
    """
    source = _download(f"{BASE_URL}/{question.file_name}", cache / question.file_name)
    pcm = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(source), "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    return Clip(name=f"bba.{question.id}", pcm=pcm, rate=rate, text="")


# ── grading ──────────────────────────────────────────────────────────────────

def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def extract_answer(reply: str, category: str) -> str | None:
    """The answer token in a spoken reply, or None if there is not one.

    Word boundaries matter more than they look: ``invalid`` contains ``valid``,
    so a substring test would score every wrong answer in that category correct.
    """
    words = _words(reply)
    if category == "formal_fallacies":
        for word in words:
            if word in ("invalid", "valid"):
                return word
        return None
    if category in ("navigate", "web_of_lies"):
        for word in words:
            if word in ("yes", "no"):
                return word
        return None
    for word in words:                       # object_counting
        if word.isdigit():
            return str(int(word))
        if word in _NUMBER_WORDS:
            return str(_NUMBER_WORDS[word])
    return None


def grade(reply: str, question: Question) -> bool:
    extracted = extract_answer(reply, question.category)
    if extracted is None:
        return False
    official = question.official_answer.strip().lower()
    if question.category == "object_counting":
        return extracted == str(int(official))
    return extracted == official


# ── the probe ────────────────────────────────────────────────────────────────

@dataclass
class SpokenQuestion:
    """Ask one question aloud and grade the spoken answer.

    A validation cell, not a ranked one. It still runs through the ordinary
    runner so it produces the same artifacts as everything else -- if the
    validation cannot itself be rechecked from files, it is not evidence.
    """

    question: Question
    clip: Clip
    settle_ms: float = 600.0
    timeout_s: float = 45.0
    name: str = "spoken_question"

    @property
    def slug(self) -> str:
        return f"{self.name}-{self.question.category}-{self.question.id}"

    async def run(self, ctx: ProbeContext) -> ProbeResult:
        await ctx.caller.play(self.clip)
        # Settle on the response completing, not merely on the audio going quiet.
        # The transcript this is graded from arrives with the response, which can
        # be after the last audio chunk -- waiting on silence alone read a
        # perfectly good answer as having no text in it.
        if not await ctx.wait_reply(self.settle_ms, timeout_s=self.timeout_s):
            return ProbeResult(self.name, void="agent never finished a reply")

        reply = " ".join(ctx.adapter.agent_text)
        if not reply.strip():
            # Audio arrived but no transcript did; grading text we do not have
            # would score a harness gap as a wrong answer.
            return ProbeResult(self.name, void="no transcript of the reply")
        correct = grade(reply, self.question)
        return ProbeResult(
            self.name,
            verdict="pass" if correct else "fail",
            values={
                "dataset": DATASET,
                "question_id": self.question.id,
                "category": self.question.category,
                "official_answer": self.question.official_answer,
                "extracted": extract_answer(reply, self.question.category),
                "reply": reply[:300],
                "heard_as": " ".join(ctx.adapter.caller_text)[:300],
                "audio_ms": round(self.clip.duration_ms, 1),
            },
        )


def summarize(cells: list[Any]) -> dict[str, Any]:
    """Accuracy overall and per category, with the voids kept separate.

    Only graded questions count. A campaign carries cells that are not questions
    -- the sentinel latency cell leads every run -- and those always pass, so
    folding them in would raise the score by however many of them there were.
    A question is identified by carrying the answer it is graded against.
    """
    questions = [c for c in cells if "official_answer" in c.values or c.void]
    scored = [c for c in questions if not c.void]
    by_category: dict[str, list[bool]] = {}
    for cell in scored:
        by_category.setdefault(cell.values.get("category", "?"), []).append(cell.verdict == "pass")
    return {
        "dataset": DATASET,
        "license": LICENSE,
        "scored": len(scored),
        "void": len(questions) - len(scored),
        "accuracy": round(sum(c.verdict == "pass" for c in scored) / len(scored), 4) if scored else None,
        "by_category": {
            name: {"n": len(hits), "accuracy": round(sum(hits) / len(hits), 4)}
            for name, hits in sorted(by_category.items())
        },
    }
