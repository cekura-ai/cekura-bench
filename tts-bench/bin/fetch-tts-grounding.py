#!/usr/bin/env python
"""Fetch the public material the TTS instrument is grounded against. Nothing is vendored.

Two sets, both fetched from public dataset hosting at run time and written
under data/tts-grounding/ (git-ignored):

* ``instrument-floor/`` -- human read speech with verified transcripts
  (LibriSpeech test-clean, CC BY 4.0). Transcribing it with the same instruments
  and normaliser used on synthesised audio gives the instrument's own error
  floor, so a TTS word error rate can be read relative to what the instrument
  gets wrong on a human. This is the calibration every round-trip WER needs
  and few publish.
* ``external-corpus/`` -- text prompts from a published TTS evaluation set
  (EmergentTTS-Eval, Apache-2.0), the "Complex Pronunciation" and "Questions"
  categories, written in this bench's corpus shape so the identical probes run
  on them. Cross-checking our cohort results against an independently authored
  hard-case set is how the corpus itself is grounded. Their spoken_reference is
  the text itself: the set ships none, so digits and symbols are scored against
  the written form only.

    python bin/fetch-tts-grounding.py --floor 50 --external 60
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROWS = "https://datasets-server.huggingface.co/rows"


def rows(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    with urllib.request.urlopen(f"{ROWS}?{query}", timeout=60) as response:
        return json.loads(response.read())


def fetch_floor(out: Path, n: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    offset = 0
    while len(manifest) < n:
        page = rows("openslr/librispeech_asr", "clean", "test", offset, min(100, n - len(manifest)))
        if not page.get("rows"):
            break
        for entry in page["rows"]:
            row = entry["row"]
            audio = row["audio"][0]["src"] if isinstance(row["audio"], list) else row["audio"]["src"]
            # Served as FLAC; kept as FLAC and named so, since the instruments read it by suffix.
            path = out / f"{row['id']}.flac"
            if not path.exists():
                data = urllib.request.urlopen(audio, timeout=120).read()
                path.write_bytes(data)
            manifest.append({"id": row["id"], "audio": path.name, "text": row["text"].lower(), "speaker_id": row["speaker_id"]})
            if len(manifest) >= n:
                break
        offset += len(page["rows"])
    (out / "manifest.json").write_text(json.dumps({
        "source": "LibriSpeech test-clean via openslr/librispeech_asr (CC BY 4.0)",
        "purpose": "ASR instrument error floor on human speech", "items": manifest}, indent=2))
    print(f"instrument floor: {len(manifest)} recordings -> {out}")


def fetch_external(out: Path, n: int, categories: tuple[str, ...]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    offset = 0
    seen = 0
    while len(items) < n and offset < 2000:
        page = rows("bosonai/EmergentTTS-Eval", "default", "train", offset, 100)
        got = page.get("rows") or []
        if not got:
            break
        for entry in got:
            row = entry["row"]
            if row.get("language") != "en" or row["category"] not in categories:
                continue
            if str(row.get("evolution_depth")) != "0":
                continue                     # the seed prompts, not their LLM-lengthened variants
            seen += 1
            slug = row["category"].lower().replace(" ", "_")
            items.append({"id": f"ext.{slug}.{seen:03d}", "cohort": f"external.{slug}",
                          "text": row["text_to_synthesize"], "spoken_reference": row["text_to_synthesize"].lower(),
                          "note": "external hard-case set; spoken form not provided"})
            if len(items) >= n:
                break
        offset += len(got)
    (out / "corpus.json").write_text(json.dumps({
        "version": "external-emergenttts-eval-seed",
        "source": "bosonai/EmergentTTS-Eval (Apache-2.0), evolution_depth 0, English",
        "cohorts": {f"external.{c.lower().replace(' ', '_')}": f"external set, category {c}" for c in categories},
        "items": items}, indent=2))
    print(f"external corpus: {len(items)} items from {categories} -> {out / 'corpus.json'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/tts-grounding")
    parser.add_argument("--floor", type=int, default=50, help="human recordings for the instrument floor (0 to skip)")
    parser.add_argument("--external", type=int, default=60, help="external hard-case prompts (0 to skip)")
    parser.add_argument("--category", action="append", default=None)
    args = parser.parse_args()
    out = Path(args.out)
    if args.floor:
        fetch_floor(out / "instrument-floor", args.floor)
    if args.external:
        fetch_external(out / "external-corpus", args.external, tuple(args.category or ("Complex Pronunciation", "Questions")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
