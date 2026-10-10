#!/usr/bin/env python
"""Export listening samples for the website: one recording per model for a few sentences.

    python bin/export-samples.py <run>... --models config/site-models.json --out <dir> [--item id]...

For each chosen sentence and each model, the first one-shot recording from the
newest run that has it is transcoded to a small mono MP3 under
``<out>/<site model id>/<item>.mp3``, and ``<out>/samples.json`` lists every
clip with its sentence, model and length. Nothing is altered: the clip is the
audio the service sent, resampled for the page. A sentence a model never
recorded is simply absent from the manifest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_bench import corpus, store  # noqa: E402

# Sentences across the cohorts, short enough to listen to in a row, with the hard parts a caller would notice.
DEFAULT_ITEMS = [
    "prose.greet", "currency.balance", "datetime.iso", "alnum.tracking", "contact.email_digits", "names.siobhan",
    "repair.actually", "heteronym.read", "abbrev.shorthand", "symbols.keypad", "symbols.email", "terms.thyroid",
    "numbers.temperature", "long.summary", "tech.error", "worded.time", "dense.renewal", "numbers.apostrophe", "repair.menu",
]
BITRATE = "40k"


def recording(run_dir: Path, item: str) -> Path | None:
    for cell in store.latest_cells(run_dir):
        if cell["probe"] == "one_shot" and cell["item"] == item and cell["repeat"] == 1 and not cell.get("void") and not cell.get("sentinel"):
            folder = run_dir / cell["artifacts"]["slug"]
            return next((p for p in (folder / "audio-main.flac", folder / "audio-main.wav") if p.exists()), None)
    return None


def seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return round(float(out), 2)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--models", required=True, help="the site's model list: display names and ids")
    parser.add_argument("--out", required=True)
    parser.add_argument("--item", action="append", help="sentences to export; default is a fixed set across the cohorts")
    args = parser.parse_args(argv)

    models = json.loads(Path(args.models).read_text())["models"]
    items = args.item or DEFAULT_ITEMS
    by_id = corpus.by_id()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Newest run first, so the first run that holds a sentence is the one that measured it last.
    runs = sorted((Path(r) for r in args.runs),
                  key=lambda r: json.loads((r / "provenance.json").read_text())["started_utc"], reverse=True)
    clips = []
    for run in runs:
        p = json.loads((run / "provenance.json").read_text())
        site = models.get(f"{p['provider']}/{p['model']}")
        if site is None:
            continue
        # The text a run read, which may predate a rewording: a clip must say the sentence the page shows.
        read = {i["id"]: i["text"] for i in json.loads((run / "corpus.json").read_text())["items"]}
        for item in items:
            if any(c["model"] == site["id"] and c["item"] == item for c in clips):
                continue
            if item not in by_id or read.get(item) != by_id[item].text:
                continue
            source = recording(run, item)
            if source is None:
                continue
            target = out / site["id"] / f"{item}.mp3"
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1", "-ar", "24000",
                            "-codec:a", "libmp3lame", "-b:a", BITRATE, str(target)], check=True)
            clips.append({"model": site["id"], "item": item, "file": f"{site['id']}/{item}.mp3", "seconds": seconds(target),
                          "run": run.name})
    manifest = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "AI-generated speech from each vendor's API, transcoded to MP3 for the page; otherwise as received.",
        "items": [{"id": i, "cohort": by_id[i].cohort, "text": by_id[i].text} for i in items if i in by_id and any(c["item"] == i for c in clips)],
        "models": [{"id": m["id"], "name": m["name"], "vendor": m["vendor"]} for m in models.values() if any(c["model"] == m["id"] for c in clips)],
        "clips": clips,
    }
    (out / "samples.json").write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(f.stat().st_size for f in out.rglob("*.mp3"))
    print(f"{len(clips)} clips, {len(manifest['items'])} sentences, {len(manifest['models'])} models, {total / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
