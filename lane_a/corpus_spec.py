"""A corpus and scenario set loaded from files, so a holdout can live elsewhere.

The public corpus is authored in ``corpus_v1.py`` because a script is easier to
read and review than a table. The hidden holdout -- same schema, different
phrasings, different probe offsets -- must not live in this repository, and a
loader that reads the same shapes from a directory is what lets the identical
probes run against it. Nothing about the harness knows which set it is running.

A holdout directory holds:

    corpus.json      {"version", "voices": {label: {vendor, voice_id, model}}, "clips": [{id, text, note}]}
    audio/<voice>/   the renders, named by the same text-stamp rule as the public set
    scenarios.json   a list of ScenarioSpec documents
    params.json      probe parameters that replace the public defaults (optional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lane_a.clips import ClipSpec, Corpus, Voice
from lane_a.scenarios import ScenarioSpec


def load_corpus(directory: str | Path) -> Corpus:
    root = Path(directory)
    spec = json.loads((root / "corpus.json").read_text())
    corpus = Corpus(root, spec.get("version", "holdout"))
    corpus.add(*(ClipSpec(c["id"], c["text"], c.get("note", "")) for c in spec["clips"]))
    corpus.add_voice(
        *(
            Voice(label=label, vendor=v["vendor"], voice_id=v["voice_id"], model=v.get("model", Voice.model))
            for label, v in spec["voices"].items()
        )
    )
    return corpus


def load_scenarios(directory: str | Path) -> list[ScenarioSpec]:
    path = Path(directory) / "scenarios.json"
    if not path.exists():
        return []
    return [ScenarioSpec.from_json(item) for item in json.loads(path.read_text())]


def load_params(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "params.json"
    return json.loads(path.read_text()) if path.exists() else {}


def dump_corpus(corpus: Corpus) -> dict[str, Any]:
    """The public corpus in the file shape, for authoring a holdout beside it."""
    return {
        "version": corpus.version,
        "voices": {label: {"vendor": v.vendor, "voice_id": v.voice_id, "model": v.model} for label, v in corpus.voices.items()},
        "clips": [{"id": c.id, "text": c.text, "note": c.note} for c in corpus.specs.values()],
    }
