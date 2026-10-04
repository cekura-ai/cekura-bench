"""Naturalness of the synthesised speech, estimated by a listener model.

Transcribers say whether the words came through; they say nothing about how
the voice sounds. The estimate here is UTMOS (the ``utmos22_strong`` model
from the VoiceMOS 2022 challenge, MIT-licensed code and weights), which was
trained to predict the mean opinion score listeners give synthetic speech on
the usual 1 to 5 scale. It is a model's guess at a listening test, and is
published as such: a difference of a tenth is noise, a difference of half a
point is a voice people would notice.

Every one-shot recording is scored once and the number journalled to
``naturalness.jsonl`` in the run, keyed by cell, so a later export never has
to run the model again and an amendment run joins cell by cell. The model
is fetched through ``torch.hub`` at a pinned tag and needs ``torch`` and
``torchaudio``, which the harness itself does not; ``bin/score-naturalness.py``
runs in its own environment (see ``requirements-naturalness.txt``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tts_bench.store import append_line, latest_cells, read_jsonl

JOURNAL = "naturalness.jsonl"
MODEL = {"name": "utmos22-strong", "hub": "tarepan/SpeechMOS:v1.2.0", "entry": "utmos22_strong", "sample_rate": 16000}


def recordings(run_dir: Path) -> list[dict[str, Any]]:
    """The one-shot cells worth scoring, each with the audio file it left behind (FLAC once archived, WAV before)."""
    out = []
    for cell in latest_cells(run_dir):
        if cell["probe"] != "one_shot" or cell.get("void") or cell.get("sentinel"):
            continue
        folder = run_dir / cell["artifacts"]["slug"]
        audio = next((p for p in (folder / "audio-main.flac", folder / "audio-main.wav") if p.exists()), None)
        if audio is not None:
            out.append({"cell_id": cell["artifacts"]["slug"], "item": cell["item"], "cohort": cell["cohort"],
                        "repeat": cell["repeat"], "audio": audio})
    return out


def scored(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / JOURNAL
    return {row["cell_id"]: row for row in read_jsonl(path)} if path.exists() else {}


def load_model() -> Any:
    import torch  # the heavy import stays inside, so the harness never needs it

    return torch.hub.load(MODEL["hub"], MODEL["entry"], trust_repo=True)


def score_run(run_dir: str | Path, model: Any = None, log: Any = print) -> int:
    """Score every unscored one-shot recording of a run, appending each result as it lands. Returns how many were scored."""
    import librosa
    import torch

    root = Path(run_dir)
    todo = [r for r in recordings(root) if r["cell_id"] not in scored(root)]
    if not todo:
        return 0
    model = model or load_model()
    done = 0
    with open(root / JOURNAL, "a") as handle:
        for rec in todo:
            wave, sr = librosa.load(rec["audio"], sr=MODEL["sample_rate"], mono=True)
            with torch.no_grad():
                score = float(model(torch.from_numpy(wave)[None], sr))
            append_line(handle, {"cell_id": rec["cell_id"], "item": rec["item"], "cohort": rec["cohort"], "repeat": rec["repeat"],
                                 "audio_s": round(len(wave) / sr, 3), "model": MODEL["name"], "hub": MODEL["hub"], "mos": round(score, 4)})
            done += 1
            if done % 100 == 0:
                log(f"{root.name}: {done}/{len(todo)}")
    return done


def rows(run_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    """Journalled scores across a run and its amendments, the latest reading of a cell winning."""
    out: dict[str, dict[str, Any]] = {}
    for root in run_dirs:
        out.update(scored(Path(root)))
    return list(out.values())


def as_json(run_dir: str | Path) -> str:
    return json.dumps(rows([Path(run_dir)]))
