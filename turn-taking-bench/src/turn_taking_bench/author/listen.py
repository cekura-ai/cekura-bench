"""Listening pack: a stereo WAV and a cue sheet per conversation, to check labels by ear.

For each conversation of a built dataset directory this writes, under
``<dataset>/listening/`` (so it stays with the gitignored build):

    <conversation_id>-<spec id>.wav       stereo 16-bit WAV at the dataset rate:
                                          caller (speaker_1) left, agent (speaker_2) right
    <conversation_id>-<spec id>.cues.csv  every labelled event in time order

The cue sheet has one row per annotation segment (what the tracks say) and one
per planted gold event (what TurnBench's scorer will test), so a listener can seek
to ``time_s`` in any audio editor and confirm the label matches what is heard:

    time_s, end_s, speaker, channel, event, label, text

``event`` is ``segment`` (annotation), ``eot_positive`` (turn ends here),
``eot_negative`` (a mid-turn pause: start..end), ``int_positive`` (a barge-in
onset) or ``int_negative`` (a backchannel: start..end); ``channel`` is ``L`` for
the caller and ``R`` for the agent. The audio is read back from the dataset's
own parquet shard, so what is heard is exactly what a model is streamed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from turn_taking_bench.data import load_audio_conversation, load_local_dataset, local_split, locate_rows

SPEAKER_NAMES = {1: "caller", 2: "agent"}
SPEAKER_CHANNELS = {1: "L", 2: "R"}
CUE_COLUMNS = ("time_s", "end_s", "speaker", "channel", "event", "label", "text")


def cue_rows(truth: dict[str, Any]) -> list[dict[str, Any]]:
    """The cue sheet rows for one conversation's ``ground_truth.json`` record, sorted by time."""
    rows: list[dict[str, Any]] = []

    def row(time_s: float, end_s: float | None, speaker: int, event: str, label: str, text: str = "") -> dict[str, Any]:
        return {
            "time_s": round(time_s, 3),
            "end_s": "" if end_s is None else round(end_s, 3),
            "speaker": SPEAKER_NAMES[speaker],
            "channel": SPEAKER_CHANNELS[speaker],
            "event": event,
            "label": label,
            "text": text,
        }

    for speaker_key, events in truth["annotations"].items():
        speaker = int(speaker_key)
        for event in events:
            rows.append(row(event["start_s"], event["end_s"], speaker, "segment", event["label"], event["text"]))
    planted = truth["planted"]
    for event in planted["eot_positive"]:
        rows.append(row(event["time_s"], None, event["speaker"], "eot_positive", "turn ends"))
    for event in planted["eot_negative"]:
        rows.append(row(event["start_s"], event["end_s"], event["speaker"], "eot_negative", "mid-turn pause"))
    for event in planted["int_positive"]:
        rows.append(row(event["time_s"], None, event["speaker"], "int_positive", "barge-in onset"))
    for event in planted["int_negative"]:
        rows.append(row(event["start_s"], event["end_s"], event["speaker"], "int_negative", "backchannel"))
    order = {"segment": 0, "int_positive": 1, "int_negative": 2, "eot_negative": 3, "eot_positive": 4}
    rows.sort(key=lambda r: (r["time_s"], order[r["event"]], r["speaker"]))
    return rows


def write_listening_pack(dataset_dir: Path, out_dir: Path | None = None, ids: list[str] | None = None) -> list[Path]:
    """Write the WAV and cue sheet for every conversation (or only ``ids``);
    returns the files written."""
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir) if out_dir else dataset_dir / "listening"
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = json.loads((dataset_dir / "ground_truth.json").read_text(encoding="utf-8"))
    dataset = load_local_dataset(dataset_dir)
    locations = locate_rows(local_split(dataset_dir))
    written: list[Path] = []
    wanted = sorted(dataset.index, key=int) if ids is None else list(ids)
    missing = [cid for cid in wanted if cid not in dataset.index]
    if missing:
        raise KeyError(f"conversations not in {dataset_dir}: {missing}")
    for cid in wanted:
        conv = load_audio_conversation(cid, locations[cid], dataset.durations[cid])  # one conversation's audio at a time
        left, rate = conv.audio(1)
        right, rate_2 = conv.audio(2)
        if rate != rate_2 or len(left) != len(right):
            raise ValueError(f"conversation {cid}: channels differ ({rate} Hz x {len(left)} vs {rate_2} Hz x {len(right)})")
        stem = f"{cid}-{truth[cid]['spec_id']}"
        wav_path = out_dir / f"{stem}.wav"
        sf.write(wav_path, np.stack([left, right], axis=1), rate, subtype="PCM_16")
        cues_path = out_dir / f"{stem}.cues.csv"
        with cues_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CUE_COLUMNS)
            writer.writeheader()
            writer.writerows(cue_rows(truth[cid]))
        written += [wav_path, cues_path]
    return written
