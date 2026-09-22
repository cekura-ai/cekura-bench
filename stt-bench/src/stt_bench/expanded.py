"""Prepare the reviewed public anchor and disjoint entity supplement."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad

from .data import RATE, FRAME_SAMPLES, prepare_audio, read_fleurs, sha256, write_json


def boundary_review(source: Path, details: dict) -> dict:
    audio, _ = sf.read(source, dtype="float64")
    retained = details["retained_samples"]
    tail = audio[retained:]
    pcm = np.rint(np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2")
    padded = np.pad(pcm, (0, (-len(pcm)) % FRAME_SAMPLES))
    # Run on the whole signal: a cold start on quiet trailing noise is unreliable.
    vad = webrtcvad.Vad(0)
    voiced = [i for i in range(0, len(padded), FRAME_SAMPLES)
              if vad.is_speech(padded[i:i + FRAME_SAMPLES].tobytes(), RATE)]
    extra = max(0, (voiced[-1] + FRAME_SAMPLES - retained) / RATE) if voiced else 0
    rms = [float(np.sqrt(np.mean(tail[i:i + FRAME_SAMPLES] ** 2)))
           for i in range(0, len(tail), FRAME_SAMPLES) if len(tail[i:i + FRAME_SAMPLES])]
    tail_db = 20 * np.log10(max(max(rms, default=0), 1e-12))
    reasons = []
    if len(tail) / RATE < .080:
        reasons.append("speech_near_source_end_possible_truncation")
    if extra > .060001:
        reasons.append("permissive_vad_detects_discarded_speech")
    if tail_db > -30:
        reasons.append("energetic_discarded_tail_needs_listening")
    return dict(status="excluded_unresolved" if reasons else "passed_signal_checks",
                method="mode-0 VAD cross-check, discarded-tail energy, source-end margin; not human listening",
                source_end_margin_ms=len(tail) / RATE * 1000,
                mode0_later_speech_ms=extra * 1000, discarded_tail_peak_frame_rms_dbfs=float(tail_db),
                reasons=reasons, human_listening_reviewed=False)


def prepare_expanded(tsv: Path, audio_dir: Path, annotations: Path, out: Path, count=100, seed=42):
    if count < 1:
        raise ValueError("count must be positive")
    records = read_fleurs(tsv, audio_dir)
    labels = json.loads(annotations.read_text())
    if labels["source_tsv_sha256"] != sha256(tsv):
        raise ValueError("Annotations belong to a different transcript file")
    by_id = {r["source_id"]: r for r in labels["records"]}
    if len(by_id) != len(labels["records"]) or set(by_id) != {r["source_id"] for r in records}:
        raise ValueError("Annotation review must cover every distinct source sentence exactly once")
    for row in records:
        label = by_id[row["source_id"]]
        if hashlib.sha256(row["reference"].encode()).hexdigest() != label["reference_sha256"]:
            raise ValueError("Reference changed since annotation review")
        if label["review_status"] != "agent_reviewed_reference_text" or label["entities"] is None:
            raise ValueError("Unreviewed annotation record")
        end = 0
        for e in label["entities"]:
            if (e["type"] not in labels["policy"]["categories"] or e["start"] < end
                    or not e["start"] < e["end"] <= len(row["reference"])
                    or row["reference"][e["start"]:e["end"]] != e["text"]):
                raise ValueError(f"Invalid/overlapping entity span: {row['source_id']}")
            end = e["end"]
    ordered = sorted(records, key=lambda r: hashlib.sha256(f"{seed}:{r['filename']}".encode()).hexdigest())
    selected, seen, rejected, cache = [], set(), [], {}

    def candidate(row):
        name = row["filename"]
        if name not in cache:
            try:
                pcm, details = prepare_audio(audio_dir / name)
                review = boundary_review(audio_dir / name, details)
                cache[name] = (pcm, details, review)
            except ValueError as e:
                rejected.append({"filename": name, "source_id": row["source_id"], "reasons": [str(e)]})
                cache[name] = None
            if cache[name] and cache[name][2]["reasons"]:
                rejected.append({"filename": name, "source_id": row["source_id"], **cache[name][2]})
        item = cache[name]
        return item if item and not item[2]["reasons"] else None

    for condition in ("public_anchor", "public_entities"):
        for row in ordered:
            if row["source_id"] in seen:
                continue
            if condition == "public_entities" and not by_id[row["source_id"]]["entities"]:
                continue
            item = candidate(row)
            if item is None:
                continue
            selected.append((row, condition, item))
            seen.add(row["source_id"])
            if condition == "public_anchor" and len(selected) == count:
                break
        if condition == "public_anchor" and len(selected) != count:
            raise ValueError("Not enough usable distinct source sentences for public anchor")
    out.mkdir(parents=True, exist_ok=False)
    (out / "audio").mkdir()
    clips = []
    for row, condition, (pcm, details, review) in selected:
        target = out / "audio" / row["filename"]
        sf.write(target, pcm, RATE, subtype="PCM_16")
        label = by_id[row["source_id"]]
        clips.append({**row, **details, "clip_id": "fleurs-en_us-" + Path(row["filename"]).stem,
                      "condition": condition, "audio": "audio/" + row["filename"],
                      "source_sha256": sha256(audio_dir / row["filename"]), "audio_sha256": sha256(target),
                      "entities": label["entities"], "entity_review_status": label["review_status"],
                      "boundary_review": review})
    manifest = dict(schema_version=2, dataset="google/fleurs", language="en_us", split="test",
                    source_url="https://huggingface.co/datasets/google/fleurs", license="CC-BY-4.0",
                    source_revision="local user-supplied files; content hashes recorded",
                    source_tsv_sha256=sha256(tsv), annotations_sha256=sha256(annotations),
                    annotation_reviewer=labels["reviewer"], entity_policy=labels["policy"],
                    selection=dict(seed=seed, count=count, algorithm="sha256(seed:filename), unique source_id, deterministic replacements after boundary exclusions"),
                    preprocessing=dict(sample_rate=RATE, encoding="linear16", channels=1, frame_ms=20,
                                       trailing_silence_ms=1000, vad="webrtcvad-wheels==2.0.14", vad_mode=1),
                    exclusions=rejected, clips=clips)
    write_json(out / "manifest.json", manifest)
    (out / "annotations.json").write_bytes(annotations.read_bytes())
    for condition in ("public_anchor", "public_entities"):
        group = [c for c in clips if c["condition"] == condition]
        print(condition, "clips:", len(group), "minutes:", round(sum(c["submitted_seconds"] for c in group) / 60, 2),
              "entity counts:", dict(Counter(e["type"] for c in group for e in c["entities"])))
    print("Excluded candidate recordings:", len(rejected))
    return out / "manifest.json"
