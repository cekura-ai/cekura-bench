"""Freeze local FLEURS audio before contacting any provider."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad

RATE = 16000
FRAME_SAMPLES = 320
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_SECONDS = 0.020
SILENCE_FRAMES = 50


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    import os
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_fleurs(tsv: Path, audio_dir: Path) -> list[dict]:
    # FLEURS TSV is not CSV-quoted. Quotes inside transcripts are literal text.
    with tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
    records, names = [], set()
    for line, row in enumerate(rows, 1):
        if len(row) != 7:
            raise ValueError(f"{tsv}:{line}: expected 7 columns, got {len(row)}")
        source_id, name, raw, transcription, _, samples, gender = row
        if Path(name).name != name or name in names:
            raise ValueError(f"Invalid or duplicate audio filename on line {line}")
        path = audio_dir / name
        if not path.is_file() or not raw.strip():
            raise ValueError(f"Missing audio or reference on line {line}: {name}")
        info = sf.info(path)
        if info.samplerate != RATE or info.channels != 1 or info.frames != int(samples):
            raise ValueError(f"Unexpected FLEURS audio format or sample count: {path}")
        names.add(name)
        records.append(dict(source_id=source_id, filename=name, reference=raw,
                            dataset_transcription=transcription, gender=gender,
                            source_samples=int(samples)))
    return records


def prepare_audio(path: Path) -> tuple[np.ndarray, dict]:
    audio, rate = sf.read(path, dtype="float64")
    if rate != RATE or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite mono 16 kHz audio: {path}")
    pcm = np.rint(np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2")
    padded = np.pad(pcm, (0, (-len(pcm)) % FRAME_SAMPLES))
    vad = webrtcvad.Vad(1)
    voiced = [i for i in range(len(padded) // FRAME_SAMPLES)
              if vad.is_speech(padded[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].tobytes(), RATE)]
    if not voiced:
        raise ValueError(f"No speech detected; inspect this clip manually: {path}")
    speech_frames = voiced[-1] + 1
    end = speech_frames * FRAME_SAMPLES
    # Preserve leading silence and internal pauses; trim only after the last VAD-positive frame.
    prepared = np.concatenate((padded[:end], np.zeros(RATE, dtype="<i2")))
    return prepared, dict(
        speech_frames=speech_frames, total_frames=speech_frames + SILENCE_FRAMES,
        source_samples=len(pcm), retained_samples=end,
        trimmed_samples=max(0, len(pcm) - end), boundary_padding_samples=max(0, end - len(pcm)),
        speech_end_seconds=end / RATE, submitted_seconds=len(prepared) / RATE,
        appended_silence_samples=RATE, appended_silence_ms=1000,
    )


def prepare(tsv: Path, audio_dir: Path, out: Path, count: int, seed: int) -> Path:
    if count < 1:
        raise ValueError("count must be positive")
    records = read_fleurs(tsv, audio_dir)
    ordered = sorted(records, key=lambda r: hashlib.sha256(f"{seed}:{r['filename']}".encode()).hexdigest())
    selected, seen = [], set()
    for row in ordered:
        # Multiple speakers read the same sentence. Pick at most one recording per source sentence.
        if row["source_id"] in seen:
            continue
        selected.append(row)
        seen.add(row["source_id"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} unique source sentences available")
    out.mkdir(parents=True, exist_ok=False)
    (out / "audio").mkdir()
    clips = []
    for row in selected:
        source = audio_dir / row["filename"]
        audio, details = prepare_audio(source)
        target = out / "audio" / row["filename"]
        sf.write(target, audio, RATE, subtype="PCM_16")
        clips.append({**row, **details, "clip_id": "fleurs-en_us-" + Path(row["filename"]).stem,
                      "condition": "public_anchor", "audio": "audio/" + row["filename"],
                      "source_sha256": sha256(source), "audio_sha256": sha256(target),
                      "entities": None})
    manifest = dict(schema_version=1, dataset="google/fleurs", language="en_us", split="test",
                    source_url="https://huggingface.co/datasets/google/fleurs", license="CC-BY-4.0",
                    source_revision="local user-supplied files; content hashes recorded",
                    source_tsv_sha256=sha256(tsv), available_clips=len(records),
                    selection={"algorithm": "sha256(seed:filename), unique source_id", "seed": seed, "count": count},
                    preprocessing={"sample_rate": RATE, "encoding": "linear16", "channels": 1,
                                   "frame_ms": 20, "trailing_silence_ms": 1000,
                                   "boundary": "last positive WebRTC VAD frame; automatic, needs listening review",
                                   "vad": "webrtcvad-wheels==2.0.14", "vad_mode": 1,
                                   "transcript_provenance": "FLEURS supplied reference text; alignment not manually audited"},
                    clips=clips)
    write_json(out / "manifest.json", manifest)
    return out / "manifest.json"


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") not in (1, 2) or not manifest.get("clips"):
        raise ValueError("Unsupported or empty manifest")
    ids = [c["clip_id"] for c in manifest["clips"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip IDs")
    for clip in manifest["clips"]:
        audio_path = (path.parent / clip["audio"]).resolve()
        if not audio_path.is_relative_to(path.parent.resolve()):
            raise ValueError("Audio must be inside the frozen dataset directory")
        if sha256(audio_path) != clip["audio_sha256"]:
            raise ValueError(f"Frozen audio changed: {clip['clip_id']}")
        pcm, rate = sf.read(audio_path, dtype="int16")
        if (rate != RATE or pcm.ndim != 1 or len(pcm) != clip["total_frames"] * FRAME_SAMPLES
                or clip["total_frames"] != clip["speech_frames"] + SILENCE_FRAMES
                or np.any(pcm[-RATE:])):
            raise ValueError(f"Invalid frame count, format or silence tail: {clip['clip_id']}")
    return manifest
