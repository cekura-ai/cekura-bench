"""Write an authored dataset as a local directory ``turnbench.data`` loads unchanged.

Layout (``resolve_dataset(<dir>)`` globs ``*.parquet`` at the top level)::

    <out>/
      <name>-00000-of-00001.parquet   one row per conversation, one row group per row
      durations.json                  {"durations": {conversation_id: seconds}}
      ground_truth.json               per conversation: spec tags (family, difficulty, pairing),
                                      placed items, segments, planted gold
      manifest.json                   sha256 of every file above + build provenance

Parquet columns mirror Sesame's released shards: ``conversation_id`` (string),
``speaker_{1,2}_audio`` (struct<bytes, path>: FLAC, 48 kHz mono, 24-bit — the
released set's format), ``speaker_{1,2}_annotation_{a,b,c}`` (list<struct<start_s,
end_s, label, text>>), ``metadata`` (struct<conversation_type,
speaker_{1,2}_actor_id, speaker_{1,2}_actor_gender>). One extra column,
``authored`` (JSON string), records the spec id, family, difficulty, tags,
renderer, transform and the fact that the three tracks are script-derived;
Sesame's loader projects only the columns it knows, so the extra is inert there.

``durations.json`` has the shape of Sesame's ``durations-{split}.json`` so the
harness can score without decoding audio; Sesame's own local-directory path
ignores it and reads the FLAC headers, which agree by construction.

``DatasetWriter`` streams rows into the shard one row group at a time, so a
dataset of hundreds of conversations (or thousands, once every degradation
condition is included) never sits in memory as a whole; ``write_dataset`` is
the build's use of it, and ``author degrade`` / ``author split`` reuse it to
copy rows between datasets.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from turn_bench import __version__, manifest
from turn_bench.author.compose import Timeline
from turn_bench.author.labels import Annotation, Planted
from turn_bench.author.render import VOICES, voice_registry

FLAC_SUBTYPE = "PCM_24"
CONVERSATION_TYPE = "Authored/Scripted"
ANNOTATION_PROVENANCE = "script-derived; tracks a, b and c are identical"

ANNOTATION_TYPE = pa.list_(
    pa.struct([("start_s", pa.float64()), ("end_s", pa.float64()), ("label", pa.string()), ("text", pa.string())])
)
AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
METADATA_TYPE = pa.struct(
    [
        ("conversation_type", pa.string()),
        ("speaker_1_actor_id", pa.string()),
        ("speaker_2_actor_id", pa.string()),
        ("speaker_1_actor_gender", pa.string()),
        ("speaker_2_actor_gender", pa.string()),
    ]
)
SCHEMA = pa.schema(
    [
        ("conversation_id", pa.string()),
        ("speaker_1_audio", AUDIO_TYPE),
        ("speaker_2_audio", AUDIO_TYPE),
        ("metadata", METADATA_TYPE),
        *[(f"speaker_{speaker}_annotation_{annotator}", ANNOTATION_TYPE) for speaker in (1, 2) for annotator in "abc"],
        ("authored", pa.string()),
    ]
)


@dataclass
class BuiltConversation:
    timeline: Timeline
    tracks: dict[int, list[Annotation]]
    planted: Planted
    channels: dict[int, np.ndarray]  # after the transform; emptied by `encode_audio`
    transform: str
    renderer: dict[str, Any]
    pauses: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # item id -> scripted vs detected breaks
    flac: dict[int, bytes] = field(default_factory=dict)  # per channel, once encoded


def flac_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(samples, dtype=np.float32), sample_rate, format="FLAC", subtype=FLAC_SUBTYPE)
    return buffer.getvalue()


def encode_audio(built: BuiltConversation) -> None:
    """Encode both channels to FLAC now and drop the float audio (the channels
    and every placed item's clip), so a build of many conversations holds
    compressed bytes rather than a minute of 48 kHz float per channel each."""
    rate = built.timeline.sample_rate
    for channel in (1, 2):
        if channel not in built.flac:
            built.flac[channel] = flac_bytes(built.channels[channel], rate)
    built.channels = {}
    built.timeline.channels = {}
    for placed in built.timeline.items:
        placed.audio = np.zeros(0, dtype=np.float32)


def conversation_row(built: BuiltConversation) -> dict[str, Any]:
    spec = built.timeline.spec
    cid = spec.conversation_id
    rate = built.timeline.sample_rate
    actor = {role: f"{built.renderer['name']}:{label}" for role, label in spec.voices.items()}
    gender = {role: VOICES[label].gender for role, label in spec.voices.items()}
    authored = {
        "spec_id": spec.id,
        "title": spec.title,
        "family": spec.family,
        "subtype": spec.subtype,
        "difficulty": spec.difficulty,
        "tags": spec.tags,
        "voices": spec.voices,
        "pairing": spec.pairing,
        "renderer": built.renderer,
        "transform": built.transform,
        "condition": built.transform,
        "annotation_provenance": ANNOTATION_PROVENANCE,
        "spec_sha256": spec.source_sha256,
    }
    encode_audio(built)
    row: dict[str, Any] = {
        "conversation_id": cid,
        "speaker_1_audio": {"bytes": built.flac[1], "path": f"{cid}_speaker_1.flac"},
        "speaker_2_audio": {"bytes": built.flac[2], "path": f"{cid}_speaker_2.flac"},
        "metadata": {
            "conversation_type": CONVERSATION_TYPE,
            "speaker_1_actor_id": actor["caller"],
            "speaker_2_actor_id": actor["agent"],
            "speaker_1_actor_gender": gender["caller"],
            "speaker_2_actor_gender": gender["agent"],
        },
        "authored": json.dumps(authored),
    }
    for speaker in (1, 2):
        for annotator in "abc":
            row[f"speaker_{speaker}_annotation_{annotator}"] = list(built.tracks[speaker])
    return row


def planted_json(planted: Planted) -> dict[str, Any]:
    return {
        "eot_positive": [{"speaker": e.speaker, "time_s": e.time_s} for e in planted.eot_positive],
        "eot_negative": [{"speaker": e.speaker, "start_s": e.start, "end_s": e.end} for e in planted.eot_negative],
        "int_positive": [{"speaker": e.speaker, "time_s": e.time_s} for e in planted.int_positive],
        "int_negative": [{"speaker": e.speaker, "start_s": e.start, "end_s": e.end} for e in planted.int_negative],
        "counts": {"all": planted.counts(), "speaker_1": planted.counts(1), "speaker_2": planted.counts(2)},
    }


def truth_record(b: BuiltConversation) -> dict[str, Any]:
    """One conversation's ``ground_truth.json`` record."""
    spec = b.timeline.spec
    return {
        "spec_id": spec.id,
        "title": spec.title,
        "family": spec.family,
        "subtype": spec.subtype,
        "difficulty": spec.difficulty,
        "tags": spec.tags,
        "voices": spec.voices,
        "pairing": spec.pairing,
        "transform": b.transform,
        "condition": b.transform,
        "renderer": b.renderer,
        **b.timeline.ground_truth(),
        "annotations": {str(channel): b.tracks[channel] for channel in (1, 2)},
        "pauses": b.pauses,
        "planted": planted_json(b.planted),
    }


class DatasetWriter:
    """Stream conversations into a dataset directory (see the module docstring).

    ``add`` takes one parquet row (``conversation_row`` shape), its ground-truth
    record and its duration; rows must arrive in ascending numeric id order
    (Sesame sorts ids with ``int``) and ids must be unique. ``finish`` writes the
    sidecars and the manifest and returns the manifest."""

    def __init__(self, out: Path, *, name: str | None = None) -> None:
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.name = name or self.out.name
        self.shard = self.out / f"{self.name}-00000-of-00001.parquet"
        self._writer = pq.ParquetWriter(self.shard, SCHEMA)
        self.ids: list[str] = []
        self.truth: dict[str, dict[str, Any]] = {}
        self.durations: dict[str, float] = {}
        self.specs: dict[str, dict[str, Any]] = {}

    def add(self, row: dict[str, Any], truth: dict[str, Any], duration_s: float, *, spec: dict[str, Any] | None = None) -> None:
        cid = row["conversation_id"]
        if cid in self.truth:
            raise ValueError(f"duplicate conversation_id {cid!r}")
        if self.ids and int(cid) <= int(self.ids[-1]):
            raise ValueError(f"conversation_id {cid!r} out of order after {self.ids[-1]!r}")
        self._writer.write_table(pa.Table.from_pylist([row], schema=SCHEMA))
        self.ids.append(cid)
        self.truth[cid] = truth
        self.durations[cid] = duration_s
        if spec is not None:
            self.specs[cid] = spec

    def finish(self, *, generated_by: dict[str, Any] | None = None, authored: dict[str, Any] | None = None) -> dict[str, Any]:
        self._writer.close()
        if not self.ids:
            raise ValueError(f"no conversations written to {self.out}")
        (self.out / "durations.json").write_text(
            json.dumps({"generated_at": _now(), "source": str(self.out), "durations": self.durations}, indent=2) + "\n", encoding="utf-8"
        )
        (self.out / "ground_truth.json").write_text(json.dumps(self.truth, indent=2) + "\n", encoding="utf-8")
        built_manifest = manifest.build_local_manifest(
            name=self.name,
            dataset_dir=self.out,
            conversation_ids=list(self.ids),
            generated_by={"harness": f"turn-bench {__version__}", **(generated_by or {})},
            authored={**(authored or {}), "specs": self.specs},
        )
        manifest.write_manifest(built_manifest, self.out / "manifest.json")
        return built_manifest


def write_dataset(out: Path, built: list[BuiltConversation], *, name: str | None = None, generated_by: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the directory described in the module docstring; returns the manifest."""
    ordered = sorted(built, key=lambda b: int(b.timeline.spec.conversation_id))
    writer = DatasetWriter(out, name=name)
    for b in ordered:
        spec = b.timeline.spec
        writer.add(
            conversation_row(b),
            truth_record(b),
            b.timeline.duration_s,
            spec={"id": spec.id, "file": spec.source_path, "sha256": spec.source_sha256},
        )
    return writer.finish(
        generated_by=generated_by,
        authored={
            "renderer": ordered[0].renderer if ordered else None,
            "transform": ordered[0].transform if ordered else None,
            "conditions": sorted({b.transform for b in ordered}),
            "sample_rate": ordered[0].timeline.sample_rate if ordered else None,
            "audio_format": f"FLAC {FLAC_SUBTYPE}",
            "annotation_provenance": ANNOTATION_PROVENANCE,
            "voices": voice_registry(ordered[0].renderer.get("model", "") if ordered else ""),
        },
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
