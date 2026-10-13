"""Memory-bounded access to the TurnBench dataset, on top of Sesame's data module.

Everything that defines the benchmark comes from ``turnbench.data``: the dataset
id and pinned revision, the conversation ids, the committed durations, and the
``Conversation`` type (whose ``audio()`` decodes a channel). Sesame's
``resolve_dataset()`` reads every shard into memory when audio is wanted, which
is the whole ~4 GB dev split; this host has 7 GB. So this module loads the
split with ``skip_audio=True`` (gold columns only, cached) and reads one
conversation's audio bytes at a time straight from the pinned parquet snapshot
(one parquet row group per conversation), wrapping them in Sesame's own
``Conversation`` so the decode path is theirs too.

A *local* dataset (a directory of parquet shards in the same schema — an
authored dataset from ``turn-bench author build``) goes through the same
functions: ``local_split`` names it, ``load_split`` reads its gold columns and
``durations.json`` (falling back to the FLAC headers, which is what Sesame's
loader does for a local directory), and ``locate_rows`` indexes its shards.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download
from turnbench.data import (  # HFDataset: the row wrapper Sesame's own loader uses
    AUDIO_COLUMNS,
    GOLD_COLUMNS,
    HFDataset,
    PINNED_REVISIONS,
    SPEAKERS,
    Conversation,
    Dataset,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from turnbench.durations import SPLITS


@dataclass(frozen=True)
class Split:
    name: str  # "dev" | "test" | a local directory's name
    source: str  # HF dataset repo id, or the local directory
    revision: str | None  # pinned commit when Sesame pins one; None for local
    local_dir: Path | None = None  # set for a local dataset directory
    labelled: bool = True  # gold available, so a run can be scored

    @property
    def kind(self) -> str:
        return "local" if self.local_dir else "huggingface"


def split_for(name: str) -> Split:
    if name not in SPLITS:
        raise KeyError(f"unknown split {name!r}; known: {', '.join(sorted(SPLITS))}")
    source = SPLITS[name]
    return Split(name=name, source=source, revision=PINNED_REVISIONS.get(source), labelled=name == "dev")


def local_split(directory: str | Path) -> Split:
    """A dataset directory of parquet shards in Sesame's schema (``turn-bench author build``)."""
    path = Path(directory).expanduser().resolve()
    if not path.is_dir() or not any(path.glob("*.parquet")):
        raise FileNotFoundError(f"{path} is not a dataset directory (no *.parquet shards)")
    return Split(name=path.name, source=str(path), revision=None, local_dir=path, labelled=True)


def local_shards(directory: Path) -> list[Path]:
    """The shards Sesame's loader would read for this directory (top level, sorted)."""
    return sorted(Path(directory).glob("*.parquet"))


def load_split(split: Split) -> Dataset:
    """The split's gold columns and durations, without audio (Sesame's scorer path)."""
    if split.local_dir is not None:
        return load_local_dataset(split.local_dir)
    return resolve_dataset(source=split.source, revision=split.revision, skip_audio=True)


def load_local_dataset(directory: Path) -> Dataset:
    """Gold columns of a local dataset directory, wrapped exactly as Sesame's
    ``resolve_dataset`` wraps them, plus durations from ``durations.json`` when
    the directory has one, else from each conversation's FLAC header (one row at
    a time, so the audio never sits in memory as a whole)."""
    shards = local_shards(directory)
    table = pa.concat_tables([pq.read_table(str(shard), columns=GOLD_COLUMNS) for shard in shards])
    rows = HFDataset(table)
    index = {row_id: i for i, row_id in enumerate(rows["conversation_id"])}
    durations_path = Path(directory) / "durations.json"
    if durations_path.is_file():
        durations = {str(k): float(v) for k, v in json.loads(durations_path.read_text(encoding="utf-8"))["durations"].items()}
    else:
        durations = {}
        for shard in shards:
            parquet = pq.ParquetFile(str(shard))
            for group in range(parquet.metadata.num_row_groups):
                batch = parquet.read_row_group(group, columns=["conversation_id", AUDIO_COLUMNS[0]])
                for cid, audio in zip(batch["conversation_id"].to_pylist(), batch[AUDIO_COLUMNS[0]].to_pylist()):
                    info = sf.info(io.BytesIO(audio["bytes"]))
                    durations[cid] = info.frames / info.samplerate
    missing = sorted(set(index) - set(durations), key=int)
    if missing:
        raise KeyError(f"{directory}: no duration for conversations {missing}")
    return Dataset(rows=rows, index=index, durations=durations)


def subset(dataset: Dataset, ids: list[str]) -> Dataset:
    """A view of `dataset` restricted to `ids`, so a partial run can be scored and
    coverage-validated against exactly the conversations it ran."""
    missing = [cid for cid in ids if cid not in dataset.index]
    if missing:
        raise KeyError(f"conversation ids not in dataset: {missing}")
    return Dataset(
        rows=dataset.rows,
        index={cid: dataset.index[cid] for cid in ids},
        durations=dataset.durations,
    )


def durations(dataset: Dataset) -> dict[str, float]:
    """{conversation_id: duration_s} via Sesame's `conversation()` (committed
    durations artifact, with their audio-header fallback)."""
    return {cid: conversation(dataset, cid).duration_s for cid in conversation_ids(dataset)}


@dataclass(frozen=True)
class RowLocation:
    """Where one conversation's row lives in the parquet snapshot."""

    path: str
    row_group: int
    row: int


def snapshot_dir(split: Split, patterns: list[str] | str = "*.parquet", *, local_only: bool = False) -> Path:
    """The local Hugging Face snapshot of the split at its pinned revision (the
    same one Sesame's loader reads), holding at least the files matching
    ``patterns``; downloaded on demand unless ``local_only``. A local split is
    its own directory."""
    if split.local_dir is not None:
        return split.local_dir
    return Path(
        snapshot_download(
            split.source,
            repo_type="dataset",
            revision=split.revision,
            allow_patterns=patterns,
            local_files_only=local_only,
        )
    )


def locate_rows(split: Split) -> dict[str, RowLocation]:
    """Map every conversation id to its parquet row group, reading only the id
    column. Uses the same snapshot (and revision) Sesame's loader would."""
    snapshot = snapshot_dir(split)
    shards = local_shards(snapshot) if split.local_dir is not None else sorted(Path(snapshot).rglob("*.parquet"))
    locations: dict[str, RowLocation] = {}
    for path in shards:
        shard = pq.ParquetFile(path)
        for group in range(shard.metadata.num_row_groups):
            ids = shard.read_row_group(group, columns=["conversation_id"])
            for row, cid in enumerate(ids["conversation_id"].to_pylist()):
                locations[cid] = RowLocation(path=str(path), row_group=group, row=row)
    return locations


def read_row(location: RowLocation) -> dict:
    """One conversation's whole parquet row (every column, audio bytes included)
    as a dict: the unit ``author degrade`` and ``author split`` copy between
    datasets, read one row group at a time."""
    table = pq.ParquetFile(location.path).read_row_group(location.row_group)
    return table.slice(location.row, 1).to_pylist()[0]


def load_audio_conversation(
    conversation_id: str, location: RowLocation, duration_s: float
) -> Conversation:
    """One conversation's audio bytes as a Sesame `Conversation` (annotations
    omitted: the runner only streams audio; gold comes from `load_split`)."""
    table = pq.ParquetFile(location.path).read_row_group(
        location.row_group, columns=["conversation_id"] + AUDIO_COLUMNS
    )
    row = table.slice(location.row, 1).to_pylist()[0]
    if row["conversation_id"] != conversation_id:
        raise RuntimeError(
            f"parquet row at {location} is conversation {row['conversation_id']}, "
            f"expected {conversation_id}"
        )
    return Conversation(
        conversation_id=conversation_id,
        duration_s=duration_s,
        annotations={},
        audio_bytes={speaker: row[f"speaker_{speaker}_audio"]["bytes"] for speaker in SPEAKERS},
    )
