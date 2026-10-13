"""Condition variants: a clean authored dataset -> the same conversations under
named degradations, with the labels unchanged.

Every conversation of the clean dataset is decoded, each channel is passed
through one of the service bench's seeded, length-preserving transforms
(``service.transforms.TRANSFORMS``, via ``build.apply_transform``), and the
result is written as a new conversation whose ground truth, annotation tracks
and planted gold are copied verbatim from the clean one — only the audio and
the ``condition`` differ. Both channels get the same transform (a phone line,
a noisy room, a bad cell link affect what the model hears of either speaker).

Conditions and their service-bench transforms (``CONDITIONS``)::

    clean       the master as rendered                         id suffix (none)
    telephone   8 kHz G.711 mu-law round trip (narrowband)     1
    noise-20db  pink noise at 20 dB SNR against the speech      2
    noise-10db  pink noise at 10 dB SNR                         3
    reverb      far-field room / speakerphone, 0.4 s decay      4
    dropouts    poor cell: 60 ms silent holes every 500 ms      5

Conversation ids: a variant's id is the clean id followed by the condition's
suffix digit (``9011`` -> ``90111`` telephone, ``90112`` noise-20db, ...), so
every clean id (4 digits: series, template, pairing) stays as it is and every
variant is unique, numeric and sorts after the clean set. The combined dataset
holds the clean rows (copied byte for byte) and every variant.

The transforms are deterministic: the noise bed and the reverb tail are
seeded once per transform (``service.transforms``), the dropout holes sit at
fixed offsets, and the telephone leg is a fixed filter chain, so the degraded
audio can be regenerated from the clean master and checked. Seeds are the
transform's own, recorded in the manifest with the transform names.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

from turn_bench.author import service_bridge
from turn_bench.author.build import apply_transform
from turn_bench.author.package import DatasetWriter, flac_bytes
from turn_bench.data import load_local_dataset, local_split, locate_rows, read_row

CONDITIONS: dict[str, dict[str, str]] = {
    "clean": {"transform": "clean", "suffix": "", "description": "the master as rendered"},
    "telephone": {"transform": "telephone", "suffix": "1", "description": "narrowband phone line: 8 kHz G.711 mu-law round trip"},
    "noise-20db": {"transform": "noise-20db", "suffix": "2", "description": "pink noise at 20 dB SNR against the speech"},
    "noise-10db": {"transform": "noise-10db", "suffix": "3", "description": "pink noise at 10 dB SNR against the speech"},
    "reverb": {"transform": "reverb", "suffix": "4", "description": "far-field / speakerphone: 0.4 s decay, 60% direct energy"},
    "dropouts": {"transform": "dropouts", "suffix": "5", "description": "poor cell link: 60 ms silent holes every 500 ms"},
}
DEFAULT_CONDITIONS = ("telephone", "noise-20db", "noise-10db", "reverb", "dropouts")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def variant_id(conversation_id: str, condition: str) -> str:
    return conversation_id + CONDITIONS[condition]["suffix"]


def degraded_row(location: Any, cid: str, condition: str) -> dict[str, Any]:
    """One conversation's parquet row under ``condition``: the source row read
    from ``location`` with both channels decoded, transformed and re-encoded
    (copied byte for byte for ``clean``). Runs in a worker process when the
    dataset is degraded with several (the reverb transform is a direct
    convolution: seconds per channel)."""
    source = read_row(location)  # one row at a time: audio, tracks, metadata, authored
    if condition == "clean":
        audio = {speaker: source[f"speaker_{speaker}_audio"]["bytes"] for speaker in (1, 2)}
    else:
        audio = {}
        for speaker in (1, 2):
            samples, rate = sf.read(io.BytesIO(source[f"speaker_{speaker}_audio"]["bytes"]), dtype="float32", always_2d=False)
            audio[speaker] = flac_bytes(apply_transform(CONDITIONS[condition]["transform"], samples, rate), rate)
            del samples
    authored = json.loads(source["authored"])
    authored.update({"condition": condition, "transform": CONDITIONS[condition]["transform"], "source_conversation_id": cid})
    new_id = variant_id(cid, condition)
    return {
        **{k: v for k, v in source.items() if not k.startswith("speaker_") or "annotation" in k},
        "conversation_id": new_id,
        "speaker_1_audio": {"bytes": audio[1], "path": f"{new_id}_speaker_1.flac"},
        "speaker_2_audio": {"bytes": audio[2], "path": f"{new_id}_speaker_2.flac"},
        "authored": json.dumps(authored),
    }


def degrade_dataset(
    src: Path,
    out: Path,
    conditions: tuple[str, ...] | list[str] = DEFAULT_CONDITIONS,
    *,
    include_clean: bool = True,
    name: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Write ``out``: the clean conversations of ``src`` (when ``include_clean``)
    plus one variant per condition, labels copied; returns the manifest.
    ``workers`` > 1 transforms conversations in that many processes; the rows
    are still written in id order, one at a time, in this process."""
    src = Path(src)
    unknown = [c for c in conditions if c not in CONDITIONS or c == "clean"]
    if unknown:
        raise KeyError(f"unknown condition(s) {unknown}; known: {[c for c in CONDITIONS if c != 'clean']}")
    for condition in conditions:
        service_bridge.transforms().get(CONDITIONS[condition]["transform"])  # fail early
    truth = json.loads((src / "ground_truth.json").read_text(encoding="utf-8"))
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    dataset = load_local_dataset(src)
    locations = locate_rows(local_split(src))
    wanted = ["clean", *conditions] if include_clean else list(conditions)
    jobs = sorted(
        ((variant_id(cid, condition), cid, condition) for cid in dataset.index for condition in wanted),
        key=lambda job: int(job[0]),
    )
    writer = DatasetWriter(out, name=name)
    pool = None
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # Fresh (spawned) workers: a forked copy of this process inherits pyarrow's
        # and numpy's thread pools, which is where a forked worker spends its time.
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        rows = _ordered_rows(pool, jobs, locations, window=8 * workers)
    else:
        rows = (degraded_row(locations[cid], cid, condition) for _, cid, condition in jobs)
    try:
        for (new_id, cid, condition), row in zip(jobs, rows):
            record = dict(truth[cid])
            record.update({"condition": condition, "transform": CONDITIONS[condition]["transform"], "source_conversation_id": cid})
            writer.add(row, record, dataset.durations[cid], spec=(src_manifest.get("authored", {}).get("specs") or {}).get(cid))
            log(f"{new_id} <- {cid} {condition}")
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    manifest = writer.finish(
        generated_by={
            "source_dataset": {"dir": str(src), "manifest_sha256": _sha256(src / "manifest.json")},
            "conditions": {c: CONDITIONS[c] for c in wanted},
        },
        authored={
            **{k: v for k, v in src_manifest.get("authored", {}).items() if k != "specs"},
            "transform": None,
            "conditions": wanted,
            "condition_ids": "clean id + suffix digit per condition (see turn_bench.author.degrade.CONDITIONS)",
        },
    )
    log(f"wrote {out}: {manifest['conversations']['count']} conversations ({len(wanted)} conditions), {manifest['total_bytes'] / 1e6:.1f} MB")
    return manifest


def _ordered_rows(pool: Any, jobs: list[tuple[str, str, str]], locations: dict[str, Any], *, window: int):
    """``degraded_row`` results in job order with at most ``window`` jobs in
    flight: a plain ``pool.map`` submits everything at once and buffers every
    finished row (megabytes each) until the slow ones ahead of it in id order
    (reverb) are done, which is a memory blow-up on a large dataset."""
    from collections import deque

    pending = deque()
    it = iter(jobs)
    for _, cid, condition in it:
        pending.append(pool.submit(degraded_row, locations[cid], cid, condition))
        if len(pending) >= window:
            break
    while pending:
        yield pending.popleft().result()
        for _, cid, condition in it:
            pending.append(pool.submit(degraded_row, locations[cid], cid, condition))
            break


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
