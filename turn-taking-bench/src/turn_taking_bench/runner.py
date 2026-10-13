"""Stream conversations through a model, enforce causality, build the submission.

``stream_speaker`` is the whole contract between the harness and a model: fixed
20 ms frames in order, audio time announced with each frame, every emitted
event checked against the audio heard so far. ``run_benchmark`` fans
conversations out over worker processes (one conversation at a time each),
writes the predictions in Sesame's format plus the raw events and a provenance
record, and scores the predictions with Sesame's scorer.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from turnbench.data import SPEAKERS, Conversation, conversation_ids
from turnbench.submission import (
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)

from turn_taking_bench import __version__
from turn_taking_bench.adapters import build_model, is_live, model_provenance, resolve_config
from turn_taking_bench.data import (
    RowLocation,
    Split,
    durations,
    load_audio_conversation,
    load_split,
    local_split,
    locate_rows,
    snapshot_dir,
    split_for,
    subset,
)
from turn_taking_bench.manifest import find_manifest, load_manifest, verification_record, verify_snapshot
from turn_taking_bench.model import EVENT_KINDS, FRAME_S, Event, TurnModel
from turn_taking_bench.report import authored_groups, score_report, write_report

HERE = Path(__file__).resolve().parent


class CausalityError(RuntimeError):
    """A model committed an event at a time past the audio it has heard."""


class EventOrderError(ValueError):
    """A model's events of one kind went backwards in time."""


class RealtimePacer:
    """Sleep so that audio time never runs ahead of wall-clock time: frame
    ``i`` is pushed no earlier than ``start + (i + 1) * FRAME_S``."""

    def __init__(self) -> None:
        self.start = time.monotonic()

    def wait_until(self, audio_time_s: float) -> None:
        delay = self.start + audio_time_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)


@dataclass
class SpeakerRun:
    """What one channel produced: raw events in emission order, plus the
    stream geometry needed to interpret them."""

    events: list[Event]
    sample_rate: int
    frames: int
    audio_time_s: float  # audio pushed, = frames * FRAME_S
    dropped_tail_samples: int  # samples past the last whole frame (never pushed)
    diagnostics: dict[str, Any] | None = None  # model-reported extras (e.g. per-frame scores)


def check_causal(events: list[Event], t_end_s: float) -> None:
    for event in events:
        if event.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {event.kind!r}")
        if not np.isfinite(event.commit_time_s) or event.commit_time_s < 0.0:
            raise ValueError(f"commit time must be finite and >= 0, got {event.commit_time_s}")
        if event.commit_time_s > t_end_s:
            raise CausalityError(
                f"{event.kind} committed at {event.commit_time_s:.4f}s but only "
                f"{t_end_s:.4f}s of audio has been heard"
            )


def stream_speaker(
    model: TurnModel, samples: np.ndarray, sample_rate: int, *, realtime: bool = False
) -> SpeakerRun:
    """Feed one channel to `model` as whole 20 ms frames, in order.

    The frame end time is ``(i + 1) * FRAME_S``, the same float expression the
    reference baseline uses for its commit times. Samples past the last whole
    frame are dropped (the reference does the same). With ``realtime`` the push
    cadence follows wall-clock time, as a live provider would see it.
    """
    samples_per_frame = int(FRAME_S * sample_rate)
    frame_count = len(samples) // samples_per_frame
    model.reset(sample_rate)
    pacer = RealtimePacer() if realtime else None
    events: list[Event] = []
    t_end_s = 0.0
    for index in range(frame_count):
        t_end_s = (index + 1) * FRAME_S
        if pacer is not None:
            pacer.wait_until(t_end_s)
        emitted = list(model.push(samples[index * samples_per_frame : (index + 1) * samples_per_frame], t_end_s))
        check_causal(emitted, t_end_s)
        events.extend(emitted)
    emitted = list(model.flush(t_end_s))
    check_causal(emitted, t_end_s)
    events.extend(emitted)
    # Optional: a model may report JSON-serialisable diagnostics (for Smart Turn,
    # its 12.5 Hz score array) that events.jsonl keeps next to the events.
    diagnostics = model.diagnostics() if hasattr(model, "diagnostics") else None
    return SpeakerRun(
        events=events,
        sample_rate=sample_rate,
        frames=frame_count,
        audio_time_s=t_end_s,
        dropped_tail_samples=len(samples) - frame_count * samples_per_frame,
        diagnostics=diagnostics,
    )


def speaker_events(events: list[Event]) -> tuple[SpeakerEvents, dict[str, int]]:
    """Raw events -> Sesame's per-speaker lists: strictly increasing per kind.

    Exact repeats of a time (two decisions committed on the same frame) are
    collapsed and counted; a time earlier than the previous one of the same
    kind is a model bug and raises."""
    times: dict[str, list[float]] = {kind: [] for kind in EVENT_KINDS}
    deduped: dict[str, int] = {kind: 0 for kind in EVENT_KINDS}
    for event in events:
        kind_times = times[event.kind]
        if kind_times:
            if event.commit_time_s == kind_times[-1]:
                deduped[event.kind] += 1
                continue
            if event.commit_time_s < kind_times[-1]:
                raise EventOrderError(
                    f"{event.kind} at {event.commit_time_s} after {kind_times[-1]}"
                )
        kind_times.append(event.commit_time_s)
    return SpeakerEvents(eot=times["eot"], interruption=times["interruption"]), deduped


# ---- per-conversation worker --------------------------------------------------


@dataclass(frozen=True)
class ConversationJob:
    conversation_id: str
    model_name: str
    model_config: dict[str, Any]
    location: RowLocation
    duration_s: float
    realtime: bool
    # Where a model that records per-conversation artifacts (an S2S session's
    # audio, timelines and provider frames) puts them: <run>/artifacts.
    artifacts_dir: Path | None = None


def skipped_run(samples: np.ndarray, sample_rate: int, reason: str) -> SpeakerRun:
    """The ``SpeakerRun`` of a channel the model declined to stream: no events,
    the stream geometry it would have had, and the reason in its diagnostics."""
    samples_per_frame = int(FRAME_S * sample_rate)
    frame_count = len(samples) // samples_per_frame
    return SpeakerRun(
        events=[],
        sample_rate=sample_rate,
        frames=frame_count,
        audio_time_s=frame_count * FRAME_S,
        dropped_tail_samples=len(samples) - frame_count * samples_per_frame,
        diagnostics={"skipped": reason},
    )


@dataclass
class ConversationResult:
    conversation_id: str
    wall_s: float
    speakers: dict[int, SpeakerRun] = field(default_factory=dict)
    deduped: dict[int, dict[str, int]] = field(default_factory=dict)
    prediction: ConversationPrediction | None = None
    error: str | None = None


def run_conversation(job: ConversationJob) -> ConversationResult:
    """Stream both channels of one conversation through a fresh model instance."""
    started = time.monotonic()
    result = ConversationResult(conversation_id=job.conversation_id, wall_s=0.0)
    try:
        conv: Conversation = load_audio_conversation(job.conversation_id, job.location, job.duration_s)
        model = build_model(job.model_name, job.model_config)
        per_speaker: dict[int, SpeakerEvents] = {}
        # Optional hook: a model that needs to know which channel is coming
        # (an S2S model opens a provider session for the caller channel and
        # replaces the agent channel) is told before ``reset``; returning False
        # skips the channel, which then contributes no events.
        begin_channel = getattr(model, "begin_channel", None)
        for speaker in SPEAKERS:
            samples, sample_rate = conv.audio(speaker)
            if begin_channel is not None and not begin_channel(job.conversation_id, speaker, job.artifacts_dir):
                run = skipped_run(samples, sample_rate, f"speaker_{speaker} not streamed by {job.model_name}")
            else:
                run = stream_speaker(model, samples, sample_rate, realtime=job.realtime)
            result.speakers[speaker] = run
            per_speaker[speaker], result.deduped[speaker] = speaker_events(run.events)
            del samples
        result.prediction = ConversationPrediction(
            conversation_id=job.conversation_id,
            speaker_1=per_speaker[1],
            speaker_2=per_speaker[2],
        )
    except Exception as exc:  # recorded per conversation; the run continues
        result.error = f"{type(exc).__name__}: {exc}"
    result.wall_s = time.monotonic() - started
    return result


# ---- provenance ---------------------------------------------------------------


def harness_git_state() -> dict[str, Any]:
    """Commit and dirty flag of the checkout this harness runs from."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(HERE.parent.parent)],
            cwd=HERE, capture_output=True, text=True, check=True,
        ).stdout
        return {"git_commit": commit, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def turnbench_provenance() -> dict[str, Any]:
    """The installed scorer: version and, for a git install, the exact commit."""
    info: dict[str, Any] = {"version": metadata.version("turnbench"), "url": None, "commit": None}
    direct = metadata.distribution("turnbench").read_text("direct_url.json")
    if direct:
        parsed = json.loads(direct)
        info["url"] = parsed.get("url")
        info["commit"] = (parsed.get("vcs_info") or {}).get("commit_id")
    return info


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- the run ------------------------------------------------------------------


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_benchmark(
    *,
    model_name: str,
    split_name: str = "dev",
    workers: int = 2,
    limit: int | None = None,
    out: Path,
    realtime: bool = False,
    config_overrides: dict[str, Any] | None = None,
    dataset_dir: Path | None = None,
    ids: list[str] | None = None,
) -> dict[str, Any]:
    """Stream every conversation of the split through the model, then write
    predictions, raw events, provenance and (for a labelled split) the scores.

    ``dataset_dir`` runs a local dataset directory (``turn-taking-bench author build``)
    instead of the Hugging Face split; everything else is identical. ``ids``
    restricts the run to those conversations, in the split's order (scored as a
    subset, like ``limit``)."""
    started_at, wall_start = utc_now(), time.monotonic()
    model_config = resolve_config(model_name, config_overrides)
    if is_live(model_name) and workers != 1:
        log(f"{model_name} opens one provider session at a time: workers={workers} -> 1")
        workers = 1
    split: Split = local_split(dataset_dir) if dataset_dir is not None else split_for(split_name)
    manifest_info = verify_split_manifest(split)
    dataset = load_split(split)
    all_ids = conversation_ids(dataset)
    if ids is not None:
        missing = [cid for cid in ids if cid not in set(all_ids)]
        if missing:
            raise KeyError(f"conversation ids not in {split.name}: {missing}")
        wanted = set(ids)
        all_ids = [cid for cid in all_ids if cid in wanted]
    if limit is not None:
        all_ids = all_ids[:limit]
    ids = all_ids
    duration_by_id = durations(dataset)
    locations = locate_rows(split)
    log(f"turn-taking-bench run: model={model_name} split={split.name} conversations={len(ids)} workers={workers}")

    artifacts_dir = out / "artifacts" if is_live(model_name) else None
    jobs = [
        ConversationJob(
            conversation_id=cid,
            model_name=model_name,
            model_config=model_config,
            location=locations[cid],
            duration_s=duration_by_id[cid],
            realtime=realtime,
            artifacts_dir=artifacts_dir,
        )
        for cid in ids
    ]
    results: dict[str, ConversationResult] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(run_conversation, job): job.conversation_id for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[result.conversation_id] = result
            status = "failed: " + result.error if result.error else "ok"
            log(f"[{done}/{len(jobs)}] conversation {result.conversation_id} {result.wall_s:.1f}s {status}")

    out.mkdir(parents=True, exist_ok=True)
    ordered = [results[cid] for cid in ids]
    succeeded = [r for r in ordered if r.prediction is not None]
    failures = [{"conversation_id": r.conversation_id, "error": r.error} for r in ordered if r.error]

    submission = Submission(
        schema_version=SCHEMA_VERSION, predictions=[r.prediction for r in succeeded]
    )
    predictions_path = out / f"predictions-{split.name}.json"
    predictions_path.write_text(submission.model_dump_json(indent=2), encoding="utf-8")
    write_events(out / "events.jsonl", ordered)

    run_info: dict[str, Any] = {
        "run_id": out.name,
        "model": {"name": model_name, "config": model_config, "provenance": model_provenance(model_name, model_config)},
        "harness": {"name": "turn-taking-bench", "version": __version__, **harness_git_state()},
        "turnbench": turnbench_provenance(),
        "dataset": {"kind": split.kind, "split": split.name, "source": split.source, "revision": split.revision, "manifest": manifest_info},
        "frame_s": FRAME_S,
        "realtime": realtime,
        "workers": workers,
        "limit": limit,
        "ids": ids,
        "python": sys.version.split()[0],
        "started_at": started_at,
        "conversations": [conversation_record(r) for r in ordered],
        "failures": failures,
        "files": {"predictions": predictions_path.name, "events": "events.jsonl"},
    }

    scored_ids = [r.conversation_id for r in succeeded]
    if split.labelled and scored_ids:
        groups = authored_groups(split.local_dir) if split.local_dir is not None else None
        report = score_report(submission, subset(dataset, scored_ids), groups)
        report["partial"] = len(scored_ids) < len(conversation_ids(dataset))
        report["conversations_in_split"] = len(conversation_ids(dataset))
        write_report(report, out, run_info)
        run_info["files"].update({"report": "report.json", "report_markdown": "report.md"})
        run_info["scores"] = report["aggregate"]
    elif scored_ids:
        log(f"split {split.name!r} has no public labels; predictions written, not scored")

    run_info["finished_at"] = utc_now()
    run_info["wall_s"] = round(time.monotonic() - wall_start, 3)
    (out / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    log(f"wrote {out}/ ({len(succeeded)} ok, {len(failures)} failed, {run_info['wall_s']:.0f}s)")
    return run_info


def verify_split_manifest(split: Split) -> dict[str, Any] | None:
    """Before any audio is streamed: the local snapshot must match the committed
    manifest for this dataset (``datasets/*/manifest.json``, or ``manifest.json``
    inside a local dataset directory). A mismatch raises ``ManifestError``; a
    dataset with no manifest (the test split) records None."""
    if split.local_dir is not None:
        path = split.local_dir / "manifest.json"
        if not path.is_file():
            log(f"no manifest.json in {split.local_dir}: dataset not verified")
            return None
        verification = verify_snapshot(load_manifest(path), split.local_dir, log=log)
        log(f"manifest for local dataset {split.name} verified: {verification.files} files, {verification.bytes / 1e6:.1f} MB")
        return verification_record(path, verification)
    found = find_manifest(split.source, split.revision)
    if found is None:
        log(f"no manifest for {split.source}: snapshot not verified")
        return None
    path, manifest = found
    snapshot = snapshot_dir(split, [entry["path"] for entry in manifest["files"]])
    verification = verify_snapshot(manifest, snapshot, log=log)
    log(
        f"manifest {path.name} for {split.source} @ {split.revision[:8]} verified: "
        f"{verification.files} files, {verification.bytes / 1e9:.2f} GB, "
        f"{verification.hashed} hashed now, {verification.seconds:.1f}s"
    )
    return verification_record(path, verification)


def conversation_record(result: ConversationResult) -> dict[str, Any]:
    record: dict[str, Any] = {
        "conversation_id": result.conversation_id,
        "status": "failed" if result.error else "ok",
        "wall_s": round(result.wall_s, 3),
        "error": result.error,
        "speakers": {},
    }
    for speaker, run in result.speakers.items():
        counts = {kind: sum(1 for e in run.events if e.kind == kind) for kind in EVENT_KINDS}
        record["speakers"][str(speaker)] = {
            "sample_rate": run.sample_rate,
            "frames": run.frames,
            "audio_time_s": run.audio_time_s,
            "dropped_tail_samples": run.dropped_tail_samples,
            "raw_events": counts,
            "deduped": result.deduped.get(speaker, {}),
        }
    return record


def write_events(path: Path, results: list[ConversationResult]) -> None:
    """One line per conversation and speaker: the raw events in emission order,
    before deduplication, so the submission can be rebuilt or audited."""
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            for speaker, run in sorted(result.speakers.items()):
                line = {
                    "conversation_id": result.conversation_id,
                    "speaker": speaker,
                    "sample_rate": run.sample_rate,
                    "audio_time_s": run.audio_time_s,
                    "events": [asdict(event) for event in run.events],
                }
                if run.diagnostics is not None:
                    line["diagnostics"] = run.diagnostics
                handle.write(json.dumps(line) + "\n")
