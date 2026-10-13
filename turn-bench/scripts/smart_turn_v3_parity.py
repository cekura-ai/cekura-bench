"""Parity checks for the Smart Turn v3 adapter against Sesame's reference.

``reference``: run Sesame's own ``baselines/smart_turn_v3/predict.py`` pipeline
(its per-conversation function, not its whole-dataset loader) on the first N
dev conversations in this environment, stream the same conversations through
the adapter, and compare events (``turn-bench parity``) and the 12.5 Hz score
arrays frame by frame. Needs a TurnBench checkout whose
``baselines/smart_turn_v3/smart_turn`` is set up as its README says (the
smart-turn clone with ``smart-turn-v3.1.onnx`` and ``silero_vad.onnx``).

    uv run --locked python scripts/smart_turn_v3_parity.py reference \
        --turnbench ~/Dev/turnbench --limit 3 --out runs/smart_turn_v3-ref-parity

``probs``: compare a finished run's per-frame scores (``events.jsonl``
diagnostics) with a committed TurnBench probs file, to attribute event
differences to the VAD/segmentation (a frame scored on one side only) or to
the classifier (both sides scored, values differ), and to measure the effect of
the rounding in the committed file.

    uv run --locked python scripts/smart_turn_v3_parity.py probs \
        --run runs/smart_turn_v3-dev --probs-eot path/to/probs-eot.json \
        --reference path/to/predictions-dev.json

``commit``: the decision logic alone, no inference. Apply the adapter's
``EdgeCommitter`` (theta_eot, theta_int, 12.5 Hz) frame by frame to Sesame's
published per-frame probabilities (``probs-eot.json`` / ``probs-int.json``,
fetched from the pinned ``freemanjiang/turnbench-baseline-probs`` snapshot
unless ``--probs-dir`` points at local copies) and check that every event
list of their committed ``predictions-dev.json`` is reproduced exactly.

    uv run --locked python scripts/smart_turn_v3_parity.py commit \
        --reference ~/Dev/turnbench/baselines/smart_turn_v3/predictions-dev.json
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Sesame's published per-frame probabilities for the smart_turn_v3 baseline
# (turnbench/probs.py fetches the same files into its baselines/ directory).
PROBS_REPO = "freemanjiang/turnbench-baseline-probs"
PROBS_REVISION = "e3cd4caa2a45b85ef4d6714d79d5c7c40471ae79"
PROBS_SUBDIR = "smart_turn_v3"


def _stub_module(name: str, **attrs) -> None:
    """record_and_predict.py imports scipy.io.wavfile for a debug path that never
    runs here; keep the smart-turn extra free of scipy by stubbing that import."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)  # transformers probes it with find_spec
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def load_reference_module(turnbench_checkout: Path):
    sys.path.insert(0, str(turnbench_checkout))
    _stub_module("scipy")
    _stub_module("scipy.io", wavfile=None)
    sys.modules["scipy"].io = sys.modules["scipy.io"]
    import baselines.smart_turn_v3.predict as reference  # chdirs into its smart_turn clone

    return reference


def patch_vad_reset_to_audio_time(reference, reset_s: float) -> None:
    """Replace the reference wrapper's 5 s wall-clock state reset with the
    adapter's audio-time rule, so the two pipelines can be compared exactly."""
    silero = reference.SileroVAD
    original_prob = silero.prob
    silero.maybe_reset = lambda self: None

    def prob(self, chunk):
        value = original_prob(self, chunk)
        self._audio_t = getattr(self, "_audio_t", 0.0) + reference.CHUNK / reference.SR
        last = getattr(self, "_last_audio_reset", 0.0)
        if reset_s and self._audio_t - last >= reset_s:
            self._init_states()
            self._last_audio_reset = self._audio_t
        return value

    silero.prob = prob


def score_with_reference(reference, ids: list[str], log) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    from turn_bench.data import durations, load_audio_conversation, load_split, locate_rows, split_for

    split = split_for("dev")
    dataset = load_split(split)
    duration_by_id = durations(dataset)
    locations = locate_rows(split)
    silero_path = reference.ensure_model(str(reference._SMART_TURN / "silero_vad.onnx"))
    scored = []
    for index, cid in enumerate(ids, start=1):
        conv = load_audio_conversation(cid, locations[cid], duration_by_id[cid])
        wav1 = reference._load_wav(*conv.audio(1))
        wav2 = reference._load_wav(*conv.audio(2))
        del conv
        s1, s2 = reference._score_conversation(wav1, wav2, silero_path)
        scored.append((cid, s1, s2, duration_by_id[cid]))
        log(f"[reference {index}/{len(ids)}] conversation {cid}: {len(s1)} frames per speaker")
    return scored


def compare_scores(ours: list[float], theirs: list[float], decimals: int | None = None) -> dict:
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if decimals is not None:
        a = np.round(a, decimals)
    scored_a, scored_b = a > 0, b > 0
    both = scored_a & scored_b
    return {
        "frames": n,
        "length_ours": len(ours),
        "length_reference": len(theirs),
        "equal": int((a == b).sum()),
        "scored_only_ours": int((scored_a & ~scored_b).sum()),
        "scored_only_reference": int((~scored_a & scored_b).sum()),
        "scored_both": int(both.sum()),
        "scored_both_differ": int((both & (a != b)).sum()),
        "max_abs_diff": float(np.abs(a - b).max()) if n else 0.0,
        "max_abs_diff_scored_both": float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0,
    }


def read_run_scores(run_dir: Path) -> dict[tuple[str, int], list[float]]:
    scores = {}
    with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            diagnostics = record.get("diagnostics") or {}
            if "scores" in diagnostics:
                scores[(record["conversation_id"], record["speaker"])] = diagnostics["scores"]
    return scores


def summarise(rows: list[dict]) -> dict:
    keys = ("frames", "equal", "scored_only_ours", "scored_only_reference", "scored_both", "scored_both_differ")
    total = {key: sum(row[key] for row in rows) for key in keys}
    total["max_abs_diff"] = max((row["max_abs_diff"] for row in rows), default=0.0)
    total["max_abs_diff_scored_both"] = max((row["max_abs_diff_scored_both"] for row in rows), default=0.0)
    return total


def cmd_reference(args: argparse.Namespace) -> int:
    from turn_bench.parity import run_parity
    from turn_bench.runner import log, run_benchmark

    out = args.out.resolve()
    reference_dir = out / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference_module(args.turnbench.resolve())
    if args.vad_reset == "audio":
        patch_vad_reset_to_audio_time(reference, args.vad_reset_s)
    log(f"reference: {reference.__file__} (vad reset: {args.vad_reset})")

    from turnbench.data import conversation_ids

    from turn_bench.data import load_split, split_for

    ids = conversation_ids(load_split(split_for("dev")))[: args.limit]
    scored = score_with_reference(reference, ids, log)
    submission = reference._make_submission(args.theta_eot, args.theta_int, scored)
    (reference_dir / "predictions-dev.json").write_text(submission.model_dump_json(indent=2), encoding="utf-8")
    reference_scores = {
        (cid, speaker): s[: reference.frame_count(duration, reference.OUT_HZ)].tolist()
        for cid, s1, s2, duration in scored
        for speaker, s in ((1, s1), (2, s2))
    }
    (reference_dir / "scores.json").write_text(
        json.dumps({f"{cid}/{speaker}": values for (cid, speaker), values in reference_scores.items()}),
        encoding="utf-8",
    )
    del scored

    overrides = {"vad_reset_s": args.vad_reset_s}
    run_benchmark(
        model_name="smart_turn_v3",
        split_name="dev",
        workers=args.workers,
        limit=args.limit,
        out=out / "ours",
        config_overrides=overrides,
    )
    status = run_parity(out / "ours", reference_dir / "predictions-dev.json")

    ours_scores = read_run_scores(out / "ours")
    rows = []
    for key, theirs in reference_scores.items():
        row = compare_scores(ours_scores[key], theirs)
        rows.append({"conversation_id": key[0], "speaker": key[1], **row})
        print(f"  scores {key[0]}/{key[1]}: {row}")
    total = summarise(rows)
    print(f"  scores total: {total}")
    (out / "score_parity.json").write_text(json.dumps({"rows": rows, "total": total}, indent=2), encoding="utf-8")
    return status


def cmd_probs(args: argparse.Namespace) -> int:
    from turnbench.submission import load_submission
    from turnbench.sweep import commit_events, load_probs

    from turn_bench.adapters.smart_turn_v3 import OUT_HZ, THETA_EOT, THETA_INT

    ours_scores = read_run_scores(args.run)
    probs = load_probs(args.probs_eot).by_conversation()
    reference = load_submission(args.reference).by_conversation() if args.reference else None
    decimals = args.decimals
    rows, event_rows = [], []
    for (cid, speaker), ours in sorted(ours_scores.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        if cid not in probs:
            continue
        theirs = getattr(probs[cid], f"speaker_{speaker}").prob
        row = compare_scores(ours, theirs, decimals=decimals)
        rows.append({"conversation_id": cid, "speaker": speaker, **row})
        if reference is not None:
            ref_events = getattr(reference[cid], f"speaker_{speaker}")
            raw = np.asarray(ours, dtype=np.float32)
            rounded = np.round(raw.astype(np.float64), decimals) if decimals is not None else raw
            for task, theta, transform, ref_list in (
                ("eot", THETA_EOT, lambda s: s, ref_events.eot),
                ("int", THETA_INT, lambda s: 1.0 - s, ref_events.interruption),
            ):
                from_raw = commit_events(transform(raw).tolist(), OUT_HZ, theta)
                from_rounded = commit_events(transform(rounded).tolist(), OUT_HZ, theta)
                from_theirs = commit_events(transform(np.asarray(theirs)).tolist(), OUT_HZ, theta)
                event_rows.append({
                    "conversation_id": cid, "speaker": speaker, "task": task,
                    "ours_raw": len(from_raw), "ours_rounded": len(from_rounded), "reference": len(ref_list),
                    "raw_equals_reference": from_raw == ref_list,
                    "rounded_equals_reference": from_rounded == ref_list,
                    "theirs_probs_reproduce_reference": from_theirs == ref_list,
                })
    total = summarise(rows)
    print(f"scores vs {args.probs_eot} (ours rounded to {decimals} decimals): {total}")
    if event_rows:
        lists = len(event_rows)
        print(
            f"event lists ({lists}): raw-scores == reference {sum(r['raw_equals_reference'] for r in event_rows)}, "
            f"rounded-scores == reference {sum(r['rounded_equals_reference'] for r in event_rows)}, "
            f"their probs reproduce reference {sum(r['theirs_probs_reproduce_reference'] for r in event_rows)}"
        )
    (args.run / "probs_parity.json").write_text(
        json.dumps({"probs_eot": str(args.probs_eot), "decimals": decimals, "total": total, "rows": rows, "events": event_rows}, indent=2),
        encoding="utf-8",
    )
    return 0


def published_probs_dir(override: Path | None) -> Path:
    if override is not None:
        return override
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(PROBS_REPO, repo_type="dataset", revision=PROBS_REVISION, allow_patterns=f"{PROBS_SUBDIR}/*")
    return Path(snapshot) / PROBS_SUBDIR


def cmd_commit(args: argparse.Namespace) -> int:
    from turnbench.submission import load_submission
    from turnbench.sweep import load_probs

    from turn_bench.adapters.smart_turn_v3 import EdgeCommitter

    probs_dir = published_probs_dir(args.probs_dir)
    reference = load_submission(args.reference).by_conversation()
    rows = []
    for task, theta, key in (("eot", args.theta_eot, "eot"), ("int", args.theta_int, "interruption")):
        probs = load_probs(probs_dir / f"probs-{task}.json")
        for cid, entry in probs.by_conversation().items():
            for speaker in (1, 2):
                values = getattr(entry, f"speaker_{speaker}").prob
                committer = EdgeCommitter(theta, probs.frame_rate_hz)
                ours = [(k + 1) / probs.frame_rate_hz for k, p in enumerate(values) if committer.step(k, p)]
                theirs = getattr(reference[cid], f"speaker_{speaker}")
                theirs = getattr(theirs, key)
                rows.append({"conversation_id": cid, "speaker": speaker, "task": task,
                             "ours": len(ours), "reference": len(theirs), "identical": ours == theirs})
    identical = sum(row["identical"] for row in rows)
    print(f"commit logic vs {args.reference} (probs from {probs_dir}): {identical}/{len(rows)} event lists identical")
    for row in rows:
        if not row["identical"]:
            print(f"  differs: {row}")
    if args.out:
        args.out.write_text(json.dumps({"probs_dir": str(probs_dir), "reference": str(args.reference),
                                        "theta_eot": args.theta_eot, "theta_int": args.theta_int,
                                        "identical": identical, "lists": len(rows), "rows": rows}, indent=2),
                            encoding="utf-8")
    return 0 if identical == len(rows) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reference", help="Sesame's pipeline vs the adapter on the first N dev conversations")
    p.add_argument("--turnbench", type=Path, required=True, help="TurnBench checkout with baselines/smart_turn_v3/smart_turn set up")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--theta-eot", type=float, default=0.0082)
    p.add_argument("--theta-int", type=float, default=0.0155)
    p.add_argument("--vad-reset", choices=["wallclock", "audio"], default="audio",
                   help="patch the reference's wall-clock Silero reset to audio time like the adapter (exact comparison), "
                        "or leave it alone (then its decisions depend on how fast this machine ran)")
    p.add_argument("--vad-reset-s", type=float, default=5.0)
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("probs", help="A run's per-frame scores vs a committed TurnBench probs file")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--probs-eot", type=Path, required=True)
    p.add_argument("--reference", type=Path, help="the committed predictions file those probs were finalised into")
    p.add_argument("--decimals", type=int, default=4, help="round our scores like the committed file (None: compare raw)")
    p.set_defaults(func=cmd_probs)

    p = sub.add_parser("commit", help="The commit rule alone: Sesame's published probs -> their committed predictions")
    p.add_argument("--reference", type=Path, required=True, help="Sesame's baselines/smart_turn_v3/predictions-dev.json")
    p.add_argument("--probs-dir", type=Path, help="directory holding probs-eot.json and probs-int.json (default: the pinned HF snapshot)")
    p.add_argument("--theta-eot", type=float, default=0.0082)
    p.add_argument("--theta-int", type=float, default=0.0155)
    p.add_argument("--out", type=Path, help="write the per-list comparison as JSON")
    p.set_defaults(func=cmd_commit)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
