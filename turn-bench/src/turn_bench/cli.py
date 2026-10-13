import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from turn_bench import __version__, assets, manifest
from turn_bench.adapters import ASSETS, model_names
from turn_bench.parity import run_parity
from turn_bench.runner import run_benchmark, turnbench_provenance, verify_split_manifest


def parse_config(pairs: list[str]) -> dict:
    """`--config key=value` overrides; values are parsed as JSON when they are."""
    config = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--config expects key=value, got {pair!r}")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming turn-taking benchmark harness scored by Sesame's TurnBench scorer"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="List registered model adapters")

    p = sub.add_parser("run", help="Stream a split through a model, write predictions, score with Sesame's scorer")
    p.add_argument("--model", required=True, choices=model_names())
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--dataset", type=Path, help="local dataset directory (from `author build`) instead of --split")
    p.add_argument("--workers", type=int, default=2, help="conversations processed concurrently")
    p.add_argument("--limit", type=int, help="only the first N conversations (scored as a subset)")
    p.add_argument("--ids", help="comma-separated conversation ids to run, in the split's order (scored as a subset)")
    p.add_argument("--out", type=Path, help="output directory; default runs/<model>-<split>-<utc time>")
    p.add_argument("--realtime", action="store_true", help="pace frames at wall-clock speed (for live providers)")
    p.add_argument("--config", action="append", default=[], metavar="KEY=VALUE", help="model config override")

    p = sub.add_parser("setup", help="Download a model's pinned assets (weights, source) into the assets directory")
    p.add_argument("model", choices=sorted(ASSETS))
    p.add_argument("--dir", type=Path, help="assets directory; default .assets/<model> or $TURN_BENCH_ASSETS_DIR/<model>")
    p.add_argument("--force", action="store_true", help="re-download assets that are already present")
    p.add_argument("--check", action="store_true", help="only verify what is installed; download nothing")

    p = sub.add_parser("manifest", help="Dataset manifests: pin the local snapshot's file hashes, or verify against them")
    msub = p.add_subparsers(dest="manifest_command", required=True)
    m = msub.add_parser("build", help="Write datasets/<name>/manifest.json from the local Hugging Face cache (no download)")
    m.add_argument("--split", default="dev", choices=["dev"])
    m.add_argument("--name", default="turnbench-dev", help="manifest directory name under datasets/")
    m = msub.add_parser("verify", help="Check the local snapshot against the committed manifest (what `run` does first)")
    m.add_argument("--split", default="dev", choices=["dev", "test"])

    p = sub.add_parser("author", help="Authored datasets: build one from conversation specs, or check one against Sesame's gold")
    asub = p.add_subparsers(dest="author_command", required=True)
    a = asub.add_parser("build", help="Render specs with TTS, compose the timelines, write a local dataset directory")
    a.add_argument("--specs", type=Path, required=True, help="directory of conversation spec JSON files")
    a.add_argument("--out", type=Path, required=True, help="dataset directory to write")
    a.add_argument("--renderer", default="mock", choices=["mock", "elevenlabs"])
    a.add_argument("--transform", default="clean", help="service-bench degradation transform per channel (default clean)")
    a.add_argument("--name", help="shard/manifest name; default the output directory's name")
    a.add_argument("--cache-dir", type=Path, help="render cache; default .cache/renders")
    a.add_argument("--max-characters", type=int, help="refuse to build when more than this many characters would be rendered")
    a.add_argument("--estimate", action="store_true", help="only print how many lines/characters the build would render")
    a = asub.add_parser("check", help="Planted-vs-derived gold table and the oracle round trip through Sesame's scorer")
    a.add_argument("--dataset", type=Path, required=True, help="dataset directory written by `author build`")
    a = asub.add_parser("degrade", help="Condition variants of a clean dataset: every conversation under each named degradation, labels unchanged, ids suffixed")
    a.add_argument("--dataset", type=Path, required=True, help="clean dataset directory written by `author build`")
    a.add_argument("--out", type=Path, required=True, help="combined dataset directory to write (clean rows + variants)")
    a.add_argument("--conditions", default=None, help="comma-separated condition names (default: all of turn_bench.author.degrade.DEFAULT_CONDITIONS)")
    a.add_argument("--no-clean", action="store_true", help="leave the clean rows out of the combined dataset")
    a.add_argument("--name", help="shard/manifest name; default the output directory's name")
    a.add_argument("--workers", type=int, default=1, help="processes transforming conversations concurrently (rows are written in order)")
    a = asub.add_parser("split", help="Dev / test / holdout datasets from a built dataset, assigned per template (stratified by family)")
    a.add_argument("--dataset", type=Path, required=True, help="built dataset directory to split")
    a.add_argument("--out", type=Path, required=True, help="directory that receives <out>/dev and <out>/test")
    a.add_argument("--holdout-out", type=Path, required=True, help="directory (gitignored) that receives the holdout dataset")
    a.add_argument("--assignment", type=Path, required=True, help="splits.json: read if it exists, else drawn with --seed and written")
    a.add_argument("--holdout-file", type=Path, required=True, help="holdout ids JSON (gitignored), next to the holdout dataset")
    a.add_argument("--seed", type=int, default=2026)
    a.add_argument("--holdout-fraction", type=float, default=0.2)
    a.add_argument("--dev-fraction", type=float, default=0.4)
    a = asub.add_parser("listen", help="Listening pack: per conversation a stereo WAV (caller L, agent R) and a cue sheet of every labelled event")
    a.add_argument("--dataset", type=Path, required=True, help="dataset directory written by `author build`")
    a.add_argument("--out", type=Path, help="output directory; default <dataset>/listening")
    a.add_argument("--ids", help="comma-separated conversation ids to write (default: every conversation)")

    p = sub.add_parser("parity", help="Compare a run's predictions and scores with a reference predictions file")
    p.add_argument("--run", type=Path, required=True, help="run directory written by `run`")
    p.add_argument("--reference", type=Path, required=True, help="reference predictions JSON")
    p.add_argument("--split", default="dev", choices=["dev"])

    p = sub.add_parser("report", help="Rescore a finished run's predictions (no model run) and rewrite report.json / report.md with scorecard and CIs")
    p.add_argument("--run", type=Path, required=True, help="run directory written by `run`")
    p.add_argument("--dataset", type=Path, help="local dataset directory; default the dataset run.json names (local dir or HF split)")
    p.add_argument("--weight", action="append", default=[], metavar="ROW=WEIGHT", help="scorecard overall weight override, e.g. turn_taking=2 (default: all equal)")

    p = sub.add_parser("compare", help="Print two runs' scorecards side by side with CIs; flag rows whose CIs do not overlap")
    p.add_argument("--run", type=Path, action="append", required=True, help="run directory with a report.json; give exactly twice (A then B)")

    p = sub.add_parser("s2s", help="Speech-to-speech providers: Track 2 interaction probes and the combined scorecard (Track 1 is `run --model s2s:<provider>`)")
    ssub = p.add_subparsers(dest="s2s_command", required=True)
    s = ssub.add_parser("interaction", help="Run the service bench's barge_in / backchannel_tolerance probes against one provider (Track 2)")
    s.add_argument("--provider", required=True, help="service-bench provider key (openai-realtime, gemini-live, grok-realtime, fake)")
    s.add_argument("--repeats", type=int, default=2)
    s.add_argument("--voice", default="f-us", help="caller corpus voice label")
    s.add_argument("--model", help="provider model id; default the service registry's")
    s.add_argument("--turn-detection", default="server_vad", help="server_vad (provider defaults) or semantic_vad")
    s.add_argument("--silence-ms", type=int, help="pin the server VAD silence duration instead of the provider's default")
    s.add_argument("--render", action="store_true", help="render the probes' caller clips with ElevenLabs if absent")
    s.add_argument("--env", help="dotenv file holding the provider credential (default turn-bench/.env, then the repo root's)")
    s.add_argument("--out", type=Path, default=Path("runs") / "s2s-interaction", help="directory that receives the service bench's run directory")
    s.add_argument("--label", default="s2s-interaction")
    s = ssub.add_parser("authored-report", help="Replay experimental authored cells from saved audio; no provider calls")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s = ssub.add_parser("authored-calibrate", help="Offline experimental acoustic calibration; original scores remain unchanged")
    s.add_argument("--run", type=Path, required=True, help="completed v1 authored run; read only")
    s.add_argument("--out", type=Path, required=True, help="fresh directory outside the source run")
    s.add_argument("--vad-model", type=Path, required=True, help="existing pinned Silero ONNX asset; never downloaded")
    for command in ("authored-plan", "authored-run"):
        s = ssub.add_parser(command, help="Experimental authored Track 2: cache-only preflight or live caller")
        s.add_argument("--specs", type=Path, required=True)
        s.add_argument("--cache-dir", type=Path, required=True, help="existing ElevenLabs render cache; never renders")
        s.add_argument("--ids", required=command == "authored-run", help="comma-separated spec or conversation IDs; required before live calls")
        s.add_argument("--split", choices=["dev", "test"], default="test", help="holdout is never selected")
        s.add_argument("--caller-voice", default="f-us")
        s.add_argument("--out", type=Path, required=command == "authored-run")
        if command == "authored-run":
            from turn_bench.s2s import S2S_PROVIDERS
            s.add_argument("--provider", choices=S2S_PROVIDERS, required=True)
            s.add_argument("--model", help="model variant; native provider turn settings and fixed prompt")
            s.add_argument("--repeats", type=int, default=1)
            s.add_argument("--env", help="dotenv credential file; values never enter reports")
            s.add_argument("--track1", type=Path, help="read-only matching v0.5 run; new combined report is written only under --out")
    s = ssub.add_parser("t2-run", help="Track 2 v0.7: authored barge-in/backchannel sessions, speech-cessation metric + no-stimulus controls")
    s.add_argument("--specs", type=Path, required=True)
    s.add_argument("--cache-dir", type=Path, required=True, help="existing ElevenLabs render cache; never renders")
    s.add_argument("--ids", help="comma-separated spec IDs (default: the split's barge_in + backchannels templates)")
    s.add_argument("--split", choices=["dev", "test"], default="test")
    s.add_argument("--caller-voice", default="f-us")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--plan-only", action="store_true", help="cache-only preflight; no provider session")
    from turn_bench.s2s import S2S_PROVIDERS as _P
    s.add_argument("--provider", choices=_P)
    s.add_argument("--model", help="model variant; native turn settings and the fixed Track 2 prompt")
    s.add_argument("--env", help="dotenv credential file; values never enter reports")
    s.add_argument("--condition", default="clean",
                   choices=["clean", "telephone", "noise-20db", "noise-10db", "reverb", "dropouts"],
                   help="caller audio condition: Track 1's seeded service transform applied to the caller stream "
                        "(clips and carrier silence); the model's output is measured as received")
    s.add_argument("--turn-detection", default="server_vad", choices=["server_vad", "semantic_vad"],
                   help="provider turn detection: server_vad = the provider's defaults (published rows); "
                        "semantic_vad where the provider offers it")
    s = ssub.add_parser("t2-rescore", help="Re-score a saved Track 2 run offline with the current speech-cessation method (no provider contact)")
    s.add_argument("--run", type=Path, required=True, help="saved t2-run directory (read only)")
    s.add_argument("--out", type=Path, required=True, help="new directory for the re-scored run")
    s.add_argument("--allow-incomplete", action="store_true", help="also re-score a run whose status is not complete")
    s = ssub.add_parser("t2-combine", help="Pool Track 2 runs of one model (e.g. two caller voices); template-cluster bootstrap")
    s.add_argument("--runs", type=Path, nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    s = ssub.add_parser("scorecard", help="Combine a Track 1 run and a Track 2 run into one scorecard (markdown + json)")
    s.add_argument("--provider", required=True)
    s.add_argument("--track1", type=Path, help="`run --model s2s:<provider>` directory (report.json)")
    s.add_argument("--track2", type=Path, help="`s2s interaction` run directory (interaction.json)")
    s.add_argument("--reference", action="append", default=[], metavar="NAME=RUN_DIR", help="Track 1 reference run on the same conversations (e.g. rms_vad=runs/...)")
    s.add_argument("--label", default="SMOKE")
    s.add_argument("--out", type=Path, required=True, help="directory for s2s-scorecard-<provider>.{md,json}")

    args = parser.parse_args()
    if args.command == "models":
        for name in model_names():
            print(name)
        return
    if args.command == "run":
        out = args.out
        if out is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            label = args.dataset.resolve().name if args.dataset else args.split
            out = Path("runs") / f"{args.model}-{label}-{stamp}"
        info = run_benchmark(
            model_name=args.model,
            split_name=args.split,
            workers=args.workers,
            limit=args.limit,
            out=out,
            realtime=args.realtime,
            config_overrides=parse_config(args.config),
            dataset_dir=args.dataset,
            ids=[i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None,
        )
        sys.exit(1 if info["failures"] else 0)
    if args.command == "s2s":
        sys.exit(run_s2s(args))
    if args.command == "parity":
        sys.exit(run_parity(args.run, args.reference, args.split))
    if args.command == "report":
        from turn_bench.rescore import rescore_run

        rescore_run(args.run, args.dataset, parse_config(args.weight) or None)
        return
    if args.command == "compare":
        from turn_bench.rescore import run_compare

        if len(args.run) != 2:
            raise SystemExit("compare expects exactly two --run directories (A then B)")
        sys.exit(run_compare(args.run[0], args.run[1]))
    if args.command == "manifest":
        run_manifest(args)
        return
    if args.command == "author":
        sys.exit(run_author(args))
    if args.command == "setup":
        directory = assets.assets_dir(args.model, args.dir)
        specs = ASSETS[args.model]
        if args.check:
            problems = assets.verify(directory, specs)
            print(f"{args.model} assets in {directory}: " + ("ok" if not problems else "; ".join(problems)))
            sys.exit(1 if problems else 0)
        print(f"{args.model} assets -> {directory}")
        record = assets.install(directory, specs, force=args.force)
        print(json.dumps(record["assets"], indent=2))
        return


def run_s2s(args) -> int:
    if args.s2s_command == "authored-calibrate":
        from turn_bench.s2s.authored_calibration import calibrate_run

        summary = calibrate_run(args.run, args.out, args.vad_model)
        print(f"calibrated {summary['cells']} cells: {summary['scored']} scored; original results unchanged")
        return 0
    if args.s2s_command == "authored-report":
        from turn_bench.s2s.authored_runner import replay_authored

        summary = replay_authored(args.run, args.out)
        print(f"reproduced {summary['scored']} scored cells from saved PCM and timelines")
        return 0
    if args.s2s_command == "t2-rescore":
        from turn_bench.s2s.track2 import rescore_run

        result = rescore_run(args.run, args.out, allow_incomplete=args.allow_incomplete)
        print((args.out / "t2-summary.md").read_text())
        print("re-score diff: " + json.dumps(result["diff"]))
        return 0
    if args.s2s_command == "t2-combine":
        from turn_bench.s2s.track2 import combine, render

        summary = combine(args.runs, args.out)
        print(render(summary, summary))
        return 0
    if args.s2s_command == "t2-run":
        from turn_bench.s2s.track2 import build_t2_plan, run_t2

        ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None
        plan = build_t2_plan(args.specs, args.cache_dir, ids=ids, caller_voice=args.caller_voice, split=args.split)
        if args.plan_only:
            if args.out.exists():
                raise FileExistsError(f"refusing to overwrite plan: {args.out}")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"counts": plan["counts"], "t2_counts": plan["t2_counts"], "exclusions": plan["exclusions"]}, indent=2))
            return 0
        if not args.provider:
            raise SystemExit("--provider is required for a live Track 2 run")
        summary = run_t2(plan, args.out, provider=args.provider, model=args.model, env_file=args.env,
                         turn_detection=args.turn_detection,
                         condition=args.condition)
        print((args.out / "t2-summary.md").read_text())
        return 0
    if args.s2s_command in ("authored-plan", "authored-run"):
        from turn_bench.s2s.authored_plan import build_plan

        ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None
        plan = build_plan(args.specs, args.cache_dir, ids=ids, caller_voice=args.caller_voice, split=args.split)
        if args.s2s_command == "authored-plan":
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                if args.out.exists():
                    raise FileExistsError(f"refusing to overwrite plan: {args.out}")
                args.out.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"counts": plan["counts"], "exclusions": plan["exclusions"]}, indent=2))
            return 0
        from turn_bench.s2s.authored_runner import run_authored

        summary = run_authored(plan, args.out, provider=args.provider, model=args.model,
                               repeats=args.repeats, env_file=args.env, track1=args.track1)
        print((args.out / "authored-interaction.md").read_text())
        print(f"wrote experimental authored results to {args.out}")
        return 0 if summary["scored"] else 1
    if args.s2s_command == "interaction":
        from turn_bench.s2s.interaction import run_interaction

        def show(cell) -> None:
            detail = cell.void or (f"stop_ms={cell.values.get('stop_ms')}" if "stop_ms" in cell.values else "")
            print(f"  {cell.variant:34} {cell.config:12} {cell.voice:5} r{cell.repeat}  {cell.verdict or '-':5} {detail}", flush=True)

        root = run_interaction(
            args.provider, args.out, repeats=args.repeats, voice=args.voice, model=args.model, label=args.label,
            render=args.render, env_file=args.env, turn_detection=args.turn_detection, silence_duration_ms=args.silence_ms,
            on_cell=show,
        )
        print(f"run -> {root}")
        print((root / "interaction.json").read_text(encoding="utf-8"))
        return 0
    if args.s2s_command == "scorecard":
        from turn_bench.s2s.scorecard import combined_scorecard, load_track1, load_track2, write_scorecard

        references = {}
        for pair in args.reference:
            name, sep, run_dir = pair.partition("=")
            if not sep:
                raise SystemExit(f"--reference expects NAME=RUN_DIR, got {pair!r}")
            references[name] = load_track1(Path(run_dir))
        card = combined_scorecard(
            args.provider,
            load_track1(args.track1) if args.track1 else None,
            load_track2(args.track2) if args.track2 else None,
            references,
            label=args.label,
        )
        json_path, md_path = write_scorecard(args.out, card)
        print(md_path.read_text(encoding="utf-8"))
        print(f"wrote {json_path} and {md_path}")
        return 0
    return 2


def run_author(args) -> int:
    if args.author_command == "check":
        from turn_bench.author.check import run_check

        return run_check(args.dataset)
    if args.author_command == "degrade":
        from turn_bench.author.degrade import DEFAULT_CONDITIONS, degrade_dataset

        conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip()) if args.conditions else DEFAULT_CONDITIONS
        degrade_dataset(args.dataset, args.out, conditions, include_clean=not args.no_clean, name=args.name, workers=args.workers)
        return 0
    if args.author_command == "split":
        from turn_bench.author.splits import assign_templates, load_assignment, split_dataset, split_summary, write_assignment

        if args.assignment.is_file():
            assignment = load_assignment(args.assignment, args.holdout_file)
        else:
            truth = json.loads((args.dataset / "ground_truth.json").read_text(encoding="utf-8"))
            families = {record["spec_id"]: record["family"] for record in truth.values()}
            assignment = assign_templates(families, seed=args.seed, holdout=args.holdout_fraction, dev=args.dev_fraction)
            write_assignment(assignment, args.assignment, args.holdout_file, seed=args.seed, holdout=args.holdout_fraction, dev=args.dev_fraction)
        outs = {"dev": args.out / "dev", "test": args.out / "test", "holdout": args.holdout_out}
        split_dataset(args.dataset, outs, assignment)
        print(json.dumps(split_summary(args.dataset, assignment), indent=2))
        return 0
    if args.author_command == "listen":
        from turn_bench.author.listen import write_listening_pack

        ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
        written = write_listening_pack(args.dataset, args.out, ids=ids)
        for path in written:
            print(path)
        return 0
    from turn_bench.author import service_bridge
    from turn_bench.author.build import build_dataset
    from turn_bench.author.render import RenderCache, make_renderer

    if args.transform not in service_bridge.transform_names():
        raise SystemExit(f"unknown transform {args.transform!r}; known: {', '.join(service_bridge.transform_names())}")
    renderer = make_renderer(args.renderer)
    if args.estimate:
        from turn_bench.author.build import estimate_characters
        from turn_bench.author.spec import load_specs

        print(json.dumps(estimate_characters(load_specs(args.specs), renderer, RenderCache(args.cache_dir)), indent=2))
        return 0
    build_dataset(
        args.specs,
        args.out,
        renderer,
        transform=args.transform,
        cache=RenderCache(args.cache_dir),
        name=args.name,
        max_characters=args.max_characters,
    )
    return 0


def run_manifest(args) -> None:
    from turn_bench.data import locate_rows, snapshot_dir, split_for

    split = split_for(args.split)
    if args.manifest_command == "verify":
        record = verify_split_manifest(split)  # raises ManifestError on mismatch
        print(json.dumps(record, indent=2))
        return
    if split.revision is None:
        raise SystemExit(f"split {split.name!r} has no pinned revision; nothing to pin")
    snapshot = snapshot_dir(split, "*", local_only=True)
    ids = sorted(locate_rows(split), key=int)
    dataset_url = manifest.HF_DATASET_URL.format(source=split.source)
    built = manifest.build_manifest(
        name=args.name,
        source=split.source,
        revision=split.revision,
        split=split.name,
        snapshot_dir=snapshot,
        conversation_ids=ids,
        gated=True,
        source_urls={
            "dataset": f"{dataset_url}/tree/{split.revision}",
            "benchmark": "https://github.com/SesameAILabs/turnbench",
        },
        license={
            "name": "Dataset Public License v1.0",
            "licensor": "Mundo AI Inc.",
            "link": f"{dataset_url}/blob/{split.revision}/LICENSE",
            "note": (
                "Non-commercial use only; voice cloning prohibited; redistribution must carry the licence in full. "
                "The recordings, transcripts, annotations and related materials are Licensed Material and are not "
                "in this repository: this manifest holds identifiers and content hashes only."
            ),
        },
        generated_by={"harness": f"turn-bench {__version__}", "turnbench": turnbench_provenance()},
    )
    out = manifest.MANIFESTS_DIR / args.name / "manifest.json"
    manifest.write_manifest(built, out)
    print(f"wrote {out}: {built['conversations']['count']} conversations, {len(built['files'])} files, {built['total_bytes'] / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
