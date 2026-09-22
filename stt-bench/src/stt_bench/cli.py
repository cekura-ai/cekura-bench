import argparse
import asyncio
from pathlib import Path

from .data import prepare
from .run import run
from .score import score
from .expanded import prepare_expanded
from .diagnostics import audit_pacing, local_probe
from .review import export_review, load_review
from .compare import compare_reports


def main():
    parser = argparse.ArgumentParser(description="Small reproducible streaming STT benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser('prepare-turns', help='Propose conversational turns and export a local listening review')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('auto-prepare-turns', help='Estimate speech ends locally and freeze valid turns without human approval')
    p.add_argument('--draft', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('freeze-turns', help='Freeze approved conversational audio/transcript pairs')
    p.add_argument('--draft', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('plan-turn-run', help='Write a reviewed dataset/model session budget; no provider calls')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--configs', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run-turns', help='Stream frozen manual or automatic turns with independent sessions and TTFT/TTFS')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pacing-check', type=Path)
    p.add_argument('--authorized-private-manifest-sha256')
    p = sub.add_parser('prepare-audio', help='Build hash-bound 24 kHz derivatives without provider calls')
    p.add_argument('--dataset', default='pipecat-stt-benchmark')
    p = sub.add_parser('import-baseline', help='Offline verification and import of historical Nova-3 batches')
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--batch-state', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('models', help='List STT models and credential presence; no network calls')
    p = sub.add_parser('prepare-check', help='Verify frozen inputs and local readiness; no transcription')
    p.add_argument('--dataset', default='pipecat-stt-benchmark')
    p.add_argument('--out', type=Path)
    p = sub.add_parser('compare-models', help='Compare model batch summaries without API calls')
    p.add_argument('--summaries', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('prepare-dataset', help='Download and freeze a named dataset and its smoke subset')
    p.add_argument('--dataset', required=True)
    p = sub.add_parser('benchmark', help='Run a named model: local timing checks, smoke, then full dataset')
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--run-id')
    p.add_argument('--resume', action='store_true')
    p = sub.add_parser("prepare", help="Freeze a deterministic local FLEURS sample")
    p.add_argument("--tsv", type=Path, default=Path("test.tsv"))
    p.add_argument("--audio-dir", type=Path, default=Path("test"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("audit-pacing", help="Diagnose saved send timing without provider calls")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("probe-pacing", help="Stream synthetic audio to a local duplex WebSocket receiver")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--receiver-delay-ms", type=float, default=0)
    p.add_argument("--stall-ms", type=float, default=0)
    p.add_argument('--receiver-stall-ms', type=float, default=0, help='Inject one blocking pause in the independent receiver')
    p.add_argument('--client-stall-ms', type=float, default=0, help='Inject one blocking pause in client message handling')
    p = sub.add_parser("export-review", help="Prepare a human listening review inventory")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("validate-review", help="Check review hashes, coverage and grouping metadata")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p = sub.add_parser("compare", help="Secondary paired comparison of two v3 results.json files")
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare-expanded", help="Freeze the reviewed anchor and entity supplement")
    p.add_argument("--tsv", type=Path, default=Path("test.tsv"))
    p.add_argument("--audio-dir", type=Path, default=Path("test"))
    p.add_argument("--annotations", type=Path, default=Path("annotations/fleurs-en-us-entities-v1.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("run", help="Stream prepared clips; a new connection per clip")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/deepgram.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="Pace audio locally; no provider, no fake transcripts")
    p.add_argument("--resume", action="store_true", help="Resume matching v2 run, preserving attempts")
    p.add_argument("--pacing-check", type=Path, help="Passing local probe pacing.json, required for live runs")
    p = sub.add_parser("score", help="Re-score saved raw events without API requests")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--review", type=Path, help="Hash-bound human listening review inventory")
    args = parser.parse_args()
    try:
        if args.command == 'prepare-turns':
            from .turns import prepare_turns
            print(prepare_turns(args.source, args.out))
        elif args.command == 'auto-prepare-turns':
            from .turn_auto import automatic_preparation
            from .turns import freeze_turns
            import tempfile
            with tempfile.TemporaryDirectory(prefix='turn-auto-') as temp:
                evidence = automatic_preparation(args.draft, Path(temp)/'preparation.json')
                manifest = freeze_turns(args.draft, evidence, args.out)
            print(manifest)
        elif args.command == 'freeze-turns':
            from .turns import freeze_turns
            print(freeze_turns(args.draft, args.review, args.out))
        elif args.command == 'plan-turn-run':
            from .turn_runner import prepare_run_plan
            print(prepare_run_plan(args.manifest, args.configs, args.out))
        elif args.command == 'run-turns':
            from .turn_runner import run_turns
            outcomes = asyncio.run(run_turns(args.manifest, args.config, args.out, dry_run=args.dry_run,
                resume=args.resume, pacing_check=args.pacing_check,
                authorized_private_manifest_sha256=args.authorized_private_manifest_sha256))
            if any(not r['valid'] for r in outcomes) or len(outcomes) != len(__import__('json').loads(args.manifest.read_text())['clips']):
                parser.exit(1, 'Turn run incomplete or invalid; inspect raw attempts and score the run.\n')
            print(args.out)
        elif args.command == 'prepare-audio':
            from .audio_formats import prepare_derivatives
            result = prepare_derivatives(args.dataset)
            print({s: len(d['clips']) for s, d in result.items()})
        elif args.command == 'import-baseline':
            from .baseline import import_baseline
            import_baseline(args.evidence_root, args.batch_state, args.out)
            print(args.out / 'summary.json')
        elif args.command == 'models':
            import json
            from .preparation import models
            print(json.dumps(models(), indent=2))
        elif args.command == 'prepare-check':
            import json
            from .preparation import check
            print(json.dumps(check(args.dataset, args.out), indent=2))
        elif args.command == 'compare-models':
            from .model_jobs import compare_summaries
            compare_summaries(args.summaries, args.out)
            print(args.out / 'comparison.md')
        elif args.command == 'prepare-dataset':
            from .huggingface_data import prepare_dataset
            print(prepare_dataset(args.dataset))
        elif args.command == 'benchmark':
            from .benchmark import benchmark
            asyncio.run(benchmark(args.dataset, args.model, args.run_id, args.resume))
        elif args.command == "prepare":
            print(prepare(args.tsv, args.audio_dir, args.out, args.count, args.seed))
        elif args.command == "prepare-expanded":
            print(prepare_expanded(args.tsv, args.audio_dir, args.annotations, args.out, args.count, args.seed))
        elif args.command == "run":
            outcomes = asyncio.run(run(args.manifest, args.config, args.out, args.dry_run, args.resume, args.pacing_check))
            if any(not r["valid"] for r in outcomes):
                parser.exit(1, "Run has failed or invalid clips; inspect saved outcomes and raw events.\n")
        elif args.command == 'audit-pacing':
            result = audit_pacing(args.run, args.out)
            print(f"{result['valid_attempts']} / {result['attempts']} attempts pass the unchanged pacing gate")
        elif args.command == 'probe-pacing':
            result = asyncio.run(local_probe(args.out, seconds=args.seconds, repeats=args.repeats,
                                             receiver_delay_ms=args.receiver_delay_ms, stall_ms=args.stall_ms,
                                             receiver_stall_ms=args.receiver_stall_ms, client_stall_ms=args.client_stall_ms))
            print(f"{sum(t['valid'] for t in result['trials'])} / {args.repeats} local trials passed; see {args.out / 'pacing.json'}")
            if any(not t['valid'] for t in result['trials']):
                parser.exit(1, 'Local probe did not meet the unchanged pacing thresholds.\n')
        elif args.command == 'export-review':
            print(export_review(args.manifest, args.out))
        elif args.command == 'validate-review':
            _, status = load_review(args.manifest, args.review)
            print(status)
            if status['status'] != 'verified':
                parser.exit(1, 'Human listening review is incomplete.\n')
        elif args.command == 'compare':
            compare_reports(args.left, args.right, args.out)
            print(args.out / 'comparison.json')
        else:
            result = score(args.run, args.out, args.review)
            print(args.out / "results.json")
            if result.get('measurement_version') == 5:
                print(f"Scored {result['accuracy']['measured_turns']} private turns")
            else:
                print(f"Scored {sum(r['scored_clips'] for r in result['results'])} clips")
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
