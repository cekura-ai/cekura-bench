import argparse
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "human-bench.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech-to-speech benchmark with real recorded callers")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("site-export", help="Write the website's data files from exported runs: leaderboard, insights, languages, skills, examples")
    p.add_argument("--runs", type=Path, required=True, help="directory holding the exported run folders (<run>/calls/<code>/<call id>/)")
    p.add_argument(
        "--public-s2s",
        type=Path,
        required=True,
        help="the public S2S benchmark's s2s-benchmark.json (model metadata, published prices), or a website checkout to read it from at origin/main",
    )
    p.add_argument("--titles", type=Path, help='scenario titles, {"calls": [{"id", "title"}]} (default: <runs>/scenario_titles.json, if present)')
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="benchmark config (default: config/human-bench.json)")
    p.add_argument("--clips", type=Path, help="recorded caller clips per scenario (default: <runs>/scenario_clips.json, if present)")
    p.add_argument(
        "--name", action="append", metavar="KIND=FILE", help="output file name for one kind (benchmark, insights, reports, languages, skills, examples)"
    )
    p.add_argument("--date", help="generatedAt date, YYYY-MM-DD (default: today)")
    p.add_argument("--exclude-unhealthy", action="store_true", help="leave out calls whose simulation-health check failed")
    p.add_argument(
        "--tasks",
        type=Path,
        help="private per-scenario task file; with it, Task Outcome is pass or fail on each scenario's main task, decided from the call's tool calls,"
        " instead of the share of expected-outcome lines met",
    )
    p.add_argument(
        "--reviewed-examples", type=Path, help="example calls a person has listened to, {site id: [call id, ...]}; without it no example calls are published"
    )
    p.add_argument("--reviewed-clips", type=Path, help="approved clip pairs per skill, {skill id: {setup, should, pass, fail}}; replaces the candidate moments")
    p.add_argument("--clip-picks", type=Path, help="also write the example moments' clip windows (with call ids) here; keep it out of the repo")

    args = parser.parse_args()
    if args.command == "site-export":
        from s2s_human_bench.site_export import run_site_export

        sys.exit(run_site_export(args))


if __name__ == "__main__":
    main()
