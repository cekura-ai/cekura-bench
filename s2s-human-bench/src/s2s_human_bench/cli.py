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
    p.add_argument("--titles", type=Path, required=True, help="the website's call list (s2s-human-bench-calls.json): scenario titles for the examples")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="benchmark config (default: config/human-bench.json)")
    p.add_argument("--clips", type=Path, help="recorded caller clips per scenario (default: <runs>/scenario_clips.json, if present)")
    p.add_argument(
        "--name", action="append", metavar="KIND=FILE", help="output file name for one kind (benchmark, insights, reports, languages, skills, examples)"
    )
    p.add_argument("--date", help="generatedAt date, YYYY-MM-DD (default: today)")
    p.add_argument("--exclude-unhealthy", action="store_true", help="leave out calls whose simulation-health check failed")
    p.add_argument("--clip-picks", type=Path, help="also write the example moments' clip windows (with call ids) here; keep it out of the repo")

    args = parser.parse_args()
    if args.command == "site-export":
        from s2s_human_bench.site_export import run_site_export

        sys.exit(run_site_export(args))


if __name__ == "__main__":
    main()
