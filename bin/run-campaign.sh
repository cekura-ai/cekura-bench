#!/usr/bin/env bash
# One full Lane A campaign against one provider: every suite, in sequence.
#
#   LANE_A_ENV=/path/to/.env bin/run-campaign.sh openai-realtime [out_dir]
#
# Open-loop suites run on every corpus voice; task suites run on one voice with
# fewer repeats, because a task cell is a whole conversation. Each suite writes
# its own run directory, audits itself, and writes its report; the campaign
# report across suites is produced afterwards with lane_a.report.
set -u
PROVIDER="${1:?provider}"
OUT="${2:-data/lane-a}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${LANE_A_PYTHON:-$HERE/.venv/bin/python}"
ENV_ARG=()
[ -n "${LANE_A_ENV:-}" ] && ENV_ARG=(--env "$LANE_A_ENV")
VOICES=(--voice f-us --voice m-us --voice f-gb)

run() {
  echo "== $(date -u +%H:%M:%SZ) $PROVIDER $*"
  "$PY" "$HERE/bin/run-lane-a.py" --provider "$PROVIDER" --out "$OUT" "${ENV_ARG[@]}" "$@"
  echo "== exit $?"
}

run --suite latency        --repeats 5 "${VOICES[@]}"
run --suite endpointing    --repeats 3 "${VOICES[@]}"
run --suite interaction    --repeats 3 "${VOICES[@]}"
run --suite transcription  --repeats 3 "${VOICES[@]}"
run --suite noise          --repeats 3 --voice f-us
run --suite robustness     --repeats 3 --voice f-us
run --suite task           --repeats 3 --voice f-us
run --suite task-text      --repeats 3 --voice f-us
run --suite task-medicare  --repeats 3 --voice f-us
run --suite task-medicare-text --repeats 3 --voice f-us
echo "== $(date -u +%H:%M:%SZ) campaign done: $PROVIDER"
