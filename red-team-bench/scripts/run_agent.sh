#!/bin/bash
# The full benchmark on one agent: every goal in the goal module, up to 5 attacks each (Crescendo with --learn), then
# the controls.
# A round resumes (finished attacks are skipped, failed ones run again), so a pass that dies is run again, up to three
# passes, until one adds no rows.
#
#   scripts/run_agent.sh <cell> <round> [lines]        e.g. scripts/run_agent.sh s2svoice-phonic phonic-1 10
#
# GOALS_MODULE names the goal module (default redteam.example_goals) and CONTROLS the controls file (default
# redteam/example_controls.json).
#
# Needs REDTEAM_ENV (the keys file), RED_TEAM_DATA (run data, outside the repository), MAX_USD (no new attack starts
# once the ledger reaches it), and TOOL_URL for platforms that call our tools over the internet (Vapi, Retell).
set -euo pipefail
CELL=$1 ROUND=$2 LINES=${3:-8}
cd "$(dirname "$0")/.."
: "${REDTEAM_ENV:?set REDTEAM_ENV to the keys file}" "${RED_TEAM_DATA:?set RED_TEAM_DATA to the run data directory}"
: "${MAX_USD:?set MAX_USD, the ledger total at which no new attack starts}"
export LATENCY_LOG=${LATENCY_LOG:-$RED_TEAM_DATA/runs/latency2.jsonl}
PY=${PY:-python3}
GOALS_MODULE=${GOALS_MODULE:-redteam.example_goals}
CONTROLS=${CONTROLS:-redteam/example_controls.json}
RUNS=$RED_TEAM_DATA/runs/$ROUND.jsonl
last=-1
for pass in 1 2 3; do
  rows=$([ -f "$RUNS" ] && wc -l < "$RUNS" || echo 0)
  [ "$rows" = "$last" ] && break
  last=$rows
  echo "$(date -u +%H:%M) pass $pass of $ROUND ($rows rows)"
  $PY -m redteam.crescendo_attack --cells "$CELL" --goals-module "$GOALS_MODULE" --goals all --attempts 5 --learn \
    --round "$ROUND" \
    --attackers deepinfra:deepseek-ai/DeepSeek-V4-Pro --concurrency "$LINES" --max-usd "$MAX_USD" || true
done
echo "$(date -u +%H:%M) controls"
$PY -m redteam.control_calls --cells "$CELL" --controls "$CONTROLS" --concurrency 5 --max-usd "$MAX_USD"
