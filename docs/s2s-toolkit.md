# S2S campaign toolkit

Four Node 20+ commands cover the repeatable parts of S2S corpus work. They do not
change agents, deployment, corpus, evaluators, or metrics. The only mutating
platform operation is an explicit one-row campaign launch. Never put API keys
in a campaign, registry, or evidence file; the launch/export commands read
`CEKURA_API_KEY` from the environment.

## 1. Campaign controller

Start from `config/s2s-campaign.example.json`. Pin each row's target agent,
scenario IDs, provider model/settings, frequency, concurrency, and exact name.
The Pipecat session config replaces stored defaults, so include every setting
under test. The controller sends top-level `agent`, scenario objects, and
`mock_tool_names: []` on the v1 Pipecat-v2 REST endpoint.

```sh
node bin/s2s-campaign.mjs --campaign campaign.json --registry local-registry.json
node bin/s2s-campaign.mjs --campaign campaign.json --registry local-registry.json --row row-key --execute
```

Preview is offline. Execute launches one result only after the registry has no
matching key/name, all `after` rows are marked validation-clear and terminal
remotely, and the exact result name is absent from the project's result list.
If the list is paginated or a local lock remains after an interrupted launch,
stop and reconcile before retrying. A remote result may exist even if a local
write failed; never retry without searching its exact name. Do not run this
against the example's placeholder project or IDs.

## 2. Cohort registry

```sh
node bin/s2s-cohort-registry.mjs init --registry local-registry.json --project-id 8197
node bin/s2s-cohort-registry.mjs import --registry local-registry.json --entry known-result.json
node bin/s2s-cohort-registry.mjs validation --registry local-registry.json --key row-key --status clear
node bin/s2s-cohort-registry.mjs link --registry local-registry.json --key row-key --url https://example.invalid/report
node bin/s2s-cohort-registry.mjs show --registry local-registry.json
```

The controller writes new entries as `pending`; a human must complete the
run-level stop-line review before marking `clear`. Imported entries require
`key`, `name`, `result_id`, `suite`, `provider`, and exact `settings` including
`scenario_ids` and `frequency`; they default to `unverified`. The registry is
local operational state, not a public artifact. Link entry is bookkeeping only:
it does not create or revoke a public share.

## 3. Privacy-safe cohort export and ZIP

```sh
node bin/s2s-export-cohort.mjs --registry local-registry.json --fetch --out fresh-safe-dir --zip fresh-safe.zip
node bin/s2s-export-cohort.mjs --registry local-registry.json --raw local-raw-capture --key row-key --out fresh-safe-dir
```

`--fetch` retrieves result/run details in memory and never fetches recordings,
logs, or traces. `--raw` accepts the local layout produced by the existing
`export-s2s-results` capture command. Both modes require terminal containers
with exactly `scenario_ids.length * frequency` unique run records and refuse
an existing output directory. The optional ZIP refuses an existing path.

The strict output allowlist contains `manifest.json`, `s2s-benchmark.json`,
`scenario-matrix.jsonl`, `tool-failure-breakdown.json`, and two
`local-diagnostics` files. Only IDs, scores, durations, selected provenance,
counts, closure, timing, tool names and argument field names are emitted. Raw
transcripts, recordings, logs, traces, caller profiles, credentials, session
IDs, and tool argument values never enter the output. `recording_based_stall_count`
is included only when the source integrity metadata supplies a count or list;
unavailable coverage is not treated as zero. The ZIP contains the same safe
files; scan it before sharing. A validation status of `stop` is retained as a
label, not silently promoted to publication-clear.

## 4. Local evidence reducer

```sh
node bin/s2s-reduce-evidence.mjs --input local-evidence.jsonl --out safe-verdicts.json
```

Input is a *local*, value-free ordered event extract, one JSON object per run.
The reducer rejects unknown fields to prevent accidental leakage. The schema:

```json
{"result_id":1,"run_id":2,"scenario_id":311705,"provider":"example-provider","coverage":{"as58_complete":true,"tool_result_complete":false,"response_group_complete":false,"vad_complete":false},"events":[{"type":"as58_pause","at_ms":1000,"pause_end_ms":3000},{"type":"phone_action","at_ms":3200,"tool_name":"update_appointment","argument_names":["phone_number"],"used_unspoken_digits":false,"caller_finished_number":true}]}
```

Only `as58_pause`, `phone_action`, `tool_call`, `tool_result`, and
`vad_interruption` events are accepted. Times are milliseconds on one run's
monotonic timeline. A local reviewer must derive `used_unspoken_digits` and
`caller_finished_number` from private evidence; this script cannot infer them
from an argument-name-only export. For MS72 (`311732`) and MS74 (`311734`),
`tool_result` can carry boolean `routing_ready`, `tool_call` can carry
`model_response_id`, and VAD events carry only the signal time. The output has
AS58 verdicts and three pattern counts with tool/argument names only. Any
missing coverage or ordering produces `unverified`/`null`, never an invented
zero. Do not put transcript text, argument values, raw log lines, or session
identifiers in this input or output.

## Scope and checks

These scripts do not run a campaign during tests. Run the synthetic narrow
check with `node --test tests/test_s2s_toolkit.mjs`. The suite exercises preview,
duplicate refusal, registry bookkeeping, safe export/ZIP, privacy exclusion,
and evidence reduction. Production API response shape and a live launch still
require a separately authorized smoke run and review.
