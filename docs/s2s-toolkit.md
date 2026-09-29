# S2S campaign toolkit

Four Node 20+ commands cover the repeatable parts of S2S corpus work. They do not
change agents, deployment, corpus, evaluators, or metrics. The only mutating
platform operation is an explicit one-row campaign launch. Never put API keys
in a campaign, registry, or evidence file; the launch/export commands read
`CEKURA_API_KEY` from the environment.

## 1. Campaign controller

Start from `config/s2s-campaign.example.json`. Pin each row's target agent,
scenario IDs, provider model/settings, frequency, concurrency, and exact name.
Pin `expected_agent_commit_prefix` when the deployed build matters; validation
then checks every recorded run against it.
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
The name check uses the backend's exact `name` filter. Session config keys are
restricted to `Settings.KEYS` in `reference-agents/pipecat-s2s/bot.py`; unknown
keys are rejected before preview or launch.
If the list is paginated or a local lock remains after an interrupted launch,
stop and reconcile before retrying. A remote result may exist even if a local
write failed; never retry without searching its exact name. Do not run this
against the example's placeholder project or IDs.

## 2. Cohort registry

```sh
node bin/s2s-cohort-registry.mjs init --registry local-registry.json --project-id 1
node bin/s2s-cohort-registry.mjs import --registry local-registry.json --entry known-result.json
node bin/s2s-cohort-registry.mjs validation --registry local-registry.json --key row-key --status clear --export fresh-safe-dir
node bin/s2s-cohort-registry.mjs link --registry local-registry.json --key row-key --url https://example.invalid/report
node bin/s2s-cohort-registry.mjs show --registry local-registry.json
```

The controller writes new entries as `pending`; a human must complete the
run-level stop-line review before marking `clear`. `clear` also requires an
export made by the cohort driver with matching result ID, settings hash, exact
scenario/repeat distribution, and zero missing agent records or configuration
mismatches. The registry stores the export manifest's SHA-256. Both mutation
commands use the same lock. Imported entries require
`key`, `name`, `result_id`, `suite`, `provider`, and exact `settings` including
`scenario_ids` and `frequency`; they default to `unverified`. The registry is
local operational state, not a public artifact. Link entry is bookkeeping only:
it does not create or revoke a public share.

## 3. Privacy-safe cohort export and ZIP

```sh
node bin/s2s-export-cohort.mjs --registry local-registry.json --campaign presentation.json --definitions private-contracts --fetch --out fresh-safe-dir --zip fresh-safe.zip
node bin/s2s-export-cohort.mjs --registry local-registry.json --campaign presentation.json --definitions private-contracts --raw local-raw-capture --key row-key --out fresh-safe-dir
```

`--fetch` invokes the canonical raw exporter with retry/backoff and with logs
and traces disabled; its temporary private capture is removed after the build.
`--raw` accepts that exporter's local directory layout. Both modes require
terminal containers with the exact scenario/repeat distribution and refuse an
existing output directory or ZIP. `--definitions` is the private expected-tool
contract directory; `--campaign` supplies the contract builder's presentation
decisions and must not contain sensitive values. One export requires a uniform
frequency across selected results.

The driver invokes `lib/s2s-export-contract.mjs`, the same schema-version-2
builder used for the website. Its shareable output contains `manifest.json`,
`s2s-benchmark.json`, `scenario-matrix.jsonl`, and
`tool-failure-breakdown.json`. The output directory also has a safe
`registry-verification.json` for the validation gate; the ZIP contains only
the four website files. Raw transcripts, recordings, logs, traces, caller
profiles, credentials, session IDs, and tool argument values stay out of both.
The scorer's raw local diagnostics live in disposable private scratch, never
in the shareable directory. Recording-based stall counts are unavailable in
the current source records; do not infer zero from that absence. A result with
missing agent data can still be exported for diagnosis, but the registry will
not mark it `clear`.

## 4. Local evidence reducer

```sh
node bin/s2s-reduce-evidence.mjs --input local-evidence.jsonl --out safe-verdicts.json --as58-scenario 1 --ms72-scenario 2 --ms74-scenario 3
```

Input is a *local*, value-free ordered event extract, one JSON object per run.
The reducer rejects unknown fields to prevent accidental leakage. The schema:

```json
{"result_id":1,"run_id":2,"scenario_id":1,"provider":"example-provider","coverage":{"as58_complete":true,"tool_result_complete":false,"response_group_complete":false,"vad_complete":false},"events":[{"type":"as58_pause","at_ms":1000,"pause_end_ms":3000},{"type":"phone_action","at_ms":3200,"tool_name":"update_appointment","argument_names":["phone_number"],"used_unspoken_digits":false,"caller_finished_number":true}]}
```

Only `as58_pause`, `phone_action`, `tool_call`, `tool_result`, and
`vad_interruption` events are accepted. Times are milliseconds on one run's
monotonic timeline. A local reviewer must derive `used_unspoken_digits` and
`caller_finished_number` from private evidence; this script cannot infer them
from an argument-name-only export. The three scenario IDs are CLI inputs, not
constants in the public repository. For the two Medicare scenarios,
`tool_result` can carry boolean `routing_ready`, `tool_call` can carry
`model_response_id`, and VAD events carry only the signal time. The output has
AS58 verdicts and three pattern counts with tool/argument names only. With
complete coverage, no phone action and an unfinished caller number are explicit
verdicts; incomplete evidence is `unverified`/`null`, never an invented zero.
The repeat-after-interruption check only considers tools already used before
the VAD signal. Do not put transcript text, argument values, raw log lines, or
session identifiers in this input or output.

## Scope and checks

These scripts do not run a campaign during tests. Run the synthetic narrow
check with `node --test tests/test_s2s_toolkit.mjs`. The suite exercises preview,
duplicate refusal, registry bookkeeping, contract export/ZIP, privacy exclusion,
and evidence reduction. The canonical exporter/scorer has its own targeted test
file. The live API launch path still requires a separately authorized smoke
run and review.
