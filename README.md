# Cekura Benchmarks

Run Cekura's Appointment and Medicare voice-agent benchmark suites against your
own Cekura-connected agent.

This repository is intentionally separate from Cekura's internal benchmark
operations repository. It contains no production credentials, phone numbers,
call logs, customer data, or internal result IDs.

## License

License selection is intentionally pending before the first public push.

## What this runner does

Given a Cekura API key and a local configuration, the runner:

1. Reads all Appointment (`AS`) and Medicare (`MS`) scenarios from your catalog
   agent.
2. Validates the target agent/run configuration.
3. Launches the complete suite against your target agent with your selected
   repetition count and concurrency.
4. Prints the result URL and a machine-readable launch record.

It is dry-run by default. Add `--execute` only after reviewing the resolved
payload.

## Prerequisites

- Node.js 20 or later.
- A Cekura project API key in `CEKURA_API_KEY`.
- A catalog agent in your project containing the benchmark scenarios.
- A target agent that can receive Cekura calls and publish its final transcript
  in the [transcript-ingestion format](docs/transcript-ingestion.md).
- A dedicated target number. For custom transcript publishers, use
  `different_numbers` so each run can be associated to its provider call.

## Quick start

```bash
cp config/benchmark.example.json benchmark.config.json
export CEKURA_API_KEY='...'
npm run benchmark -- --config benchmark.config.json
npm run benchmark -- --config benchmark.config.json --execute
```

## Configuration

```json
{
  "projectId": 1234,
  "catalogAgentId": 5678,
  "targetAgentId": 9012,
  "agentNumber": "+15555550100",
  "frequency": 3,
  "concurrencyLimit": 5,
  "numberMode": "different_numbers",
  "name": "My provider Benchmark v1"
}
```

`catalogAgentId` owns the canonical evaluator scenarios. `targetAgentId` is the
agent being measured; they must not be the same agent.

## Reference implementations

The public reference-agent area will hold sanitized, deployable examples for
LiveKit, Pipecat, OpenAI Realtime, and Gemini Live. Before adding an example,
remove production endpoints, credentials, phone numbers, result IDs, traces,
and any customer or caller data. See [reference-agents](reference-agents/README.md).

## Current scope

This first public scaffold launches an existing catalog's full suite. Importing
or provisioning the canonical catalog into a brand-new Cekura project is the
next implementation step; it needs a supported public scenario-import API so
the runner does not invent or rely on internal endpoints.
