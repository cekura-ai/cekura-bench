# Speechmatics Linden-1 combined benchmark

## Configuration

- Model: `linden-1`, explicitly selected on Speechmatics Agent STT.
- Endpoint: `wss://global.rt.speechmatics.com/v2/agent`.
- Turn detection: `external`.
- One mono 16 kHz stream per Vercel sandbox; 20 ms frames, original pacing gates.
- `ForceEndOfUtterance` is sent at the frozen speech-end boundary with an audio timestamp. The existing one-second silence tail is retained, followed by `EndOfStream` and a required `EndOfTranscript` response.
- The external boundary is supplied by the benchmark, not measured as live turn-detector quality. Private recordings remain intact; no extra internal turn detector is added.
- No separate flush acknowledgment is assumed. Final transcript receipt, deadline accuracy, and terminal completion are measured from raw events.
- Segment reconstruction follows the official SDK: each `AddSegment` appends text, `AddPartialSegment` replaces the live preview. Passed-through legacy `AddTranscript` messages do not add duplicate text.

## Frozen scope

`reports/linden-full-20260915/plan.json` contains 1,000 public Pipecat clips and 8 private longform recordings, totaling 289.1117 submitted audio minutes. Input IDs, audio hashes, references, configuration, and code hashes are frozen in the run bundle.

The benchmark uses the same public and private source manifests as the existing combined runs. It pools word-error counts across both cohorts; it does not average per-clip percentages. It retains separate public/private totals, planned/attempted/successful coverage, retries, and first-attempt timing.

## Execution

`scripts/run_vercel_linden_parallel.mjs` owns unique recording assignments and durable command IDs. It starts with one public pilot and a private pilot, ramps through 5 and 10 to at most 20 workers, and reduces concurrency on provider quota errors. At most two attempts per recording are allowed. Raw archives are checksum-verified and independently replayed before merging; duplicate attempts are rejected. Each sandbox is stopped and its network policy set to deny-all after its assigned work ends.

This is an isolated execution profile, so existing benchmark run plans are unchanged. The controller runs on the local computer and dispatches detached remote commands. The computer must stay awake and connected for continued dispatch and collection. This is not a scheduled automation.

## Run commands

Use the existing authenticated Vercel SDK directory:

```sh
export VERCEL_SANDBOX_SDK_DIR=/absolute/path/to/node_modules/@vercel/sandbox
node scripts/run_vercel_linden_parallel.mjs run reports/linden-full-20260915
node scripts/run_vercel_linden_parallel.mjs status reports/linden-full-20260915
```

Run output includes `controller.json`, immutable batch assignments, `batches/*/evidence.tar.gz`, replayed `verified.json` receipts, `results.json`, `RESULTS.md`, and `index.html`.

## Protocol sources

- https://github.com/pipecat-ai/pipecat/releases/tag/v1.10.0
- https://github.com/speechmatics/speechmatics-python-sdk/tree/e93433ce2a0f98248c9fc0bbce9bd652b1a43b45/sdk/agent_stt
- https://docs.speechmatics.com/get-started/authentication#supported-endpoints

## Account concurrency limits

The controller reduces concurrency when the provider reports quota errors. The
requested worker ceiling does not establish the account's available capacity.
Check the run's saved controller state for its actual ceiling and progress.
