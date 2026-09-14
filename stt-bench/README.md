# STT Bench

A streaming speech-to-text benchmark that measures transcription accuracy, how
quickly text becomes available, and whether each request completes reliably.
Audio is sent at real-time speed. Timestamped responses are saved so results can
be scored again without another provider request.

## Quick start

Use Python 3.12 or 3.13 and `uv`. In this repository, first run `cd stt-bench`
from the checkout root. Run all commands below from that STT project directory.
Its `.env`, `.venv`, configuration, datasets, and outputs belong in this directory.
See the [repository overview](../README.md) for the separate voice-agent runner.

```sh
uv sync --locked
uv run --locked stt-bench prepare-dataset --dataset pipecat-stt-benchmark
```

Preparation downloads and freezes the public dataset without transcription requests.
Add `DEEPGRAM_API_KEY` to your environment or a local `.env` file, then run:

```sh
uv run --locked stt-bench benchmark \
  --dataset pipecat-stt-benchmark --model deepgram-nova-3
```

This command sends audio to the provider and incurs usage charges. It checks the
host's streaming timing, runs ten smoke-test clips, and starts the full dataset
only if every smoke clip produces a valid, usable result. The full pass streams
about 168 minutes of prepared audio; allow additional time for setup, smoke tests,
connections, finalization, and retries.

To continue an interrupted workflow, use its original run ID:

```sh
uv run --locked stt-bench benchmark \
  --dataset pipecat-stt-benchmark --model deepgram-nova-3 \
  --run-id YOUR_RUN_ID --resume
```

Resume preserves successful attempts and requires matching data, configuration,
source code, and timing environment. A completed failed smoke stage remains blocked
for inspection.

## Models and credentials

Model selectors and settings live in [config/models](config/models). Inspect the
registered models and check local preparation without transcribing audio:

```sh
uv run --locked stt-bench models
uv run --locked stt-bench prepare-audio --dataset pipecat-stt-benchmark
uv run --locked stt-bench prepare-check
```

`prepare-audio` creates the 24 kHz derivatives required by OpenAI and Gradium.
The model listing shows configured models and whether their credentials are set.
A provider may require additional account access for a particular model.

| Provider | Configured models | Credential variable |
| --- | --- | --- |
| Deepgram | Nova-2, Nova-3, Flux English, Flux Multilingual | `DEEPGRAM_API_KEY` |
| OpenAI | GPT Realtime Whisper, GPT-4o Transcribe, GPT-4o Mini Transcribe | `OPENAI_API_KEY` |
| Google Gemini | Gemini 3.5 Transcribe Live | `GEMINI_API_KEY` |
| ElevenLabs | Scribe v2 Realtime | `ELEVENLABS_API_KEY` |
| Speechmatics | Standard, Enhanced | `SPEECHMATICS_API_KEY` |
| Cartesia | Ink-2 | `CARTESIA_API_KEY` |
| Google Cloud Speech | Chirp 2, Chirp 3 | `GOOGLE_APPLICATION_CREDENTIALS` |
| AssemblyAI | Universal 3.5 Pro | `ASSEMBLYAI_API_KEY` |
| Soniox | STT RT v5 | `SONIOX_API_KEY` |
| Smallest AI | Pulse | `SMALLEST_API_KEY` |
| Sarvam | Saaras v3 Realtime | `SARVAM_API_KEY` |
| Inworld | STT-1 | `INWORLD_API_KEY` |
| Gradium | Default | `GRADIUM_API_KEY` |
| Reson8 | Realtime | `RESON_API_KEY` |

Environment variables take precedence over `.env`. Accepted aliases are defined in
[credentials.py](src/stt_bench/credentials.py). Keep credentials in the environment
or the local `.env` file, which is ignored by Git. Google Cloud Speech requires
service-account credentials, separate from a Gemini API key; configure the intended
project and region in its model JSON. Inworld expects an already-encoded Basic
credential. Files under `.secrets/` are also ignored.

## Datasets

The default dataset contains all 1,000 clips from Pipecat's STT benchmark, pinned
in [config/datasets](config/datasets) to revision
`3fe50170d520c951957b86996ef082a6ab87b394`. Its upstream `train` split is used for
evaluation here. Source IDs and transcripts are preserved. Ten clips, selected
with seed 42, form a separate smoke run. They are transcribed again in the full
run, and their smoke costs are counted separately.

Preparation validates every clip and records file hashes. Incomplete preparation
fails instead of silently publishing a smaller dataset. Repeating preparation
verifies existing files.

FLEURS preparation is also available. With source audio in `test/` and metadata
in `test.tsv`, create a public anchor and a separate entity-focused supplement:

```sh
uv run --locked stt-bench prepare-expanded \
  --annotations annotations/fleurs-en-us-entities-v1.json \
  --out datasets/fleurs-local
```

The supplement is selected for entity coverage, so its accuracy is reported
separately from the anchor. Private recordings and short trial cohorts also remain
separate from the full public benchmark. Audio and run outputs are local artifacts.

## Configuration

- [config/models](config/models) contains the provider settings used by each model,
  including the endpoint, audio format, completion behavior, and pricing.
- [config/datasets](config/datasets) pins the dataset revision and sample selection.
- [config/vercel-models.json](config/vercel-models.json) records the Vercel account,
  compute settings, and model jobs. Set these for your own environment before
  launching remote jobs.

## Measurement

The common clock starts when the client finishes sending the last retained speech
frame. It excludes connection setup. Responses contribute to a deadline only if
they were received by that time.

| Metric | Meaning |
| --- | --- |
| Deadline word error rate (WER) | Accuracy of the first attempt's available text at speech end and 250, 500, and 1,000 ms afterward. |
| Eventual WER | Accuracy of the first valid completed attempt, which may be a retry. Total substitutions, insertions, and deletions are divided by total reference words. |
| Acknowledgment latency | Time until a provider acknowledges finalization, where supported. It does not establish that the complete transcript is available. |
| Completion latency | Time until the adapter's terminal completion condition, including the observation window and applicable silence or cleanup time. |
| Coverage and reliability | Planned, attempted, usable, failed, and retried clips, with reasons for exclusions. |
| Entity accuracy | Separate scoring of annotated values and exact formatting. Unsupported types or missing annotations remain unavailable. |
| Estimated cost | Configured rates applied to submitted audio, including failures and retries. Unknown rates remain unavailable. |

WER uses the pinned `jiwer` and `whisper-normalizer` dependencies. Deadline accuracy
always uses attempt one; a retry cannot improve an earlier observation. The standard
runner permits one retry per failed or invalid clip and retains every attempt.

Most adapters send 20 ms frames followed by one second of silence. AssemblyAI sends
60 ms wire packets, with 80 or 100 ms remainder packets where needed, and uses
packet-duration-aware timing checks. OpenAI and Gradium receive 24 kHz audio;
the original frozen input remains unchanged.

Completion is provider-specific: an acknowledgment, final segment, closed socket,
and completed stream are different events. Exact contracts are recorded in the
model configurations and implemented in [providers.py](src/stt_bench/providers.py)
and its adapters. Soniox is finalized at the common speech boundary without extra
pre-finalization silence. AssemblyAI retains native turn detection. These differences
must accompany comparisons. Specialized private-recording runners may also use a
different retry policy.

Sending audio too early or too late can change deadline accuracy. Reports flag
clips that fail the timing checks and deadlines with no available transcript;
inspect those flags when comparing providers. To check reference transcripts,
speech boundaries, or entity labels by listening to the audio, use `export-review`
and `validate-review`.

## Results and offline scoring

Each benchmark run saves its inputs and results under:

```text
datasets/<dataset>/<revision>/{smoke,full}/manifest.json
runs/<dataset>/<model>/<run-id>/
reports/<dataset>/<model>/<run-id>/
```

Stage reports include `results.json`, metric tables, per-clip results, and deadline
observations. They record the dataset, model configuration, measurement version,
and any model-version information returned by the provider.

Score saved raw events into a new directory without provider requests:

```sh
uv run --locked stt-bench score \
  --run runs/YOUR_DATASET/YOUR_MODEL/YOUR_RUN_ID/full \
  --out reports/rescored
```

Use `export-review` and `validate-review` to prepare and check listening reviews.
`compare` checks compatibility between paired reports; `compare-models` compares
saved model-batch summaries. Run `uv run --locked stt-bench --help` for all commands.

Build the offline dashboard from the saved comparison cohorts:

```sh
python3 scripts/build_benchmark_html.py
```

The output is `reports/benchmark-dashboard/index.html`. Its data is embedded, so
it can be viewed without a server. The generator requires locally saved reports;
a fresh clone does not include them. See the
[dashboard guide](scripts/BENCHMARK_DASHBOARD.md) for inputs and exports.

## Timing checks and remote execution

Run a local timing probe without provider requests:

```sh
uv run --locked stt-bench probe-pacing \
  --seconds 10 --repeats 3 --out reports/pacing-check
```

For the standard 20 ms protocol, send-start gaps must stay within 18–40 ms and total
span drift within 2%. Qualification is bound to the host, runtime, and source code
and expires after one hour. Every live attempt is checked again. A failed probe
means the host did not meet timing requirements; it is not a provider failure.
`audit-pacing` inspects saved events, while
[validate_harness.py](scripts/validate_harness.py) adds longer trials and injected stalls.

For Linux compute, [bundle_remote.py](scripts/bundle_remote.py) packages the current
working tree and verified inputs while excluding credentials and old run outputs.
[remote_job.py](scripts/remote_job.py) runs regression and timing checks by default;
`--live` adds provider transcription. Use each script's `--help` for arguments.

Vercel launchers use [config/vercel-models.json](config/vercel-models.json) and an
installed Sandbox SDK selected through `VERCEL_SANDBOX_SDK_DIR`. Review the target
team, project, snapshot, and selected jobs before using the launchers; the checked-in
configuration contains the original run settings. Keep the local checkpoint files
to resume or collect remote jobs, and download results before deleting a sandbox.
The [script guide](scripts/README.md) explains the preparation and launch sequence.

### Concurrent public and private runs

The dedicated [parallel controller](scripts/run_vercel_full_parallel.mjs) runs
Smallest, Gradium, Reson8, and Inworld against the full public dataset and eight
intact private recordings. It requires the local private manifests and prepared
audio referenced by [full_benchmark.py](src/stt_bench/full_benchmark.py); those
inputs are excluded from Git.

```sh
uv run --locked python -m stt_bench.full_benchmark prepare --root reports/full-parallel-new
node scripts/run_vercel_full_parallel.mjs prepare reports/full-parallel-new
node scripts/run_vercel_full_parallel.mjs run reports/full-parallel-new
node scripts/run_vercel_full_parallel.mjs status reports/full-parallel-new
```

Preparation creates the frozen inputs and compute snapshot without transcription.
`run` makes provider requests; `status` reads the local checkpoint. The controller
ramps up to ten streams per model after successful pilot results, permits at most
one recovery attempt per item, and reduces concurrency after rate-limit errors.
Authentication, credit, and model-identity errors stop new submissions for that provider.

Restart with the same plan and command to reattach saved jobs. Uncertain dispatches
require reconciliation before another request. Collected archives are checksum-checked
and replayed before entering `results.json`, `RESULTS.md`, and `index.html`. Public
deadline accuracy and private word-finalization timing use original attempts and
remain separate; recovery accuracy is reported independently.

## Development

```sh
uv run --locked pytest -q
```

Some integration tests need the locally supplied FLEURS audio. Protocol tests use
fixtures and local WebSocket servers, so they can run without provider credentials.

The main code lives in [src/stt_bench](src/stt_bench): preparation freezes inputs,
adapters capture timestamped responses, scoring replays those responses, and
reporting assembles metrics and coverage. The [script guide](scripts/README.md) describes the additional launch and reporting
tools. Regression tests live in [tests](tests).
