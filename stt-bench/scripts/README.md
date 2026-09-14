# Benchmark scripts

Run commands from `stt-bench/`. For a standard public-dataset run, use the
[CLI quick start](../README.md#quick-start). The scripts below provide remote
execution, additional benchmark studies, and report generation.

## Timing and validation

| Script | Purpose |
| --- | --- |
| [validate_harness.py](validate_harness.py) | Checks real-time audio sending, including longer trials and deliberate pauses. Used by remote runners before benchmarking. |
| [validate_providers.py](validate_providers.py) | Runs local tests and records the source version that passed them. |
| [offline_provider_checks.py](offline_provider_checks.py) | Runs provider tests in prepared sandboxes with Python network access limited to local test servers. |

These checks help distinguish a machine that cannot send audio on time from a
provider that responds slowly. The CLI also includes `probe-pacing` for checking
the current machine and `audit-pacing` for inspecting a saved run.

## Remote execution

For a Linux machine of your choice:

1. Prepare the datasets described in the main README. The current bundle also
   requires the FLEURS source and prepared audio used by the regression tests.
2. Use [bundle_remote.py](bundle_remote.py) to package code and verified inputs.
3. Extract the bundle on the remote machine and run `uv sync --locked` there.
4. Use [remote_job.py](remote_job.py) for regression and timing checks. Its `--live`
   flag also runs the provider benchmark. Its `--help` lists the required machine
   identity and run arguments.
5. Download the job's result archive before deleting the remote machine.

[verify_bundle.py](verify_bundle.py) checks a bundle against the local source and
prepared inputs. Credentials are supplied separately from the bundle.

### Vercel

The Vercel scripts use [config/vercel-models.json](../config/vercel-models.json).
It contains the original account and run settings. Set the team, project, region,
preparation sandbox, and model jobs for your own account. Give new runs distinct
run IDs and sandbox names. The scripts require an installed Sandbox SDK selected
by `VERCEL_SANDBOX_SDK_DIR` and an authenticated SDK/CLI session.

| Script | Role in the workflow |
| --- | --- |
| [prepare_vercel_models.mjs](prepare_vercel_models.mjs) | Uploads a verified bundle to an existing preparation sandbox, installs dependencies, and creates the snapshot used by model jobs. Requires `--live` and `--bundle`. |
| [vercel_models.mjs](vercel_models.mjs) | Plans, launches, and collects model jobs from that snapshot. Without `--live`, it prints the local plan. |
| [model_batches.py](model_batches.py) | Runs one model's smoke/full batches inside a prepared sandbox. |
| [vercel_batches.py](vercel_batches.py) | Shared batching, checkpoint, and scoring code used by `model_batches.py`. |
| [vercel_command_wait.mjs](vercel_command_wait.mjs) | Shared helper for waiting on remote commands. |
| [setup_vercel_chirp.mjs](setup_vercel_chirp.mjs) | Prepares Chirp-specific dependencies and sandboxes. Also provides file-upload helpers used by other launchers. |
| [setup_vercel_trial_providers.mjs](setup_vercel_trial_providers.mjs) | Prepares additional provider sandboxes and verifies their uploaded code. |
| [setup_vercel_gradium.mjs](setup_vercel_gradium.mjs), [setup_vercel_reson8.mjs](setup_vercel_reson8.mjs) | Prepare those providers and optionally run a short connection/transcription check. |
| [gradium_tiny_smoke.py](gradium_tiny_smoke.py), [reson8_tiny_smoke.py](reson8_tiny_smoke.py) | Single-phrase checks used by the corresponding setup scripts. |

Keep the generated preparation records and job checkpoints under `reports/`.
They identify the snapshots and commands needed to continue or collect a run.
The scripts do not configure a new Vercel account from scratch.

## Additional benchmark studies

These scripts reproduce specific cohorts or provider protocols. Their local
input paths and saved setup records must exist before launching. Private audio
is not distributed with this repository.

| Study | Scripts and inputs |
| --- | --- |
| AssemblyAI public dataset | [run_vercel_assemblyai_full.mjs](run_vercel_assemblyai_full.mjs) and [assemblyai_pipecat_run.py](assemblyai_pipecat_run.py). Uses the frozen Pipecat dataset; the launcher also supports the private cohort. |
| AssemblyAI private recordings | [prepare_assemblyai_private.py](prepare_assemblyai_private.py), [run_vercel_assemblyai_private.mjs](run_vercel_assemblyai_private.mjs), and [assemblyai_private_run.py](assemblyai_private_run.py). Requires the original private manifests and audio. |
| Short public/private comparison | [prepare_trial_short.py](prepare_trial_short.py), [run_vercel_trial_short.mjs](run_vercel_trial_short.mjs), and [trial_short_run.py](trial_short_run.py). Uses ten Pipecat smoke clips and ten private excerpts. |
| Limited-duration public comparison | [prepare_credit_benchmark.py](prepare_credit_benchmark.py), [run_vercel_credit_benchmark.mjs](run_vercel_credit_benchmark.mjs), and [credit_benchmark_run.py](credit_benchmark_run.py). Uses selected Pipecat and FLEURS cohorts with a 300-second audio budget per provider. |
| Concurrent full public/private comparison | [run_vercel_full_parallel.mjs](run_vercel_full_parallel.mjs) and [full_benchmark.py](../src/stt_bench/full_benchmark.py). Runs Smallest, Gradium, Reson8, and Inworld on the public dataset and eight private recordings. See the [workflow commands](../README.md#concurrent-public-and-private-runs). |

## Reporting

[build_benchmark_html.py](build_benchmark_html.py) and
[benchmark_dashboard.html](benchmark_dashboard.html) generate the offline dashboard
from saved comparison reports. The [dashboard guide](BENCHMARK_DASHBOARD.md)
describes the selected runs, required inputs, metric definitions, and exports.
