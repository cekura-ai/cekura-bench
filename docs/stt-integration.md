# STT integration

## Project layout

The voice-agent runner lives at the repository root. The Python STT project lives
in `stt-bench/`, with its own configuration, dependencies, tests, and output
folders. Run each project's commands from its respective directory.

See the [STT README](../stt-bench/README.md) for provider setup, dataset preparation,
benchmark commands, and scoring.

## Source and changes

The STT project was imported from [cekura-ai/stt-bench](https://github.com/cekura-ai/stt-bench)
at commit `821fa89a77af6ddfa792fcb04520c598be6b1865`.
The destination base was `52c45edb18e54e064a424d52b0ebab41042784e1`.
This is a snapshot import; changes do not automatically sync between repositories.

The import retains 148 source files, with these adjustments:

- Excluded the unused `google-model-access.json` and `openai-model-access.json`
  records from `comparisons/providers-20260912/`.
- Updated the STT README to explain the working directory.
- Updated `scripts/vercel_benchmark_controller.mjs` to derive its local workspace
  from the script location instead of a fixed checkout path.

The root README links to both projects. Existing voice-agent commands and
configuration paths are unchanged.

## Setup and local files

From the repository root:

```sh
cd stt-bench
uv sync --locked
uv run --locked stt-bench models
```

Keep STT credentials, the Python environment, audio, run checkpoints, and reports
inside `stt-bench/`. These local files are ignored by Git and are not part of the
import.

Some Python integration tests require FLEURS source audio in `test/` and prepared
audio matching the committed dataset manifests. Dashboard generation requires
saved comparison reports. Remote packaging requires verified prepared inputs
and FLEURS regression audio.

Vercel scripts and `config/vercel-models.json` contain deployment-specific
settings. Check the account, project, snapshot, and run IDs before using them.
Continue existing benchmark runs in their original environment; resume checks
require matching source code, configuration, data, and timing environment.

## Validation

The initial import was checked locally on 2026-09-15:

- 441 Python tests passed across the suite and fixture rechecks; 1 was skipped.
  The tests used the existing Python 3.12 environment and locally supplied public
  FLEURS audio.
- All 24 JavaScript controller tests passed.
- The STT CLI help and model listing worked from the nested project directory.
- Existing voice-agent validation and dry-run commands passed with a mocked
  scenario catalog. The actual commands require network access to Cekura.
- Source-file integrity and Git ignore rules were verified. The later removal
  of the two access records was checked for references and import integrity.

A fresh offline dependency installation could not complete because the local
cache lacked a required matplotlib wheel. Use `uv sync --locked` with package
network access for setup. Live provider calls, Vercel provisioning, complete
audio bundle generation, and the saved-report dashboard browser check were not
part of this validation.
