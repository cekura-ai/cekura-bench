# STT integration

## Layout and import provenance

The existing voice-agent runner remains at the repository root. The independent
Python STT project lives in `stt-bench/`, with its own configuration, `uv.lock`,
`.gitignore`, tests, and operating directory. No common runner or metric schema
is introduced by this import.

Source repository: `https://github.com/cekura-ai/stt-bench`

Source commit: `821fa89a77af6ddfa792fcb04520c598be6b1865`

Destination base: `52c45edb18e54e064a424d52b0ebab41042784e1`

The import retains 148 of the 150 files tracked at that source commit, exported
with `git archive`. The two historical model-access records are excluded.
It contains no nested Git repository. The original STT checkout,
its history, and its uncommitted work are unaffected. This is a snapshot import;
there is no automatic synchronization between repositories.

The source STT repository is private and this destination is public. Publishing
this branch would publish the imported source and tracked metadata. Local
credentials, private audio, generated reports, and virtual environments are not
included. Review publication scope before publishing the branch.

## Deliberate changes from the snapshot

- Removed `comparisons/providers-20260912/google-model-access.json` and
  `openai-model-access.json`. These are historical provider-access records with
  no references in the benchmark code, scripts, tests, or configuration.
- The STT README explains that commands run inside `stt-bench/`.
- `scripts/vercel_benchmark_controller.mjs` derives its local workspace from its
  script location instead of a developer's absolute checkout path. Its remote
  sandbox path, target account, and run IDs are unchanged.

The parent README links to both benchmark families. The existing voice-agent
runner, Node package settings, configuration, and documentation paths stay intact.

## Operating boundaries

Install and run STT from its own directory:

```sh
cd stt-bench
uv sync --locked
uv run --locked stt-bench models
uv run --locked pytest -q
node --test tests/*.test.mjs
```

Some Python integration tests require the original local FLEURS audio in `test/`
and the prepared audio referenced by the committed FLEURS manifests. A fresh
clone does not contain those files. Missing audio is a fixture prerequisite,
not evidence that a provider failed. Protocol tests use local fixtures and
WebSocket servers; they do not establish live provider access.

Dashboard generation requires saved comparison reports. Remote packaging also
requires verified prepared inputs and FLEURS regression audio. Neither generated
reports nor audio should be committed as part of routine development.

Vercel scripts and `config/vercel-models.json` retain deployment-specific settings
from the source project. Review the account, project, snapshot, and run IDs before
using them. This migration does not provision or validate remote compute.

The voice-agent runner's `--validate` and default dry-run modes still read the
Cekura scenario catalog over the network. They are not offline checks. Local
migration checks must stub that API or supply explicitly authorized access.

Existing benchmark runs should finish in their original execution environment.
Do not rewrite saved run identities or bypass source/configuration/timing checks
to resume a run from a different checkout. Preserve original run evidence.

After this integration is merged and adopted, make future STT changes here and
link the old repository to this location. Until then, the original remains the
working source; reconcile subsequent changes explicitly before switching over.

## Local migration validation

Validated against the imported snapshot on 2026-09-15:

- After removing the two historical access records, 148 source files remain.
  146 are byte-identical to the source commit; the two edited files are listed
  above. The cleanup was checked for references and import integrity; the
  runtime tests below were performed on the original import. All 21 existing non-README
  files in the destination are byte-identical to its base commit.
- The STT CLI help and model listing work from the nested project directory;
  the catalog lists 21 configured model selectors.
- All 11 direct and development dependency pins match the existing Python 3.12
  environment used for testing. The imported source directory was explicitly
  selected with `PYTHONPATH=src` and its import location verified.
- On the exported files without local audio, Python tests reported 433 passed,
  8 missing-fixture failures, and 1 skipped. With the original public FLEURS
  fixtures supplied, 438 passed and 3 failed because temporary audio symlinks
  crossed the loader's allowed dataset boundary. Replacing those symlinks with
  hard links inside the dataset and rerunning those exact 3 tests passed.
  Thus all 441 non-skipped Python tests passed across the suite and rechecks.
  The temporary fixture links were removed after validation.
- All 24 JavaScript controller tests passed with their local mocks. These include
  remote orchestration checks; no remote benchmark was launched.
- The existing `npm run validate -- --config config/benchmark.example.json` and
  default `npm run benchmark -- --config config/benchmark.example.json` commands
  passed with a stubbed scenario catalog and dummy credentials. Suite filtering,
  plan mode, credential redaction, and absence of a launched result were checked.
  This is offline runner validation, not proof of live Cekura access.
- Nested ignore rules were checked for credentials, private data, public audio,
  virtual environments, and generated output. A basic credential-pattern scan
  found no matches in the imported files; this is not an exhaustive security audit.

A fresh `uv sync --locked --offline --link-mode hardlink` could not complete
because the local cache lacked a required matplotlib wheel. A clean dependency
installation is therefore not claimed. The failed empty environment was removed;
run the documented `uv sync --locked` with package-network access when setting up.

Live provider calls, Vercel provisioning, full audio bundle generation, and the
saved-report dashboard browser check were not performed. Existing remote-job,
bundle-selection, and dashboard-data unit tests are included in the Python checks.
