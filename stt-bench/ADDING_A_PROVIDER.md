# Adding a provider

The steps from a new streaming speech-to-text model to a published website row.
Run every command from `stt-bench/`. A recent provider pull request is the best
template for step 2: it touches the same files listed here.

## 1. Before you start

- [ ] An API key for the provider, with enough credit for one full run (about 168
      minutes of public audio, plus the private recordings and turns) and a few
      retries. Note the account's concurrent-session limit.
- [ ] Access to the Vercel team and project in
      [config/vercel-models.json](config/vercel-models.json), a logged-in Vercel
      CLI, and the Sandbox SDK installed with `VERCEL_SANDBOX_SDK_DIR` pointing at it.
      Every published row was measured from the region set there, so new rows must
      run there too.
- [ ] The frozen inputs in `datasets/`: the public set from
      `stt-bench prepare-dataset --dataset pipecat-stt-benchmark`, and the private
      recordings and private turns restored from the team's run archive. The private
      data never goes into a commit, a pull request, or the website.

## 2. Add the model

- [ ] `src/stt_bench/<provider>.py`: the adapter. It streams 20 ms frames in real
      time, records every response with its timestamp, and states when a clip is
      finished (`completion_basis` in the config).
- [ ] `src/stt_bench/providers.py`: the model names the adapter accepts, its allowed
      hosts, validation, and replay.
- [ ] `src/stt_bench/catalog.py`, `src/stt_bench/credentials.py` and `.env.example`:
      register the model and the key's variable name.
- [ ] `config/models/<model-id>.json`: endpoint, audio format, finalization, a
      `model_source` link, and `pricing` with the vendor's public price page as
      `source` and the date you read it as `checked_on`. Use only letters, digits,
      `-` and `_` in the model id: sandbox names are built from it.
- [ ] `config/profiles/private-turns-v1/<model-id>.json` for the turn run, and the
      model id in `OPT_IN` in `src/stt_bench/turn_batch.py`. Add the provider's
      concurrency limit there if you know it.
- [ ] The model id in `FULL_MODELS` in `src/stt_bench/full_benchmark.py`, and in
      `turn_metrics.py` if the provider needs its own protocol for turn timing.
- [ ] `tests/test_<provider>.py` against a local WebSocket server, and the model in
      `tests/test_providers.py`. Run those two files:
      `uv run --locked pytest tests/test_<provider>.py tests/test_providers.py`.

## 3. Try it locally

- [ ] `uv run --locked stt-bench models` lists the model with its key set.
- [ ] `uv run --locked stt-bench prepare-check` passes.
- [ ] `uv run --locked stt-bench benchmark --dataset pipecat-stt-benchmark --model <model-id>`
      runs the ten smoke clips first and stops if any fails. Stop it after the smoke
      stage if you only want to check the adapter; this host's timing is not the
      published setup.

## 4. Run on Vercel

Finish and commit the adapter first. The frozen inputs include a hash of the source
code, so changing the code afterwards means freezing them again.

- [ ] `reports/vercel-models/preparation.json` names a ready base snapshot. Vercel
      deletes snapshots 30 days after they are taken; if it has expired, rebuild it
      with [prepare_vercel_models.mjs](scripts/prepare_vercel_models.mjs).
- [ ] Freeze the inputs:
      ```sh
      uv run --locked python -m stt_bench.full_benchmark prepare \
        --root reports/full-<name>-<date> --models <model-id> --workers 10
      uv run --locked python -m stt_bench.turn_batch prepare \
        --root reports/turns-<name>-<date> --models <model-id>
      ```
- [ ] Run the full set, then the turns. Never run them at the same time: they share
      the provider account's session limit.
      ```sh
      node scripts/run_vercel_full_parallel.mjs run reports/full-<name>-<date>
      node scripts/run_vercel_turns.mjs run reports/turns-<name>-<date>
      ```
      `status` in place of `run` shows progress. A run is finished when its
      `controller.json` says `complete`.

## 5. Keep the results

The run folders are the only copy of the per-clip transcripts and timings. The
website's examples, error mix and any later rescoring need them, and Vercel
deletes a run's sandboxes 30 days after they are created.

- [ ] Upload both run folders, whole, to the team's run archive, and add them to
      the archive's index.
- [ ] Do not commit run folders; `reports/` is ignored by Git.

## 6. Publish on the website

- [ ] Build the row:
      ```sh
      uv run --locked python scripts/site_row.py \
        --full reports/full-<name>-<date> --turns reports/turns-<name>-<date> \
        --model <model-id> > row.json
      ```
      `row` holds the word error rates, their 95% ranges, error counts and timing
      medians; `percentiles` holds the P50, P90 and P95 timings. The script scores
      the way the website always has, so the numbers line up with the published
      rows; they can differ slightly from this bench's own reports.
- [ ] In the website's `data/stt-benchmark.json`, add `row` to `benchmark.results`
      with `id`, `model` (the display name), `priceUsdPerHour`, `priceSource` and
      `priceNote`. Add `percentiles` under the top-level `percentiles` key, under the
      same id; the site build fails without it. Update the totals in `benchmark`
      (`models`, `firstAttempts`, `failed`).
- [ ] Optional examples of the model's misses, from the per-clip transcripts in the
      public set only.
- [ ] The model's vendor entry and logo, and a changelog entry with the new figures.
- [ ] Build the site locally before opening the pull request.
