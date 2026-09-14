# Offline benchmark dashboard

Generate the report with the system Python; no extra packages or network access
are needed. Run from `stt-bench/`:

```sh
python3 scripts/build_benchmark_html.py
```

Open `reports/benchmark-dashboard/index.html` in a browser, or share that single
file. Its data, styling, scripts, and benchmark charts are embedded. No source
reports, audio files, server, or internet connection are needed to view it.

## Inputs and refresh

The generator explicitly selects the ten non-Speechmatics final run summaries
under `reports/vercel-models/<run-id>/hourly/summary.json`. Nova-2 uses the corrected
`vocera-deepgram-nova-2-20260912-unformatted` run. This public comparison includes ten models; Speechmatics is not part of its
selected input set.

Rerun the generator to take a fresh snapshot of those saved inputs. It does not
download results, monitor jobs, modify inputs, or call a provider. The output
records its generation time, each source file's SHA-256, and source run IDs.
New runs require an explicit update to `MODELS` in the generator; the report
never silently substitutes the newest matching directory.

Optional `--source-root`, `--manifest`, `--private-source`, and `--out` arguments support copied
artifacts. They must still describe the same frozen Pipecat dataset and ten
selected runs. The default output is local and ignored by Git; the generator,
template, documentation, and tests are source-controlled files.

## Metric behavior

- Final WER is a ratio of total word errors to reference words, using the first
  valid completed result for each usable clip. It can include a retry.
- Deadline WER uses only the first attempt, at 0/250/500/1,000 milliseconds after
  speech end. Missing observations stay unavailable. Default deadline summaries
  retain invalid pacing as diagnostics and disclose their counts in CSV export.
- The page uses each model's usable final results and shows coverage in the table.
- The main comparison switches between final WER and median finalize latency.
  The scatter plot shows these same two metrics. Its timing axis retains the
  provider-acknowledgment definition and does not claim equivalent final-word timing.
- Highlights describe all ten models even when the legend hides individual models.
- Deadline observations, completion timing, reliability, and source identifiers
  remain available in CSV export without adding more panels to the page.
- Unknown prices remain unavailable. Recorded cost excludes smoke, setup, and
  Vercel compute. Confidence intervals and reviewed entity metrics are not shown.

The generator verifies complete status, ordered unique clip coverage, frozen
manifest identity, normalization, and the shared measurement contract. It
recomputes WER, deadline totals, timing percentiles, failures, and retry counts
and checks them against the source summaries before writing any HTML.

## Using the dashboard

Switch between accuracy and latency, toggle models in the legend, and sort the
results table. The scatter plot compares latency with accuracy. Metric definitions
and input sources are under "About this benchmark". CSV export follows the selected
models and table sorting, including missing values and source precision.

## Validation

From `stt-bench/`:

```sh
.venv/bin/python -m pytest tests/test_benchmark_html.py -q
node tests/benchmark_dashboard.browser.cjs
```

The browser check requires an installed Playwright package and Chromium. Set
`NODE_PATH` to the available package directory when Playwright is not installed
in the repository. It opens the actual HTML through `file://` with the browser
offline, verifies interactions and displayed values, downloads a filtered CSV,
and captures desktop/mobile screenshots beside the HTML. It makes no provider
calls. The Python tests use synthetic data to check report calculations.

## Separate private benchmark

The benchmark switch keeps the public Pipecat results separate from
`reports/private-longform-v1/comparison/comparison.json`. The private view shows
all 14 models and their recording coverage, WER, finalized-word delay p50/p95,
timed-word coverage, failed benchmark attempts, retries, and failed smoke attempts.
Its highlights only consider models with all eight valid recordings. The chart
switches between accuracy and finalized-word delay; missing scores stay unavailable.

The private timing definition differs from public finalize acknowledgment latency.
The views do not combine samples or compare their latency values. The private view
explains the four smoke failures, Nova-2's partial coverage, and unreviewed reference
annotations. Download CSV exports the active benchmark with its own metric labels
and source hash. Public filtering and sorting remain unchanged.

The generator validates private terminal status, model and recording counts, status
totals, saved archive-verification flags, and WER arithmetic. It embeds an explicit
allowlist of aggregate fields, without private transcripts, word annotations, or
audio. It reads saved evidence; it does not reverify remote jobs or raw archives.
The HTML remains a local offline artifact.

## September 14 refresh

The generator overlays completed recoveries from `reports/private-longform-recovery-v2`.
It checks terminal controller status, frozen manifest and configuration identity,
archive and raw-attempt checksums, coverage, and WER arithmetic. Nova-2 keeps its
seven earlier valid recordings and original first-attempt word timings. Pending
recoveries retain their prior results and are named in the dashboard status.

The **New · Short tests** tab reads the saved `trial-short-20260914/comparison.json`
and `limited-gradium-reson8-20260914/comparison.json` reports. It displays their
individual cohorts, attempted/usable coverage, WER, deadline WER, timing, and
retries. Its CSV includes source hashes and unrounded metrics. These small samples
remain separate from the full Pipecat and private long-recording comparisons.
