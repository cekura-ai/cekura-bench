# Unified English benchmark dashboard

Run from `stt-bench/` to build the offline page from locally saved evidence:

```sh
python3 scripts/build_benchmark_html.py
```

The output is `reports/benchmark-dashboard/index.html`. Share or open that single
file: data, charts, styles, and scripts are embedded. No provider calls, downloads,
server, or internet access are needed. Reports are ignored by Git and are not
included in a fresh clone. `--reports-root` accepts a copied reports directory;
`--out` changes the destination.

## Explicit source selection

`scripts/unified_benchmark.py` selects the following evidence, rather than using
file modification times or live controller progress as a score:

- Eight selected completed Pipecat summaries from `reports/vercel-models`.
- Reconciled private results from
  `reports/private-longform-recovery-v2/comparison/comparison.json`, including
  original first-attempt timing and preserved failed attempts.
- The four completed models in `reports/full-parallel-20260915/results.json`:
  Smallest, Gradium, Reson8, and Inworld. Its saved evidence audit must have passed
  and its compute-stop receipt must confirm completion.
- Nova-3's 180 FLEURS clips from `reports/deepgram-public-v3-final/results.json`.
- Gradium and Reson8's 15 FLEURS clips each from the two FLEURS cohorts under
  `reports/limited-gradium-reson8-20260914`.
- The completed AssemblyAI run from
  `reports/assemblyai-min-latency-full-20260915/results.json`: all 1,000 Pipecat
  clips and eight private recordings. It receives a rank only after the saved
  execution status, evidence audit, and compute-stop receipt confirm completion.
  An incomplete saved snapshot remains visible and unranked.
- Sarvam and Soniox's Pipecat trial observations, marked small-trial and unranked.
  Private excerpts are excluded from the full-recording suite. Trial Pipecat
  results for models with full runs are not counted again.

Each selected source has a SHA-256 in the page. The generator checks source
identities, unique item coverage, public dataset consistency, normalization,
word-count arithmetic, private archive-verification flags, and full-run audit
receipts. It recomputes pooled WER and percentiles from saved item observations.
It does not re-download archives or repeat the underlying provider benchmarks.
The default summary export contains no private transcript, audio, or individual
word annotation. Listening review is an explicit export option described below.

## Share a listening review

For a review containing all 1,180 public clips plus the eight private recordings:

```sh
.venv/bin/python scripts/build_benchmark_html.py --include-private-review --share-zip
```

Send `reports/benchmark-dashboard/benchmark-review.zip`. The recipient extracts
the entire ZIP and opens `index.html`; the `audio` folder must stay beside it.
Everything works offline. The current bundle is about 270 MB, including private
audio and transcripts. Sending the HTML alone preserves transcripts and diffs,
but audio requires the accompanying files. Nothing is uploaded by the build.
For public clips only, use `--with-clip-review --share-zip` instead. Rebuilding
without either review flag returns the HTML to the summary-only version.

The Verify clips section has dataset/model selection, reference-text and clip-ID
search, previous/next navigation, an errors-only filter, and sorting by WER.
It shows original text or the exact normalized word alignment from the scorer.
Changed, missing, and extra words have distinct highlights and a numerical
breakdown. Zero-reference-word clips have undefined individual WER; their error
counts still contribute to corpus totals. Selected recovery attempts, excluded
transcripts, and missing results are labeled separately.

The generator verifies frozen audio hashes, compresses the prepared 16 kHz WAVs
to lossless FLAC, then compares every decoded sample with the input. This
preserves the added silence; some providers used 24 kHz derivatives or session
handoffs. It recomputes alignments with the existing scorer and rejects mismatched
text, counts, model coverage, or source hashes. Each clip shows its source and
audio hashes. Private original annotation text is available in an expandable
section. Listening verifies transcript content, not network latency.

The ZIP contains only this build's selected files, so a later public-only ZIP
does not accidentally include old private audio from the output directory.

## Failure rate on a fixed denominator

The main table replaces varying clip-count columns with **Failure rate** and
**Not run**, and the comparison chart includes a failure-rate view. Every model
uses the same 1,188 planned recordings as the denominator:

- Failure rate = attempted recordings with no usable result after retries / 1,188.
- Not run = recordings with no saved attempt / 1,188.
- Missing result rate, also in CSV = (failed + not run) / 1,188.

Failures include transport errors, pacing-validation failures, and other reasons
an attempted recording was unusable. They are not all provider-caused errors.
A recovered recording counts once as usable; failed-attempt and retry counts are
retained separately. A model with no attempts has an unavailable failure rate.
Low failure with high Not run is incomplete evidence, not strong reliability.
WER continues to use usable transcripts only. Dataset details retain the usable,
failed, and not-run counts; adding a metric does not create missing observations.

## Model selection

The page retains Flux English and Nova-3 as distinct Deepgram families, and one
OpenAI entry: GPT-4o Transcribe. It removes Nova-2, Flux Multilingual, Whisper,
GPT-4o Mini, and Chirp 2. Speechmatics Standard and Enhanced remain separate
operating modes. This is an explicit display selection, not a live model-catalog
claim. Source benchmark artifacts and model configurations remain unchanged.

## Combined ranking and coverage

The suite has 1,000 Pipecat clips, 180 FLEURS clips, and eight private recordings.
Combined WER is the sum of substitutions, insertions, and deletions divided by
reference words across all usable items. Each reference word has equal weight;
dataset percentages are never averaged.

The ordinal ranking includes terminal full runs with every Pipecat and private
item attempted and usable results in both datasets. FLEURS scores enter the same
combined total wherever available. FLEURS coverage and exclusions differ across
models, so this is an **available-results ranking**, not a matched-sample ranking.
The page shows usable/planned counts for every dataset, plus failures and retries.
It keeps Chirp 3 and the two trial models visible and unranked. AssemblyAI is now
included among the 13 completed full-run rankings; it has no FLEURS observations.
Unavailable measurements contribute neither errors nor reference words.

## Timing

All four timing views show p50, p90, p95, and p99, calculated using linear
interpolation over individual observations. The chart has a percentile selector;
the table always shows all four percentiles and the measurement denominator.

- **Final transcript:** last final-text receipt minus speech end, across valid
  first attempts on Pipecat and FLEURS. This is not finalize-acknowledgment latency.
  Negative values mean final text arrived before the reference speech boundary.
- **Interim after speech end:** first nonempty partial update received after speech
  end. This saved metric is not time to first interim text from speech start.
  Missing partial observations are unavailable, not zero and not proof that a
  provider lacks interim support.
- **Finalized-word delay:** correctly aligned private word receipt minus the
  actual delivery time of that word's ending audio. Private words are never
  pooled with public clip timings. The table shows timed/reference words.
- **Stream completion:** provider completion receipt minus speech end, for valid
  first Pipecat/FLEURS attempts. Provider completion signals vary.

Accuracy may use a valid retry. Timing never substitutes a recovery for the
original first attempt. Public deadline WER includes pacing-valid first-attempt
observations at 0/250/500/1,000 ms, with measurement counts shown.

The page explains differing run dates, concurrency, session handoffs, AssemblyAI
packet timing, automatic reference boundaries, and incomplete listening review.
It does not invent confidence intervals or a composite accuracy/speed score.

## Interface and exports

The leaderboard lets users switch accuracy/timing charts, select the
timing definition and percentile, hide models, sort the table, and inspect
accuracy versus timing. Dataset details and methodology use expandable sections.
Charts and wide tables scroll internally on mobile.

CSV exports the visible sorted models, all four timing metrics and percentiles,
coverage, reference words, deadline scores, failures, retries, and source hashes.
Values retain source precision. Missing values are explicitly `Unavailable`.

## Validation

```sh
.venv/bin/python -m pytest tests/test_unified_benchmark.py tests/test_benchmark_html.py -q
node tests/benchmark_dashboard.browser.cjs
.venv/bin/python -m pytest tests/test_benchmark_clip_review.py -q
node tests/benchmark_clip_review.browser.cjs
```

The Python tests use synthetic observations to check pooling, tail percentiles,
missing data, duplicates, retry semantics, and private-data exclusion. The browser
check uses the actual HTML, offline, at desktop and mobile sizes. It verifies all
four timing views, displayed values, model selection, charts, sorting, tooltips,
CSV download, and empty states. It saves desktop/mobile screenshots beside the
HTML. Set `NODE_PATH` to an installed Playwright package directory if needed.

The legacy reducer functions in `build_benchmark_html.py` remain covered by their
existing tests; the default command now calls the unified reducer.

## Actual cost

The combined leaderboard and CSV reserve actual cost in USD for charges verified
against provider billing records and attributed to each model’s benchmark runs.
This includes smoke tests, billable failed requests, retries, and reruns; hosting
costs are separate. No reconciled billing evidence is currently available, so
`actual_cost_usd` is null and `actual_cost_status` is `unverified` for every model.
Submitted audio and configured list-rate estimates must not populate this field.
