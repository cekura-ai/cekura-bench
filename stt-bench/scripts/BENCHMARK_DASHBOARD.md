# Unified English benchmark dashboard

Run from `stt-bench/` to build the offline page from locally saved evidence:

```sh
.venv/bin/python scripts/build_benchmark_html.py
```

The output is `reports/benchmark-dashboard-combined-v1/index.html`. Share or open that single
file for the summary: data, charts, styles, and scripts are embedded. Linked audio
and detailed turn proofs require the accompanying folders. No provider calls, downloads,
server, or internet access are needed. Reports are ignored by Git and are not
included in a fresh clone. `--reports-root` accepts a copied reports directory;
`--out` changes the destination.

## Latest turn results and proof

When `reports/private-turns-consolidated-20260915/final/` is present, the builder
adds three views: **Latest results**, **Overall benchmark**, and **Proof & methodology**.
Latest results shows all 19 included turn models, each with 206 attempted clips
from four conversations. The overall score appears beside turn WER;
its fixed comparison set and weighting stay unchanged. The same calls are
not counted a second time in that overall score.

The overview supports search, sorting, p50/p90/p95 latency, and separate first-attempt
and after-recovery views. Each model's proof panel includes valid observation
counts, failures, deadline accuracy, recovery outcomes, timing statuses, and
its effective completion settings. Google Chirp's observed delay is separate
from controlled TTFS. Automatic clip selection and the lack of independent
listening review are disclosed in the methodology.

The builder verifies saved totals, per-clip timing percentiles, receipt hashes,
dataset identity, and completion checks. It writes a portable `turn-proof/`
folder with all 206 audio clips, per-model listening pages, exact result JSON,
compressed original provider logs, session metadata, and hash indexes. Original
benchmark evidence is not modified and no provider is called. The proof package
contains private audio and transcripts whenever the final turn report is present;
the older `--include-private-review` flag controls only the whole-recording review.

To share the complete integrated dashboard, use:

```sh
.venv/bin/python scripts/build_benchmark_html.py --include-private-review --share-zip
```

Extract the entire `benchmark-review.zip` and keep both `audio/` and `turn-proof/`
beside `index.html`. The dashboard and evidence work offline.

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
- The completed, audited `inworld-full-stream-end-v2-20260915` rerun replaces
  Inworld's older `full-parallel-20260915` observations in every score, timing,
  chart, table, and listening transcript. It covers the same 1,000 public clips
  and eight private recordings with corrected stream ending.
- The completed `linden-full-20260915` run has 998 usable public clips and all
  eight private recordings. Its archived logs and merged scores pass the local
  audit; its saved controller records stopped workers. Two missing ranking
  items leave Linden unranked rather than shrinking the common comparison set.
- Sarvam and Soniox's Pipecat trial observations, marked small-trial and unranked.
  Private excerpts are excluded from the full-recording suite. Trial Pipecat
  results for models with full runs are not counted again.

Each selected source has a SHA-256 in the page. The generator checks source
identities, unique item coverage, public dataset consistency, normalization,
word-count arithmetic, private archive-verification flags, and full-run audit
receipts. It validates the original totals, then recomputes WER with punctuation-only
tokens removed on both sides and calculates percentiles from saved observations.
The corrected totals and per-item before/after audit are written separately to
`reports/offline-rescore-combined-v1/`; see [the correction guide](../OFFLINE_RESCORING.md).
The historical Inworld diagnostic additionally requires its old public raw archives under
`full-parallel-20260915/batches/`; raw receipt checksums are verified locally.
Those diagnostic receipts do not supply the current Inworld score; the verified
full rerun listed above does.
It does not re-download archives or repeat the underlying provider benchmarks.
Without the final turn report described above, the default summary export contains
no private transcript, audio, or individual word annotation. Whole-recording
listening review is an explicit export option described below.

## Share a listening review

For a review containing all 1,180 public clips plus the eight private recordings:

```sh
.venv/bin/python scripts/build_benchmark_html.py --include-private-review --share-zip
```

Send `reports/benchmark-dashboard-combined-v1/benchmark-review.zip`. The recipient extracts
the entire ZIP and opens `index.html`; the `audio` folder must stay beside it.
Everything works offline. The whole-recording assets are about 270 MB, including private
audio and transcripts. Sending the HTML alone preserves transcripts and diffs,
but audio requires the accompanying files. Nothing is uploaded by the build.
For public whole-recording clips only, use `--with-clip-review --share-zip` instead;
this does not exclude the separate private turn proof package when its final report
is available. Rebuilding without either review flag removes the whole-recording
listening interface.

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

The main table shows **Failure rate**, and the comparison chart includes a
failure-rate view. The separate **Not run** column has been removed; those
counts remain in dataset details and CSV evidence. Every model
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

Flux Multilingual, GPT Realtime Whisper, GPT-4o Mini Transcribe, and Chirp 2
are restored to the overall view. Chirp 2 and Chirp 3 now use their completed
1,000-clip public summaries collected from the original sandboxes, alongside
their eight verified private recordings. Their public/private counts and
normalized transcripts are checked during the build. The sandboxes were
stopped again after reading these files; no provider calls were started.

AssemblyAI normal and minimum-latency profiles are separate. The normal
profile's saved full public run has zero scored clips, and Sarvam has a
10-public-clip pilot with no full private run. Their overall WER stays
unavailable with an explicit explanation; their 206-turn scores stay visible.

Nova-2 remains excluded. The restored models appear beside the existing
models; full-run eligibility is determined from saved coverage, rather than
an old display filter. Speechmatics Standard, Enhanced, and Linden remain
separate profiles. Source benchmark artifacts and model configurations remain
unchanged.

## Provisional combined ranking

The default **Combined · available results** chart pools each model's usable
Pipecat results and selected private recordings. It requires usable results in
both datasets, excludes FLEURS, and shows coverage and exclusions. Linden is
visible with 1,006 included recordings and two public failures. The separate
**Combined · fixed ranking set** preserves the original comparison set and
scoring rules; ranks are recomputed when eligible models are restored.

At the user's request, Inworld's selected private accuracy excludes
`conversation-04-B` because of excessive repeated output. The selected result is
3.95% across seven of eight recordings; its combined available result is 3.10%
across 1,000 public clips and seven private recordings. This is an explicitly
selected subset, not a corrected result for all eight. Original cohorts,
transcripts, fixed-set ranks and timing observations remain intact. The full
eight-recording private result is still 10.56% and remains visible in details.

The chart offers **Public Pipecat · all available** for every model's usable
results from the 1,000 planned public clips, including Linden's 998 usable clips.
Each bar shows coverage and failed clips. **Shared public clips · fixed set**
remains a separate comparison; models missing a required clip have no score there.
The all-available chart and scatter plot do not require a combined rank. Coverage
can differ, so they are not a replacement for the fixed-set ranking.

The suite has 1,000 Pipecat clips, 180 FLEURS clips, and eight private recordings.
Eligible models use exactly the same 864 public clips plus all eight
private recordings: 20,865 public and 12,555 private reference words. Combined
WER is total substitutions, insertions, and deletions divided by 33,420 words.
FLEURS does not contribute. Each word has equal weight; dataset percentages are
not averaged. The frozen IDs and per-item word counts are versioned in
`config/rankings/combined-public-private-v1.json`.

A missing or unusable ranking item makes that model unranked; it does not shrink
the set. UI filtering cannot change scores or rank. All-available public WER,
common-public WER, private WER, and their coverage remain separately visible.
The page now contains the restored full-run models alongside profiles with incomplete coverage; ranks follow the fixed-set eligibility checks. The shared public subset
excludes 136 clips, so accuracy must be read alongside reliability and coverage.

The private files are eight speaker tracks from four conversations. Their 37.6%
influence reflects word counts, not a validated production traffic mix. Pipecat
currently describes its references as Gemini-generated and human-reviewed;
review coverage for the exact frozen revision remains unverified here. Private
reference provenance and independent listening review also remain unresolved.
See [the benchmark assessment](../BENCHMARK_ASSESSMENT.md) for sensitivity checks
and prioritized follow-up work. Eventual WER alone is not a voice-agent ranking.

## Timing

All four timing views show p50, p90, p95, and p99, calculated using linear
interpolation over individual observations. The chart has a percentile selector;
the table always shows all four percentiles and the measurement denominator.

- **Last final text received after speech end:** last final-text receipt minus
  speech end, across valid first attempts on Pipecat only. Later extra text can
  extend this measurement. This is not finalize-acknowledgment latency.
  Negative values mean final text arrived before the reference speech boundary.
- **Interim after speech end:** first nonempty partial update received after speech
  end. This saved metric is not time to first interim text from speech start.
  Missing partial observations are unavailable, not zero and not proof that a
  provider lacks interim support.
- **Finalized-word delay:** correctly aligned private word receipt minus the
  actual delivery time of that word's ending audio. Private words are never
  pooled with public clip timings. The table shows timed/reference words.
- **Stream completion:** provider completion receipt minus speech end, for valid
  first Pipecat attempts. Provider completion signals vary.

Accuracy may use a valid retry. Timing never substitutes a recovery for the
original first attempt. Public deadline WER includes pacing-valid first-attempt
Pipecat observations at 0/250/500/1,000 ms, with measurement counts shown.
These deadline scores use available observations, not the common ranking set.

Public latency charts and scatterplots separate signals sent at speech end from
native endpointing/stream closure after the tail. Inworld sends `endTurn` and
Gradium sends `flush` at speech end. No silence duration is subtracted. Private
word timing preserves the original saved word mapping and denominator. The
Inworld evidence note does not report an unverified hallucination rate.

The page explains differing run dates, concurrency, session handoffs, AssemblyAI
packet timing, automatic reference boundaries, and incomplete listening review.
It does not invent confidence intervals or a composite accuracy/speed score.

## Interface and exports

The leaderboard lets users switch accuracy/timing charts, select the
timing definition and percentile, hide models, sort the table, and inspect
accuracy versus timing. Dataset details and methodology use expandable sections.
Charts and wide tables scroll internally on mobile.

CSV exports the visible sorted models, combined WER and integer counts, ranking
version and basis, common-public WER and denominator, scoring version, all timing metrics,
coverage, reference words, deadline scores, failures, retries, and source hashes.
Values retain source precision. Missing values are explicitly `Unavailable`.

## Validation

The model comparison graph has an accuracy dataset selector for combined, shared-public,
and private WER. Nova-3 and ElevenLabs use the same saved scores as the leaderboard;
there is no separate comparison card. Hiding models does not change scores or ranks.
The separate September 15 settings-trial note is diagnostic context;
it does not replace any leaderboard score or timing observation.

```sh
.venv/bin/python -m pytest tests/test_combined_ranking.py tests/test_unified_benchmark.py tests/test_benchmark_html.py -q
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

Actual cost is omitted from the page and CSV. Saved data reserves actual cost in
USD for charges verified against provider billing records and attributed to each
model’s benchmark runs.
This includes smoke tests, billable failed requests, retries, and reruns; hosting
costs are separate. No reconciled billing evidence is currently available, so
`actual_cost_usd` is null and `actual_cost_status` is `unverified` for every model.
Submitted audio and configured list-rate estimates must not populate this field.
