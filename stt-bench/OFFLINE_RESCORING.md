# Offline scoring correction

The dashboard uses `english-wer-v2-punctuation-tokens`. It removes whole
punctuation-only tokens after the pinned English normalizer, on both reference
and hypothesis. The original provider text, audio, event logs and source reports
are preserved. Decimal numbers, currencies and punctuation inside words are not
stripped by this additional step. Entity scoring is unchanged.

## Reproduce

From this repository, with the existing environment and downloaded evidence:

```sh
.venv/bin/python scripts/build_benchmark_html.py --with-clip-review --include-private-review --share-zip
```

This command makes no provider calls. It rebuilds the local dashboard and its
review ZIP. The ZIP includes the existing eight private recordings when
`--include-private-review` is supplied; omit that flag for a public-only review.
It does not deploy the page or change provider settings.

The new reports are saved separately:

- `reports/offline-rescore-combined-v1/results.json`: corrected dataset totals, common
  public scores, fixed ranking set, latency methods and source hashes.
- `reports/offline-rescore-combined-v1/audit.json`: before/after totals, corrected per-item
  and deadline alignments, code hashes, unchanged-evidence fingerprints, and
  checksum-matched Inworld receipt diagnostics. It includes normalized private
  transcript text and should be handled like the original private evidence.
- `reports/benchmark-dashboard-combined-v1/index.html` and `benchmark-review.zip`: the local
  presentation and portable listening review.

Original report totals are validated before correction. Every saved alignment
is independently checked against its stored normalized text. The correction
then aligns the cleaned text again; it does not subtract estimated errors.
Already-normalized numbers are not normalized a second time. Listening review
also checks that the original normalized text matches the raw reference and
selected provider transcript. Source files are hashed again before output.

## Score and timing definitions

The 13 eligible models are ranked on **864 frozen shared public clips plus all
8 private recordings**. The ranking adds integer substitutions, insertions and
deletions across both datasets, then divides by **33,420 reference words**:
20,865 public (62.4%) and 12,555 private (37.6%). Every reference word has equal
weight; dataset percentages are not averaged. FLEURS is excluded.

`config/rankings/combined-public-private-v1.json` pins the item IDs and each item's
reference-word count, with the source public-only result hash. A missing or
unusable ranking item leaves that model unranked; it never shrinks the comparison
set. Incomplete models remain visible with their available dataset measurements.
Hiding models changes neither membership nor rank. Exact score ties use model ID
as a deterministic secondary sort, not evidence of a quality difference.

Result schema version 3 adds `ranking` (version, basis, IDs, denominators and
manifest hash), `models[].ranking_score` (pooled counts, WER, dataset coverage),
and `public_rank`. The existing `headline` field remains common-public WER.
`rank` now means combined rank. The legacy `combined` field remains the old
all-available aggregate, including FLEURS, for compatibility; it must not be used
for the new ranking. Dashboard and CSV explicitly use `ranking_score`.

The original `reports/offline-rescore-v2/` artifacts remain unchanged. Before
refreshing the previous dashboard path, its public-only HTML is preserved locally
as `reports/benchmark-dashboard/index-public-v2.html`. The default build writes the new dashboard directory
above and a separate correction audit. No existing portable ZIP is overwritten.

Public all-available, common-public, private and FLEURS component scores remain available with coverage. Only the frozen public and private items contribute to combined rank.
The gap is private WER minus all-available public WER, expressed in percentage
points. Differences in acquisition, domain, duration and session splitting can
contribute to this gap. The frozen public set was selected from clips usable by every originally ranked
model; this does not establish whether those clips are harder. Read it alongside
reliability: the existing failure/not-run rates remain visible.

Final WER keeps the existing selected attempt. Public timing keeps eligible
first attempts, excludes FLEURS, and never substitutes recovery timing. Public
deadline WER uses the original pacing-valid first-attempt observations from
Pipecat, not the common ranking subset. All retained deadline alignments,
including FLEURS, are corrected in the audit.

“Last final text received after speech end” uses the original receipt timestamp.
Later extra text can extend it. Plots separate providers sent a finalization
signal at speech end from native endpointing/stream closure after the tail.
Inworld and Gradium belong to the signal-at-speech-end group. No silence is
subtracted. Private word timing retains the original saved word mapping,
timestamps and reference-word denominator; it is separate from corrected WER.

The Inworld note is supported by checksum-matched selected public JSONL logs.
It describes later text after stream closure and repetition already present
before speech end. It does not claim a causal effect of silence or profiling,
or report an unverified hallucination rate. Genuine inserted words remain errors.

## Expected checks

- ElevenLabs all-available public: **727 → 413 errors**, **23,065 words**,
  **3.15% → 1.79% WER**.
- ElevenLabs common-public: **385 / 20,865 = 1.85%**, public rank **3** (combined rank **6**).
- Original and corrected non-score evidence fingerprints match. Attempt
  selection, transcript text, failure classifications, timestamps and gates
  remain unchanged.

Run `.venv/bin/python -m pytest -q` from the repository root. Some transport tests
start local mock servers and need loopback access; they do not use real providers. Browser checks are
`tests/benchmark_dashboard.browser.cjs` and
`tests/benchmark_clip_review.browser.cjs`; they use Playwright in offline mode
and check the actual generated page, CSV, transcript highlights and playback.

## Reference provenance and follow-up

Pipecat’s current documentation describes Gemini-generated, human-reviewed
ground-truth transcripts:
https://github.com/pipecat-ai/stt-benchmark/blob/main/README.md . The review coverage for our exact frozen
revision has not been verified here. Matching the
reference generator's wording can favor a model family, but this has not been
established as the cause of Gemini's score. Private reference generation has not
been independently verified here. References and word boundaries have not had
independent listening verification. Pooling datasets does not eliminate this risk.

Follow-up: establish reference provenance for the exact frozen source revisions,
then review a sample by listening with provider names hidden. Record adjudicated
corrections in a new reference version and rescore saved hypotheses uniformly.
This review has not been performed by this change.

## Future provider settings

The opt-in settings and offline validation are documented in
`config/profiles/finalization-v1/README.md`. Original configurations and the saved
pilot remain unchanged. The new settings are not applied to historical timing
labels. No provider calls or deployment are needed to rebuild the combined report.

See [the benchmark assessment](BENCHMARK_ASSESSMENT.md) for the provisional
interpretation, sensitivity checks, and prioritized follow-up work.
