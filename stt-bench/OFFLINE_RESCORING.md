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

- `reports/offline-rescore-v2/results.json`: corrected dataset totals, common
  public scores, fixed ranking set, latency methods and source hashes.
- `reports/offline-rescore-v2/audit.json`: before/after totals, corrected per-item
  and deadline alignments, code hashes, unchanged-evidence fingerprints, and
  checksum-matched Inworld receipt diagnostics. It includes normalized private
  transcript text and should be handled like the original private evidence.
- `reports/benchmark-dashboard/index.html` and `benchmark-review.zip`: the local
  presentation and portable listening review.

Original report totals are validated before correction. Every saved alignment
is independently checked against its stored normalized text. The correction
then aligns the cleaned text again; it does not subtract estimated errors.
Already-normalized numbers are not normalized a second time. Listening review
also checks that the original normalized text matches the raw reference and
selected provider transcript. Source files are hashed again before output.

## Score and timing definitions

The 13 previously eligible models are ranked on the intersection of their usable
public clips: **864 clips and 20,865 reference words** in this saved snapshot.
Hiding a model in the interface does not recalculate the intersection or rank.
An empty intersection produces no score or rank. Models without full public
results stay unranked.

Public all-available, private and FLEURS scores remain separate, with coverage.
The gap is private WER minus all-available public WER, expressed in percentage
points. Differences in acquisition, domain, duration and session splitting can
contribute to this gap. A common-clip ranking excludes difficult and failed
cases for everyone; the existing failure/not-run rates remain visible.

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
- ElevenLabs common-public: **385 / 20,865 = 1.85%**, rank **3**.
- Original and corrected non-score evidence fingerprints match. Attempt
  selection, transcript text, failure classifications, timestamps and gates
  remain unchanged.

Run `.venv/bin/python -m pytest -q` from the repository root. Some transport tests
start local mock servers and need loopback access; they do not use real providers. Browser checks are
`tests/benchmark_dashboard.browser.cjs` and
`tests/benchmark_clip_review.browser.cjs`; they use Playwright in offline mode
and check the actual generated page, CSV, transcript highlights and playback.
