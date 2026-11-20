# Independent Codex failure QC

Reviewed all 92 overall-failed calls across 49 scenarios, using failed-metric explanations, transcript windows,
all tool calls/results, scenario targets and rubrics. Private source material is not included here.

- 68 calls contain at least one supported failure under the benchmark rubric.
- 24 calls across 16 scenarios have no confidently confirmed main-agent failure.
- 50 calls have at least one questionable flag; some also have a separately supported failure.

These counts describe attribution, not corrected pass/fail grades. No metric, call selection or score was changed.

The strongest checker questions include H03's accepted appointment-change workflow being rejected in favour
of cancel-and-rebook, pause verdicts contradicting their own explanations, a truthful tool-failure/retry sequence
being flagged, corrected-name readback being treated as action completion, and answered date interruptions
being reported as ignored. Consent and cancellation verdicts also warrant rubric/clock review.

Numeric pronunciation, hold alignment and speech-stop timing require acoustic verification. No independent
listening or distinct-voice identity verification was performed; speaker attribution follows scenario mapping.
A successful repeat alone would not invalidate the original verdict. Outcome-triggered diagnostic repeats
must be reported separately from the selected 210-call benchmark cohort.

Claude's earlier review covered 41 calls, including all 30 task-completion failures. It did not deeply review
all 92 overall-failed calls. The earlier review and the Codex review both leave the raw grades unchanged.
