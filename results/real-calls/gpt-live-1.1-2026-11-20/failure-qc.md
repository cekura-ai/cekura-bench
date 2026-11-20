# GPT-Live 1.1 + GPT-6 Sol: failure QC

The corrected cohort has **82 failed calls across 47 scenarios**. All 210 selected executions and recording checks pass. [Structured QC](failure-qc.json).

The exported platform rubrics run in the private local runner. Timing, hold, ASR and tool-wait input defects were repaired; all 6,300 metric entries are evaluated. Every affected input change is recorded in the report and results metadata. Source attempts and their original metric files remain preserved.

Claude Opus 5.5 HIGH and Codex reviewed all 36 prosody-targeted and 39 multi-person-targeted calls. The prosody audit reproduced lost agent confirmations, late caller endpoints and conflicting log/transcript clocks. The repaired transcript path yields 28/28 eligible passes; eight targeted cases remain not reached.

The multi-person audit corrected two single-voice target exclusions and a missed executed second voice, giving 17/39 passes. Supported failures include acting on an unverified second speaker's go-ahead. H02 privacy scope, H07 adjacent questions/branch misfires, F30 authority interpretation and H27 holder-confirmation handling remain ambiguous. H30 also has a post-commit prosody scope conflict. Historical comparison calls contain some grading inconsistencies and have not been regraded, so exact judge parity is unconfirmed.

The 77 verification reruns completed, passed recording/model/prompt/tool checks and were graded without remaining judge errors. The three generic wait/resume prompt pilots remain separate experiments. F29's same-prompt repeat still showed conversational resumption trouble. The F09 status-lookup fixture was repaired for diagnostics; the existing main cohort is preserved.

Review used transcripts, tool outcomes, playback logs, agent logs, waveform measurements and machine audio judgments. Independent human listening was not performed. Private recordings/transcripts remain in the workspace. Do not interpret the remaining failed-call count as a count of fully confirmed model errors.
