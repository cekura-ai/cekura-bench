# GPT-Live 1.1 + GPT-6 Sol: failure QC

The final cohort contains 88 overall-failed calls across 47 scenarios.
Codex reviewed the failure evidence against the configured rubrics. Claude Opus 5.5 HIGH reviewed
targeted calls and 13 diagnostic calls, using transcripts, tool outcomes, caller playback logs,
agent utterance logs and per-channel recording timing. [Structured QC](failure-qc.json).

All 13 diagnostic calls completed and were graded, with no agent/tool errors or judge errors.
Their lossless recordings were verified. Diagnostic concurrency was capped at 15 with paced starts.

Supported failure patterns include numeric read-back errors, speaking over unfinished caller digits,
incorrect value confirmation, and failure to complete the requested tool workflow.

The final metric results follow the configured rubrics: permitted bad-line repetition, courtesy-title
exceptions, actual hold timing and answers to interruptions are evaluated within their stated scope.
A corrupted caller moment is excluded from the affected skill evaluation.

Remaining limits:

- No independent listening or speaker-identity verification was performed.
- Speech-stop times near the two-second limit remain uncertain; waveform energy alone is insufficient.
- H03 uses the configured task workflow checker.
- Some F05/F26 caller branches were not reached; task interpretation requires care.
- Language conclusions from model-reported transcript fragments do not establish audible speech.

Recordings, transcripts, task values and private evidence remain in the task workspace.
