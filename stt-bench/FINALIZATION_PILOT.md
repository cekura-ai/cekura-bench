# Finalization settings pilot

This is a small live comparison of the existing settings and proposed settings.
It is separate from the completed offline scoring correction and never updates
the leaderboard or its frozen 864-clip intersection.

## What changes

| Model | Baseline | Candidate |
| --- | --- | --- |
| AssemblyAI Universal 3.5 Pro | Native turns, Terminate after the silence tail | Also send ForceEndpoint at speech end |
| Speechmatics Standard and Enhanced | Default delay; EndOfStream after the tail | Send ForceEndOfUtterance at speech end and set max_delay to 1 second |
| Inworld STT 1 | endTurn at speech end, voice profiling enabled | Same endTurn; voice profiling disabled |

The normal model configuration files remain unchanged. Optional adapter settings
allow the pilot to write its own configurations. Speechmatics changes two settings
together, so the comparison cannot attribute a difference to either setting alone.
Its max_delay_mode remains the provider default.

**AssemblyAI scope limitation:** the pilot selected the default streaming profile
`assemblyai-universal-3-5-pro`. The current dashboard uses the separate
`assemblyai-universal-3-5-pro-min-latency` profile, which also sets `mode=min_latency`
and an English language bias. Therefore this pilot can test ForceEndpoint on the
default profile, but cannot establish its effect on the published profile. Do not
promote its numbers or recommend a full rerun of the published profile based on it.

AssemblyAI has no separately identified acknowledgment for ForceEndpoint in this
adapter. Sending the command does not prove that all text has arrived. Speechmatics
acknowledgment requires an EndOfUtterance event with forced=true after the request.
All profiles still send the original one-second digital-zero tail and collect
the provider's terminal response. No timestamps are shifted or tail-subtracted.

Official protocol references:

- [AssemblyAI ForceEndpoint and Terminate](https://www.assemblyai.com/blog/raw-websocket-voice-agent-with-assemblyai-universal-3-pro-streaming)
- [Speechmatics ForceEndOfUtterance and max_delay](https://docs.speechmatics.com/api-ref/realtime-transcription-websocket)
- [Inworld voice profile configuration](https://docs.inworld.ai/stt/voice-profiles)

## Selection and execution

The pilot freezes 19 public clips selected by a deterministic hash, from clips with
submitted duration above 1 and at most 20 seconds. It adds the known Inworld
repetition example as the twentieth clip. This deliberate diagnostic example must
be identified when interpreting aggregate results; this sample is not a general
accuracy ranking.

Every model receives the same 20 clips under both configurations. Each clip gets
one attempt per configuration, for at most 160 sessions. Baseline and candidate
run next to each other; which one runs first alternates by clip. Speechmatics
Standard and Enhanced run serially to respect their shared concurrency limit.
AssemblyAI and Inworld have separate concurrent workers. Every attempt retains
the existing timing gates; a failed gate does not become a valid observation.

Before any provider connections, the host must pass three independent 10-second
local timing trials. Every capture records raw responses, audio send timing,
configuration, source hashes and the finalization request. Provider or transport
errors stop the affected model without retries. An exclusive live-start marker
prevents accidentally dispatching a duplicate run after an uncertain interruption.

The September 15 local preflight failed before any provider connections. Its
evidence is retained in `reports/finalization-pilot-20260915-v1/`. The same frozen
selection was then prepared for Vercel in
`reports/finalization-pilot-20260915-vercel-v2/`. Live provider tests run on Vercel,
as requested. The launcher uses the existing benchmark team/project and verified
public dataset snapshot, uploads only code and public pilot inputs, and allows
network access only to the three provider endpoints during capture. Credentials
are passed as command environment values and are excluded from file uploads.

```bash
.venv/bin/python scripts/finalization_pilot.py prepare \
  --out reports/finalization-pilot-20260915-v1
# Live execution requires an explicitly authorized provider experiment.
.venv/bin/python -u scripts/finalization_pilot.py live \
  --out reports/finalization-pilot-20260915-v1
```

For the Vercel run, use `scripts/run_vercel_finalization_pilot.mjs` with
`--action launch`, `status`, or `collect`, and
`--out reports/finalization-pilot-20260915-vercel-v2`.
Set `VERCEL_SANDBOX_SDK_DIR` to an existing installed `@vercel/sandbox` package.
Launch creates a sandbox with a 45-minute timeout, verifies inputs, runs the
adapter tests there, and dispatches the bounded live command. Status is read-only.
Collection checks the evidence archive hash and stops the sandbox. Neither
status nor collection can dispatch another provider run.

### Capture scheduling correction

The first Vercel capture used three workers in one Python event loop. Although
the standalone timing preflight passed, several live streams failed send timing
gates. Shared synchronous scoring/checkpoint work was a suspected source of
delayed sends. That command was interrupted; its raw files remain unchanged.

`scripts/continue_finalization_pilot.py` gives each provider its own Python
process. It performs another timing qualification and submits only sessions
without an original raw file. It copies original evidence into reconciliation
directories for scoring, preserving interrupted sessions as failures. It never
retries them. Speechmatics models still run serially. Results identify each
observation as `original` or `isolated`, so the scheduling phases can be reviewed
separately. This continuation stays inside the original 160-session budget.

The prepared plan pins all inputs and relevant source code. Do not alter them
during capture. A new experiment needs a new directory and authorization for its
scope; deleting a start marker is not a resume procedure.

## Reading the results

`results.json` contains every attempted clip and a separate summary per model.
Accuracy and latency comparisons use only clips valid under both configurations.
Planned, attempted, usable, failed and not-run counts remain visible separately.
The report includes corrected final word error rate, deadline word error rate,
last-final-text receipt time, stream completion time, and trailing insertions.
Trailing insertions are automatically aligned words, not a verified hallucination
rate. Later text remains in the transcript and word error count.

Compare old and new settings on this host. Do not compare these absolute latency
values with the historical dashboard, whose execution host and network differed.
A small improvement warrants review, not automatic replacement of the full
benchmark. Larger runs and deployment remain separate decisions after results.
