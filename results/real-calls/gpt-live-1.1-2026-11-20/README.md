# GPT-Live 1.1 + GPT-6 Sol on Real Human Calls · 20 November 2026

Aggregate results of GPT-Live 1.1 with GPT-6 Sol (reasoning low) on the Real Calls benchmark: the original 70
recorded-caller scenarios (35 banking, 35 healthcare), three runs each, scored by `s2s-human-bench` with
`config/human-bench.json` (model key `gpt-live-1.1`, runs `fullrun_gptlive11_rerunsel_r1..r3`). Every per-call score,
verdict and metric value is in [results.json](results.json); `python validate.py` recomputes the tables below.

**The cohort is 210 completed calls: 206 originals and four replacement slots.** Two original calls ran into the 900 s
call budget (r1/F15, r2/F02) and were rerun once at the user's instruction. The final execution audit then found that
every F15 call had run on a defective local tool fixture (its matcher raised `NameError` instead of answering `file_claim`),
so F15 was rerun in all three runs on a repaired build (v4) and only those three calls were graded again. r1/F15 was
therefore replaced twice. The superseded attempts are excluded from every number; see [Replaced calls](#replaced-calls).

**New comparative ELO is pending.** The rating is fitted on every model's per-scenario scores, and only the other
models' published aggregates are available to this run, so no rating or rank is claimed. The comparison with the
published GPT-Live 1 row is aggregate-to-aggregate; see [Comparability](#comparability).

## Headline

| | GPT-Live 1.1 + GPT-6 Sol (this run) | GPT-Live 1 + GPT-6 Sol (published) |
| --- | --- | --- |
| Calls passed (score ≥ 80) | 118/210 (56.2%) | 122/210 (58.1%) |
| pass³ (all three runs passed) | 21/70 (30.0%) | 27/70 (38.6%) |
| Main task completed | 180/210 (85.7%) | 166/210 (79.0%) |
| Hearing | 65.7 | 78.2 |
| Turn Taking | 71.6 | 71.0 |
| Task Handling | 84.2 | 81.8 |
| Speaking | 77.5 | 83.3 |
| Mean call score | 80.16 | — |
| Response latency p50 / p90 (passive metric) | 1,850 / 2,260 ms | 2,270 / 2,720 ms |
| Median words per agent turn | 18.0 | 15.0 |
| Cost per call (live model + delegated backend) | $0.2869 | $0.2861 |
| ELO | pending | 1122 (rank 2 of 16 rated) |

Latency covers the 208 calls on which the latency metric returned a value. It returns none on r1/H32, r2/H32:
H32's recorded caller audio carries constant loud background sound, so the metric's energy threshold finds no caller
speech to time a reply from (its rule, applied correctly; r3/H32 yields a single reply).

**Failure QC:** [failure-qc.md](failure-qc.md) reviews the failures against the evidence (raw grades unchanged). Of the
30 main-task failures, 16 are model failures and 14 trace to the scenario path, the caller engine or the task checker; for
barge-in (2.2), the metric counts GPT-Live's low-level post-speech signal as talking; re-measured at speech level, 11 of
the 22 timed failures stop within the 2 s limit and 10 more within 2.7 s.

## By run

| Run | Calls | Passed | Pass rate | Task done | Mean score | Hearing | Turn Taking | Task Handling | Speaking | Latency p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1 | 70 | 37 | 52.9% | 60 | 79.22 | 66.5 | 71.4 | 83.8 | 77.9 | 1,860 ms |
| r2 | 70 | 38 | 54.3% | 59 | 79.07 | 64.8 | 68.0 | 84.8 | 74.2 | 1,860 ms |
| r3 | 70 | 43 | 61.4% | 61 | 82.19 | 65.2 | 76.8 | 83.9 | 79.7 | 1,840 ms |
| all | 210 | 118 | 56.2% | 180 | 80.16 | 65.7 | 71.6 | 84.2 | 77.5 | 1,850 ms |

Replacement calls: r1/F15 (v4), r2/F02 (timeout rerun), r2/F15 (v4), r3/F15 (v4).

## By domain and language

| Slice | Calls | Passed | pass³ | Task done | Hearing | Turn Taking | Task Handling | Speaking | Published GPT-Live 1 passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Banking | 105 | 53 (50.5%) | 5/35 | 86 | 64.9 | 70.8 | 82.9 | 73.8 | 52/105 |
| Healthcare | 105 | 65 (61.9%) | 16/35 | 94 | 74.9 | 73.6 | 85.5 | 79.3 | 70/105 |
| English only | 165 | 86 (52.1%) | 12/55 | 140 | 65.2 | 69.6 | 84.1 | 76.8 | — |
| Switches to Spanish | 21 | 15 (71.4%) | 4/7 | 16 | 100.0 | 73.3 | 88.1 | 76.0 | — |
| Switches to Hindi | 24 | 17 (70.8%) | 5/8 | 24 | 100.0 | 73.6 | 93.2 | 76.9 | — |

## Skills

Pass rate among the calls that reached each skill (a skill counts with at least 3 reaching calls). Published
GPT-Live 1 rates are the same metrics on its own 210 calls, with its reaching-call count in brackets.

| Skill | Family | Reached | Pass rate | Published GPT-Live 1 |
| --- | --- | --- | --- | --- |
| 1.2 Entity Extraction Accuracy | Hearing | 42 | 95.2% | 76.2% (42) |
| 1.3 Low-Confidence Confirmation | Hearing | 38 | 23.7% | 31.4% (35) |
| 1.4 Side-Speech Rejection | Hearing | 45 | 100.0% | 100.0% (44) |
| 1.5 Waiting Through Hold | Hearing | 12 | 8.3% | 88.0% (25) |
| 1.6 Prosodic Cue Understanding | Hearing | 28 | 92.9% | 93.1% (29) |
| 1.7 Emotion Recognition | Hearing | 38 | 100.0% | 100.0% (42) |
| 1.8 Degraded-Channel Robustness | Hearing | 36 | 77.8% | 75.0% (32) |
| 1.10 Multi-Speaker Handling | Hearing | 36 | 27.8% | 61.5% (39) |
| 2.1 Endpointing | Turn Taking | 27 | 74.1% | 71.4% (42) |
| 2.2 Backchannel vs Barge-in Detection | Turn Taking | 32 | 25.0% | 73.3% (30) |
| 2.4 Overlapping-Speech Recovery | Turn Taking | 41 | 100.0% | 97.4% (38) |
| 2.5 Tool-Latency Handling | Turn Taking | 57 | 59.7% | 37.0% (54) |
| 2.7 Mid-Utterance Pause Tolerance | Turn Taking | 11 | 90.9% | 94.9% (39) |
| 2.8 Premature-Response Avoidance | Turn Taking | 75 | 80.0% | 52.0% (75) |
| 3.2 Correction Handling | Task Handling | 38 | 86.8% | 91.7% (36) |
| 3.3 Human Handoff | Task Handling | 36 | 66.7% | 61.1% (36) |
| 3.4 Tool Error Handling | Task Handling | 28 | 92.9% | 100.0% (22) |
| 3.5a Consent Verification | Task Handling | 186 | 78.5% | 91.1% (180) |
| 3.5b Cancellation Handling | Task Handling | 38 | 55.3% | 50.0% (36) |
| 3.6 Redundant Re-prompt Avoidance | Task Handling | 210 | 100.0% | 99.5% (210) |
| 3.7 Hallucinated-Success Avoidance | Task Handling | 210 | 99.1% | 99.0% (210) |
| 4.1 Numeric Read-back | Speaking | 176 | 73.9% | 80.6% (160) |
| 4.2 TTS Text Normalization | Speaking | 210 | 100.0% | 100.0% (210) |
| 4.3 Style Instruction Following | Speaking | 41 | 56.1% | 73.7% (38) |
| 4.4 Language Switching | Speaking | 40 | 80.0% | 78.9% (38) |

The largest gaps against the published row are 1.5 Waiting Through Hold, 2.2 Backchannel vs Barge-in Detection
and 1.10 Multi-Speaker Handling (lower here) and 2.5 Tool-Latency Handling and 2.8 Premature-Response Avoidance
(higher here). The [failure QC](failure-qc.md#gap-skills-what-the-evidence-shows) finds the 1.10 failures it reviewed to
be model behavior, the 2.2 gap to come largely from how the metric measures stops on GPT-Live's output, and the 2.5/2.8
gains to hold at speech level; hold (1.5) was reached in only 12 calls. Judge and runtime differences are not separated here.

Expected-outcome lines met (mean share per call): 87.6%. The published baseline's EO field measures main-task completion, compared in the headline table; its expected-outcome line share is unavailable. Agent ended the call with its end-call tool on 148/210 calls; stalled (>10 s of silence owed to the caller) on 33; silent throughout on 0; simulation-health failures: 0.

## Replaced calls

| Run | Scenario | Superseded attempt | Reason | Replaced by |
| --- | --- | --- | --- | --- |
| r1 | F15 | attempt 2 | stopped at the 900 s call budget | attempt 3 |
| r1 | F15 | attempt 3 | fixture-invalid: `file_claim` raised `NameError` in the local matcher | attempt 4 |
| r2 | F02 | attempt 1 | stopped at the 900 s call budget | attempt 2 |
| r2 | F15 | attempt 1 | fixture-invalid: `file_claim` raised `NameError` in the local matcher | attempt 2 |
| r3 | F15 | attempt 1 | fixture-invalid: `file_claim` raised `NameError` in the local matcher | attempt 2 |

Both kinds of replacement were authorized by the user, ran once each through the same launch path and configuration,
and completed. The F15 fixture defect: the local runtime's generated mock-tool matcher dropped two constants from the
original Human Bench matcher source, because its dependency extractor skipped a destructured assignment. Build v4
restores them; regenerating the matcher from the preserved original source reproduces both the defective and the
repaired file byte for byte, and every extracted function is identical to the source. No other call recorded a
tool-handler error. Infrastructure retries (below) are a separate policy; metric verdicts did not trigger reruns.

## Configuration

| Setting | Value |
| --- | --- |
| Live model | `gpt-live-1.1` over `wss://api.openai.com/v1/live/sessions` (full duplex) |
| Backend | `gpt-6-sol`, reasoning `low`, Responses delegation (tools `strict: false`) |
| Voice / rate | `marin`, 24 kHz |
| Agent | `reference-agents/pipecat-s2s/bot.py` `gpt-live` row at this repo's commit, imported unchanged; delegation prompt and shared rules as published |
| Scenarios | the original 70 (35 F, 35 H): recorded human caller audio, per-scenario prompt, opening, tool schemas and stateful responses (fixture set s2shb-v2-15) |
| Call budget | 900 s per call |
| Collected | 2026-11-20, 00:47–03:18 UTC; F15 fixture reruns 06:05–06:13 UTC |

The calls ran on a local runtime that hosts the recorded-caller engine and the agent on one machine over a loopback
audio connection, not on the hosted Cekura platform. Both directions are paced at real time in 20 ms frames.

### Builds

| Build | What changed | Calls in the cohort |
| --- | --- | --- |
| v1 | first local adapter | 5 (r1: F01, H16, H21, H26, H29) |
| v2 | the agent hears the caller as a continuous 20 ms line (silence when the caller is quiet), as a phone line or WebRTC receiver delivers it | 35 |
| v3 | v2 plus a 100 s pipeline setup timeout on both sides (Pipecat default 20 s), before the call connects | 167 (including the F02 timeout rerun) |
| v4 | v3 plus the mock-tool matcher's two restored boolean-word constants (nothing else in the call path) | 3 (the F15 reruns) |

GPT-Live is full duplex and its output is clocked by its input: in a direct test an opening instruction was first spoken
1.3 s after the session began when silence was streamed, and 13.3 s after when no input was sent. In v1 the caller's
transport sent audio only while the caller spoke, so the agent stalled until the caller spoke again; every v1 call on
which the reference agent's own `audio_in_starved` check fired was voided and recollected on v2/v3. The five v1 calls
kept are scenarios whose caller stream was already continuous (input drift 0.4–0.8 s, inside the agent's 1 s check).
Model, voice, backend, prompts, tools, fixtures, timings and the call budget are the same in every build.

### Collection and invalid attempts

| Run | Scenarios | Completed originals | Call-budget timeouts (replaced) | Attempts | Infrastructure-voided attempts | Agent silent |
| --- | --- | --- | --- | --- | --- | --- |
| r1 | 70 | 69 | 1 | 136 | 66 | 0 |
| r2 | 70 | 69 | 1 | 70 | 0 | 0 |
| r3 | 70 | 70 | 0 | 70 | 0 | 0 |

The 66 voided attempts are infrastructure failures, each with recorded reasons: starting 25 calls at once on one
8-CPU host (pipeline setup and connection timeouts before any conversation), the v1 input starvation, an interrupted
controller, and a three-minute window in which the provider dropped every new Live session before it started. Each was
recollected. Counts of recorded reasons (an attempt can carry several): no_conversation 39, pipeline_setup_timeout 35, agent_input_starved 23, caller_pipeline_not_ready 23, agent_not_started 7, caller_speech_error 6, audio_scheduler_slips 5, provider_session_not_started 4, orchestrator_interrupted 3.
Afterwards starts were paced and gated on host load at 10, then 15 live calls; r2 and r3 needed no recollection.
Every run meets the public usability rule (70 calls, at most 7 with a silent agent). "Completed originals" counts the
main collection; three of those completed F15 calls were later superseded as fixture-invalid (see Replaced calls).

## Grading

All 30 linked metrics of every scenario were evaluated on all 210 cohort calls (6,300 metric results, all with status evaluated), with their original definitions,
reachability rules and test profiles. Judges, as decided by the user for this run:

| Judge | Model | Calls answered |
| --- | --- | --- |
| Text skill judges (enum/binary/numeric) | deepseek/deepseek-v4-flash (provider pinned: fireworks) | 1,624 |
| Expected outcome | deepseek/deepseek-v4-flash via Fireworks | 210 |
| Audio (4.1 number read-back) | gemini-direct: gemini-3.5-flash-lite (fallback index 0, attempt 1) | 138 |
| Audio (4.1 number read-back) | gemini-direct: gemini-3.5-flash-lite (fallback index 0, attempt 2) | 2 |
| Deterministic | response latency, turn length, main-task completion (original code, no model) | — |

The three F15 replacement calls were graded later with the same judges and the same evaluator code (identical `src` tree; recorded per attempt); no other call was regraded. Judge errors in the final results: none. Calls re-graded after a judge error: 0. The audio judge hit 2 transient Gemini server errors, each answered by the same model on its retry. Text judges ran through OpenRouter with the provider pinned
(Fireworks) and fallbacks disabled; the audio judge called the Gemini API directly in the product's current order
(3.5 Flash-Lite, then 3.1 Flash-Lite, then 3.5 Flash) with the platform's audio-judge prompt, a mono slice of the
requested window and an enum result.

## Cost

$60.25 for the 210 cohort calls (usage recorded on every call; $0.2869 per call, $0.0591 per call minute): live audio (the `gpt-live` row's $0.05 per minute) $49.00, GPT-6 Sol tokens $11.25.
Rates are this repo's `pricing/prices.json` `gpt-live` row (live audio per minute plus the backend's tokens); the
OpenAI pricing page read on 2026-11-20 lists gpt-live-1.1 at the same $0.05 per active minute as gpt-live-1.
Caller-side services, superseded attempts and voided attempts are excluded.

## Comparability

- **Not the hosted platform.** The published rows were run and judged on the Cekura platform. This run used the
  same scenarios, recordings, prompts, tools, fixtures, metrics and scorer on a local runtime with the differences
  listed here; production scoring parity was not established.
- **Judges.** Text judges were DeepSeek V4 Flash and the audio judge the product's current Gemini Flash-Lite chain,
  by decision for this run. Which model judged each metric for the published rows is not recorded in the copied
  benchmark data, so part of any difference can come from the judges. The audio judge's prompt template, mono
  downmix and model order match current backend main for 4.1's path (no audio channel requested); this covers the
  judge call only, not full platform parity.
- **Caller side.** The recorded-caller engine ran locally with ElevenLabs speech recognition and voices (the four
  Cartesia caller voices mapped to their ElevenLabs clones) and its decision models through OpenRouter.
- **Replacements.** Four of the 210 calls are reruns: one replaced a call stopped at the call budget, three replaced
  F15 calls whose tool fixture raised an error (build v4).
- **ELO pending**, see above.
- **Builds.** 5 calls ran on v1 (continuous caller stream in those scenarios), 35 on v2, 167 on v3 and 3 on v4.
- **No example calls.** Recordings and transcripts stay private; none is published with this snapshot.

## Execution audit

All 210 selected calls reached `completed`, both parties spoke in every call, every main-agent process exited
with code 0, every call used GPT-Live 1.1 with GPT-6 Sol low and the configured voice and rate, and all 210
lossless recordings decoded with matching PCM checksums, frame counts, rates and channels. Tool-handler errors in the
selected calls: 0 (the F15 matcher defect is resolved by the v4 replacements). Longest call:
659.2 s, below the 900 s budget. Largest absolute input drift: 812 ms (limit 1 s).
Two calls had one isolated audio-scheduling slip each (r1/F08, r2/H10), below the exclusion threshold; ten calls ended
through the reference agent's goodbye backstop, which is deliberate. Details: [execution-audit.json](execution-audit.json).

## Files

- [results.json](results.json): the cohort definition; aggregates overall, per run, per domain and per language group;
  skills; latency; turn length; cost; grading provenance; collection counts; the published GPT-Live 1 aggregates used
  above; and one row per cohort call with its score, family scores, skill verdicts and all 30 metric values and statuses
  (no explanations, transcripts, recordings or task targets).
- [validate.py](validate.py): checks the cohort (3 × 70 scenarios, 210 calls, 6,300 metric results, roles and builds,
  no superseded attempt) and recomputes passed calls, pass³, task completion, mean score, family scores and latency
  percentiles overall, per run, per domain and per language group.
- [execution-audit.json](execution-audit.json): execution and recording checks for every selected call, the superseded
  attempts and the resolved fixture defect.
- [failure-qc.md](failure-qc.md) and [failure-qc.json](failure-qc.json): the failure review, with per-call QC verdicts
  (score-only; no transcripts, explanations or tool values).
