# Failure QC · GPT-Live 1.1 + GPT-6 Sol on Real Calls

A quick, evidence-based review of the failures in this snapshot's final 210 calls. **The scores in
[results.json](results.json) and the [report](README.md) are the raw grades and are not changed by this
review.** QC verdicts sit beside them. Per-call verdicts, the cohort-wide measurements and the inventory
are in [failure-qc.json](failure-qc.json).

## What was reviewed, and how

- **Inventory:** all 92 calls below the pass threshold (score < 80) and all 30 deterministic main-task
  failures, grouped by scenario and by failed skill.
- **Deep review of 41 unique calls:**
  - all 30 main-task failures;
  - the three corrected F15 calls;
  - six metric-only failures covering hold, multi-speaker, low-confidence confirmation, consent,
    cancellation and style;
  - two passes checking the apparent tool-latency and premature-response improvements.

  The other 51 failed calls were covered by the inventory and the cohort-wide checks only, not reviewed
  one by one.
- **Evidence per call:**
  - the original scenario (caller script, expected outcome, main-task target);
  - the call transcript;
  - every tool call and result compared with the scenario's fixture rows;
  - the caller engine's own condition-matching log;
  - the judge's explanation;
  - per-channel voice activity and energy measured on the lossless recording.
- **Cohort-wide checks over all 210 calls:**
  - tool-handler exceptions;
  - whether each call applied its scenario's tool delays;
  - unscripted caller replies;
  - non-requested languages in agent speech;
  - a speech-level re-measurement of every barge-in (2.2) verdict and every slow-tool wait (2.5);
  - a sampled check of response-latency onsets.
- **Limits:**
  - No utterance was listened to. Audio conclusions rest on measured timing and energy, not on hearing
    the audio.
  - Transcripts are speech-recognised (caller) or model-reported (agent) text.
  - "Model failure" means the evidence supports the judge's verdict. It doesn't mean every failed line
    was re-judged.

## Inventory

| | Calls |
| --- | --- |
| Below the pass threshold | 92 of 210 (70 of them completed the main task) |
| Main task not completed | 30 of 210 |
| Scenarios failing in all three runs | F02, F05, F20, F22, F23, F25, F27, H02, H03, H07, H09, H16, H23, H24 |
| Scenarios failing in two of three | F01, F03, F08, F14, F15, F18, F26, F29, H06, H12, H17, H21, H29, H30, H32 |

Skills with the most failures (failed of reached calls): Numeric Read-back 46/176, Consent Verification
40/186, Low-Confidence Confirmation 29/38, Multi-Speaker Handling 26/36, Backchannel vs Barge-in 24/32,
Tool-Latency Handling 23/57, Style Instruction Following 18/41, Cancellation Handling 17/38,
Premature-Response Avoidance 15/75, Human Handoff 12/36, Waiting Through Hold 11/12. A failed call
usually fails two skills (median).

## Main-task failures (30, all reviewed)

| QC verdict | Calls | Generic cause |
| --- | --- | --- |
| Model failure | 16 | **Mishearing a name, ID or digit** (F11, F16, F17, F23, F33, H30, H35, and F08 r2), so identity or lookup failed and the task never opened. **Committing too early or racing a take-back:** in F08 r1/r3 and H33 the committing tool ran 1–2 s after the caller said stop, or before their go-ahead. **Acting on a second speaker's go-ahead** (F02 r2/r3). **Incomplete tool arguments:** the account or date was never collected (F22, F24, both judged likely). **Treating a successful result as uncertain** after an unnecessary status lookup (F09). |
| Scenario path | 8 | **Overlapping caller conditions:** one agent turn satisfied both the main-task cue and a closing or test cue, and the caller engine chose the other one, so the main task was never requested (F05 r1/r3, H14 r2/r3). **Go-ahead answering an unrelated question:** the scripted go-ahead arrived as the answer to the agent's "anything else?", and the scripted cancel followed before any payment (F14 r2/r3). **Eligibility question with no scripted answer:** the tool's description limits sick notes to a recent visit, the agent asked about one, the script had no line for it and the caller engine improvised "no", blocking the task (H16 r1/r2). |
| Testing agent | 3 | The caller engine's matcher scored the correct scripted condition below its gate and generated an unscripted reply (F14 r1, H22 r3), or the call fell into a mutual-wait loop the caller engine ended (F26 r2). |
| Deterministic checker disagreement | 3 | H03 (all runs): the agent moved the appointment with the fixture-supported change tool (exact fixture row, confirmed). The main-task definition accepts only cancel-and-book, so it scores this as not done. |

So 14 of the 30 main-task failures trace to the scenario path, the caller engine or the task checker
rather than the model. The published baselines use the same scenarios and caller design, so these causes
may affect other models too. This run did not establish how often.

## Corrected F15 calls

F15 was rerun in all three runs on build v4, which only restores the matcher's boolean-word constants:
- `file_claim` is now answered by the scenario fixture in every run: a free-text value first gets the
  fixture's "no match", then an exact match files the claim.
- The main task is now done in all three runs.
- r3 scored 100.
- r1 (75) and r2 (62.5) still fail. The causes are behavioral and match failures seen in other
  scenarios:
  - the plate is read back without a phonetic spelling;
  - in r2, nothing is done about the caller's "you're cutting out";
  - barge-in, discussed below.
- Execution evidence is clean: no tool-handler or NameError exceptions, the configured model and
  settings, exit 0, verified PCM checksums, both channels active.

## Gap skills: what the evidence shows

**Backchannel vs barge-in (2.2), 24 failures.** 22 of the 24 are "the agent kept talking more than 2 s
after the caller cut in". The metric locates speech with an energy threshold relative to each channel's
quiet level. GPT-Live leaves a low-level, non-speech signal on its channel after it stops talking (about
25–35 dB, against about 65–70 dB for speech), and the metric counts it as talking. Re-measured at speech
level (agent below 50 dB for at least 1 s):
- the median stop time falls from **4.6 s** (metric) to **2.0 s**;
- **11 of the 22** stop within the 2 s limit;
- **10** stop in 2.0–2.7 s;
- **one** (F15 r2, 4.1 s) is clearly late.

So much of this gap likely comes from how the metric measures stops on this model's lossless output, not
from the model failing to yield. Whether the hosted platform's audio path removes this low-level signal
was not verified. The grades are unchanged.

**Tool-latency handling (2.5) and premature responses (2.8), the apparent gains.**
- **Tool delays:** every call applied its scenario's original tool-result delays (63 calls have them).
- **2.5 passes at speech level:** the low-level signal could mask silences, but only 2 of 34 passes
  (F10 r2/r3) have a speech-level silence over 5 s during a slow tool.
- **2.8:** the low-level signal can only count against it, never for it.
- **Behavior behind the gains:** GPT-Live speaks a short working message while its backend runs a tool,
  and waits through caller pauses.

**Multi-speaker (1.10), low-confidence confirmation (1.3), consent (3.5a), cancellation (3.5b), style
(4.3).** The reviewed failures support the judges:
- **1.10:** acting on, or sharing details with, a second speaker who isn't authorised.
- **1.3:** reading back a guessed or transposed digit.
- **3.5a:** committing before a clear go-ahead, or announcing success before the tool ran.
- **3.5b:** a committing tool 1–2 s after a take-back.
- **4.3:** ignoring a requested form of address.

**Waiting through hold (1.5).** It was reached in only 12 calls and failed in 11. One of two reviewed
failures is a model failure (it kept talking 2.2 s into the caller's hold). The other is unknown: the
transcript merges the hold request into a later line and the timing alone does not place it.

**Response latency.** It isn't distorted by the low-level signal. On 554 replies in 40 random calls,
speech starts 30 ms (median) after the metric's agent start. The latency metric correctly returns no value
for H32 in r1 and r2. The caller channel there carries constant loud background audio (20th percentile
about 60 dB), so caller speech never clears the metric's 18 dB margin and no caller-then-agent pair
qualifies.

## Other observations

- **Non-requested language.** 8 calls contain short Mandarin fragments in agent speech, in English,
  Spanish or Hindi conversations: r1/F15, r1/F31, r1/H07, r2/F12, r2/F26, r2/H04, r2/H32, r3/H22. They
  range from single words to one full sentence. Some agent transcripts also use full-width punctuation;
  that is text only.
- **Unscripted caller replies are routine.** 1,561 of 3,306 caller replies were generated rather than
  scripted, mostly to agent questions the script doesn't cover. Only a few derailed a call: the F14 r1,
  H22 r3 and F26 r2 cases above.
- **No further harness corruption.** No other retained call had a tool-handler exception. Every call
  applied its scenario's tool delays. Nothing found here calls for more reruns under the user's rerun
  policy.

## Follow-ups

1. **Barge-in metric (2.2):** measure stops at speech level, or gate low-level output, before comparing
   GPT-Live with models whose silence is digital zero. Or confirm what the hosted audio path delivers.
2. **Scenario cues:** F05, H14 and F14 have conditions a single agent turn can satisfy together, and the
   caller engine's choice can make the main task unreachable. H16's tool description invites an
   eligibility question the script doesn't answer.
3. **H03 task definition:** decide whether an equivalent fixture-supported change should count as done.
4. **Mandarin fragments:** report to the model provider as a model behavior.

## Reviewed calls

| Call | Sample | Basis | QC verdict |
| --- | --- | --- | --- |
| r1/F01 | metric-only failure | 1.5, 1.3, EO | model failure |
| r2/F02 | main-task failure | main task, 1.10, 3.5a | model failure |
| r3/F02 | main-task failure | main task, 1.10, 3.5a | model failure |
| r1/F05 | main-task failure | main task | scenario path |
| r3/F05 | main-task failure | main task | scenario path |
| r1/F08 | main-task failure | main task, 3.5a, 3.5b, 1.3 | model failure |
| r2/F08 | main-task failure | main task | model failure |
| r3/F08 | main-task failure | main task, 3.5b | model failure |
| r1/F09 | main-task failure | main task | model failure |
| r1/F11 | main-task failure | main task, 1.3 | model failure |
| r1/F14 | main-task failure | main task | testing agent |
| r2/F14 | main-task failure | main task | scenario path |
| r3/F14 | main-task failure | main task | scenario path |
| r1/F15 | corrected F15 | 2.2, EO | mixed (see above) |
| r2/F15 | corrected F15 | 1.8, 2.2, EO | model failure |
| r3/F15 | corrected F15 | execution | pass |
| r1/F16 | main-task failure | main task, 3.7 | model failure |
| r2/F17 | main-task failure | main task, 1.2 | model failure |
| r1/F22 | main-task failure | main task | model failure |
| r2/F23 | main-task failure | main task, 1.2 | model failure |
| r2/F24 | main-task failure | main task | model failure |
| r2/F26 | main-task failure | main task | testing agent |
| r1/F30 | metric-only failure | 1.10, 3.5b | model failure |
| r1/F32 | metric-only failure | 1.5 | unknown |
| r2/F33 | main-task failure | main task, 1.2 | model failure |
| r1/H01 | metric-only failure | 3.5a, 4.1 | model failure |
| r1/H03 | main-task failure | main task | checker disagreement |
| r2/H03 | main-task failure | main task | checker disagreement |
| r3/H03 | main-task failure | main task | checker disagreement |
| r1/H07 | metric-only failure | 1.10, 4.1, 2.2 | model failure |
| r1/H09 | metric-only failure | 4.3 | model failure |
| r2/H10 | improvement check | 2.5 | pass confirmed |
| r3/H13 | improvement check | 2.8 | pass confirmed |
| r2/H14 | main-task failure | main task | scenario path |
| r3/H14 | main-task failure | main task | scenario path |
| r1/H16 | main-task failure | main task | scenario path |
| r2/H16 | main-task failure | main task | scenario path |
| r3/H22 | main-task failure | main task | testing agent |
| r1/H30 | main-task failure | main task, 1.2 | model failure |
| r3/H33 | main-task failure | main task, 3.5a | model failure |
| r3/H35 | main-task failure | main task, 1.2 | model failure |
