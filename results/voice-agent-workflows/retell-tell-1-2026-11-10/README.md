# Retell tell-1 · 10 November 2026

This snapshot publishes the data behind the approved [tell-1 report](report.html).
It contains all 246 calls from 82 canonical scenarios, with three runs per
scenario and concurrency 5 per workflow, over PSTN. No runs are review-required.

| Workflow | Passed calls | Pass¹ | All three passed | Pass³ |
| --- | --- | --- | --- | --- |
| Appointments | 170/177 | 96.05% | 54/59 | 91.53% |
| Medicare | 66/69 | 95.65% | 20/23 | 86.96% |
| Combined | 236/246 | 95.93% | 74/82 | 90.24% |

Pass¹ counts passing calls. Pass³ counts scenarios whose **three actual calls**
all pass; it is not Pass¹ cubed. The ten failures comprise six Expected Outcome
failures and four Tool Call Accuracy failures. Every call has an evidence link
in [results.json](results.json).

## Scoring policy

| Gate | Pass threshold | Numeric coverage |
| --- | --- | --- |
| Expected Outcome | = 5 | 246/246 |
| Infrastructure Issues | true, normalized to 5 | 246/246 |
| Interruption Score | ≥ 1 | 246/246 |
| Latency (in ms) | ≤ 10,000 | 246/246 |
| Tool Call Accuracy | = 5 | 237/237 applicable calls |

AS12, AS49 and MS79 require no mock tools: their nine tool scores are null and
the tool gate does not apply. A missing score on any applicable call cannot
count as a pass.

These five gate names and thresholds match the published PolyAI and Smallest AI
policies. Retell uses **standard** Tool Call Accuracy; Smallest's published
comparison uses its documented list-text-accepted variant. None of Retell's four
tool failures concerns list formatting. Transcription Accuracy and Voice Tone +
Clarity are recorded on all calls and do not gate success. Repetition, call
termination and Tool Call Success are excluded from this publication policy;
later counterfactual scoring is a separate analysis.

Task completion in the report's graphs means full-credit Expected Outcome
(240/246, 97.56%), while Pass¹ combines all applicable gates (236/246, 95.93%).
Response time is the mean Cekura latency metric across the 246 calls (945.91 ms).
It is separate from the reference harness's agent-stamped timing columns.

## Configuration and provenance

Project 8197: [Appointments result 1068690](https://dashboard.cekura.ai/8197/results/1068690)
(agent 20898, published Retell version 8) and
[Medicare result 1068691](https://dashboard.cekura.ai/8197/results/1068691)
(agent 20897, published version 7).

The existing Retell agents changed from gpt-5.5, fast tier, high priority enabled,
to tell-1, standard tier, high priority disabled. Prompts, first messages, tools,
voice and other runtime settings were identical to the historical agents.
Speech settings remain Cartesia Cimo, `stt_mode=accurate`, en-US, with voice
speed, temperature and volume all 1. This snapshot records selected configuration
fields; it omits credentials, raw provider exports and transcripts.

The final scores include the AS45 Expected Outcome re-evaluations and the
AS17 Expected Outcome-only re-evaluation. AS17 has two passes and one failure;
its non-outcome metrics were unchanged.

The report is byte-identical to the approved Downloads artifact, verified by
the SHA-256 in `results.json`. All seven graphs are retained. Its ten peer rows
remain the frozen website snapshot of 3 November 2026, checked against the live
site when the report was prepared. Source commits and prompt hashes are in the
snapshot. Medicare matches the repository definition; the historical appointment
prompt differs from current main in one injection-guardrail sentence. The report
therefore presents tell-1 as a reference comparison. Historical peer rows and
provider configuration files are not replaced by this result.

## Reproduce the counts

From the repository root, with Python 3:

```sh
python3 results/voice-agent-workflows/retell-tell-1-2026-11-10/validate.py
python3 -m pytest tests/test_tell_1_results.py -q
```

The validator checks membership, unique calls, three attempts per scenario,
applicable numeric scores, every five-gate verdict, per-workflow and combined
rates, all seven metric aggregates, and the approved report hash. It reads the
frozen snapshot and makes no API calls or changes to Cekura verdicts.
