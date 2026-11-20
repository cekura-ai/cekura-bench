# GPT-Live 1.1 + GPT-6 Sol on Real Human Calls · 20 November 2026

The Real Calls benchmark covers 70 recorded-human scenarios (35 banking, 35 healthcare), three runs each: **210 completed calls and 6,300 evaluated metric entries**. [Full results](results.json).

**ELO 1154, rank 1/16; 95% interval 1115–1204.**

## Results

| Metric | Result |
| --- | ---: |
| Calls passed | 128/210 (61.0%) |
| Pass³ | 23/70 (32.9%) |
| Main task completed | 180/210 (85.7%) |
| Mean call score | 83.97 |
| Hearing / Turn Taking / Task Handling / Speaking | 82.5 / 73.4 / 85.6 / 83.2 |
| Latency P50 / P90 / P95 | 1850 / 2260 / 2400 ms (208 eligible calls) |
| Median words per turn | 17.0 |
| Cost per call / minute | $0.2869 / $0.0591 |
| Selected-call cost | $60.25 |

Pass³ counts scenarios that pass all three repetitions. The 23 scenarios above equal 32.9%; counts and percentages have different denominators.

## By run

| Run | Passed | Task completed | Mean score |
| --- | ---: | ---: | ---: |
| r1 | 45/70 | 60/70 | 84.88 |
| r2 | 41/70 | 59/70 | 83.05 |
| r3 | 42/70 | 61/70 | 83.97 |

## Domain and language

| View | Scenarios | Passed calls | Pass³ | ELO |
| --- | ---: | ---: | ---: | ---: |
| banking | 35 | 57/105 | 8/35 | 1168 |
| healthcare | 35 | 71/105 | 15/35 | 1142 |
| all | 70 | 128/210 | 23/70 | 1154 |
| en | 55 | 96/165 | 16/55 | 1148 |
| es | 7 | 14/21 | 3/7 | 1175 |
| hi | 8 | 18/24 | 4/8 | 1211 |

## ELO ratings

| Rank | Model | ELO | 95% interval |
| ---: | --- | ---: | --- |
| 1 | GPT-Live 1.1 + GPT-6 Sol | 1154 | 1115–1204 |
| 2 | Grok Voice 3.0 | 1151 | 1110–1194 |
| 3 | GPT-Live 1 + GPT-6 Sol | 1111 | 1068–1160 |
| 4 | GPT Realtime 2.1 | 1079 | 1033–1123 |
| 5 | Phonic-1 | 1060 | 1020–1104 |
| 6 | Gemini 3.1 Flash Live Preview | 1040 | 994–1083 |
| 7 | Azure Realtime | 1036 | 989–1080 |
| 8 | Grok Voice Think Fast 2.0 | 1033 | 994–1075 |
| 9 | Gemini 3.8 Live Extended Thinking | 996 | 958–1035 |
| 10 | Eleven Realtime v1 Max | 992 | 947–1037 |
| 11 | Gemini 3.8 Live | 962 | 914–1007 |
| 12 | Eleven Realtime v1 Mini | 950 | 899–996 |
| 13 | GPT Realtime 2.1 Mini | 909 | 870–945 |
| 14 | Higgs Realtime | 874 | 825–920 |
| 15 | Deepslate Opal | 837 | 777–889 |
| 16 | Amazon Nova 2 Sonic | 816 | 764–859 |

Cascade is an unranked reference at 970.

Ratings average each model's three call scores per scenario, apply a five-point tie margin, and jointly fit 16 ranked models using Bradley–Terry on the 400-point ELO scale (mean 1000; 200 iterations). Cascade is anchored afterwards. Intervals use 1,000 scenario bootstrap draws, seed 20261103, and 60 fitting iterations per draw. Every model's ELO and interval is refitted when an input vector changes.

[Numeric inputs and fits](elo.json). `python recompute-elo.py --bootstrap` reproduces the domain/language fits and confidence intervals.

## Configuration and evaluation

- Main agent: `gpt-live-1.1` with delegated `gpt-6-sol`, reasoning low, marin voice, 24 kHz, provider-controlled turns.
- Caller: recorded-human replay through the isolated local Pipecat loopback runtime.
- Metric code and rubrics: the exported platform definitions, run locally against transcripts, recordings, timing logs and tool events.
- Text and expected-outcome judges: DeepSeek V4 Flash through OpenRouter, Fireworks pinned.
- Audio judge: configured to the product's current Gemini Flash-Lite chain (3.5 Flash-Lite → 3.1 Flash-Lite → 3.5 Flash). The original grading pass observed 3.5 Flash-Lite; corrected-pass fallback events remain in the private evidence.
- Scoring: the official `s2s-human-bench` family weights, task blending and pass threshold.
- Source cohort: `fullrun_gptlive11_final_r1`, `fullrun_gptlive11_final_r2`, `fullrun_gptlive11_final_r3`; existing run/attempt/build membership is preserved.
- Limits: 100 s setup timeout; 900 s call budget.

## Corrected grading evidence

The existing recordings were regraded with numeric agent-log timestamps, recovered caller clip timing, 59 executed holds across 50 calls, and fresh word-timed transcription of the recorded agent channel (ElevenLabs scribe_v3, automatic language). Source metric files remain unchanged; all 210 hashes were verified and all 210 caller clocks were available. Six same-call scoring corrections, including one corrupted-caller exclusion, are recorded in [results.json](results.json).

The interruption checker separates a yielded reply from a later answer at measured word gaps of at least two seconds. The prosody checker uses the same word-gap split to separate later confirmations, and uses the recovered transcript because the local log-pairing path dropped real confirmations and included caller detector hangover. The tool-latency checker uses actual paired agent-tool events and evaluates all 57 delayed-tool calls; caller-side holds and unrelated tool-protection notices cannot hide those tests.

Prosodic cue understanding: **28/28 eligible calls (100%)**, with eight targeted calls not reached. Multi-speaker handling: **17/39 (43.6%)**; the audit corrected two single-voice target judgments and one missed executed second voice. Remaining interpretation and simulator limitations are detailed in the QC.

Recovered caller timing includes 435 played clips with interpolated metadata and 1,242 fallback transcript rows. The recording activity threshold is `max(-55 dBFS, p20 + 18 dB)`, with 120 ms minimum activity; hosted VAD equivalence is unconfirmed. Historical judge and caller-model provenance is incomplete. Cached comparison calls exhibit some of the same checker defects; their historical verdicts have not been regraded.

## QC and verification

[Failure QC](failure-qc.md) · [Execution audit](execution-audit.json) · [Validation](validate.py).

All selected calls completed and their recording formats/checksums passed. The 77 targeted verification calls and three separate wait/resume prompt trials also completed and were graded. The benchmark uses the existing 210-call cohort. Claude Opus 5.5 HIGH and Codex reviewed all 36 prosody-targeted calls and 39 multi-person-targeted calls. Several multi-person cases remain ambiguous; this report does not assert that every remaining failure is a confirmed model error.

The original F09 status-lookup fixture defect, occasional simulator branch misfires, and uncertainty about a post-commit prosody target remain relevant execution/interpretation limitations. Ten original calls ended through the harness; incomplete closing-reference diagnostics used a longer watchdog, so those cases require care. Repaired F09 and closing-reference diagnostics are retained separately. The overlapping ELO intervals and three-point lead over Grok Voice 3.0 do not establish a decisive ranking difference. No independent human acoustic listening was performed.

`python validate.py` checks 210 calls, 6,300 entries, cohort membership, all aggregates and ELO metadata. Private transcripts, recordings, task targets and provider credentials are excluded from these artifacts.
