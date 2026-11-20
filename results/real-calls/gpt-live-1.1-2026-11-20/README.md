# GPT-Live 1.1 + GPT-6 Sol on Real Human Calls · 20 November 2026

The Real Calls benchmark covers 70 recorded-human scenarios (35 banking, 35 healthcare), three runs each. The final results contain **210 completed calls and 6,300 metric entries**. Full results: [results.json](results.json).

**ELO 1095, rank 3/16; 95% interval 1043–1145.**

## Results

| Metric | Result |
| --- | ---: |
| Calls passed | 122/210 (58.1%) |
| Pass³ | 23/70 (32.9%) |
| Main task completed | 180/210 (85.7%) |
| Mean call score | 80.77 |
| Hearing / Turn Taking / Task Handling / Speaking | 68.2 / 72.6 / 84.2 / 79.3 |
| Latency P50 / P90 / P95 | 1850 / 2260 / 2400 ms (208 eligible calls) |
| Median words per turn | 18.0 |
| Cost per call / minute | $0.2869 / $0.0591 |
| Selected-call cost | $60.25 |

## By run

| Run | Passed | Task completed | Mean score |
| --- | ---: | ---: | ---: |
| r1 | 37/70 | 60/70 | 79.46 |
| r2 | 41/70 | 59/70 | 80.2 |
| r3 | 44/70 | 61/70 | 82.66 |

## Domain and language

| View | Scenarios | Passed calls | Pass³ | ELO |
| --- | ---: | ---: | ---: | ---: |
| banking | 35 | 54/105 | 6/35 | 1068 |
| healthcare | 35 | 68/105 | 17/35 | 1124 |
| en | 55 | 88/165 | 13/55 | 1077 |
| es | 7 | 15/21 | 4/7 | 1113 |
| hi | 8 | 19/24 | 6/8 | 1255 |

## ELO ratings

| Rank | Model | ELO | 95% interval |
| ---: | --- | ---: | --- |
| 1 | Grok Voice 3.0 | 1152 | 1111–1194 |
| 2 | GPT-Live 1 + GPT-6 Sol | 1116 | 1073–1165 |
| 3 | GPT-Live 1.1 + GPT-6 Sol | 1095 | 1043–1145 |
| 4 | GPT Realtime 2.1 | 1083 | 1038–1126 |
| 5 | Phonic-1 | 1063 | 1022–1107 |
| 6 | Gemini 3.1 Flash Live Preview | 1042 | 996–1085 |
| 7 | Azure Realtime | 1040 | 992–1083 |
| 8 | Grok Voice Think Fast 2.0 | 1035 | 998–1075 |
| 9 | Gemini 3.8 Live Extended Thinking | 1001 | 963–1039 |
| 10 | Eleven Realtime v1 Max | 997 | 953–1040 |
| 11 | Gemini 3.8 Live | 966 | 918–1010 |
| 12 | Eleven Realtime v1 Mini | 954 | 904–1000 |
| 13 | GPT Realtime 2.1 Mini | 914 | 875–950 |
| 14 | Higgs Realtime | 880 | 833–925 |
| 15 | Deepslate Opal | 841 | 782–892 |
| 16 | Amazon Nova 2 Sonic | 823 | 772–866 |

Cascade is an unranked reference at 976.

The rating averages each model’s three call scores per scenario, uses a five-point tie margin, and jointly fits 16 ranked models with Bradley–Terry on the 400-point ELO scale (mean 1000; 200 iterations). Cascade is anchored afterwards. Intervals use 1,000 scenario bootstrap draws, seed 20261103, 60 fitting iterations per draw.

Numeric inputs and fits: [elo.json](elo.json). Run `python recompute-elo.py --bootstrap` to validate every domain/language fit and confidence interval.

## Configuration and evaluation

- Main agent: `gpt-live-1.1` with delegated `gpt-6-sol`, reasoning low, marin voice, 24 kHz.
- Caller: recorded-human audio replay in the isolated local loopback runtime.
- Text and expected-outcome judges: DeepSeek V4 Flash through OpenRouter, Fireworks pinned.
- Audio judge: the product’s current Gemini Flash-Lite chain; selected calls used Gemini 3.5 Flash-Lite.
- Scoring: configured `s2s-human-bench` family weights, task blending and pass threshold; failure QC covers reviewed failures and targeted diagnostic calls under those rubrics.
- Source runs: `fullrun_gptlive11_final_r1`, `fullrun_gptlive11_final_r2`, `fullrun_gptlive11_final_r3`.
- Build mix: v1 5 calls, v2 35, v3 167, v4 3. Functions and classes preserve the original fixture source; archive container byte identity is not claimed.
- Runtime limits: 100 s setup timeout; 900 s call budget.

## QC and verification

[Failure QC](failure-qc.md) · [Execution audit](execution-audit.json) · [Metric and aggregate validation](validate.py).

All 210 selected calls completed, recording formats/checksums passed, and no selected call has a tool-handler error. Thirteen targeted diagnostic calls also completed and were graded without judge errors.

`python validate.py` verifies 210 calls, 6,300 metric entries, cohort membership, all report aggregates and ELO metadata. `python recompute-elo.py --bootstrap` reproduces the ratings and intervals.

Response latency is unavailable on r1/H32 and r2/H32 because their caller recording has constant loud background sound; 208 calls contribute to latency statistics.

Historical hosting and judge parity is unconfirmed: this is local replay, and published comparison models may have different hosting/judges. Audio-dependent QC uses logs and waveform timing; independent listening was not performed. Some scenario/checker interpretation limits are documented in the QC.

Provider credentials, transcripts, recordings and private task targets are excluded from published artifacts.
