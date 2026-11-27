# Production voice quality: ElevenLabs auto-shop agents

This snapshot refreshes ElevenLabs with three production auto-shop agents tested
November 24–25. Each received the same 15 generic scenarios, three times each.
The other 16 provider entries, including Cisco's November 27 result, are unchanged
from the [previous snapshot](../2026-11-27/README.md).

| Agent | Selected calls | Composite /100 | NISQA continuity /5 |
|---|---:|---:|---:|
| 1st Class Auto Service | 45 | 97.97 | 4.42512 |
| Omega Auto Werks | 45 | 98.12 | 4.45050 |
| Laurel Heights Automotive | 45 | 97.97 | 4.42316 |
| **ElevenLabs** | **135** | **98.02** | **4.43293** |

The provider score gives the three tested phone lines equal weight, using the
existing twelve metric weights and normalization. Latency remains a separate
measurement with zero composite weight. Expected Outcome is diagnostic only
and is not imported. These are live agents serving different businesses; the
comparison describes the tested deployments rather than all agents on a platform.

## Call selection

135 calls are selected from 146 executions. Six caller-execution defects were
replaced with validated reruns of the same scenarios; five additional retry
attempts are excluded. Selection is based on caller validity, not agent scores.
Each shop retains 15 scenarios ×3. `elevenlabs-selection.json` records every
selected call and its result/scenario lineage, replacement mappings and excluded
attempts. `elevenlabs-calls.json` contains the thirteen published metric values
and qualifying timing events for each selected call. All required metric records
are present. N/A is excluded from applicable denominators, never treated as a pass.

## Audio and timing

All 135 calls use the established isolated-AI NISQA pipeline and checkpoint:
`7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c`.
Only AI responses after the substantive caller request and before caller goodbye
are selected. At least five unique seconds are required; shorter model inputs
are repeated to exactly 30 seconds and long inputs are capped at 60 seconds.
The shortest eligible recording contains 5.823 seconds of selected AI speech.
`elevenlabs-nisqa-validation.json` records scores, selected spans, durations,
channel attribution and recording/model-input hashes. Audio and signed URLs are
not committed.

Timing uses the same qualifying-event export as the existing benchmark. Entry,
handoffs, responses to caller interruptions, unscripted idle checks, harmless
backchannels and terminal caller goodbyes are excluded where applicable. Latency
is the median of qualifying events within each number, then averaged across
numbers. Descriptive Stop Time is the mean of qualifying events within each
number, then averaged across numbers. Stop-time quality retains the fixed
300–3,000 ms anchors and call/scenario/number balancing.

## Rebuild and validate

```sh
python3 results/production-voice-quality/2026-11-27-elevenlabs/build_results.py
python3 results/production-voice-quality/2026-11-27-elevenlabs/validate.py
python3 -m pytest tests/test_elevenlabs_auto_quality_results.py -q
```

The full snapshot contains 17 providers, 34 tested phone lines and 1,530 selected
calls. Previous snapshots are retained. The website import contains aggregate
scores and coverage only, with no phone numbers, transcripts, recordings or call IDs.
