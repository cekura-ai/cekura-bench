# Production voice quality: Cisco rerun, November 27

This snapshot replaces Cisco's November 24 calls with a rerun on November 27, after Cisco
changed the number and tenant of its agent. The other 16 providers keep their November 12
reference scores, exactly as in the [November 24 snapshot](../2026-11-24/README.md).

**Voice tone is final.** All 45 calls use isolated-AI NISQA scoring with the same
checkpoint and audio preparation as the November 24 snapshot. `results.json`
records `cisco_audio_source: nisqa`. The displayed Cisco phone number is omitted
at Luis's request; the result and call links identify the tested data.

## Result

| | November 24 | November 27 |
|---|---|---|
| Composite (12 weighted metrics) | 93.95, 3rd of 17 | 93.08, 5th of 17 |
| Median qualifying turn latency | 3,760 ms over 77 events | 1,920 ms over 82 events |
| Mean interruption stop time | 507 ms over 6 events | 793 ms over 6 events |

## Selection and scoring

- 45 calls: the same 15 scenarios, three repetitions each, all from one result.
- The caller stayed on script in every call, so no call was replaced.
- Every call has all 14 metrics scored. Six calls have no provider call details. Scoring reads
  Cekura's own transcript and audio, so those calls are still scored.
- Each call is rebuilt from the platform run with the same rules as November 24. On the
  November 24 calls, that rebuild reproduced 12 of the 13 metric keys and both event lists
  exactly. The one difference was voice tone.
- Aggregation, weights, normalization and stop-time anchors come unchanged from the
  November 24 builder.

## NISQA scoring and validation

The 45 recordings were scored using the same local pipeline and checkpoint as #94:
`7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c`.
Only the isolated AI response spans after the substantive caller request and before
caller goodbye were used. Source speech shorter than 30 seconds was repeated to
30 seconds; inference was capped at 60 seconds. All calls exceeded the five-second
minimum (shortest 11.324 seconds). No provisional scores, exclusions, replacements,
or manual metric overrides were needed. Mean audio continuity is 4.4479775/5.

[cisco-nisqa-validation.json](cisco-nisqa-validation.json) records each model score,
AI channel, response spans, durations and recording hash. Recordings and signed URLs
are not committed. The other twelve metric keys and all timing events are unchanged.
The audio-model coverage count includes Cisco's 45 newly scored calls (1,387 total).
The 16 reference providers and previous snapshots remain unchanged.

Rebuild, validate and test:

```sh
python3 results/production-voice-quality/2026-11-27/build_results.py
python3 results/production-voice-quality/2026-11-27/validate.py
python3 -m pytest tests/test_cisco_quality_results_2026_11_27.py tests/test_cisco_quality_results.py -q
```

Website PR #235 imports this final snapshot. The existing November 24 HTML renderer
contains its historical findings; it was not reused to present November 27 calls.
