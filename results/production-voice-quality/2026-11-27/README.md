# Production voice quality: Cisco rerun, November 27

This snapshot replaces Cisco's November 24 calls with a rerun on November 27, after Cisco
changed the number and tenant of its agent. The other 16 providers keep their November 12
reference scores, exactly as in the [November 24 snapshot](../2026-11-24/README.md).

**Voice tone is provisional.** Each call's voice-tone value is Cekura's platform Voice Tone +
Clarity score. The other providers, and Cisco's November 24 calls, use the isolated-AI NISQA
score. Until those NISQA scores replace the provisional values, the composite and rank here are
not comparable for publication. `results.json` records this as `cisco_audio_source`.

## Result

| | November 24 | November 27 |
|---|---|---|
| Composite (12 weighted metrics) | 93.95, 3rd of 17 | 93.19, 4th of 17 (provisional voice tone) |
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

## Finishing the snapshot

1. Score the agent's isolated speech in each call with the NISQA checkpoint used for
   November 24 (`7ec4cf93…`).
2. Replace each call's `audio` value in `cisco-calls.json` with that score.
3. Set `AUDIO_SOURCE = 'nisqa'` in `build_results.py`.
4. Rebuild, validate and test:

```sh
python3 results/production-voice-quality/2026-11-27/build_results.py
python3 results/production-voice-quality/2026-11-27/validate.py
python3 -m pytest tests/test_cisco_quality_results_2026_11_27.py -q
```

The test that pins the provisional source will then need updating. The website report and
the website import come after this step.
