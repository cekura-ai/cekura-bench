# STT benchmark ELO

One rating that combines clean accuracy, noise conditions, short replies and entities, fitted the same way as the
Speech-to-Speech Real Human Calls rating. Every published WER figure stays as it is; the ELO adds one ranking on top.

## Measures and weights
| Measure | Weight | Clips |
|---|---|---|
| WER, Ocular private turns (clean) | 10% | 206 turns |
| WER, Pipecat public set | 10% | 1,000 clips |
| Semantic WER (Pipecat judge prompt, verbatim) | 10% | Ocular turns |
| Phone line (G.711, 8 kHz) | 12% | 206 turns |
| Call-center noise (chatter 10 dB below speech, G.711) | 10% | 206 turns |
| Babble noise (6 voices, 5 dB below speech) | 8% | 206 turns |
| TV noise (5 dB below speech) | 5% | 206 turns |
| Reverb (0.6 s) | 5% | 206 turns |
| Short replies (< 5 words), error count | 10% | Ocular turns |
| Entities captured exactly | 20% | Ocular turns |

A measure with no data for any model is dropped and the rest renormalized.

## Per-clip rules
- Every clip with reference words counts. Clip score = 100 − WER; two models draw if their scores are within 5 points.
  Short replies (< 5 words) also form their own measure, scored by error count.
- Matches are weighted by the clip's word count; each measure's matches are scaled to its weight.
- The per-measure WER in the output table is plain WER: all word errors over all reference words of the model's valid
  clips, with no floor. Only the match scores are floored at 0.
- A failed clip (model error or timeout) loses to a valid one. Clips that failed for our own reasons (sender pacing,
  provider concurrency limits, or rows flagged `our_side_failure` after checking the raw provider events) are missing,
  not losses. Transcripts the harness failed to stitch are rebuilt from the raw events first (ElevenLabs Scribe v3 Live
  uses `elevenlabs-committed-segments-v4`, set in its model config and its turn profile).
- Short replies: score = −(errors); no margin. Entities: count of labelled entities captured exactly in the normalised
  transcript (same normaliser as WER); no margin. Only entities present in the normalised reference are scored.

## Fit
Bradley–Terry on all weighted matches; ELO = 1000 + 400·log10(p). 95% ranges come from a bootstrap
that resamples whole Ocular conversations (one draw shared by every measure built on them, since turns of a
conversation share speakers and audio) and the Pipecat clips; a model's rank is 1 + the number of models whose lower bound is above its upper bound. Latency and price
are not in the ELO.

## Running it
`python -m stt_bench.elo sources.json elo.json`, where `sources.json` lists, per measure and model, the per-clip
`results.json` files (oldest first; the newest valid row per clip wins), plus `entity_labels`. Set
`STT_ELO_JUDGMENTS` to the directories holding semantic-judge outputs (`{"id", "errors"}` JSON lines; pair id =
sha1(reference + "\0" + transcript)[:16]). `DRAWS` sets the bootstrap draws (default 200).

Condition runs: `STT_TURN_MANIFEST=datasets/private-turns-v1/<set>/manifest.json` selects the frozen noise set;
`STT_TURN_SMOKE_POLICY=timeout-counts-as-failure-v1` lets a genuine model failure on a smoke clip count as a lost clip
instead of blocking the model.
