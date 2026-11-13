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
- Clip score = 100 − WER, on clips with 5+ reference words. Two models draw if their scores are within 5 points.
- Matches are weighted by the clip's word count; each measure's matches are scaled to its weight.
- A failed clip (model error or timeout) loses to a valid one. Clips that failed for our own reasons (sender pacing,
  provider concurrency limits) are missing, not losses.
- Short replies: score = −(errors); no margin. Entities: count of labelled entities captured exactly in the normalised
  transcript (same normaliser as WER); no margin. Only entities present in the normalised reference are scored.

## Fit
Bradley–Terry on all weighted matches; ELO = 1000 + 400·log10(p). 95% ranges come from resampling clips
(bootstrap); a model's rank is 1 + the number of models whose lower bound is above its upper bound. Latency and price
are not in the ELO.

## Running it
`python -m stt_bench.elo sources.json elo.json`, where `sources.json` lists, per measure and model, the per-clip
`results.json` files (oldest first; the newest valid row per clip wins), plus `entity_labels`. Set
`STT_ELO_JUDGMENTS` to the directories holding semantic-judge outputs (`{"id", "errors"}` JSON lines; pair id =
sha1(reference + "\0" + transcript)[:16]). `DRAWS` sets the bootstrap draws (default 200).

Condition runs: `STT_TURN_MANIFEST=datasets/private-turns-v1/<set>/manifest.json` selects the frozen noise set;
`STT_TURN_SMOKE_POLICY=timeout-counts-as-failure-v1` lets a genuine model failure on a smoke clip count as a lost clip
instead of blocking the model.
