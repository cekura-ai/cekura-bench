# Gradium voices

Ten voices Gradium named in its order of preference, each measured on the full
suite with `scripts/sandbox.mjs launch --providers gradium --voice <id>`, one
label per voice: `campaign-1018b-gradium-<voice>` on harness `664e183` for the
first six, `campaign-1019-gradium-<voice>` on `8b4c736` for the last four (the
harness change between them touches only Inworld). Same model (`default`), site
(Vercel iad1) and corpus (0.3.0) throughout.
The summaries are `bin/export-website.py summarize` output: records only, no
audio.

Gradium's key allows two active sessions, so the voices ran two at a time.

| Voice | | First audio (median / p90) | Streaming, after the last word | Hard parts read right | Codes read right | Word errors | Naturalness |
|---|---|---|---|---|---|---|---|
| Everly | F | 76 / 105 ms | −64 ms | 80.8% | 76.2% | 2.7% | 4.51 |
| Zoey | F | 78 / 106 ms | −62 ms | 78.5% | 69.0% | 3.0% | 4.52 |
| Harper | F | 76 / 105 ms | −62 ms | 75.8% | 61.9% | 2.9% | 4.52 |
| Nora | F | 76 / 112 ms | −62 ms | 79.5% | 71.4% | 2.6% | 4.51 |
| Hadley | F | 76 / 107 ms | −62 ms | 81.3% | 78.6% | 2.5% | 4.53 |
| Russell | M | 76 / 89 ms | −64 ms | 78.5% | 66.7% | 3.0% | 4.50 |
| Garrett | M | 76 / 90 ms | −63 ms | 77.6% | 64.3% | 2.7% | 4.50 |
| Sterling | M | 76 / 96 ms | −62 ms | 77.2% | 64.3% | 2.8% | 4.50 |
| Marcus | M | 75 / 90 ms | −62 ms | 77.6% | 64.3% | 2.6% | 4.50 |
| Damon | M | 76 / 112 ms | −62 ms | 78.1% | 64.3% | 3.0% | 4.51 |

Reading it:

* Accuracy is Deepgram Nova-3's alone. The OpenAI key was rejected (HTTP 401)
  for this campaign, so the Whisper half of each summary is all instrument
  errors and is left out of every row.
* Codes rest on 42 spans, so a few points is a few spans. The published Harper
  run (`campaign-1013`) read 69.0% of codes and 78.1% of hard parts on the same
  instrument, against 61.9% and 75.8% here: that spread is run-to-run noise.
