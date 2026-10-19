# Gradium voices

Ten voices Gradium named in its order of preference, each measured on the full
suite with `scripts/sandbox.mjs launch --providers gradium --voice <id>`, one
label per voice (`campaign-1018b-gradium-<voice>`). Same model (`default`),
site (Vercel iad1), corpus (0.3.0) and harness commit (`664e183`) as each other.
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
| Garrett | M | 76 / 90 ms | not measured | 78.5% | 64.3% | 2.6% | 4.50 |
| Sterling | M | 76 / 96 ms | not measured | 78.1% | 61.9% | 2.7% | 4.50 |

Reading it:

* Accuracy is Deepgram Nova-3's alone. The OpenAI key was rejected (HTTP 401)
  for this campaign, so the Whisper half of each summary is all instrument
  errors and is left out of every row.
* Codes rest on 42 spans, so a few points is a few spans. The published Harper
  run (`campaign-1013`) read 69.0% of codes and 78.1% of hard parts on the same
  instrument, against 61.9% and 75.8% here: that spread is run-to-run noise.
* Garrett and Sterling ran out of Gradium credits partway: one-shot and repeat
  cells are complete, streamed input and continuation were refused, so they
  have no streaming figure. Marcus (`r2sIQdqqoqgRJuXw`) and Damon
  (`KUpE0JVhjiIzp1Fk`) are not yet measured.
