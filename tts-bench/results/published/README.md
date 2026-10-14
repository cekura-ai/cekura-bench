# Published results

One summary per model on the website's text-to-speech page, as written by
`bin/export-website.py summarize --amend`. They hold the run's records only:
no audio, no transcripts. Every run was measured from Vercel iad1 on corpus
0.3.0.

The website's data file is these summaries combined. Microsoft's two MAI models
are in `config/site-models.json` but not measured yet, so they are left out of
the model list until they are:

```bash
jq '.models |= with_entries(select(.key | startswith("azure-speech/") | not))' config/site-models.json > /tmp/site-models.json
uv run --locked python bin/export-website.py combine results/published/*.site.json --models /tmp/site-models.json --out tts-benchmark.json
```

The output equals the published file apart from its `generatedAt`. To add a
model, summarize its run, drop the summary here and combine again.

## Runs behind each summary

A model's base run is its full measurement; the amending run re-measured the
21 sentences reworded or added in corpus 0.3.0, and its reading of a sentence
replaces the base run's.

| Models | Base run | Amended by |
|---|---|---|
| Inworld TTS-2 Flash, Cartesia Sonic 3.6 and 3.5, Deepgram Aura-2 and Flux, ElevenLabs Flash v2.5, Smallest Lightning v3.1 Pro | `campaign-0928` | `campaign-1004` |
| ElevenLabs v3 Conversational, Gemini 3.8 Flash, 3.8 Flash-Lite and 3.1 Flash, GPT-4o mini TTS, Qwen3-TTS | `campaign-0928-fix` | `campaign-1004` |
| Grok TTS | `campaign-0928-fix` | `campaign-1004-xai` |
| ElevenLabs v4 Turbo and v4 | `campaign-0928-v4` | `campaign-1004` |
| Soniox TTS RT v2 | `campaign-1004-soniox` | none |

`campaign-0928-fix` re-ran the models whose recordings an early stream-end bug
had cut short, and `campaign-0928-v4` added the two v4 models. Grok's
`campaign-1004` run was refused on every request and was re-run as
`campaign-1004-xai`; Soniox was first measured on 2026-10-04, as one full run.
Each summary names its runs in `run` and `amendments`.
