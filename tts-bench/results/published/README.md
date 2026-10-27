# Published results

One summary per model on the website's text-to-speech page, as written by
`bin/export-website.py summarize --amend`. They hold the run's records only:
no audio, no transcripts. Every run was measured from Vercel iad1 on corpus
0.3.0.

The website's data file is these summaries combined:

```bash
uv run --locked python bin/export-website.py combine results/published/*.site.json --models config/site-models.json --out tts-benchmark.json
```

The output equals the published file apart from its `generatedAt`. To add a
model, summarize its run, drop the summary here and combine again.

## Runs behind each summary

A model's base run is its full measurement; the amending run re-measured the
21 sentences reworded or added in corpus 0.3.0, and its reading of a sentence
replaces the base run's.

| Models | Base run | Amended by |
|---|---|---|
| Cartesia Sonic 3.6 and 3.5, Deepgram Aura-2 and Flux, ElevenLabs Flash v2.5, Smallest Lightning v3.1 Pro | `campaign-0928` | `campaign-1004` |
| ElevenLabs v3 Conversational, Gemini 3.8 Flash, 3.8 Flash-Lite and 3.1 Flash, GPT-4o mini TTS, Qwen3-TTS | `campaign-0928-fix` | `campaign-1004` |
| Grok TTS | `campaign-0928-fix` | `campaign-1004-xai` |
| ElevenLabs v4 Turbo and v4 | `campaign-0928-v4` | `campaign-1004` |
| Soniox TTS RT v2 | `campaign-1004-soniox` | none |
| Rime Coda and Mist v3, Fish Audio S2.1-Pro | `campaign-1013` | none |
| Microsoft MAI-Voice-2.1 and MAI-Voice-2.1-Flash | `campaign-1013-azure` | none |
| Fish Audio Drama 3 (preview) | `campaign-1013-drama3` | none |
| Hume Octave 2 (preview) | `campaign-1014-hume` | none |
| Gradium (default), voice Everly | `campaign-1018b-gradium-everly` | none |
| Inworld TTS-2 Flash and TTS-2 (creative delivery) | `campaign-1019-inworld` | none |
| Bland TTS V3, voice Alexandra | `campaign-1027-bland-alexandra` | none |

`campaign-0928-fix` re-ran the models whose recordings an early stream-end bug
had cut short, and `campaign-0928-v4` added the two v4 models. Grok's
`campaign-1004` run was refused on every request and was re-run as
`campaign-1004-xai`; Soniox was first measured on 2026-10-04, as one full run.
The `campaign-1013` runs and later were measured on corpus 0.3.0 from the start,
so they need no amending run. `campaign-1019-inworld` re-measured Inworld after
its new checkpoint rolled out, with the voice Sarah; it replaces the
`campaign-0928` Flash run. Gradium is shown with Everly, its first choice of the
ten voices it recommended, in place of the `campaign-1013` Harper run. The
sandbox's OpenAI key had been rejected for both campaigns, so their Whisper
transcripts were scored offline with `bin/score-tts.py`.
Bland is shown with Alexandra: a first run, `campaign-1027-bland`, used the
featured voice Karen, which is narrow-band (no energy above 4 kHz) and so says
more about the voice than the model.
Each summary names its runs in `run` and `amendments`.
