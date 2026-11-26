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
| ElevenLabs v4 Turbo and v4 (text normalization on) | `campaign-1110-elevenlabs-norm` | none |
| Soniox TTS RT v2 | `campaign-1004-soniox` | none |
| Rime Coda and Mist v3, Fish Audio S2.1-Pro | `campaign-1013` | none |
| Microsoft MAI-Voice-2.1 and MAI-Voice-2.1-Flash | `campaign-1013-azure` | none |
| Fish Audio Drama 3 (preview) | `campaign-1013-drama3` | none |
| Hume Octave 2 (preview) | `campaign-1014-hume` | none |
| Gradium (default), voice Everly | `campaign-1018b-gradium-everly` | none |
| Inworld TTS-2 Flash and TTS-2 (creative delivery) | `campaign-1019-inworld` | none |
| Murf Falcon 2.1, voice Judith | `murf-falcon-judith-20261113` | none |
| SpeechifyAI Simba 3.5 and Simba 3.2, voice Harper | `speechify-simba-20261109` | none |
| SpeechifyAI Simba 3.0, voice Harper | `speechify-simba-3.0-20261117` | none |
| SpeechifyAI Simba 3.5, 3.2 and 3.0 over HTTP streaming, voice Harper | `speechify-http-20261117` | none |
| Mistral Voxtral TTS 3 (preview), voice Margaret | `mistral-voxtral-margaret-us-20261106` | none |

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
`campaign-1110-elevenlabs-norm` re-ran ElevenLabs v4 Turbo and v4 with
`apply_text_normalization=on`, as ElevenLabs recommends for v4; it replaces their
`campaign-0928-v4` runs and their `campaign-1004` amendment.
SpeechifyAI's site rows come from two endpoints of the same model and voice, as
`config/site-models.json` says with `streamedFrom`: every one-shot figure from
`speechify-http-20261117` (`POST /v1/audio/stream`, the endpoint SpeechifyAI asked
us to time), and the streamed-input and continuation figures, which a
whole-text request cannot measure, from the websocket runs
(`speechify-simba-20261109`, `speechify-simba-3.0-20261117`). Both sets of
summaries are published; the site row names the HTTP run in `run` and the
websocket run in `streamedFrom`.
Each summary names its runs in `run` and `amendments`.

On 2026-11-23 every summary except AssemblyAI TTS 1's was rescored from its
runs' stored transcripts with `rescore_run`, after hard parts began accepting
more written forms of a correct reading ("6.05 a.m.", "36°F", "the 2-4 window",
"control plus shift plus T"). Only hard parts changed. AssemblyAI TTS 1 keeps
its earlier hard parts until its run's records are rescored.

On 2026-11-26 every summary except AssemblyAI TTS 1's was rescored again, with
`bin/score-tts.py --rescore --arbitrate`:

- Word error puts written shorthand into words the same way on both sides
  (times written "2.15am", emails and web addresses, "+", "(555)", "5'11''",
  "co-pay", "o'clock") and chooses between the text as sent, its spoken
  reference and any accepted reading word by word, so "Chapter four covers
  Henry VIII" costs nothing. Three readings were accepted: "Eastern Time" for
  "ET", "Rylee" for "Riley", and "I verified" for "I've verified", which sound
  alike.
- Hard parts get one verdict each instead of each transcriber's check counting
  apart: a term must be heard right by both transcribers; any other kind is
  decided by the majority, and where Deepgram and Whisper split, Zoom Scribe
  (`zoom-scribe-en-pro-v2.1`) transcribed that one-shot recording. 227
  recordings across 44 runs went to Zoom; none is left unresolved. The new
  `hardParts` field of a summary holds the verdicts; `accuracy` keeps each
  transcriber's own figures.

Only `accuracy`, `hardParts` and `instrumentDisagreements` changed.
