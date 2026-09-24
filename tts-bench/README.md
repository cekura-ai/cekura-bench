# TTS component bench

A text-to-speech service measured at its own interface: text in, timestamped
audio chunks out. Transport, trunk, codec and telephony platform are out of the
benchmark. The unit measured is **"provider model under configuration X"**, and
every number is recomputable from the audio and the per-chunk arrival times each
cell writes to disk.

Methodology version: `tts/0.1`. Corpus version: `0.1.0`.

## What it measures

A voice agent feeds the service text as an LLM produces it, plays the audio the
instant it arrives, cuts it off when the caller interrupts, and has to say
phone numbers, confirmation codes and dollar amounts correctly. A first-byte
latency for a whole sentence on a warm socket answers none of that. This bench
measures the things an agent depends on:

| # | metric | what it answers |
|---|---|---|
| 1 | **TTFA** | how long a listener waits for the first *audible* sample: round-trip to the first chunk plus the silence inside the stream before speech |
| 2 | **streamed-input TTFA** | with words arriving at an LLM's cadence, how long from the first word, and how long from the last one; whether speech began before the text ended |
| 3 | **playout margin / underruns / stall** | once speech starts, does the stream keep up with a player consuming at exactly realtime |
| 4 | **cancel latency** | after a stop request, when the last chunk lands and how much audio arrived after the request |
| 5 | **continuation consistency** | one utterance sent whole versus as sentence frames with pauses: duration and onset deltas, both recordings kept for transcript comparison |
| 6 | **determinism** | the same text twice: byte-identical or not, duration and TTFA deltas |
| 7 | **tail under load** | N concurrent one-shots on N connections under one key |
| 8 | **round-trip WER and digit accuracy**, per cohort | did the words, and separately the digits, survive; scored offline |
| 9 | **telephony-native output** | capability metadata: does the service emit 8 kHz mu-law itself |

## Conventions, stated once

**t0 is the first text frame.** Connecting, authenticating and any per-context
setup frame happen before it. What each protocol excludes is recorded on the
adapter (`setup_excluded`) and printed in every report header. A cold
connection is not a service property and is not measured.

**Arrival is the anchor.** A chunk that arrives at `t` is playable from `t`;
sample `k` within it is heard at `t + k/rate`. A provider that batches a
sentence into one frame is later for the listener and is scored later.

**Leading silence is measured by a fixed rule**: the first 10 ms window whose
DC-removed RMS exceeds 1% of full scale, at 1 ms hops. A fixed threshold is
comparable across providers in a way an adaptive one is not. An adaptive
detector is run on the same audio and published beside it as a second opinion
(`leading_silence_adaptive_ms`), so a disagreement is visible in the row.

**The playout clock starts at the first audible sample**, not at the request.
Starting it at the request would count generation latency as underrun, which
TTFA already reports.

**One connection per cell, warmed before t0.** No cell's number depends on what
the previous cell did to the socket. The `repeat` probe measures the warm
second synthesis explicitly.

**Exclusions are declared, not discovered.** A probe names the features it
needs (`streamed_input`, `cancel`, `continuation`); an adapter that lacks one
publishes the cell as an exclusion with the reason and never connects.

**The transcription instrument is not the subject.** Scoring is a separate
offline pass (`bin/score-tts.py`) so the instrument can change without
re-running the provider. Two instruments run where two credentials exist; a row
on which they disagree by more than 10 points of WER is flagged, never averaged.
The instrument's own error on human speech with verified transcripts (the
*instrument floor*, `--floor`) is measured with the same normaliser and quoted
beside any WER.

**Either rendering is correct.** Every corpus item carries a `spoken_reference`
(the text written as it should sound). A transcript is scored against both the
text and the spoken form after one pinned normaliser, and the better match is
kept. Identifiers compare by their letters and digits however they were split;
digit sequences are checked separately because a task depends on them.

## Corpus

61 items in 12 cohorts (`tts_bench/corpus.py`): prose, currency, datetime,
phone, alnum, spelled, contact, names, repair, units, questions, long.
Production-shaped, invented values, published. Reported per cohort, never
pooled into one number; the prose cohort is the baseline the others are read
against. Any corpus in the same JSON shape runs through the identical probes
(`--corpus path/to/corpus.json`), which is how the hidden holdout and the
external grounding set are run.

## Grounding

`bin/fetch-tts-grounding.py` fetches two public sets at run time (nothing is
vendored):

- **Instrument floor**: human read speech with verified transcripts
  (LibriSpeech test-clean, CC BY 4.0). Scored with `bin/score-tts.py --floor`,
  it gives each transcription instrument's own error rate under the same
  normaliser. A TTS WER is read relative to this.
- **External hard cases**: English seed prompts from a published TTS
  evaluation set's "Complex Pronunciation" and "Questions" categories
  (EmergentTTS-Eval, Apache-2.0), written in this bench's corpus shape. Running
  the identical probes on an independently authored set is how the in-house
  cohorts are checked for authoring bias.

## Providers

| key | protocol | streamed input | cancel | continuation | native 8 kHz mu-law | notes |
|---|---|---|---|---|---|---|
| `elevenlabs` | websocket, multi-context | yes | `close_context` | yes | yes | option `auto_mode=true|false` (default false: generate on flush) |
| `cartesia` | websocket, `context_id` | yes | `cancel` | yes | yes | |
| `deepgram` | websocket, one utterance at a time | yes | `Clear` | yes | yes | the voice is the model string |
| `openai` | HTTP streaming | no | no | no | no | whole text per request |
| `gemini` | HTTP server-sent events | no | no | no | no | whole text per request; audio arrives in one or a few parts |

Adapters are keyed by wire protocol: adding a model on a known protocol is a
registry entry. Pending protocols are listed in `tts_bench/registry.PENDING`.

## Running

Run every command from `tts-bench/`, with Python 3.12 or 3.13 and `uv`. Keys are
read from the environment or from a dotenv file passed with `--env`; see
`.env.example` for the names.

```bash
uv sync --locked
uv run --locked bin/run-tts.py --provider elevenlabs --suite latency --repeats 3     # ELEVENLABS_API_KEY
uv run --locked bin/run-tts.py --provider cartesia --suite streaming --cohort phone   # CARTESIA_API_KEY
uv run --locked bin/run-tts.py --provider deepgram --probe cancel --probe concurrency
uv run --locked bin/run-tts.py --provider elevenlabs --probe streamed_input --option auto_mode=true
uv run --locked bin/score-tts.py data/tts/<run> --instrument deepgram --instrument openai-whisper
uv run --locked python -m tts_bench.report data/tts/<run>
uv run --locked pytest                                                                # no provider requests
```

Suites: `smoke`, `latency` (one_shot, repeat), `streaming` (streamed_input at
30 and 10 words/s, continuation), `interaction` (cancel at 300 and 1000 ms),
`load` (concurrency 8), `full`. Every run starts with the **sentinel**: three
one-shots of one fixed prose item, published beside the results and never
folded in. Its spread across runs is the noise a ranking gap has to exceed.

## What a run writes

```
data/tts/<run>/
  provenance.json         provider, model, voice, options, capabilities, corpus and methodology versions, harness commit
  plan.json               every planned cell, sentinel first
  cells.jsonl             one line per finished cell (appended as it happens)
  summary.json            counts, rewritten after every cell
  scores.jsonl            offline transcription scores, one per cell (after bin/score-tts.py)
  report.md / report.json
  <probe>/<item>/rN/
    audio-<context>.wav   every synthesised utterance, at the provider's native rate
    syntheses.json        per context: t0, every text frame, first/last chunk, cancel, done, and the per-chunk arrival timeline
    events.jsonl          the normalised event log
    raw.jsonl             every provider frame, audio payloads elided to their size
    cell.json             the self-contained record: identity, text and spoken reference, capabilities, result, error, environment
```

## Statistics

Per probe variant × cohort: P50 with a clustered bootstrap interval (2000
draws, seed 7) and P90; P95/P99 withheld under n=30; pass rates with intervals;
voids tallied by class with exclusions listed first. Word error is pooled per
cohort (sum of errors over sum of reference words) per instrument, with the
digit-match rate beside it. No composite score, no cross-cohort mean.

## Not in this bench

A naturalness rating: a small-N preference board would be noise wearing a
number, and if naturalness is wanted later the honest form is an expert-rated
pass on this corpus reported as a cohort pass rate with the rater count. A
single "best TTS" score. Anything below the service interface.
