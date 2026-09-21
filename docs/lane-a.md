# Lane A — controlled interaction

Lane A measures a **provider realtime service under one named configuration**. The
caller is ours, the audio is ours, and the connection goes straight to the
provider's websocket. No orchestration framework sits in the measured path.

It is one of two ranked lanes. Lane B measures a complete reference agent with
tools and state over real telephony. Results are never ranked across lanes and
there is no composite score, because "the model is fast" and "the deployment is
fast" are different claims and collapsing them loses both.

## The one structural advantage

**The caller audio is authored here, so the caller-side boundary is exact.**

A latency is the distance between two instants: when the caller stopped talking
and when the agent became audible. Recover both from a recording and the
measurement inherits the detector's error at each end. Here the first instant is
a sample index in a file written for the purpose, so it carries no error at all,
and only the agent side is detected. Published latency therefore resolves to
about ±10 ms on clean audio rather than twice that.

That single fact is what makes the rest of this lane possible, including the
manual-commit configuration below, which has no meaning without it.

## What it measures

| Probe | Question | Output |
|---|---|---|
| `response_latency` | Authored speech-end to first audible agent sample | ms, per configuration |
| `endpointing_ladder` | Does a mid-utterance pause of *n* ms get cut off? | a curve, one rung per run, two ladders |
| `filled_pause` | Does "um, let me think" inside the pause buy the caller time? | pass/fail per gap |
| `barge_in` | Does the agent yield the floor when spoken over, and how fast? | pass/fail + ms, at several offsets |
| `simultaneous_start` | Both start talking at once — who yields? | pass/fail + ms |
| `barge_in_correction` | The caller interrupts to correct themselves | pass/fail + ms |
| `backchannel_tolerance` | Does "mm hmm" mid-reply derail the agent? | pass/fail |
| `false_trigger` | Noise on the line, caller silent — does it speak anyway? | rate per minute |
| `caller_transcription` | How accurately did it hear us? Digits scored apart from words | WER + digit match |
| `task` | Twenty closed-loop scenarios against the published tool contracts | tool-trace pass/fail |

Every probe also runs under seven published **degradation transforms** of the
caller audio (pink noise at two SNRs, a narrowband mu-law phone leg, a far-field
room, clipping, dropouts), reported as the **delta from clean** and never as an
absolute. The transforms are seeded and regenerable from the clean master
(`lane_a/transforms.py`), so the degraded audio can be reproduced by anyone.

### The two turn-detection configurations

Every provider is run twice, and the pair is the point.

- **Native VAD** — what a customer gets out of the box.
- **Manual commit** — because we authored the tape, we can declare the turn over
  at the exact sample the speech ends. That yields a **generation-latency floor
  with zero endpointing error**.

Native minus manual is that provider's **endpointing cost, measured**. This
subtraction is legitimate where subtracting transport overhead was not: same
adapter, same audio, same connection, same lane. Only the boundary decision
changes. Both numbers are always published; the difference never replaces them.

An early run of the latency suite, three repeats on one voice, shows the shape:

| Configuration | Latency (median of 3) | Endpointing cost |
|---|---|---|
| manual commit | ~975 ms | — floor |
| native VAD, 200 ms silence | ~1099 ms | ~124 ms |
| native VAD, 500 ms silence | ~1333 ms | ~358 ms |
| semantic VAD | ~5284 ms | ~4309 ms |

Three repeats is a shakedown, not a result. Publication needs the repeat count,
confidence intervals and void rules described below.

### Exclusions are cells

A configuration a provider has no equivalent for — semantic turn detection on a
provider that offers none, a numeric VAD threshold where only sensitivity
levels exist — is declared by the adapter up front and recorded as a **voided
cell with its reason**, without a connection being opened. The gap is then
visible in the table as an exclusion, rather than as a row that quietly went
missing.

## How the caller works

A **continuous realtime carrier** holds the stream open for the whole call and
clips are dropped into it. This is not a stylistic choice: server-side
endpointers decide a turn ended by observing silence *in the stream*, so a caller
that simply stops sending is never heard to stop talking and the provider waits
forever. It is also what a phone line does, which keeps this lane and the
telephony back-end identical in shape.

Turns are **anchored on observed events**, not on absolute offsets. When a real
caller would interrupt depends on when the agent started talking and how long it
talks, so a fixed tape punishes a verbose model with artificial overlap and
rewards a silent one with an unrealistic pause. Fixed stimuli survive only where
nothing the agent does should change what the caller says next.

Two clocks are kept apart on purpose:

- **Control** — when to speak next — runs off live events and the playout model. Coarse.
- **Measurement** — what gets published — runs offline over the recorded audio.

## The reference client

Providers deliver audio faster than realtime. One ships close to a second of
speech in its first frame and the rest of a reply within the next second; its
`turnComplete` arrives seconds later, paced to when playout would finish. So the
harness models the listener explicitly, and every number that concerns the *end*
of agent speech is measured on that model rather than on arrival times:

- A chunk starts playing at its arrival, or when the previous chunk drains,
  whichever is later (`AudioTimeline.playout_*`). Onset still uses arrival,
  because nothing is queued ahead of a reply's first sample.
- The client **clears its buffer** when the provider reports that the caller
  started speaking, or that it interrupted its own reply. That is what these
  protocols document a client should do on those events, and it is applied
  identically to every provider. The cut is recorded on the timeline
  (`cuts` in `timelines.json`) and in the event log (`cleared_playback`).

Barge-in is therefore *caller onset → the listener stops hearing the agent*. A
provider whose endpointer reacts promptly stops the listener promptly, however
much reply it had already pushed down the wire; one that never signals is heard
to the end of whatever it sent, because a client has no other way to know. The
audio that signal threw away — delivered, paid for, never heard — is published
as `discarded_ms`, and whether the provider also stopped generating as
`provider_cancelled`.

Without this model, arrival times would have credited a provider that dumped
its whole reply in one burst with finishing before anyone heard the end of it,
and could never have credited it with stopping at all. The caller uses the same
model to decide when the agent has finished, so it no longer speaks into a reply
the listener is still hearing.

## Timing anchors

Stated once, and published with every result set:

- **Outbound.** A chunk handed to the socket at `t` carries its first sample at
  `t`; sample *k* within it is at `t + k/rate`. Sends are paced in 20 ms chunks,
  well inside detector error, and pacing slip is measured. A run where this host
  fell behind is voided rather than blamed on the provider.
- **Inbound.** A chunk arriving at `t` is playable from `t`; sample *k* within it
  reaches a listener at `t + k/rate`. **Arrival, not generation, is the anchor.**
  A provider that batches its first reply into one large frame *is* later for the
  listener, and is scored later.

## Detector limits

Calibrated against signals whose boundaries are exact by construction.
Resolution on clean audio is about **±10 ms**. Under noise it runs systematically
**late** — about +53 ms at 10 dB SNR, and it fails entirely at 0 dB.

Left unchecked that would have manufactured "providers are slower under noise"
out of the instrument. It does not bite in Lane A, because the noise sits on the
caller channel while the provider's returned audio comes back clean. The rule is
explicit: **this detector may not be pointed at a noisy channel.** Lane B's phone
leg needs harmonicity-based detection instead.

Gaps smaller than detector error are reported as ties.

## Statistics and voids

The rules live in one place, `lane_a/report.py`, and run over `cells.jsonl`:

- The sampling unit is the scenario-repeat, minimum five for publication.
  Success is published two ways: the **per-run rate** over all repeats and the
  **observed all-repeats rate**, the share of scenarios in which every repeat
  passed. The second is what was seen, never a rate raised to the power of the
  repeat count — that transform produces a confident-looking number out of a
  handful of runs while describing nothing that happened.
- Confidence intervals are **clustered bootstrap** intervals (2000 draws, fixed
  seed), resampling scenarios for tasks and repeats for a single probe variant.
- Latency is **P50 and P90** until n reaches 30; P95 and P99 are withheld below
  that rather than reported from too few.
- Gaps inside the detector's resolution (10 ms on clean audio) are **ties**.
- Strata — configuration, caller voice, transform, modality — are never
  averaged across. Degradation is a delta from the clean stratum.
- A **sentinel** cell (response latency, `open.book`, native VAD at 500 ms, one
  voice, three repeats) leads every campaign whatever its suite. Its spread
  across campaigns is the run-to-run noise of instrument plus provider, and is
  published beside the rankings; a gap smaller than it is not a ranking.

Void rules, fixed before running:

- Provider error before the first agent audio → **void**, rerun, void count published.
- Caller-side failure, including pacing slip on our host → **void**.
- Agent connected and never spoke → **fail**, not void.

## Artifacts, and rechecking a number without trusting us

A run produces a directory, not a number. Per cell:

| File | What it carries |
|---|---|
| `caller.wav` / `agent.wav` | exactly what was sent and what came back |
| `timelines.json` | per-chunk timestamps: when each sample became audible |
| `events.jsonl` | the normalized event stream |
| `raw.jsonl` | every provider frame verbatim |
| `cell.json` | the whole cell, self-contained: see below |

The timeline file is the one that is easy to forget and fatal to omit. Audio and
a transcript show *what* was said; only per-chunk timestamps show *when it could
be heard*, and every latency here is a difference between two of those instants.

`cell.json` is written so that a **single cell directory answers every question
about itself** with no run root, no repository and no access to us. It carries
the probe's own parameters, the session as requested *and* as the provider
acknowledged it, the caller clips with checksums and sample boundaries, both
transcripts, the tool calls with their arguments and the contract's replies,
token usage, pacing slip, the harness commit with a dirty flag, the detector
constants, and a checksum for every other file beside it.

Three of those are there because the alternative is a rerun:

- **The acknowledged session, not only the requested one.** We asked for server
  VAD at 500 ms; the provider came back with `threshold 0.5`, `prefix_padding_ms
  300`, `idle_timeout_ms null`, `create_response true`, `interrupt_response
  true`. Those are part of the configuration under test and none of them were
  ours. A cell recording only our request would publish a configuration nobody
  actually ran.
- **The detector constants.** They decide where the agent-side boundary lands.
  Change `margin_db` and every published latency moves while the commit stays the
  same, so a result is only meaningful next to the values that produced it.
- **The uncommitted patch.** A commit hash identifies the code only when the tree
  is clean. When it is not, `harness.patch` travels with the run and the record
  says so rather than implying a clean checkout.

## Nothing waits until the end

A campaign against a live provider costs money, takes real time, and cannot be
reproduced later because the model behind the endpoint changes. So the plan is
written before the first connection, each cell's record lands as that cell
finishes, and the run summary is rewritten after every cell. An interrupted run
is a partial result, not a lost one, and `run.json` names the cells that never
ran instead of leaving a short directory looking complete.

Then the run audits itself, and the CLI exits non-zero if it does not pass:

```bash
python -m lane_a.audit data/lane-a/<run>
```

It checks that every planned cell exists, that the files present match what the
record says happened, that every checksum still holds, and that each published
latency still recomputes from the files. Finding a gap now costs one rerun;
finding it in a month costs a rerun against a provider that has changed
underneath the result, which is not the same measurement.

`lane_a/recompute.py` derives the published latency from those files alone. It
imports nothing from the runner and opens no socket:

```bash
python -m lane_a.recompute data/lane-a/<run>/response_latency-open.book/manual/f-us/clean/r1
```

Across the live cells checked so far it reproduces the published figure exactly.
Disagreeing with our detector is also supported — the audio and the boundaries
are right there, so run your own over the same files.

Provenance is per cell rather than per release for a practical reason: a harness
and the results it produced drift apart as soon as either can change without the
other, and a number whose exact method cannot be recovered has stopped being a
measurement. Stamping each cell keeps the two attached.

## Corpus

Scenario content is authored here, because these clips are replies to *our* agent
in *our* scenarios and a corpus of unrelated recordings cannot answer a question
the agent has just asked. Public audio is used where it genuinely fits and nowhere
else: content-free interaction tokens, where prosody carries the meaning and the
words carry none, and noise beds.

One canonical master per clip at 24 kHz, resampled to each provider's rate by the
published filter in `lane_a/audio.py`. Rendering separately per rate would ship
differently filtered audio to different providers, which measures the resampler
as much as the provider.

Provenance travels with each clip and therefore with each cell: `tts:<vendor>/<voice>`
or `human:<speaker>`. v1 may publish on TTS audio so labelled; a human re-record
of the same script is the first sensitivity release after it. If re-recording
moves a ranking, that is the finding.

Results are **stratified by voice**, never averaged over it. Voice-sensitive
endpointing is itself a finding and a single-voice corpus hides it completely.

Corpus v1 is 54 clips in three voices. Task scenarios are **data**
(`lane_a/scenarios.py`): an opening clip, an ordered routing table of literal
patterns that picks the caller's next line from the agent's last sentence, the
tool trace that counts as success, tools that must not be called, and what
ends the conversation. Twenty scenarios cover the appointments contract
(booking with a named provider, a full day, a date range, a mid-sentence
correction, a new patient, a failed lookup, cancellations including one that
fails, reschedules that must book before they cancel, a lookup, an emergency
that must call no tool) and the intake contract (a qualified handoff, a refused
consent, a member-services redirect). The routing table is published with the
scenario, because a caller that improvises is a second model in the
measurement.

A **hidden holdout** is authored beside the public set in the same schema —
every clip id re-phrased, the scenarios with `.h` ids, ladder gaps and barge-in
offsets shifted — and runs through the identical probes with `--holdout <dir>`.
Anything published becomes training data; the holdout is what a ranking is
checked against that it cannot have been fitted to. Runs on it carry their own
corpus version and label so the two are never mixed in a report.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lane-a.txt

# render the caller corpus (needs ELEVENLABS_API_KEY)
.venv/bin/python bin/render-corpus.py

# measure one suite, stratified by voice
.venv/bin/python bin/run-lane-a.py --provider openai-realtime --suite latency --repeats 5 \
    --voice f-us --voice m-us --voice f-gb

# or the whole campaign for one provider, every suite in sequence
LANE_A_ENV=.env bin/run-campaign.sh gemini-live

# aggregate, and compare runs across providers (ties declared inside detector resolution)
.venv/bin/python -m lane_a.report data/lane-a/<run-a> data/lane-a/<run-b> --out comparison.md
```

Suites: `smoke`, `latency`, `endpointing`, `interaction`, `noise`,
`transcription`, `robustness`, `task`, `task-text`, `task-medicare`,
`task-medicare-text`. `--transform <name>` runs any suite under a degradation;
`--scenario <id>` narrows a task suite; `--holdout <dir>` swaps in the hidden set.
Every run writes `report.md` and `report.json` beside its cells.

The harness is tested against a **scripted agent** with known reply timing before
any provider is involved (`tests/test_caller.py`). That ordering matters: if the
first time a barge-in anchor runs is against a live model, a harness bug and a
model behaviour look identical, and an untested harness has to be taken on trust.
It has already earned its keep — it caught a barge-in metric that counted the
provider's *next* reply as the tail of the interrupted one, which would have
reported instant yielding as nearly a second of talking over the caller.

## Grounding the instrument

Everything above is measured with audio we wrote. That is what makes the
caller-side boundary exact — and it also means a fault in our own audio path
would be invisible, because the only thing checking it is us. Audio sent at the
wrong rate, truncated, or badly resampled does not fail a test here: it looks
like a provider that reasons less well than it does.

So the harness is checked against **Big Bench Audio** (MIT, 1,000 spoken
reasoning questions adapted from BIG-bench Hard, audio included,
`ArtificialAnalysis/big_bench_audio` on HuggingFace). Answers are closed-form —
`valid`/`invalid`, `Yes`/`No`, or a count — so grading is exact match on an
extracted token rather than a model's judgement. Checking one instrument with
another instrument is not a check.

```bash
python bin/validate-harness.py --provider openai-realtime --per-category 10
```

Two things this is not. It is not one of our rankings: it is a single-turn quiz
with no interaction in it, and as the most widely circulated audio set in the
field it is also the most likely to have been trained on. And it is not a test of
the model. It is a test of us, and what makes it one is that the answers are
known and guessing is cheap to price.

Three of the four categories are binary and the fourth is a count, so an
instrument that has destroyed the audio scores near 50% on the first three and
near zero on the last. A working path scores far above that. The gap between
those two outcomes is tens of points, which is the whole reason this check can
be trusted at a sample size a credential can afford.

The run writes ordinary cells, so the validation is itself recomputable rather
than a number in a terminal.

### Result, 2026-09-19

100 questions per provider, 25 from each of the four categories, seed 11,
native VAD at 500 ms.

| provider | model | scored | void | accuracy |
|---|---|---|---|---|
| OpenAI Realtime | `gpt-realtime-2.1` | 99 | 1 | 91.9% |
| Gemini Live | `gemini-2.5-flash-native-audio-preview-12-2025` | 98 | 2 | 94.9% |
| xAI Grok | `grok-voice-think-fast-2.0` | 0 | 103 | — |

| provider | formal fallacies | navigate | object counting | web of lies |
|---|---|---|---|---|
| OpenAI Realtime | 91.7% | 100% | 96.0% | 80.0% |
| Gemini Live | 95.8% | 100% | 91.7% | 92.0% |

**What this does and does not establish.** The fault it exists to catch — audio
sent at the wrong rate, truncated, or badly resampled — costs tens of points, not
a handful. Both providers that ran score far above what a destroyed audio path
could reach on a set where three categories in four are a coin flip, and they do
it with about one void in a hundred. The audio path, the adapter and the grading
are sound.

What it does not establish is any ordering between the two. At n≈99 the sampling
error alone is around ±5 points, wider than the distance between them, so these
two numbers are one result and not two. Reading a ranking out of this table would
be exactly the mistake the rest of this document is built to avoid.

OpenAI's 80% on web-of-lies against 100% on navigate is a real spread across
categories. The same audio path carries all four, so it is the model's.

Grok did not run: every session was refused with HTTP 403 at the websocket
handshake, on two attempts and on a plain smoke run. Full campaigns against the
same credential the previous day had no connect failures at all, so this is the
credential, not the harness. Its row stays empty until the account is usable
again — an empty row being the point of recording a void with its reason rather
than a score.

## Providers

| Provider | Status |
|---|---|
| `fake` | a scripted agent with known reply timing — runs the whole harness with no API key |
| OpenAI Realtime (`gpt-realtime-*`) | implemented; text arm is text in, text out |
| Gemini Live (`gemini-*-live*`, `*-native-audio-*`) | implemented over the raw websocket; 16 kHz in, 24 kHz out; no turn-detection events; reasons before replying by default (thought tokens recorded); the native-audio models refuse text output, so the text arm is text in, audio out |
| xAI Grok (`grok-voice-*`) | implemented; OpenAI-shaped protocol; reports no token usage (billed per minute); no text-only output, so the text arm is text in, audio out |
| OpenAI `gpt-live-1` | separate product at `/v1/live/sessions`, delegated backend model; needs its own adapter and its row must disclose backend cost |
| Qwen Omni Realtime | needs a DashScope key |
| Nova Sonic | needs AWS credentials and Bedrock model access |

Adapters are written per **wire protocol**, not per model: one OpenAI Realtime
adapter serves every `gpt-realtime-*`. That ratio is the cost argument for direct
adapters over a framework — and the correctness argument is stronger. A
framework's per-provider integration maturity varies, so a framework-mediated
comparison measures its polish as much as the provider's quality.

## Dry run, 2026-09-18

One full campaign (`bin/run-campaign.sh`) per provider, run in parallel from one
host over about four hours: 609 cells each for OpenAI Realtime (`gpt-realtime-2.1`)
and Gemini Live (`gemini-2.5-flash-native-audio-preview-12-2025`), 606 for Grok
(`grok-voice-think-fast-2.0`). Latency cells have five repeats per voice, every
other cell three. **This is a shakedown of the instrument, not a ranking**: three
repeats resolve nothing finer than a provider's own sentinel spread, and two
metrics were corrected during the run (below), so the interaction suite was
repeated on the corrected rule for all three providers and those runs are the
ones tabulated. Every number here is recomputable from the run directories.

#### Sentinel (same cell in every run: open.book, VAD 500 ms, f-us, n=3 per run)

| provider | runs | sentinel P50 per run (ms) |
|---|---|---|
| OpenAI | 9 | 1394 · 1286 · 1264 · 1302 · 1420 · 1441 · 2039 · 1362 · 1358 |
| Gemini | 9 | 3483 · 3926 · 3160 · 3446 · 2991 · 3422 · 3954 · 3399 · 3331 |
| Grok | 8 | 2067 · 2093 · 2027 · 2039 · 2023 · 2211 · 2589 · 2108 |

#### Response latency P50 [95% CI] ms, clean, per turn-detection configuration (n=5 per voice; three voices pooled by median of medians)

| config | clip | OpenAI | Gemini | Grok |
|---|---|---|---|---|
| manual | open.book | 962 (932–1001) | 3774 (3053–3975) | 1320 (994–1357) |
| manual | open.digits | 1256 (1144–1301) | 2892 (2872–2988) | 1223 (1211–1262) |
| manual | open.question | 938 (921–1014) | 3196 (2886–3547) | 1283 (1257–1332) |
| server_vad-200ms | open.book | 1177 (1102–1182) | 3320 (2968–4202) | 2052 (1831–2171) |
| server_vad-200ms | open.digits | 1403 (1278–1526) | 3707 (3416–3852) | 2163 (2122–2247) |
| server_vad-200ms | open.question | 1154 (1055–1159) | 3487 (3100–3924) | 2169 (2018–2223) |
| server_vad-500ms | open.book | 1505 (1292–1603) | 3505 (3322–4208) | 2388 (2130–2546) |
| server_vad-500ms | open.digits | 1733 (1663–1738) | 4098 (3795–4437) | 2591 (2326–2829) |
| server_vad-500ms | open.question | 1436 (1373–1526) | 3921 (3572–3940) | 2365 (2344–2754) |
| semantic_vad | open.book | 5278 (5154–5327) | excluded | excluded |
| semantic_vad | open.digits | 4365 (1937–4757) | excluded | excluded |
| semantic_vad | open.question | 5191 (5098–5238) | excluded | excluded |

#### Endpointing: turn ended inside the gap? (pass = agent waited; VAD 500 ms; 3 voices × 3 repeats)

| probe | gap ms | OpenAI | Gemini | Grok |
|---|---|---|---|---|
| endpointing_ladder | 400 | 8/8 | 9/9 | 9/9 |
| endpointing_ladder | 600 | 9/9 | 9/9 | 9/9 |
| endpointing_ladder | 800 | 9/9 | 9/9 | 9/9 |
| endpointing_ladder | 1000 | 9/9 | 9/9 | 9/9 |
| endpointing_ladder | 1500 | 2/9 | 9/9 | 9/9 |
| endpointing_ladder | 2000 | 0/9 | 9/9 | 9/9 |
| endpointing_ladder_date | 400 | 9/9 | 9/9 | 9/9 |
| endpointing_ladder_date | 600 | 9/9 | 9/9 | 8/8 |
| endpointing_ladder_date | 800 | 9/9 | 9/9 | 9/9 |
| endpointing_ladder_date | 1000 | 9/9 | 9/9 | 9/9 |
| endpointing_ladder_date | 1500 | 1/9 | 9/9 | 9/9 |
| endpointing_ladder_date | 2000 | 0/9 | 9/9 | 8/9 |
| filled_pause | 1500 | 9/9 | 9/9 | 9/9 |
| filled_pause | 2000 | 9/9 | 9/9 | 9/9 |

#### Barge-in: caller onset → listener stops hearing the agent, P50 ms (VAD 500 ms; corrected rule; latest interaction run)

| variant | voice | OpenAI | Gemini | Grok |
|---|---|---|---|---|
| barge_in-at400ms | f-us | 272 [267, 283] | 2201 [2124, 5057] | 1102 [1042, 1285] |
| barge_in-at400ms | m-us | 289 [277, 303] | 2205 [1919, 2413] | 1287 [1285, 1293] |
| barge_in-at400ms | f-gb | 318 [286, 460] | 2357 [2155, 2582] | 1537 [1262, 1596] |
| barge_in-at900ms | f-us | 290 [290, 338] | 1962 [1895, 3268] | 1057 [996, 1328] |
| barge_in-at900ms | m-us | 269 [266, 291] | 2377 [2175, 4805] | 1333 [1305, 1381] |
| barge_in-at900ms | f-gb | 289 [288, 667] | 2303 [2175, 2699] | 1394 [1290, 1586] |
| barge_in-at1500ms | f-us | 277 [271, 282] | 2231 [2229, 2450] | 1283 [1225, 1288] |
| barge_in-at1500ms | m-us | 294 [288, 308] | 3035 [2538, 3401] | 1301 [1279, 1319] |
| barge_in-at1500ms | f-gb | 284 [284, 285] | 2992 [2429, 4450] | 1343 [1320, 1590] |
| simultaneous_start-at0ms | f-us | 395 [377, 428] | 2626 [2129, 3290] | 1080 [1077, 1297] |
| simultaneous_start-at0ms | m-us | 396 [358, 423] | 2305 [2166, 2489] | 1282 [1184, 1297] |
| simultaneous_start-at0ms | f-gb | 345 [291, 438] | 2540 [2445, 6552] | 1548 [1548, 1578] |
| barge_in_correction-at900ms | f-us | 298 [285, 308] | 2503 [2311, 2522] | 1362 [1286, 1575] |
| barge_in_correction-at900ms | m-us | 290 [288, 312] | 2288 [1826, 2424] | 1040 [983, 1089] |
| barge_in_correction-at900ms | f-gb | 281 [275, 289] | 2337 [2306, 2830] | 1486 [1285, 1581] |

#### Backchannel tolerance (pass = agent kept talking through 'mm hmm'; voids = reply too short to test)

| variant | voice | OpenAI | Gemini | Grok |
|---|---|---|---|---|
| backchannel_tolerance-at400ms | f-us | 0/1 (void 2) | 3/3 | 2/3 |
| backchannel_tolerance-at400ms | m-us | 0/0 (void 3) | 3/3 | 3/3 |
| backchannel_tolerance-at400ms | f-gb | 0/0 (void 3) | 3/3 | 2/3 |
| backchannel_tolerance-at900ms | f-us | 0/0 (void 3) | 3/3 | 3/3 |
| backchannel_tolerance-at900ms | m-us | 0/0 (void 3) | 3/3 | 2/3 |
| backchannel_tolerance-at900ms | f-gb | 0/0 (void 3) | 3/3 | 1/3 |

#### Caller transcription (pass = digits exact when present, else WER ≤ 0.25; 3 voices × 3 repeats, clean)

| clip | OpenAI | Gemini | Grok |
|---|---|---|---|
| open.book | 9/9 | 8/8 | 9/9 |
| open.digits | 9/9 | 8/8 | 9/9 |
| task.identify | 9/9 | 0/9 | 9/9 |
| identify.alt | 9/9 | 0/9 | 0/9 |
| task.cancel | 9/9 | 6/9 | 9/9 |

#### Robustness: task.identify transcription pass under each transform (f-us, 3 repeats) and response latency delta from clean

| transform | OpenAI pass · ΔP50 | Gemini pass · ΔP50 | Grok pass · ΔP50 |
|---|---|---|---|
| clean | 3/3 ·  | 0/3 ·  | 3/3 ·  |
| noise-20db | 3/3 · 28.8 | 0/3 · 232.6 | 3/3 · 154.3 |
| noise-10db | 3/3 · -33.9 | 0/3 · 283.8 | 3/3 · -358.6 |
| telephone | 3/3 · 69.3 | 0/3 · 828.4 | 3/3 · -313.6 |
| reverb | 2/2 · -0.8 | 0/3 · -8.2 | 3/3 · -360.8 |
| clipping | 3/3 · -13.9 | 0/3 · -601.3 | 3/3 · -371.4 |
| dropouts | 3/3 · 266.3 | 0/3 · -208.0 | 3/3 · -277.0 |

#### Tasks (clusters = scenarios; VAD 500 ms, f-us, 3 repeats)

| suite | OpenAI per-run · all-repeats · not all-pass | Gemini per-run · all-repeats · not all-pass | Grok per-run · all-repeats · not all-pass |
|---|---|---|---|
| task | 0.92 · 0.82 · book.distraction, book.fullday, book.newpatient | 0.38 · 0.29 · book.correction, book.digits.spoken, book.distraction, book.fullday, book.morning, book.provider, book.range, cancel.fails, cancel.pick, cancel.single, reschedule.july8, reschedule.july9 · 1 void | 1.00 · 1.00 · none |
| task-text | 0.94 · 0.88 · book.distraction, book.fullday | 0.82 · 0.59 · book.distraction, book.fullday, book.range, cancel.fails, cancel.pick, reschedule.july8, reschedule.july9 · 1 void | 0.96 · 0.88 · book.distraction, book.fullday |
| task-medicare | 0.67 · 0.67 · intake.qualified | 0.25 · 0.50 · intake.qualified · 5 void | 0.78 · 0.67 · intake.qualified |
| task-medicare-text | 0.67 · 0.67 · intake.qualified | 0.57 · 0.67 · intake.qualified · 2 void | 0.67 · 0.67 · intake.qualified |

#### Voids by class, whole campaign

cells per provider: OpenAI 609, Gemini 609, Grok 606

| class | OpenAI | Gemini | Grok |
|---|---|---|---|
| caller pacing slipped | 3 | 0 | 3 |
| excluded: gemini-live has no semantic turn detection | 0 | 45 | 0 |
| excluded: grok-realtime has no semantic turn detection | 0 | 0 | 45 |
| provider closed the session mid-call | 0 | 9 | 0 |
| provider returned no transcript of the caller | 1 | 2 | 0 |
| reply too short to test the backchannel | 17 | 0 | 0 |


#### Reading the tables

- **The sentinel spread is the noise floor.** OpenAI 1264–2039 ms, Grok 2023–2589,
  Gemini 2991–3954 across nine runs of the identical cell. No gap between two
  configurations of one provider smaller than that spread is a finding.
- **Manual commit is the generation floor; native VAD adds its endpointing cost on
  top.** OpenAI: 962 → 1177 (200 ms VAD) → 1505 (500 ms VAD) on the same clip. Grok:
  1320 → 2052 → 2388, an endpointing cost roughly twice the configured silence.
  Gemini: 3774 with no clear VAD dependence; its floor is its own thinking, which
  is on by default and disclosed in the provider table. OpenAI's semantic VAD
  waited 5.2 s on a statement it judged unfinished.
- **Endpointing.** All three hold a turn through a 1 s mid-sentence gap. OpenAI's
  500 ms VAD ends the turn inside a 1.5 s gap in 7 of 9 cells and inside 2 s in all
  9; Gemini and Grok wait through 2 s, and Grok's own VAD ignores the configured
  silence. A filled pause ("um, let me think") holds the turn everywhere.
- **Barge-in, on the listener's clock.** OpenAI stops within 270–320 ms of the
  caller's first word at every offset and voice. Grok 1.0–1.5 s. Gemini 2.0–3.0 s,
  and 2.3–2.6 s when the caller starts at the same instant as the agent. The rule
  was corrected mid-run: stop time is the end of the last stretch of agent audio
  that overlapped the caller's utterance, so a provider that fell silent between
  chunks and resumed is no longer credited with 0 ms.
- **Backchannel.** Gemini talks through "mm hmm" in 18 of 18; Grok in 13 of 18.
  OpenAI's replies to this clip end before the backchannel is fully spoken in 17
  of 18 cells, so the cell is void rather than scored: nothing was tested.
- **Caller transcription.** OpenAI 45/45. Grok fails one identifier clip on every
  voice and repeat. Gemini fails both identifier clips everywhere, on every
  transform including clean, because its input transcription collapses repeated
  digits ("two zero two, five five five, zero one eight eight" comes back as nine
  digits). This is the side-channel transcript, not necessarily what the model
  acted on; the task suites are where that is tested.
- **Degradation.** No provider's identifier transcription changed under any
  transform relative to clean; the latency deltas sit inside the sentinel spread.
  The transforms are recorded as strata; nothing here separates the providers yet.
- **Tasks.** Grok completes all 17 appointment scenarios on all three repeats in
  voice. OpenAI 0.92 per run / 0.82 all repeats: it loops on a caller who asks a
  medication question mid-booking, refuses to book when the requested provider has
  no morning slot, and once gave no reply to the opener. Gemini 0.38 / 0.29 in
  voice against 0.82 / 0.59 in text: in voice its long bookings end in a reply of
  leaked control tokens with no audio at about 5 000 tokens of context, and nine
  medicare and booking sessions were closed by its server mid-reply (codes 1007 and
  1011) with nothing but paced caller audio on the wire from our side. The
  `intake.qualified` medicare scenario fails for all three providers the same way:
  each saves the qualification record without asking age band and current coverage,
  which the contract's prompt lists and the mock's record requires, so the mock
  answers that no record matched. That is a fair fail of the same instruction by
  three models, and the scenario stays.
- **Voids are the record, not the residue.** 90 declared exclusions (semantic VAD
  on Gemini and Grok), 9 provider closes, 17 untestable backchannels, 6 host pacing
  slips, 3 missing caller transcripts. Each is in its own class and none is scored.

#### What the run changed in the harness

Found by the run and fixed before the affected suites were repeated:

- A session the provider closed mid-call left the probe waiting on a caller segment
  that would never finish; one cell held its campaign for 40 minutes. The caller
  now releases every waiting segment, the waits stop when the session is gone, and
  the cell is voided as "provider closed the session" in its own class.
- The sentinel was planned only for audio campaigns and inherited the campaign's
  prompt and tools. It is now the identical cell (audio, default prompt, no tools)
  at the head of every run, which is the only way its spread means anything.
- Barge-in stop time, as above.
- A declared exclusion was recorded with a traceback and counted as an error.
