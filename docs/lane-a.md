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
| `endpointing_ladder` | Does a mid-utterance pause of *n* ms get cut off? | a curve, one rung per run |
| `barge_in` | Does the agent yield the floor when spoken over, and how fast? | pass/fail + ms |
| `backchannel_tolerance` | Does "mm hmm" mid-reply derail the agent? | pass/fail |
| `false_trigger` | Noise on the line, caller silent — does it speak anyway? | rate per minute |

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

### What the other probes found on the same shakedown

The endpointing ladder is the clearest argument for publishing a curve rather
than a number. Under native VAD configured with 500 ms of silence, the agent did
not interrupt a mid-utterance pause of 400, 600, 800, 1000 or 1500 ms, and did
interrupt at 2000 ms, on both repeats. **The configured silence duration is not
the observed patience threshold**, and a single latency figure would never have
shown that.

Backchannel tolerance splits into two behaviours that a pass/fail alone would
merge. The agent does *not* stop mid-reply when the caller says "mm hmm" — but it
then answers the backchannel as though it were a turn, about 1.2 seconds later,
on all three repeats. The provider took roughly 300 ms to register the
backchannel as speech at all.

Barge-in: the floor was yielded every time, 330–690 ms after the caller's first
authored speech sample. False triggers: none in a 20 second window of pink noise
at −30 dBFS with the caller silent.

The booking task passed three of three in voice and three of three in text, with
identical tool traces. That is the control arm behaving as designed: on this
scenario the speech pathway costs nothing in task terms, so a future failure in
voice but not text is attributable rather than ambiguous.

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

- **Control** — when to speak next — runs off live event and audio arrival. Coarse.
- **Measurement** — what gets published — runs offline over the recorded audio.

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

The sampling unit is the scenario-repeat, minimum five. Published: per-run success
rate **and** observed all-repeats success, both with clustered bootstrap
confidence intervals. Reliability over repeats is reported as what was observed,
never as a success rate raised to the power of the repeat count: that transform
produces a confident-looking number out of a handful of runs while describing
nothing that actually happened. P50/P90 until a declared minimum n.

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
python -m lane_a.recompute data/lane-a/<run>/response_latency-open.book/manual/f-us/r1
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

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lane-a.txt

# render the caller corpus (needs ELEVENLABS_API_KEY)
.venv/bin/python bin/render-corpus.py

# measure
.venv/bin/python bin/run-lane-a.py --provider openai-realtime --suite latency --repeats 5
```

Suites: `smoke`, `latency`, `endpointing`, `interaction`, `noise`.

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
the model — a published score already exists for models we can run, so the point
is the gap. Land near the published figure and the audio path, the adapter and
the scoring are sound. Land far below it and the fault is ours, which is much
cheaper to discover here than in a published ranking.

The run writes ordinary cells, so the validation is itself recomputable rather
than a number in a terminal.

## Providers

| Provider | Status |
|---|---|
| `fake` | a scripted agent with known reply timing — runs the whole harness with no API key |
| OpenAI Realtime (`gpt-realtime-*`) | implemented |
| Gemini Live | credential verified, adapter pending |
| xAI Grok | credential verified, adapter pending |
| OpenAI `gpt-live-1` | separate product at `/v1/live/sessions`, delegated backend model; needs its own adapter and its row must disclose backend cost |
| Qwen Omni Realtime | needs a DashScope key |
| Nova Sonic | needs AWS credentials and Bedrock model access |

Adapters are written per **wire protocol**, not per model: one OpenAI Realtime
adapter serves every `gpt-realtime-*`. That ratio is the cost argument for direct
adapters over a framework — and the correctness argument is stronger. A
framework's per-provider integration maturity varies, so a framework-mediated
comparison measures its polish as much as the provider's quality.
