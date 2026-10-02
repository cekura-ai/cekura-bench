# Private conversational turns

This workflow prepares one audio clip and reference transcript per
conversational turn, using automatic speech-end detection or optional listening
review. Each benchmark turn gets a new provider session. It is
separate from the eight full-recording measurements and their existing ranking.

## Automatic preparation (no manual approval required)

The automatic mode accepts the supplied transcript segments as candidate turns,
keeps pauses within them, and estimates the last speech boundary with local
Silero VAD 6.2.1. It does not split at every silence or ask you to approve every
candidate. It also does not claim to resolve the conversational meaning of every
interruption: turn structure comes from the supplied segments.

```bash
uv sync --locked --extra turn-preparation
uv run --locked --extra turn-preparation stt-bench auto-prepare-turns \
  --draft reviews/private-turns-v1/revision-2/draft.json \
  --out datasets/private-turns-v1/automatic-v1
```

This command runs entirely locally, saves preparation evidence and exclusions,
and freezes the passing clips. No transcription service is used. The optional
Silero/PyTorch dependency is needed on the preparation machine only; a benchmark
host can use the frozen dataset with the regular dependencies.

The detector uses a 0.5 speech threshold, a 100 ms minimum speech duration,
300 ms minimum silence, and 100 ms speech padding. The shorter minimum speech
duration keeps short replies eligible. Each candidate's search window is bounded
by neighboring same-speaker annotations, with up to 300 ms additional audio at
the end. Channels remain separate. The last detected speech segment supplies
the final boundary; internal detected gaps do not split a turn.

The validator preserves the supplied reference and word identities. It excludes
invalid timestamps, conflicting text, same-speaker overlaps, clips containing
unassigned words, no-speech detections, and boundaries that would remove an
annotated word. It may expand a segment boundary to contain its valid supplied
word annotations. It does not invent new word timestamps or transcribe text.
Excluded candidates do not block passing candidates from being benchmarked.
If no candidate passes, freezing fails instead of creating an empty benchmark.

Frozen `review.json` holds **automatic preparation evidence**, not reviewer
attestations. The manifest uses `preparation_mode: automatic-silero-v1`, turn
schema version 2, and `listening_review_verified: false`. All human approval
fields remain false. The detector/model hash, source hashes, conversions,
estimated boundaries, and exclusion reasons are retained. Reports show that
references are supplied and boundaries are automatically estimated.

Use the resulting `manifest.json` with the existing `plan-turn-run` and
`run-turns` commands below. There is no human-approval check for this mode;
hash verification, word coverage checks, pacing requirements, smoke gating,
and the explicit private-manifest authorization check still apply.

Automatic and manually reviewed datasets cannot be interchanged on resume.
Existing reviewed datasets and historical reports keep their interpretation.

### Relationship to Coval

Like the supplied Coval code, this mode uses Silero to estimate a shared final
speech boundary. It is not an exact copy: we preserve short replies, retain the
existing 20 ms framing and provider-specific silence tails, and calculate TTFS
from actual boundary-frame delivery. Coval subtracts a stored audio offset from
audio-start-to-final time. VAD settings and transcript-boundary conflicts are
recorded rather than presented as human truth.

## Current delivery

The automatic dataset is frozen at
`datasets/private-turns-v1/automatic-v1/manifest.json`: **206 passing clips**
from all four conversations and eight speaker recordings, with **94 excluded
candidates** retained in the evidence. Total submitted audio is 3,187.44 seconds
(53.12 minutes) per model, including the configured silence tails. This is an
automatically prepared dataset, not a completed live benchmark.

Exclusions: 47 VAD/word-end conflicts, 33 invalid word timestamps, 10 candidates
containing unassigned words, 2 empty references after non-speech-tag removal,
1 invalid segment timestamp, and 1 segment/word text disagreement. None of the
passing clips requires a human approval to enter `run-turns`.

The current four conversations can be imported without new transcription. Their
eight mono speaker files already have supplied segment and word timestamps.
`prepare-turns` proposes all segments, tracks every word, and flags timing and
text problems. Candidates are **not** automatically verified conversational turns.

The current review for this checkout is `reviews/private-turns-v1/revision-2/index.html`.
This draft includes the final annotation checks; the earlier unapproved draft is preserved.
It is private and ignored by Git. Do not publish it or the frozen dataset as
part of a public source-code change. Review media links reference local source
files; they are not a portable sharing bundle.

## 1. Optional manual preparation and listening review

```bash
uv run --locked stt-bench prepare-turns \
  --source private-dataset --out reviews/private-turns-v1/revision-2
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory reviews/private-turns-v1
```

Open `http://127.0.0.1:8765/revision-2/`. The page requests no external resources. Choose a
conversation and turn, play the two aligned speakers, and play the selected
speaker alone. Recording offsets align playback; they never shift source crops.
Stereo playback selects the mapped channel without mixing it with the other.

Split at a source-word boundary or merge with the next contribution by the same
speaker. Thinking pauses can remain inside a turn. Short replies and overlapping
contributions remain separate eligible clips. A backchannel does not automatically
split the other speaker's ongoing contribution.

Check start/end times and reference text. The end is the last audible speech,
not a later timeout or an arbitrary segment edge. Fix invalid word annotations
under **Source words and timing corrections**, with a reason. Source units without
word timing can be merged or bounded, but cannot be split into smaller word units
in v1; exclude with a reason if their supplied timing cannot support the intended
turn. Text corrections must have notes; the original text remains in the dataset.

Approve boundaries and transcript separately after listening. Each excluded turn
requires a reason. Every source word or untimed segment must appear exactly once
across included and excluded rows. The validator rejects duplicate, missing,
reordered, cross-speaker, overlapping, or out-of-bounds assignments.

Export `review.json`. Exporting incomplete progress is allowed; freezing and live
runs require complete review. Import a saved review to resume. Browser edits are
in memory until exported. Do not close the page without exporting your progress.

## 2. Freeze manually reviewed inputs

```bash
uv run --locked stt-bench freeze-turns \
  --draft reviews/private-turns-v1/revision-2/draft.json \
  --review /absolute/path/to/exported-review.json \
  --out datasets/private-turns-v1/reviewed-v1
```

This writes a new directory atomically; it will not overwrite a prior dataset.
All source hashes are checked. Corrections, exclusions, original source sample
boundaries, conversation timestamps, and reviewer attestations are retained.

Audio outputs are a source-rate lossless mono WAV crop, 16 kHz signed 16-bit PCM,
and a 24 kHz derivative. FFmpeg must be installed for preparation; its version
and explicit resampling filter settings are recorded. The 24 kHz conversion uses
the existing deterministic repository converter. Execution hosts need only the
frozen artifacts and Python dependencies, not FFmpeg or the original recordings.

Leading context is at most 100 ms, bounded by earlier same-speaker annotations
and reviewed turns. The reviewed end is rounded outward to a 20 ms frame. Actual
speech is never removed by an automatic voice detector. Exactly one second of
artificial digital silence is appended. Provider-specific transmission rules
(including Inworld's configured zero-tail option) remain explicitly recorded.

Source-rate masters and derivatives require free disk space; the command checks
capacity before conversion. No audio cleanup or deletion occurs automatically.

## 3. Prepare a run budget and validate locally

```bash
uv run --locked stt-bench plan-turn-run \
  --manifest datasets/private-turns-v1/reviewed-v1/manifest.json \
  --configs config/profiles/private-turns-v1/deepgram-nova-3.json \
  --out reports/private-turns-v1/run-plan.json

uv run --locked stt-bench run-turns \
  --manifest datasets/private-turns-v1/reviewed-v1/manifest.json \
  --config config/profiles/private-turns-v1/deepgram-nova-3.json \
  --out runs/private-turns-v1/local-dry-run --dry-run

uv run --locked stt-bench score \
  --run runs/private-turns-v1/local-dry-run \
  --out reports/private-turns-v1/local-dry-run
```

Dry-run sends no provider requests and produces no invented transcripts or
latency values. It exercises all frozen turns and the actual local sender, with
AssemblyAI's own packetization when selected. Synthetic automated tests are
separate from listening review of the real conversations.

`plan-turn-run` accepts an explicit list of model profile paths. It records their
hashes, the dataset hash, session count, and submitted-audio budget. It does not
authorize or dispatch provider calls. Use the resulting submitted-audio budget
as a floor, not a wall-time or billing estimate; connections, rate limits and
session-duration billing can add overhead.

## 4. Live execution (a separately authorized step)

After reviewing the plan and authorizing the exact private manifest/model scope,
run the local pacing probe on the execution host and pass its `pacing.json` as
`--pacing-check`. Live `run-turns` also requires
`--authorized-private-manifest-sha256` matching the frozen manifest hash. It uses
the existing credential loader; never put credentials in a dataset or review.

The runner orders up to ten deterministic smoke turns first, covering every
included speaker recording. Smoke turns count toward the total once. Only after
all smoke turns are complete, pacing-valid, and have the supported metrics does
it submit the remaining turns. More than ten included speaker recordings needs
a smaller run dataset because the smoke coverage guarantee would not fit.

One original attempt per turn is the default and required v1 contract. `--resume`
retains every recorded attempt, including failures, and starts only previously
unstarted turns. Source code, dataset, config, measurement version, smoke
selection, and timing host must match. Failed smoke cannot be bypassed by resume;
a recovery experiment needs a new output directory and an explicit new budget.

## Measurement and compatibility

Measurement version **5**, profile **private-turns-controlled-v1**:

| Metric | Definition |
| --- | --- |
| TTFT | First nonempty transcript receipt minus first audio-packet send start, after connection setup. First text is classified as partial or final. |
| TTFS | Last change to the completed final transcript minus delivery of the reviewed speech-end frame. |
| Signed TTFS | The same difference before clamping. Early finals remain negative here; displayed TTFS is clamped to zero with a flag. |
| Deadline accuracy | Existing available-text scoring at 0, 250, 500, and 1,000 ms after speech end. |
| Final accuracy | Pooled word-error counts from valid first attempts, including incorrect transcripts. |

Completion must be confirmed before accepting TTFS. Finalization acknowledgment
and stream-close time remain separate diagnostics. Duplicate final messages do
not change the clock; late final additions and revisions do. Unsupported replay,
missing text, incomplete streams, and invalid pacing have explicit unavailable
statuses. TTFT and TTFS never select only correctly recognized words.

Profiles live under `config/profiles/private-turns-v1/`. Existing supported manual
commands are retained. AssemblyAI enables ForceEndpoint and Speechmatics enables
ForceEndOfUtterance while retaining their other base settings. Google Chirp has
no equivalent control in the adapter, so its observed end-to-final delay is an
exception metric and not controlled TTFS. Packetization and finalization behavior
still differ across providers; this is not exact replication of Coval's host,
data, model configuration, or timing implementation.

`score` dispatches version 5 to a distinct report with JSON and a listening HTML
dashboard. It shows conversation/recording/turn counts, planned/attempted/valid/
failed/not-run counts, latency percentiles and sample sizes, deadline accuracy,
and exclusions. Confidence intervals resample whole conversations; only four
conversations provide limited evidence about a broader population. Local audio
links are optional during offline replay and are hash-checked when present.

Historical measurement versions retain their existing scorer and interpretation.
The old combined ranking does not automatically ingest turn results or change
its weighting. Full-recording raw logs cannot reproduce independently streamed
turn experiments.

## Input format for new recordings

Provide a directory containing `input.json`, audio files and one transcript JSON
per speaker. Audio must be lossless integer PCM WAV/FLAC, with one or two channels.
Channel indices are zero-based. Example stereo mapping:

```json
{
  "conversations": [{
    "conversation_id": "call-001",
    "audio_file": "call-001.wav",
    "speakers": [
      {"label": "A", "channel": 0, "metadata_file": "speaker-a.json"},
      {"label": "B", "channel": 1, "metadata_file": "speaker-b.json"}
    ]
  }]
}
```

For paired mono inputs, put `audio_file` on each speaker entry and supply
`starts_at_seconds` in that entry or its transcript metadata. The existing
`private-dataset/metadata.json` layout is also supported without conversion.

Each transcript file has a `segments` array:

```json
{
  "speaker_label": "A",
  "speaker_id": "speaker-a",
  "starts_at_seconds": 0,
  "transcript": "Yes please",
  "segments": [{
    "start": 0.1, "end": 0.5, "text": "Yes please",
    "words": [
      {"word": "Yes", "start": 0.1, "end": 0.2},
      {"word": "please", "start": 0.3, "end": 0.5}
    ]
  }]
}
```

Times are seconds relative to that speaker's source file. No transcription API,
speaker inference, or automatic transcript alignment is invoked.

## Turn run continuation and empty responses

The Vercel controller keeps the original attempt for each model and clip. A
continuation may import verified raw results and submit only unstarted clips.
Any explicitly repeated failed attempt is a recovery experiment and stays out
of the primary comparison. `scripts/consolidate_turn_results.py` takes run roots
in chronological order, verifies their archived evidence, and selects the first
attempt for each model and clip.

The optional frozen-plan policy `completed-empty-allowed-v1` permits a successful,
complete, pacing-valid smoke response with no transcript. That response remains
an accuracy error; TTFT and TTFS remain unavailable. It does not permit incomplete
streams, invalid pacing, unsupported reconstruction, or invented zero latency.
The default smoke policy remains unchanged. The worker checks the policy against
the hashed run plan before running a full shard.

Collector corrections are versioned. ElevenLabs committed-segments v3 recognizes
plain commits, their timestamp annotations, and a late duplicate partial without
duplicating words. Speechmatics empty-silence-ranges v2 prevents a silent timestamp
range from falsely overlapping a real word. Historical profiles keep their prior
interpretation. Corrected reports replay saved receipts without changing receipt
timestamps or resubmitting audio.
