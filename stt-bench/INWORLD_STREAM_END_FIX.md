# Inworld stream-ending fix

## What was wrong

The adapter sent `endTurn` at the speech boundary, then streamed one second of
artificial silence, then sent `closeStream`. Inworld frequently returned another
final transcript for that trailing portion, adding words such as “Oh”, “I”, and
“I'm not sure”. Those words were present in the provider's raw responses.

This is different from the ElevenLabs punctuation-token scoring correction.
Filtering Inworld's words out of the scorer would hide real returned output.

## Changed behavior

The default Inworld configuration now declares `transmitted_silence_frames: 0`.
The adapter sends all prepared speech frames, sends `endTurn`, then sends
`closeStream`. It sends no audio after `endTurn`. It continues collecting every
final segment, including segments received after closure was requested, until
the provider's terminal usage response. An ordinary final transcript is not
treated as a finalize acknowledgment or terminal completion.

The dataset and speech boundary are unchanged. The sender verifies that the
omitted tail contains exactly the expected zero samples; it refuses to omit a
nonzero tail. Leading silence, internal pauses, and speech are preserved.

Capture, dry runs, replay assessment, reports, and pacing audits use the explicit
transmitted-tail setting. The 18–40 ms send-gap and 2% timing-drift gates remain
unchanged. Historical configurations without the new field still mean 50 silence
frames. Resume fingerprints reject changed configurations. Other providers retain
their existing transport behavior.

The voice-profile-off Inworld profile also inherits the transport fix. Voice
profiling was held **on** in both arms of this experiment, isolating the change
from the earlier profiling experiment.

## Live validation on Vercel

Evidence: `reports/inworld-stream-end-20260915-vercel-v2/`.

The test used eight frozen public Pipecat clips: two previously observed trailing
phrase failures, one known repetition failure, and five deterministic selections.
Each clip ran once under each setting, with alternating baseline/candidate order.
There were **16 sessions, zero retries, and eight usable pairs**. All three
10-second preflight trials and all 16 capture timing/completion checks passed.

| Measurement | Old: 50 silence frames | New: no silence tail |
| --- | ---: | ---: |
| All eight clips: word errors / reference words | 234 / 204 | 214 / 204 |
| All eight clips: WER | 114.71% | 104.90% |
| Seven clips excluding the preselected repetition failure: errors / words | 18 / 191 | 2 / 191 |
| Those seven clips: WER | 9.42% | 1.05% |
| Those seven clips: inserted words | 16 | 0 |
| All eight clips: median last-final-text delay after speech end | 1,027.87 ms | 51.15 ms |
| Transmitted audio | 91.32 seconds | 83.32 seconds |

The all-eight WER exceeds 100% because the deliberately selected failure returns
212 “tap” words against a 13-word reference. The candidate still returns that
repetition, with 199 insertions and 13 substitutions. The baseline adds four
more words after it. Across all eight pairs, the new sequence removes 20 inserted
words while substitution and deletion totals remain unchanged.

This supports the transport fix for extra trailing output on these clips. It
does **not** establish a general 1.05% WER or resolve the separate repetition
failure. It also does not isolate the provider's internal decoding mechanism:
removing the tail necessarily moves the close request earlier.

The original laptop attempt failed timing qualification before making any
provider calls. Its failed preflight is preserved separately under
`reports/inworld-stream-end-20260915-v2/`.

## Verification and historical results

- The full Python suite passed 551 tests before default configuration promotion.
- After promotion, 123 targeted provider, profile, reporting, and scoring tests passed.
- Vercel passed 82 adapter/pacing tests before the live experiment.
- Historical source-report checksums remain unchanged: 22 of 22.
- The experiment has its own results and raw logs; it does not change the
  dashboard, rank, or historical WER. A full new run is required for a new ranking.

The collected evidence verification and sandbox-stop receipt are recorded in
`reports/inworld-stream-end-20260915-vercel-v2/verification.json` and `vercel.json`.

## Reproduce

Use a fresh output directory. The test is public-only and limited to 16 provider
sessions. The live-start marker prevents an uncertain invocation from being
silently repeated. The live action requires a passing timing preflight.

```sh
.venv/bin/python scripts/inworld_stream_end_probe.py prepare --out reports/inworld-stream-end-NEW
.venv/bin/python scripts/inworld_stream_end_probe.py live --out reports/inworld-stream-end-NEW
```

The Vercel launcher uses the existing configured account/project and prepared
snapshot. It enables egress only to `api.inworld.ai` for the live command, downloads
and verifies the evidence archive, and stops the sandbox when collection completes:

```sh
node scripts/run_vercel_inworld_stream_end.mjs --action launch --out reports/inworld-stream-end-NEW
node scripts/run_vercel_inworld_stream_end.mjs --action status --out reports/inworld-stream-end-NEW
node scripts/run_vercel_inworld_stream_end.mjs --action collect --out reports/inworld-stream-end-NEW
```

Set `VERCEL_SANDBOX_SDK_DIR` to the existing Sandbox SDK installation. Credentials
are resolved through the existing credential helper and are never written into
the plan or uploaded source files.

Protocol reference: [Inworld manual turn control](https://docs.inworld.ai/stt/turn-detection).
