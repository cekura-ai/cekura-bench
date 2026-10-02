# Finalization at speech end: opt-in profiles v1

These five configurations are prepared for future runs. They have offline mock
validation only and do not replace saved benchmark configurations or results.
Select the JSON explicitly in a runner that accepts a configuration path; do not
copy it over `config/models/` or resume an old run with changed settings.
The existing full-run launchers are not switched to these profiles automatically.

| Profile | Change from its base configuration |
| --- | --- |
| AssemblyAI default | Send `ForceEndpoint` at speech end |
| AssemblyAI `min_latency` | Same request; retain `mode=min_latency`, English bias, 60 ms packets, model verification and session-start limits |
| Speechmatics Standard / Enhanced | Send `ForceEndOfUtterance` at speech end; `max_delay=1.0`, `max_delay_mode=flexible` |
| Inworld | Retain `endTurn`; disable voice profiling |

`profile_version` identifies these settings; `base_config` identifies the
unchanged original. `model_id` continues to identify the same provider model.
The standard provider dispatch now selects AssemblyAI's dedicated adapter when
`mode=min_latency`, including during replay. This preserves that mode's wire
parameters and server identity checks.

The benchmark requests finalization at t=0 (speech end). The AssemblyAI and
Speechmatics profiles still send the original one-second silence tail. The
Inworld profile now inherits the base configuration's `transmitted_silence_frames: 0`
fix and sends no audio after `endTurn`. All profiles wait for terminal completion. Completion
is measured separately; the request does not guarantee immediate final text.
Speechmatics acknowledges only a forced `EndOfUtterance` after our request.
AssemblyAI and Inworld have no distinct finalization acknowledgment in these
adapters. Later text remains collected and scored. `max_delay_mode=flexible`
retains the provider's formatting behavior; 1 second is not a strict timing cap.

The original pilot is retained unchanged. Its AssemblyAI default profile cannot
validate the published `min_latency` profile. Speechmatics changed two controls
together, so it cannot isolate either control's causal effect. Inworld profiling
changes do not establish a fix for repetition.

The later [Inworld stream-ending comparison](../../../INWORLD_STREAM_END_FIX.md)
tests omission of the artificial silence tail separately from voice profiling.
Saved pilot configurations and evidence remain unchanged.

## Offline verification

```sh
.venv/bin/python -m pytest -q tests/test_finalization_profiles.py
```

Mocks verify the outgoing settings, request-before-tail order, forced
acknowledgment, terminal-response requirement, and retention of late text.
Live provider validation requires a separately authorized run using a new output
directory; no provider calls are part of this change.

Protocol sources:

- [AssemblyAI turn detection](https://www.assemblyai.com/docs/streaming/turn-detection)
- [Speechmatics realtime protocol](https://docs.speechmatics.com/api-ref/realtime-transcription-websocket)
- [Inworld voice profiles](https://docs.inworld.ai/stt/voice-profiles)
