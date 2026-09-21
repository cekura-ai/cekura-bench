# Reson8: stop sending artificial silence after the speech flush

The legacy transport sends a speech-end flush and then streams another one
second of artificial silence. The versioned no-tail profile stops sending
audio at speech end while preserving the existing completion checks.

`transport_profile: reson8-stop-after-flush-v2` and
`transmitted_silence_frames: 0` select the corrected behavior. The prepared file
stays unchanged. The sender verifies that the omitted tail contains only zero
samples, sends every speech frame with the existing pacing, and stops audio
after the speech-end flush. Both correlated flush confirmations remain required.

The private-turn Reson8 profile selects the correction. Public short-clip runs
can select `config/profiles/reson8-no-tail-v2/reson8-realtime.json`. The legacy
base model config and saved configs without the new profile retain their prior
behavior, so historical evidence can still be replayed accurately. Other
providers and continuous whole-recording protocols are unchanged.

Use new run directories and fresh results; do not subtract a fixed number from
old latencies.
