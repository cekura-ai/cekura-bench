# Lane B — the reference agent on a real call

Lane A measures a provider's realtime service on its own, over a direct
websocket, with audio we authored and boundaries that are exact by construction.
Lane B measures the other thing anyone actually ships: a complete, readable agent
configuration — prompt, tools, framework, transport — doing real work on a real
phone call.

**The two lanes are never ranked against each other.** "The model is fast" and
"the deployment is fast" are different claims, and a single number mixing them
answers neither. Where both lanes run the same cell, the difference is published
as transport excess, on its own, and never subtracted from one to produce the
other: the boundaries are not the same boundaries, so the subtraction has no
meaning even when the arithmetic works.

| | Lane A | Lane B |
|---|---|---|
| unit named by a result | provider realtime service under configuration X | reference-agent configuration X |
| caller boundary | exact — we authored the samples | detected on the phone leg |
| channel | direct websocket, clean both ways | μ-law 8 kHz, line noise, codec |
| what it can rank | latency, endpointing, interaction | task completion, tool-call traces, efficiency |

## The agent under test

[`reference-agents/pipecat-s2s/`](../reference-agents/pipecat-s2s/) — one
readable file on a pinned Pipecat, loading its prompt, greeting and tools from
`agent-definitions/` and answering them from the same `mock_tools/` contract
server Lane A uses. One contract and one server across both lanes: two
implementations would drift, and then a difference between lanes could be our own
servers disagreeing rather than anything about the agents.

## Detecting speech on a phone leg

Lane A's detector thresholds frame energy against an estimated noise floor, and
Lane A's own calibration says where that stops working: systematically late at
10 dB SNR, no reliable detection at 0 dB. A phone leg is permanently in that
regime — comfort noise, mains hum and codec artifacts are on the channel whether
or not anyone is speaking. Lane A's rule is therefore that its detector **may not
be pointed at a noisy channel**, and Lane B needs its own.

What separates speech from line noise is not level, it is **periodicity**. Voiced
speech repeats at the speaker's pitch; hum repeats far below that band and
comfort noise does not repeat at all. `lane_b/detector.py` requires a frame to be
both loud enough and periodic enough, measuring periodicity as peak normalized
autocorrelation over 70–400 Hz.

Autocorrelation is the right estimator for this channel in particular: the
telephone band starts near 300 Hz, so most voices lose their fundamental
entirely — but the harmonics still repeat at the period of the missing
fundamental. Anything looking for energy *at* f0 would score the audio this lane
exists to measure as unvoiced.

### Calibration

`python -m lane_b.calibrate` — constructed signals with sample-exact boundaries,
put through the path the audio actually takes: resampled to 8 kHz, μ-law encoded
and decoded, mixed with comfort noise and mains hum. 60 trials per row.

Onset error in ms, against the true onset:

| SNR | onset | periodicity bias | P95 | misses | energy bias | P95 | misses |
|---|---|---|---|---|---|---|---|
| 30 dB | voiced | 3.8 | 7.8 | 0 | 4.3 | 8.2 | 0 |
| 30 dB | unvoiced | 120.5 | 125.0 | 0 | −4.7 | 9.0 | 0 |
| 20 dB | voiced | 12.2 | 16.5 | 0 | 17.7 | 23.2 | 0 |
| 20 dB | unvoiced | 129.3 | 136.3 | 0 | −2.8 | 8.0 | 0 |
| 10 dB | voiced | 43.5 | 49.7 | 0 | 55.3 | 64.2 | 0 |
| 10 dB | unvoiced | 161.5 | 168.5 | 0 | 172.8 | 179.8 | 0 |
| 5 dB | voiced | 67.8 | 80.0 | 0 | 87.8 | 105.0 | 0 |
| 0 dB | either | — | — | 36–40 of 60 | — | — | 60 of 60 |

And the case a bias table hides completely — a line where nobody spoke, carrying
only comfort noise, hum, and the transients a line produces on its own:

| | false onsets in 60 |
|---|---|
| periodicity | **0** |
| energy | **47** |

That is the measurement the detector exists for. A late boundary is an error with
a size, and it can be corrected for. A boundary invented out of a switching click
is a reply latency measured from nothing, and it lands in the published
percentiles looking like data.

### What this costs, stated rather than tuned away

A turn opening on a fricative is detected at its first voiced frame, about
**120 ms late**, because a fricative is not periodic. That is a bias with a known
size. The alternative — accepting any loud frame regardless of periodicity —
is exactly what produces the 47 false onsets above.

Below **5 dB SNR** neither detector is usable. Lane B voids those cells rather
than publishing them.

### Not yet a licence to publish latency

The calibration above is against constructed signals through a simulated phone
path. It bounds the detector, not the transport. **No Lane B latency may be
published until an injected-tone round-trip has been measured on the actual
carrier and transport in use** — a tone of known start time played into the call
and recovered from the returned audio, which is the only thing that establishes
what the carrier itself adds and whether the two directions are aligned.

## Status

| Piece | State |
|---|---|
| reference agent, pinned Pipecat, tools, tracing | built; verified end to end against OpenAI Realtime |
| phone-leg detector | built and calibrated offline |
| deterministic caller over telephony | in progress |
| injected-tone transport calibration | not started — gates all Lane B latency |
| caller conditions | not started |
| deployment + a phone number | blocked on credentials |
