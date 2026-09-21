# The agent bench — the reference agent on a real call

The service bench measures a provider's realtime service on its own, over a direct
websocket, with audio we authored and boundaries that are exact by construction.
The agent bench measures the other thing anyone actually ships: a complete, readable agent
configuration — prompt, tools, framework, transport — doing real work on a real
phone call.

**The two lanes are never ranked against each other.** "The model is fast" and
"the deployment is fast" are different claims, and a single number mixing them
answers neither. Where both lanes run the same cell, the difference is published
as transport excess, on its own, and never subtracted from one to produce the
other: the boundaries are not the same boundaries, so the subtraction has no
meaning even when the arithmetic works.

| | the service bench | the agent bench |
|---|---|---|
| unit named by a result | provider realtime service under configuration X | reference-agent configuration X |
| caller boundary | exact — we authored the samples | detected on the phone leg |
| channel | direct websocket, clean both ways | μ-law 8 kHz, line noise, codec |
| what it can rank | latency, endpointing, interaction | task completion, tool-call traces, efficiency |

## The agent under test

[`reference-agents/pipecat-s2s/`](../reference-agents/pipecat-s2s/) — one
readable file on a pinned Pipecat, loading its prompt, greeting and tools from
`agent-definitions/` and answering them from the same `mock_tools/` contract
server the service bench uses. One contract and one server across both lanes: two
implementations would drift, and then a difference between lanes could be our own
servers disagreeing rather than anything about the agents.

## Detecting speech on a phone leg

The service bench's detector thresholds frame energy against an estimated noise floor, and
The service bench's own calibration says where that stops working: systematically late at
10 dB SNR, no reliable detection at 0 dB. A phone leg is permanently in that
regime — comfort noise, mains hum and codec artifacts are on the channel whether
or not anyone is speaking. The service bench's rule is therefore that its detector **may not
be pointed at a noisy channel**, and the agent bench needs its own.

What separates speech from line noise is not level, it is **periodicity**. Voiced
speech repeats at the speaker's pitch; hum repeats far below that band and
comfort noise does not repeat at all. `agent/detector.py` requires a frame to be
both loud enough and periodic enough, measuring periodicity as peak normalized
autocorrelation over 70–400 Hz.

Autocorrelation is the right estimator for this channel in particular: the
telephone band starts near 300 Hz, so most voices lose their fundamental
entirely — but the harmonics still repeat at the period of the missing
fundamental. Anything looking for energy *at* f0 would score the audio this lane
exists to measure as unvoiced.

### Calibration

`python -m agent.calibrate` — constructed signals with sample-exact boundaries,
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

Below **5 dB SNR** neither detector is usable. The agent bench voids those cells rather
than publishing them.

### Not yet a licence to publish latency

The calibration above is against constructed signals through a simulated phone
path. It bounds the detector, not the transport. **No the agent bench latency may be
published until an injected-tone round-trip has been measured on the actual
carrier and transport in use** — a tone of known start time played into the call
and recovered from the returned audio, which is the only thing that establishes
what the carrier itself adds and whether the two directions are aligned.

## Putting the caller on a phone

The caller side of the agent bench is the same authored audio the service bench sends. Twilio places
an outbound call to the agent's number, the TwiML it fetches hands the call's
media to a socket we run, and from there the phone leg is implemented as an
ordinary `RealtimeAdapter` (`agent/adapters/twilio_stream.py`). The caller, the
probes, the scenarios and the record format then work over a phone call
unchanged — so a difference between the lanes is a difference in the channel,
not a difference between two harnesses that were written twice.

`<Connect><Stream>` rather than `<Start><Stream>`: `<Start>` forks a copy of the
audio for listening and cannot speak back into the call, and a deterministic
caller has to be heard.

What a phone does not have is declared rather than worked around, and the
declaration is what produces an exclusion instead of a wrong number:

| | |
|---|---|
| no manual commit | there is no turn-boundary message on a call; the agent's endpointer decides. Probes needing an exact commit are excluded, not silently run under a label they did not get. |
| no text modality | there is no text channel to a phone number, so the text control arm stays in the service bench. |
| no VAD events | the carrier does not report what the agent's endpointer decided. Probes reading those events void. |

Every inbound frame carries the carrier's own millisecond timestamp as well as
our arrival time, and both are recorded. The carrier clock removes our receive
jitter from the inbound direction; it does not remove carrier latency, and it
shares no origin with our outbound clock.

## Transport calibration

`agent/tone.py`. A linear chirp sweeping 400–3000 Hz is played into the call at
a known instant and located in the audio that comes back, by normalized
cross-correlation. The far end is a loopback endpoint we run
(`agent/telephony.py`, `/loopback`) which returns each inbound frame and does
nothing else, so what the measurement contains is the carrier, the codec, and no
reasoning.

A sweep rather than a steady tone: every cycle of a tone looks like every other,
so its correlation peak is a plateau several milliseconds wide, and a sweep
correlates sharply against exactly one alignment while still surviving the
telephone band.

Measured against constructed signals through μ-law with line noise, 60 trials
per row:

| chirp vs line noise | found | error bias | P95 |
|---|---|---|---|
| +20 dB | 60/60 | −0.06 ms | 0.12 ms |
| +10 dB | 60/60 | −0.07 ms | 0.12 ms |
| 0 dB | 60/60 | −0.06 ms | 0.11 ms |
| −5 dB | 59/60 | −0.06 ms | 0.12 ms |
| −10 dB | 0/60 | refuses | — |

And on 60 recordings of speech with no chirp in them at all, it reported a
location **0** times. Both halves matter: an instrument that finds a calibration
signal inside the agent's own voice would manufacture a transport correction out
of nothing, and one that guesses when the signal is lost would do it quietly.

One-way delay is reported as half the round trip. The two directions are separate
paths and need not be symmetric, so that halving is published as an assumption
beside the number rather than folded into it.

The full chain below the carrier — framing, μ-law, the loopback, the instrument —
is exercised locally in `tests/test_telephony_loopback.py`. The carrier is the
one part that cannot be tested from a laptop, which is precisely why it is the
part that has to be measured.

## Status

| Piece | State |
|---|---|
| reference agent, pinned Pipecat, tools, tracing | built; verified end to end against OpenAI Realtime |
| phone-leg detector | built, calibrated offline |
| phone leg as a service-bench adapter | built, tested against the carrier's message shapes |
| media socket + loopback endpoint | built, tested locally end to end |
| injected-tone instrument | built, calibrated offline |
| tone measured on a real carrier | **blocked** — gates every agent-bench latency |
| caller conditions | not started |
| deployment + a phone number | **blocked** on credentials |

Nothing in the agent bench may publish a latency until the tone has been measured on the
carrier and transport actually in use.
