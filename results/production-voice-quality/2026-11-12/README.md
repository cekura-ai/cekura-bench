# Production Voice Agent Quality Benchmark · 12 November 2026

This snapshot contains **1,395 selected calls across 31 verified production
phone numbers and 16 providers**, with a maximum of three agreed phone numbers per provider. Each number has the same 15 generic support
scenarios, with three repetitions per scenario and concurrency three per number.
Cognigy, demo numbers and identity-blocked numbers are excluded.

Open [report.html](report.html) for the composite-versus-latency plot, ranked
provider table, all metric charts, metric definitions and linked calls.
[results.json](results.json) contains the selected run IDs, result IDs,
scenario membership, scores, timing events, number-level aggregates and provider
aggregates. Dashboard evidence links require access to the Cekura project.

## Scenarios

The original ten cover baseline delivery, concise explanation, requested
repetition, preference change, clarification interruption, direction-change
interruption, a pause within the caller turn, contextual follow-up, silence,
and closing. Five extensions cover office noise, a listening acknowledgment
(“uh-huh”), an audible cough, difficult-day distress and sick-family distress.
The requests assess general conversational behavior rather than a provider's
business workflow. DTMF navigation precedes the scenario on routes requiring it.

## Composite quality

| Category | Weight | Metrics |
| --- | --- | --- |
| Audio quality | 20% | Voice tone and clarity 15%; speech integrity 5% |
| Turn-taking and reliability | 40% | Caller interruption avoidance 10%; interruption recovery 10%; interruption stop time 10%; infrastructure reliability 10% |
| Response quality | 25% | Response relevance 10%; spoken organization 10%; unnecessary repetition 5% |
| Conversational adaptation | 15% | Conversational style fit 5%; emotional acknowledgment 5%; conversational cue handling 5% |

Graded metrics scale from 0–5 to 0–100. Binary checks use 100 for a pass and 0
for a failure. Stop-time quality is 100 at 300 ms or less, 0 at 3,000 ms or more,
and linear between. Event scores are averaged within the call, then within
the scenario. All metrics average repetitions within eligible scenarios,
eligible scenarios within each number, and eligible numbers equally within
each provider. The composite is the weighted sum of the twelve normalized
metrics. It is a composite score, not Elo.

**Latency is separate:** the descriptive latency and stop-time values are means
of individual qualifying events. Navigation, human handoffs, caller-initiated
interruptions (for response latency), and testing-agent idle check-ins are
excluded. Quality scores use fixed anchors rather than field min-max scaling.

N/A means the required evidence was absent and is excluded from that metric's
applicable denominator. It never counts as a pass. Every metric includes
scored-call and eligible-number coverage. Each provider has an aggregate for
every weighted metric; individual calls need not have every opportunity.

Voice tone and clarity uses the predefined NISQA discontinuity model on isolated
interactive AI speech, excluding menus and human handoffs. At least five seconds
of AI speech are required. Speech integrity separately checks malformed
synthesis or meaningless generated speech; proper names and foreign-language
speech are not failures by themselves.

Phone numbers are production access points, not a claim of distinct underlying
AI models. Provider counts are Bland 2, Crescendo 3, Cresta 2, Decagon 3,
Dialpad 2, ElevenLabs 1, Giga 1, Intercom 2, Parloa 1, PolyAI 3, Regal 1,
Retell 3, Sierra 3, Synthflow 1, Talkdesk 2 and Vapi 1.

## Validate

From the repository root:

```sh
python3 results/production-voice-quality/2026-11-12/validate.py
python3 -m pytest tests/test_production_quality_results.py -q
```

The validator checks scenario parity, unique selected run IDs, evidence URLs,
coverage, all metric averages, equal-number aggregation, composite scores and
the report's SHA-256. The JSON omits credentials, signed recording URLs and raw
provider exports. This snapshot follows the repository's existing results
publication layout and extends the website's Production Voice Agents quality
presentation; it does not replace the separate workflow benchmark data.
