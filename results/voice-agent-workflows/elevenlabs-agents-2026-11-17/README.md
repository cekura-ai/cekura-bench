# ElevenLabs Agents · 17 November 2026

This snapshot publishes the data behind ElevenLabs Agents' Voice Agent Workflows row.
It contains all 246 counted calls from the 82 scenarios, three runs per scenario at
concurrency 5 per workflow, over PSTN. No runs are review-required.

| Workflow | Passed calls | Pass¹ | All three passed | Pass³ |
| --- | --- | --- | --- | --- |
| Appointments | 162/177 | 91.53% | 50/59 | 84.75% |
| Medicare | 62/69 | 89.86% | 17/23 | 73.91% |
| Combined | 224/246 | 91.06% | 67/82 | 81.71% |

Pass¹ counts passing calls. Pass³ counts scenarios whose **three actual calls**
all pass; it is not Pass¹ cubed. The 22 failures comprise 16 Tool Call Accuracy
failures, 3 Expected Outcome failures and 3 calls that failed both. Every call has
an evidence link in [results.json](results.json).

## Scoring policy

| Gate | Pass threshold | Numeric coverage |
| --- | --- | --- |
| Expected Outcome | = 5 | 246/246 |
| Infrastructure Issues | true, normalized to 5 | 246/246 |
| Interruption Score | ≥ 1 | 246/246 |
| Latency (in ms) | ≤ 10,000 | 246/246 |
| Tool Call Accuracy | = 5 | 237/237 applicable calls |

AS12, AS49 and MS79 require no mock tools: their nine tool scores are null and
the tool gate does not apply. A missing score on any applicable call cannot
count as a pass.

These gates match the published Retell, PolyAI and Smallest AI policies.
ElevenLabs uses **standard** Tool Call Accuracy, as Retell does; its tools accept
list arguments, so the list-text variant scores the same (also recorded).
Transcription Accuracy, Voice Tone + Clarity, Unnecessary Repetition, Appropriate
Call Termination and Tool Call Success are recorded on all applicable calls and do
not gate success. Response time is the mean Cekura latency metric across the 246
calls (1,109.81 ms).

## Configuration and provenance

Project 8197: [Appointments result 1120601](https://dashboard.cekura.ai/8197/results/1120601)
(agent 27807) and [Medicare result 1120422](https://dashboard.cekura.ai/8197/results/1120422)
(agent 27808).

| Setting | Value |
| --- | --- |
| LLM | deepseek-v41-flash, temperature 0, max 400 tokens |
| Speech to text | Scribe v3 Realtime (`scribe_v3_live`) |
| Text to speech | Eleven v4 Turbo, voice Lauren |
| Turn taking | `turn_v3` |
| Call cap | 600 s |

This is the same stack and voice as ElevenLabs Agents on the Real Human Calls
benchmark. Prompts, first messages and tools come from `agent-definitions/`; ASR
keywords and the end-call tool from `provider-configurations/elevenlabs-agent-config.json`.
Every counted call was answered by the intended agent with this stack, checked
against each call's ElevenLabs usage record.

The calls ran on the date-shifted copies of the scenarios, as Smallest AI's did:
every July 2026 date moved forward 22 weeks to December 2026, weekdays unchanged.
The Medicare prompt matches the repository exactly. The Appointments prompt
differs only in its current-date line (July 1, 2026 → December 2, 2026). Prompt
hashes are in the snapshot.

One call was replaced. In AS39 (run 4627606) Cekura's simulated caller went silent
45 seconds in, so Cekura recorded 47 seconds and could not match the ElevenLabs
conversation, which ran 96 seconds with the agent handling the injected
scheduling error. It was rerun once in its slot ([result 1120991](https://dashboard.cekura.ai/8197/results/1120991)),
and the replacement counts, as for every provider. The replaced run is listed in
`replaced_runs`. No other call was affected: ElevenLabs recorded exactly 246
conversations for the two runs, with no other caller dropouts.

All calls were rescored once with the four later-added metrics (Tool Call Success,
Unnecessary Repetition, Appropriate Call Termination and the list-text Tool Call
Accuracy), so each carries the same 12 metrics as Smallest AI's runs.

## Reproduce the counts

From the repository root, with Python 3:

```sh
python3 results/voice-agent-workflows/elevenlabs-agents-2026-11-17/validate.py
python3 -m pytest tests/test_elevenlabs_agents_results.py -q
```

The validator checks membership, unique calls, three attempts per scenario, the
replaced run and its replacement, applicable numeric scores, every five-gate
verdict, per-workflow and combined rates, and all metric aggregates. It reads the
frozen snapshot and makes no API calls or changes to Cekura verdicts.
