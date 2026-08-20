# Cekura Benchmarks

Run Cekura's Appointment and Medicare voice-agent benchmark suites against your
own Cekura-connected agent.

## What this runner does

Given a Cekura API key and a local configuration, the runner:

1. Reads the selected Appointment (`AS`) or Medicare (`MS`) scenarios from your
   catalog agent.
2. Validates the target agent/run configuration.
3. Launches the complete suite against your target agent with your selected
   repetition count and concurrency.
4. Prints the result URL and a machine-readable launch record.

It is dry-run by default. Add `--execute` only after reviewing the resolved
payload.

## Current runner behavior

On `--execute`, the runner provisions a target agent when `targetAgentId` is
omitted, launches the selected suite, then stays active while the result set
completes. It waits for the suite's conservative completion window (84 minutes
for Medicare; 192 minutes for Appointments), polls the result set, creates a
public report, and writes its links to `data/benchmark-report-<result-id>.md`.

Watching is enabled by default. Set `"watchResults": false` to return as soon
as the run launches. `watchInitialWaitMinutes` and `watchPollSeconds` can tune
the wait and polling cadence.

## Prerequisites

- Node.js 20 or later.
- A Cekura project API key in `CEKURA_API_KEY`.
- A catalog agent in your project containing the benchmark scenarios.
- A dedicated target number. For custom transcript publishers, use
  `different_numbers` so each run can be associated to its provider call.

For Vapi, Retell, ElevenLabs, or Synthflow setup, also export that provider's
API key. LiveKit setup needs `LIVEKIT_API_KEY` and the secret environment
variable named in `agentSetup.livekit.apiSecretEnv`. Pipecat and custom
phone-connected agents must publish the final transcript and tool-call events
using the [transcript-ingestion format](docs/transcript-ingestion.md).

## Quick start

```bash
cp config/benchmark.example.json benchmark.config.json
export CEKURA_API_KEY='...'
npm run benchmark -- --config benchmark.config.json
npm run benchmark -- --config benchmark.config.json --execute
```

## Configuration

```json
{
  "projectId": 1234,
  "catalogAgentId": 5678,
  "targetAgentId": 9012,
  "agentNumber": "+15555550100",
  "suite": "appointments",
  "frequency": 3,
  "concurrencyLimit": 5,
  "numberMode": "different_numbers",
  "name": "My provider Benchmark v1"
}
```

`catalogAgentId` owns the canonical evaluator scenarios. `targetAgentId` is an
optional existing Cekura agent. `suite` is required and must be either
`appointments` or `medicare`; each launch runs only that suite.

## Optional provider setup

The same command can create the target agent and launch the benchmark. Omit
`targetAgentId` and provide `agentSetup`. For the native integrations below,
export the provider API key, then add the provider and agent ID to the same
benchmark config:

```bash
export VAPI_API_KEY='...'
npm run benchmark -- --config benchmark.config.json --execute
```

```json
{
  "projectId": 1234,
  "catalogAgentId": 5678,
  "agentNumber": "+15555550100",
  "suite": "appointments",
  "agentSetup": {
    "provider": "vapi",
    "providerAgentId": "your-vapi-assistant-id"
  }
}
```

Supported native providers and their default key variables are:

| Provider | `agentSetup.provider` | Environment variable |
| --- | --- | --- |
| Vapi | `vapi` | `VAPI_API_KEY` |
| Retell | `retell` | `RETELL_API_KEY` |
| ElevenLabs | `elevenlabs` | `ELEVENLABS_API_KEY` |
| Synthflow | `synthflow` | `SYNTHFLOW_API_KEY` |

Set `providerApiKeyEnv` when your key uses a different environment-variable
name. The runner imports native-provider details, waits for the agent setup to
finish, then launches the selected suite as one command.

Provider setup is optional. If you supply only the phone number (and omit both
`targetAgentId` and `agentSetup`), the runner creates a phone-connected target
record and launches the suite. To score those calls, the agent must publish its
final transcript and native tool calls in the
[transcript-ingestion format](docs/transcript-ingestion.md). Reach out to us if
you need help wiring that connection.

For [LiveKit](provider-configurations/livekit-agent.py) and
[Pipecat](provider-configurations/pipecat-bot.py), setup remains optional and
uses their respective provider fields. LiveKit requires `LIVEKIT_API_KEY` plus
the secret variable named by `apiSecretEnv`; Pipecat requires `agentName`.
These settings are sent directly when the runner creates the Cekura agent.

```json
{
  "agentSetup": {
    "provider": "livekit",
    "providerApiKeyEnv": "LIVEKIT_API_KEY",
    "livekit": {
      "url": "wss://your-livekit-host",
      "apiSecretEnv": "LIVEKIT_API_SECRET",
      "agentName": "your-livekit-agent",
      "config": {}
    }
  }
}
```

```json
{
  "agentSetup": {
    "provider": "pipecat",
    "pipecat": {
      "agentName": "your-pipecat-agent",
      "webhookUrl": "https://your-service.example/cekura",
      "config": {},
      "roomProperties": {}
    }
  }
}
```

## Reference implementations

Reference-agent examples for LiveKit, Pipecat, OpenAI Realtime, and Gemini
Live are described in [reference-agents](reference-agents/README.md).

## Agent definition and mock-tool contract

The configuration required to score an agent is in
[agent-definitions](agent-definitions/README.md). It includes the canonical
system prompt, required first message, tool schemas, and mock data for both
Appointment and Medicare runs. Providers may host equivalent tools themselves
or use their platform's mock-tool mechanism.

## Methodology

### The agents under test

The benchmark exercises two canonical phone-agent workflows:

- **Appointment Booking**: identify a caller, look up their record, check
  availability, book, cancel, reschedule, and confirm appointments while
  handling errors and changes of mind.
- **Insurance (Medicare)**: distinguish member-service from sales requests,
  follow disclosure and consent requirements, collect only safe qualification
  information, and create the appropriate routing or callback handoff.

Each workflow is evaluated against the same scenario catalog, caller behavior,
test-profile context, mock-tool contract, and configured scoring rubric. A
provider may choose its own runtime and speech/model configuration; the
benchmark measures the submitted configuration as it behaves end to end on a
telephone call.

Every evaluated agent is reached over telephony. The benchmark therefore
includes the complete delivered call path - turn detection, speech services,
agent runtime, tool dispatch, and the applicable telephony transport - rather
than a text-only simulation.

The configuration used for a benchmark run is fixed before scoring. We do not
modify a provider's model, prompt, voice, tools, or runtime settings in
response to its benchmark results. Reference configurations are available in
[`provider-configurations/`](provider-configurations/).

### Test cases

The suite contains 82 caller situations: 59 Appointment scenarios and 23
Medicare scenarios. The public scenario coverage summary describes their
intent at a high level. We do not publish the exact evaluator dialogue,
conditional logic, fixtures, or assertions because systems could then optimize
for the test rather than general voice-agent behavior.

Appointment coverage includes:

- core booking, cancellation, rescheduling, lookup, and service-recovery
  paths;
- emergency and medical-advice boundaries;
- multi-step scheduling, corrections, ambiguity, silence, and abandonment;
- background conversation, accents, coughs, packet loss, interruptions, and
  other speech/transport robustness cases; and
- privacy, prompt-injection, and authority-boundary cases.

Medicare coverage includes:

- member-services versus licensed-sales routing and callback disposition;
- no-advice, price, eligibility, and coverage-assurance boundaries;
- safe qualification, corrected or incomplete information, and sensitive-data
  minimization;
- multiple or changing intents; and
- required disclosures, consent refusal, scope sequencing, and caregiver
  authority.

The normal benchmark configuration uses three repetitions per scenario. This
helps distinguish a one-off success from a workflow that is repeatable under
the same caller situation.

### Metrics

The benchmark reports several complementary measures rather than reducing a
call to one language-model judgment.

| Measure                   | What it represents                                                       | How to interpret it                                                                                                                                                                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task completion**       | Whether the call passed the configured rubric end to end.                | This is a gated result, not merely whether the transcript sounds plausible. The rubric combines the expected outcome with operational checks such as required tool use, infrastructure, appropriate termination, and applicable conversation-quality checks. A call must satisfy every configured gate to count as a pass. |
| **Infrastructure Issues** | Whether the telephone interaction remained operationally responsive.     | This binary check is intended to surface no-connects, missing call evidence, and prolonged dead air after the caller speaks. It is a reliability signal, separate from whether the agent knew the right business procedure.                                                                                                |
| **Interruption Score**    | Whether the agent allowed the caller to speak without talking over them. | This measures turn-taking behavior. It helps distinguish an agent that completes work from one that does so with disruptive conversational timing.                                                                                                                                                                         |
| **Voice Tone + Clarity**  | Delivered speech quality on the benchmark call path.                     | This record-only score reflects audible clarity, tone, and timing stability in the final phone-call recording. It is not a task-completion gate, not a measure of human-likeness, and not a pure TTS-model score: telephony, codecs, and the full delivery path can affect it.                                             |

The exact rubric and metric availability can vary with the catalog version. A
result should therefore always be read with its metric coverage and rubric
configuration, especially when a recording or provider call was unavailable.

### Evaluator refinement and fairness

Writing reliable conversational tests is iterative. Initial scenario prompts
are deliberately reviewed against actual test runs to identify cases where the
**testing agent**, rather than the agent under test, behaves nondeterministically
or departs from the intended script.

When that happens, the evaluator is refined before treating a result as
benchmark evidence. Refinements can include:

- adding catch-all conditions to conditional-action evaluators so reasonable
  variations do not send the simulated caller off script;
- adding test profiles that provide the testing agent with the relevant desired
  provider, date, personal information, callback details, and other fixture
  context; and
- clarifying conditional branches so they preserve the intended caller goal
  while accommodating natural conversational variation.

These changes improve test validity; they are not a limitation of any tested
platform. When a refinement materially changes a scenario's behavior or
scoring conditions, the affected scenario is rerun across every provider so
the compared evidence uses the same evaluator version and context. This keeps
the test harness fair while acknowledging that good voice-agent evaluation is
an empirical test-design process, not a one-shot prompt.
