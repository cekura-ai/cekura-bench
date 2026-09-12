# Cekura Bench

Cekura Bench is the open-source runner and agent contract for reproducing
Cekura's telephone-based **Appointment** and **Medicare** voice-agent
benchmarks. It launches the published scenario catalog in your Cekura project
against an agent you configure, then records the exact launch payload and links
to the resulting report. It is not a local simulator: calls traverse your
agent's real telephony, speech, model, and tool path.

## Two ways to use this repository

### Option 1: Set up a benchmark-compatible agent

Use this option first if you want to build or inspect a compatible agent. It
does not require a Cekura account, API key, or the runner. Follow the public
agent definition below, deploy the agent on a telephone number, and exercise
the fixture-backed tools in your own environment.

### Option 2: Run the official Cekura benchmark

Use this option after the agent is ready. The runner launches Cekura's private
scenario catalog and records results in Cekura, so it requires a Cekura
workspace, an API key, a catalog agent ID, and a reachable target number. New
to Cekura? [Create a Cekura account](https://dashboard.cekura.ai/sign-up?utm_source=benchmarks), then obtain the project API key and benchmark catalog details from your workspace.

The command-line runner is a launch client, not the agent setup mechanism. Its
dry-run is useful for inspecting the request it will make, but it still calls
the Cekura API to read the scenario catalog.

## 1. Set up the agent under test

Choose one suite per agent run: `appointments` or `medicare`. Load the matching
directory from [agent-definitions](agent-definitions/README.md) into the agent:

1. Set `system-prompt.txt` as the system prompt.
2. Use `first-message.txt` verbatim as the opening message.
3. Register every function from `tool-definitions.json`, without renaming a
   function or changing its input schema.
4. Back those functions with the matching fixture data in `mock-tools.json`,
   either locally or through your provider's mock-tool feature.
5. Put the agent behind the dedicated `agentNumber` in your configuration.

Keep the suites separate. The evaluator expects the selected suite's prompt,
opening message, functions, and fixture outputs as one compatible contract.

For a self-hosted or custom-provider agent, publish the completed transcript
and native tool calls to Cekura and associate them with the run before hangup.
Follow the [transcript-ingestion contract](docs/transcript-ingestion.md).

## 2. Run the benchmark through Cekura

Clone the repository and create a local launch configuration:

```bash
git clone https://github.com/cekura-ai/cekura-bench.git
cd cekura-bench
cp config/benchmark.example.json benchmark.config.json
```

`CEKURA_API_KEY` is required for every command. Keep it in your shell or secret
manager; do not put it in `benchmark.config.json` or commit it.

```bash
export CEKURA_API_KEY='your-project-api-key'
```

Create a configuration from the example. These are the required fields:

```json
{
  "projectId": 1234,
  "catalogAgentId": 5678,
  "targetAgentId": 9012,
  "agentNumber": "+15555550100",
  "suite": "appointments"
}
```

`catalogAgentId` is the Cekura agent that owns the canonical scenarios;
`targetAgentId` is the existing Cekura record for the agent being measured.
`agentNumber` must be able to receive benchmark calls. `numberMode` defaults to
`different_numbers`, which is the appropriate mode for custom transcript
publishers because it supports reliable run-to-call association.

Useful optional fields are `frequency` (default `3` repetitions per scenario),
`concurrencyLimit` (default `5`), `name`, `watchResults`,
`watchInitialWaitMinutes`, and `watchPollSeconds`. Set `watchResults` to
`false` if you want the command to return immediately after launch; otherwise
it waits, polls to completion, creates a public report, and writes its links to
`data/benchmark-report-<result-id>.md`.

Use `npm run validate -- --config benchmark.config.json` to validate the
configuration and selected catalog without launching a run. (It still needs
`CEKURA_API_KEY` because it reads the catalog.)

## 3. Optional: have the runner register an agent

Instead of `targetAgentId`, omit it and provide `agentSetup`. Do not provide
both. The runner registers/imports the target agent on `--execute`, waits for
an import if necessary, then launches the selected suite.

For native providers, export the provider credential and identify the existing
provider-side agent:

```bash
export VAPI_API_KEY='your-vapi-api-key'
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

| Provider | `agentSetup.provider` | Default credential variable |
| --- | --- | --- |
| Vapi | `vapi` | `VAPI_API_KEY` |
| Retell | `retell` | `RETELL_API_KEY` |
| ElevenLabs | `elevenlabs` | `ELEVENLABS_API_KEY` |
| Synthflow | `synthflow` | `SYNTHFLOW_API_KEY` |

Set `providerApiKeyEnv` when your secret has a different name. For LiveKit,
use `provider: "livekit"` with `livekit.url`, `livekit.agentName`, and
`livekit.apiSecretEnv`, plus `LIVEKIT_API_KEY` (or `providerApiKeyEnv`). For
Pipecat, use `provider: "pipecat"` and `pipecat.agentName`; its `webhookUrl`,
`config`, and `roomProperties` are optional. See the submitted
[provider configurations](provider-configurations/README.md) for examples.

If you supply neither `targetAgentId` nor `agentSetup`, the runner creates a
phone-connected `self_hosted` target record. This is suitable only when your
agent already meets the transcript-ingestion contract.

## 4. Optional: serve the mock data from tool endpoints

The benchmark does not require a particular tool-hosting model. The simplest
option is your platform's native mock-tool feature. Alternatively, expose one
endpoint per function (or a single dispatcher endpoint) in your own service.
The files in `agent-definitions/<suite>/` are the source of truth:

- `tool-definitions.json` defines the function names and JSON input schemas.
- `mock-tools.json` maps each exact fixture input to the output the endpoint
  must return, including no-match and error cases.

For example, an Appointment `lookup_patient` endpoint must accept the exact
schema for `lookup_patient` and return the fixture output for the normalized
caller phone number. Preserve tool names, required arguments, result fields,
and tool-call ordering. Do not replace fixture data with a live CRM or alter it
between repetitions: the scenarios are scored against this fixed contract.

The repository deliberately does not run a hosted mock-data service or set a
tool URL in the runner. Configure the endpoint URL in your provider/agent
runtime, then verify a call reaches it before launching the benchmark. The
included [LiveKit](provider-configurations/livekit-agent.py) and
[Pipecat](provider-configurations/pipecat-bot.py) examples show an alternative:
they load `mock-tools.json` directly and resolve functions in-process.

## 5. Execute and find the results

```bash
# Inspect the resolved scenario count, target, and payload; no calls are made.
npm run benchmark -- --config benchmark.config.json

# Launch the selected suite.
npm run benchmark -- --config benchmark.config.json --execute
```

Every invocation writes a machine-readable launch record to
`data/benchmark-launch-<timestamp>.json`. An executed run prints the Cekura
dashboard URL. With result watching enabled, it also creates a shareable report
and writes `data/benchmark-report-<result-id>.md` when the result set reaches a
terminal state. Conservative default wait windows are 84 minutes for Medicare
and 192 minutes for Appointments; these reflect the full call batch, not a
local command failure.

## Reference implementations

Reference-agent examples for LiveKit, Pipecat, OpenAI Realtime, and Gemini
Live are described in [reference-agents](reference-agents/README.md).

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
