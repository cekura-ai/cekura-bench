# Agent definitions

This directory is the reproducible agent contract for the benchmark. Configure
one agent with exactly one of these suite directories:

- [`appointments/`](appointments/) for `"suite": "appointments"`
- [`medicare/`](medicare/) for `"suite": "medicare"`

Do not combine files across suites. The benchmark evaluates the selected
prompt, opening message, tools, and fixture data as one contract.

## Configure an agent

For the suite you are running:

1. Copy `system-prompt.txt` into the agent's system instruction.
2. Configure `first-message.txt` as the exact initial spoken message. Do not
   replace it with a provider default greeting.
3. Register every function in `tool-definitions.json`. Keep each function name,
   description, required property, and JSON schema intact.
4. Configure every function to resolve against the matching entries in
   `mock-tools.json`.
5. Test the complete telephone path, including the tool call and result, before
   submitting a benchmark run.

The runner selects the suite with `"suite": "appointments"` or
`"suite": "medicare"`; see the [repository README](../README.md) for the
launch configuration.

## File contract

| File | Required use |
| --- | --- |
| `system-prompt.txt` | Canonical behavioral instruction for the agent. |
| `first-message.txt` | Required opening message. |
| `tool-definitions.json` | Public function names, descriptions, and input schemas the agent may call. |
| `mock-tools.json` | Deterministic input/output fixtures, including expected error and no-match responses. |

`mock-tools.json` is organized by tool. Each entry has a `name`, optional
`freetext_params`, and `mock_data` records with `input` and `output` values.
An implementation may normalize benign representations (for example, a phone
number's punctuation) when matching inputs, but must preserve the fixture's
meaning and return the documented output shape.

## Optional mock-tool endpoints

Serve `mock-tools.json` through provider-native mock tools, an in-process mock
server, or your own endpoints. The endpoint(s) must use the names and schemas
in `tool-definitions.json` and return the matching fixture response.

Match inputs with normalized exact matching first, then the Cekura mock
server's fuzzy fallback (closest match at a score of at least 30). This lets
minor speech-to-text variation resolve to the intended fixture without
overriding explicit error or no-match cases. Do not use live data or mutate
fixtures during a run.

The [LiveKit](../provider-configurations/livekit-agent.py) and
[Pipecat](../provider-configurations/pipecat-bot.py) examples are in-process
mock-tool servers; use them as references if you host your own implementation.

For self-hosted and custom-provider agents, also follow the
[transcript-ingestion contract](../docs/transcript-ingestion.md). A missing
tool call, tool result, transcript, or run association can make an otherwise
correct call unscorable.
