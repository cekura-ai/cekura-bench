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

## Optional tool-endpoint setup

You may use provider-native mock tools, an in-process fixture resolver, or
your own HTTPS tool endpoints. The benchmark does not prescribe endpoint URLs
and this repository does not host one.

If you host endpoints yourself, use `tool-definitions.json` as the request
contract and `mock-tools.json` as the response fixture table. One endpoint per
tool or a dispatcher endpoint are both fine, provided the agent exposes the
same tool names to its model. In particular:

- accept only the schema defined for the invoked tool;
- return the matching fixture output, including deliberate error/no-match
  outputs;
- preserve native tool-call names, arguments, results, and ordering in the
  transcript sent to Cekura; and
- do not substitute live production data or change fixture responses during a
  benchmark run.

The [LiveKit](../provider-configurations/livekit-agent.py) and
[Pipecat](../provider-configurations/pipecat-bot.py) reference configurations
load these fixtures directly, which is a useful model if your platform does
not support remote mock tools.

For self-hosted and custom-provider agents, also follow the
[transcript-ingestion contract](../docs/transcript-ingestion.md). A missing
tool call, tool result, transcript, or run association can make an otherwise
correct call unscorable.
