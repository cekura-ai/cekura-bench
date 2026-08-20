# Agent definition contract

Configure the agent under test with the files for the suite you run:

- `appointments/`
- `medicare/`

Each directory contains the canonical `system-prompt.txt`, `first-message.txt`,
`tool-definitions.json`, and `mock-tools.json`. The tool definitions and mock
data are the contract used for scoring. You may serve them from your own tool
endpoint or configure equivalent provider-native mock tools; the externally
observable tool names, input schemas, and returned data must match.

Do not combine the two directories in one agent run. Select the corresponding
`suite` in the runner configuration.
