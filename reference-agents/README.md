# Reference agents

The agent under test in the agent bench: a complete, readable configuration doing real
work on a real call, published so a result names something anyone can inspect
and run rather than a private deployment.

| Agent | Status |
|---|---|
| [`pipecat-s2s/`](pipecat-s2s/) | speech-to-speech over Pipecat — OpenAI Realtime, Gemini Live, Grok |
| LiveKit | planned |
| OpenAI Realtime (SIP/sideband) | planned |

Each one loads its prompt, greeting and tools from `agent-definitions/`, answers
tools from the shared contract server in `mock_tools/`, and publishes transcripts
and traces through the Cekura SDK. Pin the framework version: a benchmark result
names a configuration, and an agent that changes underneath it makes two results
incomparable without either looking wrong.

Do not commit credentials or call data.
