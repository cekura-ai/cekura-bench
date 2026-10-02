# Reference agents

This directory contains the artifacts used to run the published benchmark agents. Each provider has its own folder so runtime source and provider-supplied configuration remain easy to inspect.

| Provider | Artifact | Runtime |
| --- | --- | --- |
| [ElevenLabs](elevenlabs/) | Submitted agent export | Provider-managed |
| [Retell](retell/) | Submitted agent export | Provider-managed |
| [Vapi](vapi/) | Submitted assistant export | Provider-managed |
| [LiveKit](livekit/) | Canonical Ava worker, fixture backend, and frozen definitions | LiveKit Agents |
| [Pipecat](pipecat/) | Canonical Ava worker and frozen definitions | Pipecat Cloud |
| [Pipecat S2S](pipecat-s2s/) | Speech-to-speech agent: OpenAI Realtime (and Mini), GPT-Live, Gemini Live (and Flash), Grok, Nova Sonic, Qwen, Phonic, plus cascade rows | Pipecat Cloud |
| [GPT Realtime](gpt-realtime/) | OpenAI Realtime SIP/sideband benchmark harness | Vercel + Twilio |
| [Gemini Live](gemini-live/) | Gemini Live benchmark harness | Cloud Run + Twilio |

The JSON exports are provided by the corresponding provider. The self-hosted agents and harnesses are source snapshots of the code used for the benchmark. Credentials, call logs, generated deployment state, and local environment files are intentionally excluded.

The `appointments` and `insurance` definition directories are frozen benchmark inputs. The top-level [agent definitions](../agent-definitions/) remain the public contract for configuring a compatible agent.

The speech-to-speech agent in [`pipecat-s2s/`](pipecat-s2s/) loads its prompt, greeting and tools from `agent-definitions/`, answers tools from the shared contract server in `mock_tools/`, and publishes transcripts and traces through the Cekura SDK. Pin the framework version: a benchmark result names a configuration, and an agent that changes underneath it makes two results incomparable without either looking wrong.

Do not commit credentials or call data.
