# Reference agents

This directory contains the artifacts used to run the published benchmark agents. Each provider has its own folder so runtime source and provider-supplied configuration remain easy to inspect.

| Provider | Artifact | Runtime |
| --- | --- | --- |
| [ElevenLabs](elevenlabs/) | Submitted agent export | Provider-managed |
| [Retell](retell/) | Submitted agent export | Provider-managed |
| [Vapi](vapi/) | Submitted assistant export | Provider-managed |
| [LiveKit](livekit/) | Canonical Ava worker, fixture backend, and frozen definitions | LiveKit Agents |
| [Pipecat](pipecat/) | Canonical Ava worker and frozen definitions | Pipecat Cloud |
| [GPT Realtime](gpt-realtime/) | OpenAI Realtime SIP/sideband benchmark harness | Vercel + Twilio |
| [Gemini Live](gemini-live/) | Gemini Live benchmark harness | Cloud Run + Twilio |

The JSON exports are provided by the corresponding provider. The self-hosted agents and harnesses are source snapshots of the code used for the benchmark. Credentials, call logs, generated deployment state, and local environment files are intentionally excluded.

The `appointments` and `insurance` definition directories are frozen benchmark inputs. The top-level [agent definitions](../agent-definitions/) remain the public contract for configuring a compatible agent.
