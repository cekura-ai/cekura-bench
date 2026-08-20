# Provider configurations

This directory contains the configurations used for the benchmark. They let
readers inspect the tested runtime choices rather than infer them from results.

| Provider | Artifact | Format |
| --- | --- | --- |
| ElevenLabs | `elevenlabs-agent-config.json` | Provider agent configuration JSON |
| Vapi | `vapi-agent-config.json` | Provider assistant configuration JSON |
| Retell | `retell-agent-config.json` | Provider agent configuration JSON |
| Pipecat | `pipecat-bot.py` | Submitted runtime source configuration |
| LiveKit | `livekit-agent.py` | Submitted runtime source configuration |

The Pipecat and LiveKit submissions are code-defined agents, so their runtime
configuration lives in source rather than a provider JSON object. These are
not converted into a synthetic JSON format.

The artifacts show the tested agent and runtime configuration. They omit
credentials and the evaluator suite.

GPT Realtime and Gemini Live use reference configurations; their runtime
selections are documented in the main methodology rather than represented as
files here.
