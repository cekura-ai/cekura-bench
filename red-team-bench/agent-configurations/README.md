# Agent configurations

The exact setup each agent ran on, one file per agent, so anyone reading or reviewing the results knows what was
tested. Each file records:

- **What was tested:** the model id, any backend model, the connection, the session settings sent (voice, reasoning
  or thinking level, audio format, turn detection, tools) and the instructions it was given.
- **How a turn was played:** how the kept turns reached it, how the caller's line was sent, and how its reply was
  read.

The files are taken from the harness's own constants (`redteam/voice_clients.py`, `azure_realtime_client.py`,
`deepslate_client.py`, `phonic_client.py`, `reference_bot_settings.py`); the code is what runs, so change a file together with the client it describes.

| File | Agent |
|---|---|
| `gpt-realtime-2.1.json` | GPT Realtime 2.1 (high reasoning) |
| `gemini-3.8-live-extended-thinking.json` | Gemini 3.8 Live Extended Thinking (high) |
| `grok-voice-3.0.json` | Grok Voice 3.0 (high reasoning) |
| `eleven-realtime-v1-max.json` | Eleven Realtime v1 Max |
| `gpt-live-1.json` | GPT-Live 1 + GPT-6 Sol (low reasoning) |
| `gpt-live-1.1.json` | GPT-Live 1.1 + GPT-6.2 Sol (low reasoning) |
| `azure-realtime.json` | Azure Realtime |
| `deepslate-opal.json` | Deepslate Opal |
| `phonic-1.1.json` | Phonic-1.1 (high intelligence) |
| `vapi-gpt-4.1.json` | Vapi (GPT-4.1) |
| `retell-tell-1.json` | Retell (Tell-1), guarded, and the same agent unguarded |
| `elevenlabs-agents.json` | ElevenLabs Agents (DeepSeek V4.1 Flash, reasoning off) |

The platforms' own agent configs, which the code reads and builds each platform agent from, sit beside these:
`vapi-agent-config.json`, `retell-agent-config.json`, `elevenlabs-agent-config.json`, and ElevenLabs' own overlay
`elevenlabs-redteam-overlay.json`. Each platform's file above says what the harness overrides on top of its config.

Every agent gets the same agent: `redteam/tau2_airline/system-prompt.txt` (Sierra's tau2 airline prompt and policy), its 14
tools, the greeting "Hi! How can I help you today?", and the voice bot's shared rules (`redteam/reference_bot_settings.py`).
