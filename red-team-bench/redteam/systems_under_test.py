"""The systems under attack: the airline agent on each voice system, attacked as rt-airline-<cell>. VOICE_CELLS are
the speech-to-speech models, (provider, model); CASCADE_CELLS are the platforms, (provider, model, the platform's
agent config). voice_clients.client() picks each one's client by provider; agent-configurations/ records each one's
exact setup."""
from .harness_config import CONFIGS

VOICE_CELLS = {"s2svoice-openai": ("openai", "gpt-realtime-2.1"),
               "s2svoice-gemini": ("gemini", "models/gemini-3.8-live-extended-thinking"),
               "s2svoice-grok": ("xai", "grok-voice-3.0"),
               "s2svoice-eleven": ("elevenlabs", "eleven_realtime_v1_max"),
               "s2svoice-gptlive": ("openai-live", "gpt-live-1"),
               "s2svoice-gptlive11": ("openai-live", "gpt-live-1.1"),
               # Microsoft's own model on Azure Voice Live, Deepslate Opal, and Phonic-1.1 (azure_realtime_client.py,
               # deepslate_client.py, phonic_client.py), each set up as the speech-to-speech benchmark's bot sets it up
               "s2svoice-azurert": ("azure-realtime", "azure-realtime"),
               "s2svoice-deepslate": ("deepslate", "opal"),
               "s2svoice-phonic": ("phonic", "phonic-1.1")}
# Cascaded platform agents, built from their agent config in agent-configurations/ with the airline's prompt and
# tools. Only the clinic-specific transcription hints are swapped for the airline's.
CASCADE_CELLS = {"cascade-vapi": ("vapi", "gpt-4.1-2025-04-14", CONFIGS / "vapi-agent-config.json"),
                 "cascade-retell": ("retell", "tell-1", CONFIGS / "retell-agent-config.json"),  # Retell's own LLM
                 # the same, with Retell's Trust & Safety handbook prompts (AI disclosure, scope boundaries) on
                 "cascade-retell-guard": ("retell-guard", "tell-1", CONFIGS / "retell-agent-config.json"),
                 # ElevenLabs Agents with ElevenLabs' red-team overlay (voice_clients.ELEVEN_AGENT_OVERLAY) merged on
                 "cascade-elevenagents": ("eleven-agent", "deepseek-v41-flash", CONFIGS / "elevenlabs-agent-config.json"),
                 # Webex AI Agent Studio's agent on Webex AI Pro-US 2.0, over the phone (webex_client.py); built in
                 # Webex, so its config is a record (webex-ai-agent.json), not read by the code
                 "cascade-webex": ("webex", "webex-ai-pro-us-2.0", CONFIGS / "webex-ai-agent.json")}


def targets() -> dict[str, dict]:
    out = {f"rt-airline-{cell}": {"provider": provider, "model": model, "voice": True, "config": config}
           for cell, (provider, model, config) in CASCADE_CELLS.items()}
    out |= {f"rt-airline-{cell}": {"provider": provider, "model": model, "voice": True}
            for cell, (provider, model) in VOICE_CELLS.items()}
    return out


TARGETS = targets()


def named(cells: str) -> list[str]:
    """--cells as given: full ids or the short ones (s2svoice-phonic), comma-separated."""
    out = [c if c.startswith("rt-airline-") else f"rt-airline-{c}" for c in cells.split(",") if c.strip()]
    unknown = [c for c in out if c not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown cells {unknown}; known: {sorted(TARGETS)}")
    return out
