"""Speech-to-speech (S2S) models as turn-taking models: the ``s2s:<provider>`` family.

An S2S model has no end-of-turn detector a client can ask; it simply starts
talking. TurnBench therefore scores such models with a *generative*
protocol (``baselines/gemini_vad``): the model converses with the dataset's
caller in real time, its output audio is recorded sample-aligned with the input,
and every onset of model speech while the caller is quiet is read as the model's
decision that the caller's turn had ended — an EOT event committed at the onset.
Interruption (INT) has no readout for a full-duplex model and is left undefined.

**Track 1 — end of turn** (``turn_taking_bench.s2s.model``) reproduces that protocol
inside this harness: per conversation one provider session, opened by the
service bench's adapter (``service/adapters``), is fed the caller channel
(speaker_1) at real-time pace as a continuous carrier — silences included,
because a server endpointer decides a turn ended by hearing silence — and the
model's reply audio is recorded with arrival timestamps. Onsets are detected
with the service bench's energy detector on the arrival-anchored audio clock
(``service.audio.AudioTimeline``). An onset inside a scripted caller pause is a
cut-in, scored as a false positive by TurnBench's scorer; an onset within the
window after a real turn end is a hit with its latency. The agent channel of the
dataset (speaker_2) is not streamed: the S2S model *is* the agent.

For providers whose protocol reports its own endpointer (OpenAI-shaped:
``speech_stopped`` / ``speech_started``) the model also records those events
and can emit them instead (``readout: vad_events``), which is exactly how
TurnBench's ``openai_server_vad`` / ``openai_semantic_vad`` baselines derive their
events; ``mode: vad_only`` reproduces those baselines' session (text output, no
responses) for calibration against their committed predictions.

**Track 2 — interaction** (``turn_taking_bench.s2s.interaction``) runs the service
bench's own ``barge_in`` and ``backchannel_tolerance`` probes, unchanged, and
maps their verdicts onto the scorecard's interruption / backchannel rows.

``turn_taking_bench.s2s.scorecard`` combines both tracks into one scorecard per
provider.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Providers with an adapter: the service bench's own (openai-realtime,
# gemini-live, grok-realtime, fake) and turn-taking-bench's (turn_taking_bench/s2s/*, same
# interface, registered in model.EXTRA_PROVIDERS). Model variants of a provider
# (e.g. gpt-realtime-2.1-mini) are run with --config model=<id>.
S2S_PROVIDERS: tuple[str, ...] = (
    "openai-realtime", "gemini-live", "grok-realtime", "higgs-realtime", "fake",
    "azure-realtime", "gpt-live", "nova-sonic", "phonic", "deepslate", "gemini-live-thinking",
)

MODEL_PREFIX = "s2s:"

# One fixed, generic phone-agent prompt for every provider, recorded with every
# run (the text and its digest). Short replies keep the model from talking over
# the caller's next turn for long; the content is a plausible deployment, not a
# turn-taking instruction — the prompt must not coach the behaviour under test.
SYSTEM_PROMPT = (
    "You are the phone assistant for a medical practice's appointments line. "
    "Callers ring to book, change or cancel appointments and to ask simple questions. "
    "Reply briefly, in one or two short sentences, then wait for the caller to speak. "
    "Speak naturally, as on a phone call. Never mention that you are an AI."
)

# Track 2 (authored interaction, v0.7) only: the same deployment and wording
# with the brevity constraint removed, so replies are long enough to be
# interrupted. Identical for every model; it never mentions interruptions,
# backchannels or turn-taking. Track 1 keeps SYSTEM_PROMPT unchanged.
SYSTEM_PROMPT_TRACK2 = (
    "You are the phone assistant for a medical practice's appointments line. "
    "Callers ring to book, change or cancel appointments and to ask simple questions. "
    "Answer each question helpfully and completely. "
    "Speak naturally, as on a phone call. Never mention that you are an AI."
)

READOUTS: tuple[str, ...] = ("onset", "vad_events")
MODES: tuple[str, ...] = ("agent", "vad_only")

# The model ids Cekura's published speech-to-speech benchmark names, pinned here
# so the two benchmarks measure the same models (the service registry's own
# defaults differ for Gemini). The id the run actually resolved to is written to
# run.json under model.provenance.model. Later additions: gpt-realtime-2.1-mini,
# gemini-3.8-live-extended-thinking, gemini-3.1-flash-live-preview.
PINNED_MODELS: dict[str, str] = {
    "openai-realtime": "gpt-realtime-2.1",  # GPT Realtime 2.1
    "gemini-live": "gemini-3.8-live",  # Gemini 3.8 Live (reference-agents/pipecat-s2s/bot.py, pricing/prices.json)
    "grok-realtime": "grok-voice-think-fast-2.0",  # Grok Voice Think Fast 2.0
    "higgs-realtime": "higgs-realtime",  # Boson AI Higgs Realtime
}


def prompt_sha256(text: str = SYSTEM_PROMPT) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_config(provider: str) -> dict[str, Any]:
    """The ``s2s:<provider>`` model's default configuration (``turn-taking-bench run --config``
    overrides any key)."""
    return {
        "provider": provider,
        # The pinned id above; None = the service registry's default for the provider.
        "model": PINNED_MODELS.get(provider),
        # "agent": audio replies under SYSTEM_PROMPT (TurnBench's generative protocol).
        # "vad_only": TurnBench's openai_* baselines' session — text output, no
        # responses, events from the provider's endpointer (OpenAI-shaped only).
        "mode": "agent",
        # "onset": EOT = onset of model speech while the caller is quiet.
        # "vad_events": EOT = the provider's speech_stopped, INT = speech_started.
        "readout": "onset",
        # Which dataset channels get a session. The caller is speaker_1; the
        # agent channel is replaced by the model. Both for TurnBench's dev split,
        # whose two speakers are both humans.
        "channels": [1],
        # The provider's own ("native") turn detection, as the service bench
        # names it: server_vad with no parameters = the provider's defaults.
        # semantic_vad where offered; silence_duration_ms to pin a duration.
        "turn_detection": "server_vad",
        "silence_duration_ms": None,
        "voice": None,  # None = the service registry's default output voice
        "transcribe_input": True,
        # Silence streamed after the last frame so the provider can endpoint the
        # caller's final turn and the last reply can arrive (TurnBench waits 5 s).
        "trailing_grace_s": 3.0,
        # Onset readout: model speech segments closer than this are one onset
        # (TurnBench's gemini_vad MERGE_GAP_S), and an onset while the caller's own
        # channel is active is not an EOT decision (TurnBench's rule).
        "merge_gap_s": 0.5,
        "caller_active_filter": True,
        "caller_merge_gap_s": 0.0,
        # Per-conversation artifacts (caller/agent wav, timelines, events, raw
        # frames, readout) under <run>/artifacts/<conversation>/speaker_<k>/.
        "artifacts": True,
        # Extra constructor arguments for the adapter (the fake adapter's reply
        # timing, for the offline tests).
        "adapter_kwargs": {},
    }


def build(config: dict[str, Any]):
    """Model factory for the registry; imports the live machinery lazily so the
    registry itself needs neither ``websockets`` nor the service bench."""
    from turn_taking_bench.s2s.model import S2SModel

    return S2SModel(config)


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    from turn_taking_bench.s2s.model import model_provenance

    return model_provenance(config)
