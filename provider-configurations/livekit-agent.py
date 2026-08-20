"""Canonical "Ava" benchmark agent on LiveKit Agents (self-hosted), two personas.

One worker serves BOTH benchmark agent definitions; the persona is selected per
call via Cekura's dispatch config (`{"test_persona": "appointments" | "insurance"}`
in the Cekura agent's LiveKit Config JSON, delivered through dispatch job
metadata). Definitions live in ./definitions/<persona>/:
  system-prompt.txt, first-message.txt, tool-definitions.json, mock-tools.json

Constant stack (updated 2026-07-30 per "Cekura - LiveKit recommended setup"):
  LLM  LiveKit Inference, LLM_MODEL env (default openai/gpt-4.1), temp 0  |  STT  Deepgram nova-3 (en)
  TTS  Cartesia sonic-3 (voice 9626c31c-bec5-4cca-baa8-f8ba9e84c8bc)
  Turn detection: LiveKit Inference TurnDetector via TurnHandlingOptions
  Noise cancellation: Krisp BVC Telephony on SIP calls; ai-coustics QUAIL_L otherwise

Tools are built dynamically from each persona's tool-definitions.json (the fixed
cross-platform tool contract) via function_tool(raw_schema=...) and resolve
against that persona's mock-tools.json — the worker IS the fake backend.

Run (needs LIVEKIT_* + vendor creds in .env):  python agent.py dev
Cekura wiring: cekura[livekit] LiveKitTracer (scaffold: cekura-python
examples/livekit-appointment-agent). track_session correlates via dispatch job
metadata (scenario_id/run_id — no PSTN-style matching) and, when the Cekura agent
has mock tools configured, injects them in place of the same-named local tools
(disable with CEKURA_MOCK_TOOLS_ENABLED=false to force local mock_backend).
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    TurnHandlingOptions,
    inference,
    JobContext,
    cli,
    function_tool,
    get_job_context,
    RunContext,
    StopResponse,
    room_io,
)
from livekit.plugins import noise_cancellation, openai

try:
    # Optional: ai-coustics enhancement for non-SIP audio. Not in this repo's
    # pyproject/uv.lock (only the deployed worker's env has it), so degrade to
    # Krisp BVC when it's absent instead of failing at import time.
    from livekit.plugins import ai_coustics
except ImportError:
    ai_coustics = None
from cekura.livekit import LiveKitTracer

from mock_backend import MockBackend

logger = logging.getLogger("livekit-demo")

ROOT = Path(__file__).resolve().parent
DEFINITIONS = ROOT / "definitions"
# override=True: the repo .env is authoritative — the shell exports a different
# CEKURA_API_KEY (other org) which otherwise silently wins and 403s every tracer call
# Prefer .env; fall back to .env.local (common local override filename).
_env_file = ROOT / ".env" if (ROOT / ".env").exists() else ROOT / ".env.local"
load_dotenv(_env_file, override=True)


def _env(key, default=None):
    v = os.environ.get(key, default)
    return v.strip() if isinstance(v, str) else v


# Full LiveKit Inference model id for the platform path (e.g. "openai/gpt-4.1",
# "google/gemma-4-31b-it"). Azure fallback keeps its own deployment name below.
LLM_MODEL = _env("LLM_MODEL", "openai/gpt-4.1")
LLM_DEPLOYMENT = _env("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0") or 0)
STT_MODEL = _env("STT_MODEL", "deepgram/nova-3")
TTS_MODEL = _env("TTS_MODEL", "cartesia/sonic-3")
VOICE_ID = _env("VOICE_ID", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")  # Cartesia Sonic-3 default voice
COMPONENT_METRICS_ENABLED = (_env("BENCHMARK_COMPONENT_METRICS", "true") or "").lower() in (
    "1",
    "true",
    "yes",
)
COMPONENT_METRICS_JSONL = _env("BENCHMARK_COMPONENT_METRICS_JSONL", "component-metrics.jsonl")

# --- Cekura observability/simulation tracer (no-op until an agent id is set).
# Appointments and insurance are SEPARATE Cekura agents, so the tracer is built
# per job once the persona is resolved: CEKURA_AGENT_ID_APPOINTMENT /
# CEKURA_AGENT_ID_INSURANCE, with CEKURA_AGENT_ID as the fallback for both. ---
CEKURA_AGENT_IDS = {
    "appointments": int(_env("CEKURA_AGENT_ID_APPOINTMENT") or _env("CEKURA_AGENT_ID") or 0),
    "insurance": int(_env("CEKURA_AGENT_ID_INSURANCE") or _env("CEKURA_AGENT_ID") or 0),
}


def _cekura_tracer(persona: str):
    """Build the tracer for this persona's Cekura agent (disabled if no id)."""
    agent_id = CEKURA_AGENT_IDS.get(persona, 0)
    tracer = LiveKitTracer(
        api_key=_env("CEKURA_API_KEY", ""),
        agent_id=agent_id,
        host=_env("CEKURA_HOST", "https://api.cekura.ai"),
        enabled=bool(agent_id),
    )
    return tracer, agent_id


def _metric_to_dict(metric) -> dict:
    if hasattr(metric, "model_dump"):
        return metric.model_dump(mode="json", exclude_none=True)
    if hasattr(metric, "dict"):
        return metric.dict(exclude_none=True)
    return dict(metric or {})


def _metric_component(metric_type: str) -> str:
    if metric_type.startswith("stt_"):
        return "stt"
    if metric_type.startswith("llm_") or metric_type.startswith("realtime_"):
        return "llm"
    if metric_type.startswith("tts_"):
        return "tts"
    return "other"


def _jsonl_path() -> Path | None:
    if not COMPONENT_METRICS_JSONL:
        return None
    path = Path(COMPONENT_METRICS_JSONL)
    return path if path.is_absolute() else ROOT / path


def _record_component_metric(payload: dict) -> None:
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    logger.info("BENCHMARK_COMPONENT_METRIC %s", line)
    path = _jsonl_path()
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("component metric jsonl write failed: %s", exc)


def _attach_component_metrics(stt, llm, tts, meta_fn) -> None:
    if not COMPONENT_METRICS_ENABLED:
        return

    def attach(component: str, emitter) -> None:
        on = getattr(emitter, "on", None)
        if not callable(on):
            logger.warning("component metrics unavailable for %s: no event emitter", component)
            return

        def on_metrics(metric):
            metric_obj = getattr(metric, "metrics", metric)
            raw = _metric_to_dict(metric_obj)
            metric_type = raw.get("type") or type(metric_obj).__name__
            try:
                meta = meta_fn()
            except Exception as exc:
                meta = {"meta_error": str(exc)}
            payload = {
                "schema": "benchmark_component_metric_v1",
                "platform": "livekit",
                "component": component or _metric_component(metric_type),
                "metric_type": metric_type,
                "observed_at": time.time(),
                "meta": meta,
                "metric": raw,
            }
            _record_component_metric(payload)

        try:
            on("metrics_collected", on_metrics)
        except Exception as exc:
            logger.warning("component metrics attach failed for %s: %s", component, exc)

    attach("stt", stt)
    attach("llm", llm)
    attach("tts", tts)


# --- persona selection & loading -------------------------------------------------
def _resolve_persona(ctx) -> str:
    """Pick the test persona. Call AFTER ctx.connect().

    Sources, in priority order:
    1. LiveKit simulation userdata (`lk agent simulate` scenarios set
       userdata.test_persona; read via ctx.simulation_context(), agents >= 1.6.6)
    2. Room metadata - Cekura delivers the agent's "LiveKit Config (JSON)"
       (e.g. {"test_persona": "insurance"}) as room metadata (cekura SDK exposes
       it as get_simulation_data()['additional_config']; room metadata is only
       populated after ctx.connect())
    3. Job/dispatch metadata (carries scenario_id/run_id/test_profile_data)
    4. TEST_PERSONA env, then the appointments default
    """

    def find(d):
        if isinstance(d, dict):
            v = d.get("test_persona")
            if isinstance(v, str):
                return v
            for x in d.values():
                r = find(x)
                if r:
                    return r
        elif isinstance(d, list):
            for x in d:
                r = find(x)
                if r:
                    return r
        return None

    persona = None

    # 1. LiveKit simulation scenarios (lk agent simulate) pass userdata
    sim_fn = getattr(ctx, "simulation_context", None)
    sim = sim_fn() if callable(sim_fn) else None
    if sim is not None:
        try:
            ud = sim.userdata() or {}
            if isinstance(ud, dict):
                persona = ud.get("test_persona")
        except Exception as exc:
            logger.debug("could not read simulation userdata: %s", exc)
        if persona:
            logger.info("persona from simulation userdata: %s", persona)

    for raw in (
        getattr(getattr(ctx, "room", None), "metadata", None),  # LiveKit Config (JSON)
        getattr(getattr(ctx, "job", None), "metadata", None),  # dispatch metadata
    ):
        if persona:
            break
        if raw and raw.strip():
            try:
                persona = find(json.loads(raw))
            except (ValueError, TypeError):
                persona = None
    persona = (persona or _env("TEST_PERSONA", "appointments")).strip().lower()
    if persona not in ("appointments", "insurance"):
        logger.warning("unknown test_persona %r; defaulting to appointments", persona)
        persona = "appointments"
    return persona


def _make_mock_tool(tdef: dict, backend: MockBackend):
    """Wrap one entry of tool-definitions.json (the fixed cross-platform tool
    contract) as a raw-schema function tool resolving against the mock backend."""
    name = tdef["name"]

    async def handler(raw_arguments: dict) -> dict:
        return backend.resolve(name, raw_arguments)

    return function_tool(
        handler,
        raw_schema={
            "name": name,
            "description": tdef.get("description", ""),
            "parameters": tdef.get("parameters", {"type": "object", "properties": {}}),
        },
    )


# --- native end-call. The framework ships an EndCallTool (livekit.agents.beta.tools),
#     but it is still beta, so we define our own: the hang-up is deleting the room
#     (disconnects everyone, ends the SIP/telephony leg). Defined in-code (not in the
#     per-persona tool-definitions.json — that's the cross-platform CUSTOM-tool
#     contract); parallels the Pipecat end_call.
#     closing_message is a required argument: a hang-up should never be silent,
#     and making the sign-off part of the tool call guarantees the caller hears
#     a final utterance before the line drops. The tool speaks it, then ends the
#     call. ---
@function_tool
async def end_call(ctx: RunContext, closing_message: str) -> str:
    """Hang up the call, speaking `closing_message` as your final utterance first.

    ALWAYS end the call with this tool once the caller has confirmed they need
    nothing else, or when the caller says goodbye / has to go. Saying goodbye
    without calling this tool leaves the phone line open. Do not call it while
    the caller still has an open request.

    closing_message (REQUIRED): your complete, contextual final sign-off — it is
    spoken aloud to the caller and nothing after it will be heard. Author it for
    THIS call:
    - acknowledge where the call ended up (e.g. recap that they're all set after
      a booking, or acknowledge that they have to go / are deferring);
    - if the caller abandoned, deferred, or must leave, include an explicit
      callback / return-path offer (e.g. "No problem — call us back anytime and
      we can pick this up.");
    - close with a brief, warm goodbye.
    closing_message must NEVER contain a question — the caller cannot answer it.
    If you want to ask "is there anything else?", ask it as a normal message and
    do NOT call this tool yet. Call this tool at most ONCE per call; put the
    entire sign-off in this argument instead of saying it as a separate message."""
    session = ctx.session
    # The call ends once. In simulations the session outlives the hang-up (the
    # room is not deleted, see below), so the caller may keep exchanging
    # pleasantries and the model may invoke end_call again. Answer those trailing
    # turns with a brief fixed goodbye — repeating the full sign-off is noisy,
    # and going silent leaves the caller talking to a dead line.
    if getattr(session, "_end_call_guard", False):
        try:
            await session.say("Goodbye.")
        except Exception:
            pass
        raise StopResponse()
    setattr(session, "_end_call_guard", True)
    # let any in-flight agent speech finish, then speak the authored sign-off
    current = getattr(session, "current_speech", None)
    if current is not None:
        try:
            await current.wait_for_playout()
        except Exception:
            pass
    msg = (closing_message or "").strip()
    if msg:
        try:
            await session.say(msg)
        except Exception as exc:
            logger.warning("end_call: failed to speak closing message: %s", exc)
    job_ctx = get_job_context()
    # In `lk agent simulate` runs, deleting the room would tear down the
    # simulated caller mid-run; the simulation framework ends the conversation
    # itself. Record the hang-up intent and suppress the follow-up LLM reply
    # (StopResponse) — the sign-off has already been spoken, so a generated
    # response to the tool output would just duplicate it.
    sim_fn = getattr(job_ctx, "simulation_context", None)
    if callable(sim_fn) and sim_fn() is not None:
        logger.info("end_call invoked in simulation - skipping room deletion")
        raise StopResponse()
    # Real calls: return the tool result (observability tooling records it) and
    # delete the room, which disconnects all participants and ends the call.
    await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
    return "The call has ended."


async def _wait_until_answered(ctx: JobContext, timeout: float = 45.0) -> rtc.RemoteParticipant:
    """Block until there is a remote participant who can actually hear us.

    Returns the first remote participant, so the caller can branch on its kind
    (e.g. SIP vs WebRTC for noise-cancellation selection).

    on_enter speaks the fixed first message the moment the session starts. For
    agent-initiated (OUTBOUND) SIP calls the agent is in the room before the
    callee picks up, so without this gate the greeting plays into ringing /
    early media. We wait for a remote participant and, if it is an outbound
    SIP participant, for sip.callStatus == "active" (answered).

    INBOUND calls must NOT be gated: they join the room with
    sip.callStatus == "ringing", and the SIP bridge holds the 200 OK until the
    SIP participant subscribes to remote audio (waitSubscribe in
    sip/pkg/sip/inbound.go) — i.e. until WE publish via session.start().
    Waiting for "active" here deadlocks until this timeout fires, adding ~45s
    of ring time for the caller and ~45s of leading silence in egress
    recordings.
    """
    participant = await ctx.wait_for_participant()
    if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return participant
    if participant.attributes.get("sip.callDirection") != "outbound":
        # inbound (or direction attribute missing): the bridge answers once we
        # publish audio — start the session immediately instead of deadlocking.
        return participant

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = participant.attributes.get("sip.callStatus")
        if status is None or status == "active":
            return participant  # answered (or attribute not provided - don't block)
        if status == "hangup":
            logger.info("call ended before it was answered (callStatus=hangup)")
            return participant
        await asyncio.sleep(0.2)
    logger.warning("timed out after %.0fs waiting for SIP answer; continuing", timeout)
    return participant


def _load_persona(persona: str):
    d = DEFINITIONS / persona
    prompt = (d / "system-prompt.txt").read_text().rstrip("\n")
    first_message = (d / "first-message.txt").read_text().strip()
    backend = MockBackend(d / "mock-tools.json")
    tool_defs = json.loads((d / "tool-definitions.json").read_text())
    tools = [_make_mock_tool(t, backend) for t in tool_defs]
    tools.append(end_call)
    return prompt, first_message, tools


class BenchmarkAgent(Agent):
    def __init__(self, instructions: str, tools: list, first_message: str):
        super().__init__(instructions=instructions, tools=tools)
        self._first_message = first_message

    async def on_enter(self):
        # Fixed first message: speak it deterministically via session.say() so it
        # plays immediately with zero LLM latency. Left interruptible (default).
        await self.session.say(self._first_message)


def _build_llm():
    # LLM-source-v2 (2026-07-03, Tarush mandate): platform provisioning wherever the
    # platform offers it -> LiveKit Inference gateway. temperature passed via
    # extra_kwargs (gateway forwards OpenAI-compatible params).
    if _env("LLM_SOURCE", "platform") == "platform":
        from livekit.agents import inference
        return inference.LLM(model=LLM_MODEL,
                             extra_kwargs={"temperature": LLM_TEMPERATURE})
    # fallback: BYO Azure (pre-mandate config, kept for rollback/A-B)
    return openai.LLM.with_azure(
        azure_deployment=LLM_DEPLOYMENT,
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_API_KEY") or _env("OPENAI_API_KEY"),
        api_version=_env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=LLM_TEMPERATURE,
    )


server = AgentServer()

@server.rtc_session(agent_name="cekura-benchmark-agent")
async def entrypoint(ctx: JobContext):
    # Connect FIRST: the persona comes from the Cekura agent's "LiveKit Config
    # (JSON)", which arrives as room metadata and is empty until ctx.connect().
    await ctx.connect()

    # Access job/dispatch metadata (set by Cekura in automated flows) and room
    # metadata (the Cekura agent's "LiveKit Config (JSON)") — log both verbatim
    # so every run records exactly what the worker was dispatched with.
    def _parse_meta(raw, label):
        if not (raw and raw.strip()):
            return {}
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("unparseable %s metadata: %r", label, raw[:200])
            return {}

    job_metadata = _parse_meta(getattr(ctx.job, "metadata", None), "job")
    room_metadata = _parse_meta(getattr(ctx.room, "metadata", None), "room")
    logger.info(
        "dispatch metadata received",
        extra={"job_metadata": job_metadata, "room_metadata": room_metadata},
    )

    persona = _resolve_persona(ctx)
    logger.info("resolved test persona: %s", persona)
    prompt, first_message, tools = _load_persona(persona)
    cekura, cekura_agent_id = _cekura_tracer(persona)
    logger.info("cekura agent id for persona %s: %s", persona, cekura_agent_id or "unset")

    # A LiveKit simulation (`lk agent simulate`) is NOT a Cekura call: the Cekura
    # tracer's track_session injects dashboard mock tools over our local ones —
    # including an end_call mock that returns text and never terminates the call,
    # which hangs simulation scenarios that end via end_call. Skip Cekura tracing
    # entirely under a simulation context so our own (sim-aware) tools run.
    _sim_fn = getattr(ctx, "simulation_context", None)
    in_simulation = callable(_sim_fn) and _sim_fn() is not None

    stt = inference.STT(
        model=STT_MODEL,
        language="en",
    )
    llm = _build_llm()
    tts = inference.TTS(
        model=TTS_MODEL,
        voice=VOICE_ID,
    )

    def metric_meta() -> dict:
        try:
            sim = cekura.get_simulation_data(ctx) if cekura_agent_id else None
        except Exception:
            sim = None
        return {
            "agent": "livekit-demo",
            "persona": persona,
            "cekura_agent_id": cekura_agent_id or None,
            "room": getattr(ctx.room, "name", None),
            "run_id": (sim or {}).get("run_id"),
            "scenario_id": (sim or {}).get("scenario_id"),
        }

    _attach_component_metrics(stt, llm, tts, metric_meta)

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        # config revision 2026-07-30 (Cekura - LiveKit recommended setup):
        # LiveKit Inference turn detector — contextually aware end-of-utterance model,
        # served via LiveKit Inference (no local model download or VAD prewarm), plus
        # preemptive generation (LLM/TTS start on interim transcripts before EOU confirm).
        # Interruptions/endpointing stay at framework defaults (no benchmark-specific tuning).
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            preemptive_generation={"enabled": True},
        ),
    )
    agent = BenchmarkAgent(prompt, tools, first_message)

    if cekura_agent_id and not in_simulation:
        # Cekura voice calls: transcript/tool export keyed by dispatch metadata
        # (scenario_id/run_id) + dashboard mock-tool injection into `agent`
        await cekura.track_session(ctx, session, agent)
    elif in_simulation:
        logger.info("LiveKit simulation context — skipping Cekura tracing/mock injection")
    else:
        logger.warning("no Cekura agent id for persona %s — running without Cekura tracing", persona)

    if cekura_agent_id:
        sim = cekura.get_simulation_data(ctx)
        if sim:
            logger.info(f"Cekura simulation call: run_id={sim.get('run_id')} scenario_id={sim.get('scenario_id')}")

    # outbound: don't speak the first message until the callee has answered
    # (inbound passes through immediately — see _wait_until_answered docstring)
    remote = await _wait_until_answered(ctx)

    if cekura_agent_id and not in_simulation:
        # production/observe: audio recording (dual-channel egress) + call log.
        # Started AFTER the answer gate so recordings never open with
        # pre-answer ringing silence — Cekura's call matching breaks on the
        # ~45s of leading dead air that an early egress start produces.
        await cekura.observe_session(ctx, session)

    # Input audio cleanup on the caller's track before it reaches STT:
    #   SIP calls  -> Krisp BVC Telephony (background voice cancellation tuned
    #                 for narrowband/PSTN audio; requires LiveKit Cloud)
    #   WebRTC / simulate -> ai-coustics QUAIL_L when the plugin is installed,
    #                        else Krisp BVC (wideband model)
    if remote is not None and remote.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        nc = noise_cancellation.BVCTelephony()
    elif ai_coustics is not None:
        nc = ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_L)
    else:
        nc = noise_cancellation.BVC()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=nc,
            ),
        ),
    )
    # first message is spoken in BenchmarkAgent.on_enter (fixed greeting, zero LLM latency)


if __name__ == "__main__":
    cli.run_app(server)
