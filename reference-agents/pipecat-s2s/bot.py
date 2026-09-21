"""The agent bench reference agent: one realtime speech-to-speech model as the whole agent.

The service bench measures a provider's realtime service on its own, over a direct
websocket. This agent is the other half of the picture: the same models doing
real work -- tools, a system prompt, a task to finish -- on a real phone call,
inside a real orchestration framework. The two are never ranked against each
other. "The model is fast" and "the deployment is fast" are different claims, and
a single number that mixes them answers neither.

The agent under test here is the whole configuration: this file, the Pipecat
version pinned in requirements.txt, the transport, and the provider. Anyone can
read it, run it, and disagree with a choice in it -- which is the point of a
reference agent, and the reason it is a small single file rather than a framework.

It runs two stacks. Native is one realtime model doing everything. Cascade is
speech-to-text, a text model and text-to-speech, in the same file with the same
prompt, tools and transport, so the only difference between a native row and its
cascade counterpart is the speech path. That is what makes "native or cascaded?"
answerable rather than a matter of opinion.

Deliberately not included: no cascade *fallback* inside a run, no barge-in
tuning, no custom turn strategies, no retries. Every one of those would improve
the agent and make the result harder to attribute. What is measured should be
the provider plus the plainest sensible wiring around it.

Run it::

    export S2S_PROVIDER=openai-realtime AGENT_DIR=appointments
    python bot.py                       # local dev runner
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndTaskFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# The mock-tool contract is shared with the service bench rather than reimplemented here.
# Two implementations of one contract would drift, and a difference between lanes
# could then be our two servers disagreeing rather than anything about the agents.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mock_tools.server import MockToolServer  # noqa: E402

load_dotenv(override=True)


# ── providers ────────────────────────────────────────────────────────────────

def _openai(api_key: str, model: str, voice: str, instructions: str) -> LLMService:
    from pipecat.services.openai.realtime.events import AudioConfiguration, AudioOutput, SessionProperties
    from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService, OpenAIRealtimeLLMSettings

    return OpenAIRealtimeLLMService(
        api_key=api_key,
        settings=OpenAIRealtimeLLMSettings(
            model=model,
            system_instruction=instructions,
            session_properties=SessionProperties(
                audio=AudioConfiguration(output=AudioOutput(voice=voice)),
            ),
        ),
    )


def _gemini(api_key: str, model: str, voice: str, instructions: str) -> LLMService:
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiLiveLLMSettings

    return GeminiLiveLLMService(
        api_key=api_key,
        settings=GeminiLiveLLMSettings(model=model, system_instruction=instructions, voice=voice),
    )


def _grok(api_key: str, model: str, voice: str, instructions: str) -> LLMService:
    from pipecat.services.xai.realtime.events import SessionProperties
    from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService, GrokRealtimeLLMSettings

    return GrokRealtimeLLMService(
        api_key=api_key,
        settings=GrokRealtimeLLMSettings(
            model=model,
            system_instruction=instructions,
            session_properties=SessionProperties(voice=voice),
        ),
    )


def _gpt_live(api_key: str, model: str, voice: str, instructions: str) -> LLMService:
    """The live model plus the backend it hands reasoning to.

    This one is not a single model. ``gpt-live-1`` converses, and delegates
    search, reasoning and tool work to a *separate text model*. Leaving the
    delegation unset is a supported mode, and the wrong one here: delegated work
    is then dropped, so a scenario that needs a tool fails for want of a backend
    rather than for anything about the model. The backend is therefore named
    explicitly, pinned, and written into the build record, because a row that
    does not disclose it is not comparable with one model's row.
    """
    from pipecat.services.openai.live.llm import OpenAILiveLLMService, OpenAILiveLLMSettings
    from pipecat.services.openai.responses.llm import OpenAIResponsesLLMSettings

    return OpenAILiveLLMService(
        api_key=api_key,
        settings=OpenAILiveLLMSettings(model=model, system_instruction=instructions, voice=voice),
        delegation=OpenAILiveLLMService.ResponsesDelegation(
            settings=OpenAIResponsesLLMSettings(model=backend_model()),
        ),
    )


def _nova_sonic(credential: str, model: str, voice: str, instructions: str) -> LLMService:
    """Nova Sonic over Bedrock, with either credential AWS issues.

    An API key is what a team is issued first, and it is presented as a bearer
    token. Pipecat's service signs with SigV4 only, so that form goes through
    ``nova_bearer``, which swaps the client's auth scheme and leaves the wire
    protocol and the event stream untouched.

    An access-key pair is read from the variables AWS itself documents, so a
    reader who already has working AWS credentials in their environment runs
    this agent without re-encoding them into a shape only this file understands.
    The pair wins when both are present: it carries a session token, so it is
    the form that works with temporary credentials.
    """
    from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService, AWSNovaSonicLLMSettings

    settings = AWSNovaSonicLLMSettings(model=model, system_instruction=instructions, voice=voice)
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if access_key_id and secret_access_key:
        return AWSNovaSonicLLMService(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            region=aws_region(),
            settings=settings,
        )

    from nova_bearer import BearerTokenNovaSonic

    return BearerTokenNovaSonic(token=credential, region=aws_region(), settings=settings)


def backend_model() -> str:
    """The text model ``gpt-live-1`` delegates to.

    Pinned rather than left to the API's own default, for the same reason every
    other version here is pinned: a backend that changes underneath a run makes
    two results incomparable without either of them looking wrong.
    """
    return os.getenv("S2S_BACKEND_MODEL", "gpt-5.4-mini")


def aws_region() -> str:
    """The region Bedrock is called in, and the region the record names.

    One accessor because those two must be the same string. A Bedrock API key is
    scoped by region and a model is served in some regions only, so a record
    naming a different region than the call used would describe a run nobody made.
    """
    return os.getenv("AWS_REGION") or "us-east-1"


@dataclass(frozen=True)
class Provider:
    build: Callable[[str, str, str, str], LLMService]
    input_rate: int
    default_model: str
    default_voice: str
    # Every variable that can carry this provider's credential; one of them must
    # be set. A tuple rather than a string because a vendor may document more
    # than one form, and inventing a house format to squeeze them into one
    # variable makes working credentials unusable until they are re-encoded.
    credential_env: tuple[str, ...]
    # The module holding this provider's SDK. Named so it can be imported before
    # a call arrives rather than inside the window being measured; the builders
    # import lazily so that one provider's SDK is not a hard dependency of all.
    module: str
    # What this provider must put on the record beyond the common fields. A
    # provider that delegates part of the work, or that can run in more than one
    # place, is not comparable with one that does not unless it says so -- and
    # declaring it here is what stops a sixth provider being added without it.
    discloses: Callable[[], dict[str, str]] = dict


# ``input_rate`` is load-bearing, not a tuning knob. These services do not
# resample: each base64-encodes the audio frame it is handed and declares a rate
# separately. Open the pipeline at the wrong rate and the model hears the caller
# sped up or slowed down, transcribes it badly, and the run looks like a model
# failure. The telephony serializer resamples the 8 kHz phone leg to whatever the
# pipeline declares, so this is the only place the rate needs to be correct.
PROVIDERS: dict[str, Provider] = {
    "openai-realtime": Provider(
        _openai, 24000, "gpt-realtime-2.1", "marin", ("OPENAI_API_KEY",),
        "pipecat.services.openai.realtime.llm",
    ),
    "gemini-live": Provider(
        _gemini, 16000, "models/gemini-2.5-flash-native-audio-preview-12-2025", "Charon",
        ("GEMINI_API_KEY", "GEMINI_AUTHORIZATION"), "pipecat.services.google.gemini_live.llm",
    ),
    "grok-realtime": Provider(
        _grok, 16000, "grok-voice-latest", "eve", ("XAI_API_KEY",),
        "pipecat.services.xai.realtime.llm",
    ),
    # GPT-Live is the exception to the paragraph above: it resamples what it is
    # handed. The rate is still declared, so the record says what was sent.
    "gpt-live": Provider(
        _gpt_live, 24000, "gpt-live-1", "marin", ("OPENAI_API_KEY",),
        "pipecat.services.openai.live.llm",
        discloses=lambda: {"s2s_backend_model": backend_model()},
    ),
    # Nova Sonic listens at 16 kHz and speaks at 24 kHz. The pipeline runs at the
    # input rate and the service resamples its own output.
    "nova-sonic": Provider(
        _nova_sonic, 16000, "amazon.nova-2-sonic-v1:0", "matthew", ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID"),
        "pipecat.services.aws.nova_sonic.llm",
        discloses=lambda: {"aws_region": aws_region()},
    ),
}


# ── the cascade, for comparison ──────────────────────────────────────────────
#
# "Is a native speech model better than the pipeline it replaces?" is the one
# question a mixed board can answer and nothing else can. It is only answerable
# if the two sides differ in one thing. So the cascade is this same file, this
# same prompt, these same tools and this same transport, with the speech path
# swapped: speech-to-text, a text model, text-to-speech, instead of one model
# doing all three.
#
# The speech-to-text and text-to-speech services are held fixed across every
# cascade row, and only the text model changes. That is deliberate and it is
# also a limit worth stating: a cascade's latency is dominated by when its
# endpointer decides the caller stopped and how fast its voice starts, not by
# the text model. Holding both fixed makes the text model the only variable
# between cascade rows, and makes the pipeline itself common to all of them --
# so a cascade row says "this vendor's intelligence, delivered through one
# named pipeline", never "cascades are like this".

CASCADE_STT_MODEL = "flux-general-en"
CASCADE_TTS_MODEL = "eleven_flash_v2_5"
CASCADE_TTS_VOICE = "21m00Tcm4TlvDq8ikWAM"
CASCADE_RATE = 16000


@dataclass(frozen=True)
class TextModel:
    """A vendor's text model, as the counterpart to its speech model."""

    build: Callable[[str, str], LLMService]
    default_model: str
    credential_env: tuple[str, ...]
    module: str
    # The speech model this one is the counterpart to, or None for the neutral
    # baseline that belongs to no vendor.
    counterpart_to: str | None = None


def _openai_text(api_key: str, model: str) -> LLMService:
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(api_key=api_key, model=model)


def _google_text(api_key: str, model: str) -> LLMService:
    from pipecat.services.google.llm import GoogleLLMService

    return GoogleLLMService(api_key=api_key, model=model)


def _grok_text(api_key: str, model: str) -> LLMService:
    from pipecat.services.grok.llm import GrokLLMService

    return GrokLLMService(api_key=api_key, model=model)


def _qwen_text(api_key: str, model: str) -> LLMService:
    from pipecat.services.qwen.llm import QwenLLMService

    return QwenLLMService(api_key=api_key, model=model)


TEXT_MODELS: dict[str, TextModel] = {
    # The neutral baseline: the pipeline every cascade row shares, with a text
    # model chosen for being widely understood rather than for matching anyone.
    "cascade-baseline": TextModel(
        _openai_text, "gpt-4.1", ("OPENAI_API_KEY",), "pipecat.services.openai.llm",
    ),
    "cascade-openai": TextModel(
        _openai_text, "gpt-4.1", ("OPENAI_API_KEY",), "pipecat.services.openai.llm",
        counterpart_to="openai-realtime",
    ),
    "cascade-google": TextModel(
        _google_text, "gemini-2.5-flash", ("GEMINI_API_KEY", "GEMINI_AUTHORIZATION"),
        "pipecat.services.google.llm", counterpart_to="gemini-live",
    ),
    "cascade-grok": TextModel(
        _grok_text, "grok-4", ("XAI_API_KEY",), "pipecat.services.grok.llm",
        counterpart_to="grok-realtime",
    ),
    # Qwen has no realtime service to be a counterpart to yet, so this row
    # stands alone until one exists. Recorded as such rather than left out:
    # a vendor present on one side of the comparison and absent on the other
    # is a fact about the board, not a gap to hide.
    "cascade-qwen": TextModel(
        _qwen_text, "qwen-plus", ("DASHSCOPE_API_KEY",), "pipecat.services.qwen.llm",
    ),
}


def build_cascade(text: TextModel, credential: str, model: str, instructions: str):
    """Speech-to-text, a text model, text-to-speech -- the three the native model replaces."""
    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    stt = DeepgramFluxSTTService(api_key=os.environ["DEEPGRAM_API_KEY"], model=CASCADE_STT_MODEL)
    llm = text.build(credential, model)
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        voice_id=os.getenv("CASCADE_TTS_VOICE", CASCADE_TTS_VOICE),
        model=CASCADE_TTS_MODEL,
    )
    return stt, llm, tts


# ── the agent definition ─────────────────────────────────────────────────────

def load_agent() -> MockToolServer:
    """Prompt, greeting, tool schemas and mock data, from the published contract."""
    suite = os.getenv("AGENT_DIR")
    if not suite:
        # No default. A deployment that runs the wrong agent definition produces
        # a full set of plausible, scored, wrong results, and nothing in the
        # transcript says which contract it was answering.
        raise ValueError("AGENT_DIR is not set; it must name a directory under agent-definitions/")
    return MockToolServer(suite, root=REPO_ROOT / "agent-definitions")


# Two things the agent must be able to *do* rather than look up: leave the call,
# and hand it over. Neither is in the published lookup tables, because neither
# returns a record -- but the prompt instructs the agent to end a call and to
# announce a transfer, and a scored call is judged on whether it terminated
# appropriately. An agent with no way to hang up fails that for a reason having
# nothing to do with the model.
#
# The transfer is a mock: there is no second leg in a benchmark deployment, so
# the call completes after the announced handover, which is what handing over
# amounts to from the caller's side.

CALL_CONTROL = {
    "end_call": (
        "End the phone call. Use only after a brief closing message, once the "
        "caller has nothing else or asks to hang up."
    ),
    "transfer_call": (
        "Connect the caller to the arranged transfer destination. Use only after "
        "a routing tool returned a ready live transfer, any handoff record has "
        "been created, and the transfer has been announced to the caller. Never "
        "use it when only a callback or a redirect was arranged."
    ),
}


def build_tools(server: MockToolServer) -> ToolsSchema:
    published = [
        FunctionSchema(
            name=spec.name,
            description=spec.description,
            properties=spec.parameters.get("properties", {}),
            required=spec.parameters.get("required", []),
        )
        for spec in server.tool_specs()
    ]
    control = [
        FunctionSchema(name=name, description=description, properties={}, required=[])
        for name, description in CALL_CONTROL.items()
    ]
    return ToolsSchema(standard_tools=published + control)


def register_tools(llm: LLMService, server: MockToolServer) -> None:
    """Answer every declared tool from the contract's lookup table.

    An input the table does not know returns an explicit miss rather than an
    invented record: the contract's own wording says a no-match means no record
    was found. Inventing one would let an agent that asked for the wrong thing
    score like an agent that asked for the right thing.
    """

    async def handler(params: FunctionCallParams) -> None:
        result = server.call(params.function_name, params.arguments or {})
        logger.info("tool {} -> {}", params.function_name, "hit" if server.calls[-1].matched else "miss")
        await params.result_callback(result)

    async def end_call(params: FunctionCallParams) -> None:
        logger.info("end_call -- closing the call")
        await params.result_callback({"status": "ending_call"})
        await params.llm.push_frame(EndTaskFrame())

    async def transfer_call(params: FunctionCallParams) -> None:
        logger.info("transfer_call -- mock handover, closing the call")
        await params.result_callback({"status": "transferred"})
        await params.llm.push_frame(EndTaskFrame())

    for name in server.tool_names:
        llm.register_function(name, handler)
    llm.register_function("end_call", end_call)
    llm.register_function("transfer_call", transfer_call)


def opening_messages(first_message: str) -> list[dict[str, Any]]:
    """A realtime model has no text-to-speech to hand a greeting to.

    So the greeting becomes an instruction in the opening turn instead. The
    double quotes are folded because they would close the quoted span the
    instruction opens, and the model would then read the punctuation aloud or
    improvise past it.
    """
    greeting = (first_message or "").replace('"', "'").strip()
    if not greeting:
        return [{"role": "user", "content": "Greet the caller in one short sentence, then help them."}]
    return [
        {
            "role": "user",
            "content": f'Open the call by saying exactly this, word for word, and nothing else: "{greeting}"',
        }
    ]


# ── what answered the call ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _commit() -> str:
    """This agent's commit, resolved once for the process.

    Cached because ``run_bot`` is per call, not per process: a fork+exec between
    the transport connecting and the greeting going out would land inside the
    window the service bench is measuring.

    A deployed image carries no git history, so the build stamps the commit in
    instead. That value wins: it is what was actually built, whereas a checkout
    that happens to sit beside the image could be anything.
    """
    stamped = os.getenv("AGENT_COMMIT", "").strip()
    if stamped and stamped != "unknown":
        return stamped
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- a container without git is a caveat, not a crash
        return "unknown"


@lru_cache(maxsize=None)
def _version(package: str) -> str:
    """Cached for the same reason as ``_commit``: it scans installed metadata."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # noqa: BLE001
        return "unknown"


def build_record(provider_key: str, provider: "Provider", model: str, voice: str, server: Any) -> dict[str, Any]:
    """Everything needed to say which build answered a given call.

    A phone call cannot be replayed and a provider endpoint moves underneath us,
    so a recording whose configuration is unknown is not evidence of anything.
    This travels with the run as trace metadata and is logged once at startup, so
    the answer survives even when only the container logs do.

    The pipeline sample rate is in here for a specific reason: these realtime
    services do not resample, and a wrong rate makes the model hear the caller at
    the wrong speed. That failure looks exactly like a bad model unless the rate
    that was actually used is on the record.
    """
    return {
        **_common_record(server),
        "stack": "native",
        "s2s_provider": provider_key,
        "s2s_model": model,
        "s2s_voice": voice,
        # Whatever this provider says it must disclose. Absent rather than empty
        # for a provider with nothing to add: an empty value would read as a
        # field that went unrecorded.
        **provider.discloses(),
        "pipeline_sample_rate": provider.input_rate,
        "config": provider_key,
    }


def _common_record(server: MockToolServer) -> dict[str, Any]:
    """The fields that describe the task rather than the stack.

    Identical for a native row and a cascade row by construction, which is what
    lets the two be compared: if these digests differ, the two agents were not
    given the same job and no difference between them means anything.
    """
    prompt = server.system_prompt or ""
    return {
        "lane": "B",
        "agent_commit": _commit(),
        "agent_definition": server.suite,
        "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "first_message_sha256": hashlib.sha256((server.first_message or "").encode("utf-8")).hexdigest()[:16],
        "tools": ",".join(server.tool_names),
        "pipecat_version": _version("pipecat-ai"),
        "cekura_version": _version("cekura"),
        "cekura_mode": os.getenv("CEKURA_MODE", "track"),
    }


def cascade_record(key: str, text: TextModel, model: str, server: MockToolServer) -> dict[str, Any]:
    """What answered the call when three services answered it instead of one.

    All three are named. A cascade row that recorded only its text model would
    hide the two components that actually decide when it starts speaking and how
    fast its voice begins -- which is most of what a latency column measures.
    """
    return {
        **_common_record(server),
        "stack": "cascade",
        "llm_model": model,
        "stt_model": CASCADE_STT_MODEL,
        "tts_model": CASCADE_TTS_MODEL,
        "counterpart_to": text.counterpart_to,
        "pipeline_sample_rate": CASCADE_RATE,
        "config": key,
    }


# ── the bot ──────────────────────────────────────────────────────────────────

def _credential(variables: tuple[str, ...], label: str) -> str:
    found = next((v for v in map(os.getenv, variables) if v), None)
    if not found:
        raise ValueError(f"none of {', '.join(variables)} is set; {label} cannot start")
    return found


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    server = load_agent()
    name = os.getenv("S2S_PROVIDER", "openai-realtime")
    cascade = TEXT_MODELS.get(name)

    if cascade is not None:
        model = os.getenv("S2S_MODEL", cascade.default_model)
        credential = _credential(cascade.credential_env, name)
        record = cascade_record(name, cascade, model, server)
        logger.info("agent bench reference agent: {}", record)

        stt, llm, tts = build_cascade(cascade, credential, model, server.system_prompt)
        register_tools(llm, server)
        context = LLMContext(
            [{"role": "system", "content": server.system_prompt}, *opening_messages(server.first_message)],
            tools=build_tools(server),
        )
        aggregators = LLMContextAggregatorPair(context)
        # Three services where the native path has one. Everything either side of
        # them -- transport, context, tools, greeting -- is the same code.
        stages = [transport.input(), stt, aggregators.user(), llm, tts, transport.output(), aggregators.assistant()]
        rate = CASCADE_RATE
    else:
        if name not in PROVIDERS:
            known = sorted([*PROVIDERS, *TEXT_MODELS])
            raise ValueError(f"unknown S2S_PROVIDER {name!r}; expected one of {known}")
        provider = PROVIDERS[name]
        credential = _credential(provider.credential_env, name)
        model = os.getenv("S2S_MODEL", provider.default_model)
        voice = os.getenv("S2S_VOICE", provider.default_voice)
        record = build_record(name, provider, model, voice, server)
        logger.info("agent bench reference agent: {}", record)

        llm = provider.build(credential, model, voice, server.system_prompt)
        register_tools(llm, server)
        context = LLMContext(opening_messages(server.first_message), tools=build_tools(server))
        aggregators = LLMContextAggregatorPair(context, realtime_service_mode=True)
        # No separate speech-to-text or text-to-speech: the realtime model is the
        # whole agent, so the pipeline is the transport, the context and the model.
        stages = [transport.input(), aggregators.user(), llm, transport.output(), aggregators.assistant()]
        rate = provider.input_rate

    pipeline = Pipeline(stages)
    params = PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
        audio_in_sample_rate=rate,
        audio_out_sample_rate=rate,
    )
    task = create_task(pipeline, context, params, runner_args, transport, record)

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        # Every call opens from the agent, and this first context frame is also
        # what installs the tools on the realtime session.
        logger.info("caller connected")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("caller disconnected")
        await task.cancel()

    await PipelineRunner(handle_sigint=False).run(task)


def create_task(pipeline, context, params, runner_args, transport, record) -> PipelineTask:
    """Wrap the pipeline in Cekura tracing when credentials are present.

    Tracing is what makes an agent-bench run inspectable afterwards: transcripts, tool
    calls, logs and spans land against the run rather than in a container's
    stdout. Without credentials the agent still runs and still answers the phone,
    it is simply not observed -- a missing key must not be the reason a benchmark
    call fails.
    """
    api_key, agent_id = os.getenv("CEKURA_API_KEY"), os.getenv("CEKURA_AGENT_ID")
    if not (api_key and agent_id):
        logger.info("Cekura tracing off: CEKURA_API_KEY or CEKURA_AGENT_ID unset")
        return PipelineTask(pipeline, params=params)

    try:
        from cekura.pipecat import PipecatTracer

        tracer = PipecatTracer(
            api_key=api_key,
            agent_id=int(agent_id),
            host=os.getenv("CEKURA_HOST", "https://api.cekura.ai"),
        )
        metadata = dict(record)
        # "track" correlates a scenario run and captures transcripts and metadata;
        # "observe" additionally uploads the call audio and starts evaluation.
        # A benchmark run is dispatched with its own run id, so track is the
        # default and runner_args is passed through untouched to carry it.
        if os.getenv("CEKURA_MODE", "track") == "observe":
            return tracer.observe_and_create_task(
                pipeline, context, runner_args=runner_args, transport=transport,
                custom_metadata=metadata, params=params,
            )
        return tracer.track_and_create_task(
            pipeline, context, runner_args=runner_args, transport=transport,
            custom_metadata=metadata, params=params,
        )
    except Exception as exc:  # noqa: BLE001 -- observability must never fail a call
        logger.warning("Cekura tracing disabled: {}", exc)
        return PipelineTask(pipeline, params=params)


def warm() -> None:
    """Do the once-per-process work now, before a caller is on the line.

    Everything here is paid exactly once, and left alone it would be paid on the
    first call the process answers -- which is inside the window this benchmark
    exists to measure. Importing a provider SDK is the expensive one, a few
    hundred milliseconds for some of them, and resolving the commit forks the
    interpreter. On a replica that scales up mid-campaign, that first call is a
    *scored* call, and nothing in the result would distinguish the cost from a
    slow model.

    Failures are logged, not raised: every one of these is recoverable later, and
    a warm-up that refuses to start the process would turn a slow first call into
    no call at all. A genuinely unusable configuration is refused in ``run_bot``,
    where the error can name what is missing.
    """
    import importlib

    _commit()
    _version("pipecat-ai")
    _version("cekura")
    name = os.getenv("S2S_PROVIDER", "openai-realtime")
    selected = PROVIDERS.get(name) or TEXT_MODELS.get(name)
    if selected is None:
        return  # run_bot raises with the list of valid names
    modules = [selected.module]
    if name in TEXT_MODELS:
        # A cascade loads three services, and the two it does not name are the
        # ones that decide when it starts speaking and how fast its voice begins.
        modules += ["pipecat.services.deepgram.flux.stt", "pipecat.services.elevenlabs.tts"]
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 -- the builder will raise this again, in context
            logger.warning("could not pre-import {} for {}: {}", module, name, exc)


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point used by the Pipecat runner and by Pipecat Cloud."""
    telephony = lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)  # noqa: E731
    transport = await create_transport(
        runner_args,
        {
            "twilio": telephony,
            "telnyx": telephony,
            "daily": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
            "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        },
    )
    await run_bot(transport, runner_args)


warm()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
