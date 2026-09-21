"""Lane B reference agent: one realtime speech-to-speech model as the whole agent.

Lane A measures a provider's realtime service on its own, over a direct
websocket. This agent is the other half of the picture: the same models doing
real work -- tools, a system prompt, a task to finish -- on a real phone call,
inside a real orchestration framework. The two are never ranked against each
other. "The model is fast" and "the deployment is fast" are different claims, and
a single number that mixes them answers neither.

The agent under test here is the whole configuration: this file, the Pipecat
version pinned in requirements.txt, the transport, and the provider. Anyone can
read it, run it, and disagree with a choice in it -- which is the point of a
reference agent, and the reason it is a small single file rather than a framework.

Deliberately not included: no cascade fallback, no barge-in tuning, no custom turn
strategies, no retries. Every one of those would improve the agent and make the
result harder to attribute. What is measured should be the provider plus the
plainest sensible wiring around it.

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
from pipecat.frames.frames import LLMRunFrame
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

# The mock-tool contract is shared with Lane A rather than reimplemented here.
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
        settings=OpenAILiveLLMSettings(system_instruction=instructions, voice=voice),
        delegation=OpenAILiveLLMService.ResponsesDelegation(
            settings=OpenAIResponsesLLMSettings(model=backend_model()),
        ),
    )


def _nova_sonic(credential: str, model: str, voice: str, instructions: str) -> LLMService:
    """Nova Sonic over Bedrock, with either credential form.

    Bedrock accepts an API-key bearer token as well as an access-key pair, and
    the token is what a team is issued first. Pipecat's service signs with
    SigV4 only, so a bearer token is applied by swapping the client's auth
    scheme -- the wire protocol and the event stream are untouched, only who
    signs the request changes.

    A bearer token is recognised by the absence of a secret: the pair form is
    given as ``access_key_id:secret_access_key`` in the same variable.
    """
    from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService, AWSNovaSonicLLMSettings

    region = os.getenv("AWS_REGION") or "us-east-1"
    settings = AWSNovaSonicLLMSettings(model=model, system_instruction=instructions, voice=voice)
    if ":" in credential:
        access_key_id, secret_access_key = credential.split(":", 1)
        return AWSNovaSonicLLMService(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
            settings=settings,
        )

    from nova_bearer import BearerTokenNovaSonic

    return BearerTokenNovaSonic(token=credential, region=region, settings=settings)


def backend_model() -> str:
    """The text model ``gpt-live-1`` delegates to. Pinned, and disclosed."""
    return os.getenv("S2S_BACKEND_MODEL", DEFAULT_BACKEND_MODEL)


# Pinned rather than left to the API's own default, for the same reason every
# other version here is pinned: a backend that changes underneath a run makes two
# results incomparable without either of them looking wrong.
DEFAULT_BACKEND_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class Provider:
    build: Callable[[str, str, str, str], LLMService]
    input_rate: int
    default_model: str
    default_voice: str
    credential_env: str


# ``input_rate`` is load-bearing, not a tuning knob. These services do not
# resample: each base64-encodes the audio frame it is handed and declares a rate
# separately. Open the pipeline at the wrong rate and the model hears the caller
# sped up or slowed down, transcribes it badly, and the run looks like a model
# failure. The telephony serializer resamples the 8 kHz phone leg to whatever the
# pipeline declares, so this is the only place the rate needs to be correct.
PROVIDERS: dict[str, Provider] = {
    "openai-realtime": Provider(_openai, 24000, "gpt-realtime-2.1", "marin", "OPENAI_API_KEY"),
    "gemini-live": Provider(
        _gemini, 16000, "models/gemini-2.5-flash-native-audio-preview-12-2025", "Charon",
        "GEMINI_API_KEY",
    ),
    "grok-realtime": Provider(_grok, 16000, "grok-voice-latest", "eve", "XAI_API_KEY"),
    # GPT-Live is the exception to the paragraph above: it resamples what it is
    # handed. The rate is still declared, so the record says what was sent.
    "gpt-live": Provider(_gpt_live, 24000, "gpt-live-1", "marin", "OPENAI_API_KEY"),
    # Nova Sonic listens at 16 kHz and speaks at 24 kHz. The pipeline runs at the
    # input rate and the service resamples its own output.
    "nova-sonic": Provider(
        _nova_sonic, 16000, "amazon.nova-2-sonic-v1:0", "matthew", "AWS_BEARER_TOKEN_BEDROCK",
    ),
}


# ── the agent definition ─────────────────────────────────────────────────────

def load_agent() -> MockToolServer:
    """Prompt, greeting, tool schemas and mock data, from the published contract."""
    return MockToolServer(os.getenv("AGENT_DIR", "appointments"), root=REPO_ROOT / "agent-definitions")


def build_tools(server: MockToolServer) -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=spec.name,
                description=spec.description,
                properties=spec.parameters.get("properties", {}),
                required=spec.parameters.get("required", []),
            )
            for spec in server.tool_specs()
        ]
    )


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

    for name in server.tool_names:
        llm.register_function(name, handler)


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
    window Lane A is measuring.
    """
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
    prompt = server.system_prompt or ""
    # 16 characters, matching Lane A's digest of the same strings: the two lanes
    # are compared on whether they were given the same prompt, and that check
    # fails silently if one side truncates differently.
    return {
        "lane": "B",
        "agent_commit": _commit(),
        "s2s_provider": provider_key,
        "s2s_model": model,
        "s2s_voice": voice,
        # Only one provider has a backend, and its row is not readable without
        # knowing which one: the reasoning is not done by the model named above.
        **({"s2s_backend_model": backend_model()} if provider_key == "gpt-live" else {}),
        **({"aws_region": os.getenv("AWS_REGION") or "us-east-1"} if provider_key == "nova-sonic" else {}),
        "pipeline_sample_rate": provider.input_rate,
        "agent_definition": server.suite,
        "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "first_message_sha256": hashlib.sha256((server.first_message or "").encode("utf-8")).hexdigest()[:16],
        "tools": ",".join(server.tool_names),
        "pipecat_version": _version("pipecat-ai"),
        "cekura_version": _version("cekura"),
        "cekura_mode": os.getenv("CEKURA_MODE", "track"),
    }


# ── the bot ──────────────────────────────────────────────────────────────────

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    server = load_agent()
    name = os.getenv("S2S_PROVIDER", "openai-realtime")
    if name not in PROVIDERS:
        raise ValueError(f"unknown S2S_PROVIDER {name!r}; expected one of {sorted(PROVIDERS)}")
    provider = PROVIDERS[name]

    api_key = os.getenv(provider.credential_env)
    if not api_key:
        raise ValueError(f"{provider.credential_env} is not set; {name} cannot start")

    model = os.getenv("S2S_MODEL", provider.default_model)
    voice = os.getenv("S2S_VOICE", provider.default_voice)
    record = build_record(name, provider, model, voice, server)
    logger.info("lane B reference agent: {}", record)

    llm = provider.build(api_key, model, voice, server.system_prompt)
    register_tools(llm, server)

    context = LLMContext(opening_messages(server.first_message), tools=build_tools(server))
    aggregators = LLMContextAggregatorPair(context, realtime_service_mode=True)

    # No separate speech-to-text or text-to-speech: the realtime model is the
    # whole agent, so the pipeline is the transport, the context and the model.
    pipeline = Pipeline(
        [
            transport.input(),
            aggregators.user(),
            llm,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    params = PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
        audio_in_sample_rate=provider.input_rate,
        audio_out_sample_rate=provider.input_rate,
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

    Tracing is what makes a Lane B run inspectable afterwards: transcripts, tool
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


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
