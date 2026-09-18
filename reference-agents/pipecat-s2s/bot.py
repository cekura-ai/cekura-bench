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

import os
import sys
from dataclasses import dataclass
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
    logger.info("lane B reference agent: {} {} voice={} agent={}", name, model, voice, server.suite)

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
    task = create_task(pipeline, context, params, runner_args, transport, name, model, voice, server.suite)

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


def create_task(pipeline, context, params, runner_args, transport, provider, model, voice, suite) -> PipelineTask:
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
        metadata = {
            "lane": "B",
            "s2s_provider": provider,
            "s2s_model": model,
            "s2s_voice": voice,
            "agent_definition": suite,
        }
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
