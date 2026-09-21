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

One deployment serves every configuration. Which provider, which model, which
voice and which agent definition are decided *per call*, by the session that
starts it, so a cohort is a set of run configurations rather than a set of
images. Credentials are the deliberate exception and stay in the environment:
a key that travels with a request is a key that ends up in a log.

Run it::

    export S2S_PROVIDER=openai-realtime AGENT_DIR=appointments
    python bot.py                       # local dev runner: the environment stands in
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

# The mock-tool contract is shared with the service bench rather than reimplemented here.
# Two implementations of one contract would drift, and a difference between lanes
# could then be our two servers disagreeing rather than anything about the agents.
#
# Two layouts have to work, and they disagree about where "beside the agent" is.
# In the repository this file sits three directories below the root that holds
# the contract; in the image everything is flattened into one working directory.
# So the root is whichever candidate actually carries the contract, rather than a
# count of parent directories that is correct in exactly one of the two places.
def _repo_root(here: Path | None = None) -> Path:
    here = (here or Path(__file__)).resolve()
    for candidate in (here.parent, here.parent.parent.parent):
        if (candidate / "agent-definitions").is_dir():
            return candidate
    # Nothing found: keep the repository layout, so the failure names the path it
    # expected rather than a directory that happens to exist.
    return here.parent.parent.parent


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))

from mock_tools.server import MockToolServer  # noqa: E402

load_dotenv(override=True)

# One process answers more than one call: the platform reuses a warm worker
# across sessions. The first call on a worker is the one that pays whatever
# start-up cost is left after the warm-up below, and nothing else in a record
# distinguishes it -- so the worker and its call count are on every record, and
# a first-call outlier can be identified rather than guessed at.
INSTANCE = uuid.uuid4().hex[:12]
_calls_answered = 0


# ── what this call asked for ─────────────────────────────────────────────────

class Settings:
    """The configuration for one call: the session first, the environment second.

    One image answers for every provider, so what is being measured cannot be
    baked into it. The platform starts each session with a body, and these keys
    arrive in it; the environment is the fallback, which is what makes ``python
    bot.py`` on a laptop work unchanged and lets a deployment carry a default.

    Two rules, both load-bearing:

    *Credentials are never read from here.* They come from the environment only
    (see ``_credential``). A key sent per call would be copied into every log,
    trace and session record that quotes the request body.

    *Only the keys below are accepted from the session.* The platform flattens a
    scenario's own variables into the same body, so an unfiltered read would let
    a fixture field named like one of these silently change what was measured --
    a scored run against the wrong agent definition, with nothing saying so.
    """

    KEYS = (
        "s2s_provider",
        "s2s_model",
        "s2s_voice",
        "agent_dir",
        "s2s_backend_model",
        "aws_region",
        "qwen_region",
        "qwen_workspace_id",
        "cascade_tts_voice",
        "cekura_mode",
    )

    def __init__(self, body: Any = None) -> None:
        raw = body if isinstance(body, dict) else {}
        self._session = {
            key.lower(): str(value).strip()
            for key, value in raw.items()
            if isinstance(key, str)
            and key.lower() in self.KEYS
            and isinstance(value, (str, int, float))
            and str(value).strip()
        }

    def get(self, key: str, default: str | None = None) -> str | None:
        """What this call asked for, or what the environment says, or the default."""
        if key in self._session:
            return self._session[key]
        return (os.getenv(key.upper()) or "").strip() or default

    def source(self, key: str) -> str:
        """Which of the two decided a key -- recorded, so a row says how it was configured."""
        return "session" if key in self._session else "environment"


# ── providers ────────────────────────────────────────────────────────────────

def _openai(api_key: str, model: str, voice: str, instructions: str, settings: Settings) -> LLMService:
    from pipecat.services.openai.realtime.events import (
        AudioConfiguration,
        AudioInput,
        AudioOutput,
        InputAudioTranscription,
        SessionProperties,
    )
    from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService, OpenAIRealtimeLLMSettings

    return OpenAIRealtimeLLMService(
        api_key=api_key,
        settings=OpenAIRealtimeLLMSettings(
            model=model,
            system_instruction=instructions,
            session_properties=SessionProperties(
                audio=AudioConfiguration(
                    # Asked for, because this service transcribes the caller only
                    # when told to. See ``CALLER_TRANSCRIPTION`` above the table.
                    input=AudioInput(transcription=InputAudioTranscription()),
                    output=AudioOutput(voice=voice),
                ),
            ),
        ),
    )


def _gemini(api_key: str, model: str, voice: str, instructions: str, settings: Settings) -> LLMService:
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiLiveLLMSettings

    return GeminiLiveLLMService(
        api_key=api_key,
        settings=GeminiLiveLLMSettings(model=model, system_instruction=instructions, voice=voice),
    )


def _grok(api_key: str, model: str, voice: str, instructions: str, settings: Settings) -> LLMService:
    from pipecat.services.xai.realtime.events import (
        AudioConfiguration,
        AudioInput,
        InputAudioTranscription,
        SessionProperties,
    )
    from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService, GrokRealtimeLLMSettings

    return GrokRealtimeLLMService(
        api_key=api_key,
        settings=GrokRealtimeLLMSettings(
            model=model,
            system_instruction=instructions,
            session_properties=SessionProperties(
                voice=voice,
                # Named rather than defaulted: this provider streams caller
                # transcripts only under its own transcription model, and leaving
                # the field unset yields a call with no caller text at all.
                audio=AudioConfiguration(
                    input=AudioInput(transcription=InputAudioTranscription(model=GROK_TRANSCRIBE_MODEL)),
                ),
            ),
        ),
    )


def _qwen_realtime(credential: str, model: str, voice: str, instructions: str, settings: Settings) -> LLMService:
    """Qwen Omni Realtime, through the service in ``qwen_realtime``.

    Pipecat has no service for this provider, so the protocol is spoken
    directly. See that module for why it is not a subclass of the OpenAI one.

    A workspace id is part of the endpoint, not a credential: the same key
    reaches a different workspace's models, so it is a run configuration and it
    is disclosed with the row.
    """
    from qwen_realtime import QwenRealtimeLLMService

    return QwenRealtimeLLMService(
        api_key=credential,
        workspace_id=qwen_workspace(settings),
        model=model,
        voice=voice,
        instructions=instructions,
        region=qwen_region(settings),
    )


def _gpt_live(api_key: str, model: str, voice: str, instructions: str, settings: Settings) -> LLMService:
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
            settings=OpenAIResponsesLLMSettings(model=backend_model(settings)),
        ),
    )


def _nova_sonic(credential: str, model: str, voice: str, instructions: str, settings: Settings) -> LLMService:
    """Nova Sonic over Bedrock, signed with SigV4.

    Credentials are read from the variables AWS itself documents, so a reader
    who already has working AWS credentials in their environment runs this agent
    without re-encoding them into a shape only this file understands.

    There is deliberately no API-key path. An API key authenticates as a bearer
    token, and AWS excludes ``InvokeModelWithBidirectionalStream`` -- Nova
    Sonic's only invocation -- from bearer authentication, so such a key opens
    no session in any region however its policy is written. Accepting one would
    pass every start-up check and fail on the first call of a campaign.
    """
    from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService, AWSNovaSonicLLMSettings

    nova_settings = AWSNovaSonicLLMSettings(model=model, system_instruction=instructions, voice=voice)
    return AWSNovaSonicLLMService(
        access_key_id=credential,
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        session_token=os.getenv("AWS_SESSION_TOKEN"),
        region=aws_region(settings),
        settings=nova_settings,
    )


def qwen_region(settings: Settings) -> str:
    """Which Alibaba region the session runs in.

    A workspace lives in one region and a key authenticates against one region,
    so this picks the endpoint and the credential together. Singapore is the
    default because it is the international endpoint.
    """
    return settings.get("qwen_region", "singapore")


def qwen_workspace(settings: Settings) -> str:
    """The workspace whose endpoint answers.

    Part of the URL rather than a credential, and not disclosed with the row for
    the same reason no other row discloses its base URL: it says where the
    service was reached, not what was measured. It has no default, because a
    guessed workspace resolves to a host that does not exist and fails as a
    connection error rather than as a missing setting.
    """
    workspace = settings.get("qwen_workspace_id")
    if not workspace:
        raise ValueError(
            "qwen_workspace_id was not set by the session and QWEN_WORKSPACE_ID is unset; "
            "it names the Alibaba workspace whose endpoint serves the model"
        )
    return workspace


def backend_model(settings: Settings) -> str:
    """The text model ``gpt-live-1`` delegates to.

    Pinned rather than left to the API's own default, for the same reason every
    other version here is pinned: a backend that changes underneath a run makes
    two results incomparable without either of them looking wrong.
    """
    return settings.get("s2s_backend_model", "gpt-5.4-mini")


def aws_region(settings: Settings) -> str:
    """The region Bedrock is called in, and the region the record names.

    One accessor because those two must be the same string. Credentials are
    scoped by region and a model is served in some regions only, so a record
    naming a different region than the call used would describe a run nobody made.

    The default is a region where the model is served *and* our credentials are
    granted, verified by opening a real bidirectional stream: the two conditions
    fail identically from the outside, so a plausible-looking default that
    satisfies only one costs a debugging session per person who hits it.
    """
    return settings.get("aws_region", "us-west-2")


@dataclass(frozen=True)
class Provider:
    build: Callable[[str, str, str, str, "Settings"], LLMService]
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
    discloses: Callable[["Settings"], dict[str, str]] = lambda _settings: {}


# Whether the caller's own words reach the record is a per-provider decision,
# and it is not a detail: a scored run needs both halves of the conversation, and
# a transcript holding only the agent reads as a caller who never spoke. Three of
# these services transcribe the caller only when asked, and each asks
# differently; three do it themselves. Nothing warns about the difference,
# because a session without transcription is a working session.
#
#   openai-realtime   asked for  -- an input transcription config, default model
#   grok-realtime     asked for  -- same shape, but only under its own ASR model
#   qwen-realtime     asked for  -- NOT YET WIRED, see ``qwen_realtime``
#   gemini-live       automatic  -- the service configures both directions itself
#   gpt-live          automatic  -- the protocol is transcript-driven throughout
#   nova-sonic        automatic  -- the service emits caller transcripts natively
#
# The tool calls travel separately, through the context aggregator, which is why
# a run can show resolved tools and still carry no speech.
GROK_TRANSCRIBE_MODEL = "grok-transcribe"


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
        discloses=lambda settings: {"s2s_backend_model": backend_model(settings)},
    ),
    # Nova Sonic listens at 16 kHz and speaks at 24 kHz. The pipeline runs at the
    # input rate and the service resamples its own output.
    "nova-sonic": Provider(
        _nova_sonic, 16000, "amazon.nova-2-sonic-v1:0", "matthew", ("AWS_ACCESS_KEY_ID",),
        "pipecat.services.aws.nova_sonic.llm",
        discloses=lambda settings: {"aws_region": aws_region(settings)},
    ),
    # Qwen listens at 16 kHz and speaks at 24 kHz, and no framework service
    # exists for it -- see ``qwen_realtime``.
    "qwen-realtime": Provider(
        _qwen_realtime, 16000, "qwen3-omni-flash-realtime", "Ethan", ("DASHSCOPE_API_KEY",),
        "qwen_realtime",
        discloses=lambda settings: {"qwen_region": qwen_region(settings)},
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
    "cascade-qwen": TextModel(
        _qwen_text, "qwen-plus", ("DASHSCOPE_API_KEY",), "pipecat.services.qwen.llm",
        counterpart_to="qwen-realtime",
    ),
}


def build_cascade(text: TextModel, credential: str, model: str, instructions: str, settings: Settings):
    """Speech-to-text, a text model, text-to-speech -- the three the native model replaces."""
    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    stt = DeepgramFluxSTTService(api_key=os.environ["DEEPGRAM_API_KEY"], model=CASCADE_STT_MODEL)
    llm = text.build(credential, model)
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        voice_id=settings.get("cascade_tts_voice", CASCADE_TTS_VOICE),
        model=CASCADE_TTS_MODEL,
    )
    return stt, llm, tts


# ── the agent definition ─────────────────────────────────────────────────────

def load_agent(settings: Settings) -> MockToolServer:
    """Prompt, greeting, tool schemas and mock data, from the published contract."""
    suite = settings.get("agent_dir")
    if not suite:
        # No default. A deployment that runs the wrong agent definition produces
        # a full set of plausible, scored, wrong results, and nothing in the
        # transcript says which contract it was answering.
        raise ValueError("agent_dir was not set by the session and AGENT_DIR is unset; "
                         "it must name a directory under agent-definitions/")
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


class ToolTrace:
    """What the agent asked of its tools, recorded by us rather than the provider.

    The framework already traces tool calls, and it traces them unevenly: two of
    the providers on this board emit arguments and results on their spans, the
    other three emit no model-level spans at all, and the two that do truncate
    the fields. A board that compares tool use across providers cannot rest on
    a record whose completeness depends on which provider produced it.

    This side of the call is ours. Every tool on the board is answered by the
    same server in this same process, so recording the call here produces one
    row of the same shape for every provider, and it costs a dictionary.

    Times are offsets in milliseconds from the first tool call rather than
    timestamps: what a reader needs is how long the agent waited and in what
    order things happened, and a wall-clock time would additionally pin the run
    to a date it does not need published.
    """

    def __init__(self) -> None:
        self._calls: list[dict[str, Any]] = []
        self._origin: float | None = None

    def _offset(self) -> float:
        now = time.monotonic()
        if self._origin is None:
            self._origin = now
        return round((now - self._origin) * 1000, 1)

    def record(self, name: str, arguments: dict, matched: bool, output: Any, requested_ms: float) -> None:
        self._calls.append(
            {
                "name": name,
                "arguments": arguments,
                "matched": matched,
                "output": output,
                "requested_ms": requested_ms,
                "answered_ms": self._offset(),
            }
        )

    def as_metadata(self) -> dict[str, Any]:
        """The summary a row is scored on, plus the calls it is derived from.

        ``matched`` is the contract's own word: the arguments named a row in the
        published table. A miss is a legitimate answer rather than an error, so
        both counts are reported and neither is called a failure here -- what
        counts as failing a scenario is decided by the scenario, not by us.
        """
        return {
            "tool_calls": self._calls,
            "tool_call_count": len(self._calls),
            "tool_calls_matched": sum(1 for call in self._calls if call["matched"]),
        }


def register_tools(llm: LLMService, server: MockToolServer, trace: ToolTrace) -> None:
    """Answer every declared tool from the contract's lookup table.

    An input the table does not know returns an explicit miss rather than an
    invented record: the contract's own wording says a no-match means no record
    was found. Inventing one would let an agent that asked for the wrong thing
    score like an agent that asked for the right thing.
    """

    async def handler(params: FunctionCallParams) -> None:
        requested = trace._offset()
        arguments = params.arguments or {}
        result = server.call(params.function_name, arguments)
        matched = server.calls[-1].matched
        trace.record(params.function_name, arguments, matched, result, requested)
        logger.info("tool {} -> {}", params.function_name, "hit" if matched else "miss")
        await params.result_callback(result)

    async def end_call(params: FunctionCallParams) -> None:
        requested = trace._offset()
        logger.info("end_call -- closing the call")
        result = {"status": "ending_call"}
        # Recorded like any other call: whether the agent terminated the call
        # appropriately is scored, so the row has to say whether it tried.
        trace.record("end_call", params.arguments or {}, True, result, requested)
        await params.result_callback(result)
        await params.llm.push_frame(EndTaskFrame())

    async def transfer_call(params: FunctionCallParams) -> None:
        requested = trace._offset()
        logger.info("transfer_call -- mock handover, closing the call")
        result = {"status": "transferred"}
        trace.record("transfer_call", params.arguments or {}, True, result, requested)
        await params.result_callback(result)
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
    that happens to sit beside the image could be anything. The stamp arrives
    either as a build argument or, where the builder takes none, as an
    ``agent-commit`` file written into the build context by ``deploy.sh``.
    """
    stamped = os.getenv("AGENT_COMMIT", "").strip()
    if stamped and stamped != "unknown":
        return stamped
    written = Path(__file__).with_name("agent-commit")
    if written.exists():
        stamped = written.read_text().strip()
        if stamped:
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


def build_record(provider_key: str, provider: "Provider", model: str, voice: str, server: Any,
                 settings: Settings) -> dict[str, Any]:
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
        **_common_record(server, settings),
        "stack": "native",
        "s2s_provider": provider_key,
        "s2s_model": model,
        "s2s_voice": voice,
        # Whatever this provider says it must disclose. Absent rather than empty
        # for a provider with nothing to add: an empty value would read as a
        # field that went unrecorded.
        **provider.discloses(settings),
        "pipeline_sample_rate": provider.input_rate,
        "config": provider_key,
    }


def _common_record(server: MockToolServer, settings: Settings) -> dict[str, Any]:
    """The fields that describe the task rather than the stack.

    Identical for a native row and a cascade row by construction, which is what
    lets the two be compared: if these digests differ, the two agents were not
    given the same job and no difference between them means anything.
    """
    prompt = server.system_prompt or ""
    return {
        "bench": "agent",
        "agent_commit": _commit(),
        "agent_definition": server.suite,
        "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "first_message_sha256": hashlib.sha256((server.first_message or "").encode("utf-8")).hexdigest()[:16],
        "tools": ",".join(server.tool_names),
        "pipecat_version": _version("pipecat-ai"),
        "cekura_version": _version("cekura"),
        "cekura_mode": settings.get("cekura_mode", "track"),
        # Whether this call was configured by the session that started it or by
        # the image it started in. One deployment answers for every provider, so
        # a row that does not say which is a row nobody can place.
        "config_source": settings.source("s2s_provider"),
        "worker_instance": INSTANCE,
        "worker_call": _calls_answered,
    }


def cascade_record(key: str, text: TextModel, model: str, server: MockToolServer,
                   settings: Settings) -> dict[str, Any]:
    """What answered the call when three services answered it instead of one.

    All three are named. A cascade row that recorded only its text model would
    hide the two components that actually decide when it starts speaking and how
    fast its voice begins -- which is most of what a latency column measures.
    """
    return {
        **_common_record(server, settings),
        "stack": "cascade",
        "llm_model": model,
        "stt_model": CASCADE_STT_MODEL,
        "tts_model": CASCADE_TTS_MODEL,
        "tts_voice": settings.get("cascade_tts_voice", CASCADE_TTS_VOICE),
        "counterpart_to": text.counterpart_to,
        "pipeline_sample_rate": CASCADE_RATE,
        "config": key,
    }


# ── the bot ──────────────────────────────────────────────────────────────────

def _credential(variables: tuple[str, ...], label: str) -> str:
    """From the environment only. A credential is the one thing a session never sends."""
    found = next((v for v in map(os.getenv, variables) if v), None)
    if not found:
        raise ValueError(f"none of {', '.join(variables)} is set; {label} cannot start")
    return found


# The detector runs at one rate; the pipeline runs at the provider's. Silero
# accepts 8 or 16 kHz and refuses everything else, and two of these services open
# the pipeline at 24 kHz, so handing it the pipeline's audio unchanged does not
# degrade the measurement -- it raises on the first call and takes the whole call
# with it. Resampling in front of the detector keeps one instrument, at one rate,
# behind every row, which is the point: a detector whose behaviour varied with
# the provider's audio rate would put its own variance into the latency column.
VAD_RATE = 16000


class BenchVAD(SileroVADAnalyzer):
    """Silero at a fixed rate, fed by a resampler when the pipeline differs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline_rate = VAD_RATE
        self._resampler = SOXRStreamAudioResampler()

    def set_sample_rate(self, sample_rate: int) -> None:
        # The pipeline announces its rate here. Remember it for the conversion
        # and hold the detector at its own.
        self._pipeline_rate = sample_rate
        super().set_sample_rate(VAD_RATE)

    async def analyze_audio(self, buffer: bytes):
        if self._pipeline_rate != VAD_RATE:
            buffer = await self._resampler.resample(buffer, self._pipeline_rate, VAD_RATE)
        return await super().analyze_audio(buffer)


def user_aggregator_params(realtime: bool) -> LLMUserAggregatorParams:
    """Parameters for the half of the context that holds what the caller said.

    Two separate things are being arranged here, and both were missing.

    The first is who decides where a caller's turn ends. A realtime service
    endpoints on its own server and announces the result as a *proposal*; a
    proposal only becomes a turn if some strategy adopts it, and the default
    strategies listen for local voice activity instead. With no local voice
    activity to listen to, nothing ever adopted the proposals, so no turn ever
    ended, so the caller's transcript was aggregated and never handed over --
    which reads downstream as a caller who said nothing. Naming the external
    strategies makes the service's own endpointing the authority, which is also
    the only defensible arrangement for this bench: provider endpointing is part
    of what a row is measuring, so it must not be replaced by ours.

    The second is the speech clock. Turn frames say a turn happened; they do not
    say when the caller fell silent, and the response-time figure is measured
    from exactly that instant. Only a voice-activity detector marks it. So one
    runs here purely as an instrument: no strategy consults it, it cannot end a
    turn, and it changes nothing about the conversation -- it only timestamps
    it. Without it every latency cell on the board is empty.
    """
    return LLMUserAggregatorParams(
        # Same detector, same settings, every row: an instrument that varied by
        # provider would put its own variance into the column it is measuring.
        vad_analyzer=BenchVAD(),
        # Cascade rows leave this unset on purpose. Their speech-to-text service
        # recommends its own strategies when it announces itself, and naming
        # strategies here would override that recommendation.
        user_turn_strategies=ExternalUserTurnStrategies() if realtime else None,
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    # What is being measured is decided here, by the session that started this
    # call, and not by the image -- one deployment answers for every row.
    global _calls_answered
    _calls_answered += 1
    settings = Settings(getattr(runner_args, "body", None))
    server = load_agent(settings)
    trace = ToolTrace()
    name = settings.get("s2s_provider", "openai-realtime")
    cascade = TEXT_MODELS.get(name)

    if cascade is not None:
        model = settings.get("s2s_model", cascade.default_model)
        credential = _credential(cascade.credential_env, name)
        record = cascade_record(name, cascade, model, server, settings)
        logger.info("agent bench reference agent: {}", record)

        stt, llm, tts = build_cascade(cascade, credential, model, server.system_prompt, settings)
        register_tools(llm, server, trace)
        context = LLMContext(
            [{"role": "system", "content": server.system_prompt}, *opening_messages(server.first_message)],
            tools=build_tools(server),
        )
        aggregators = LLMContextAggregatorPair(
            context, user_params=user_aggregator_params(realtime=False)
        )
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
        model = settings.get("s2s_model", provider.default_model)
        voice = settings.get("s2s_voice", provider.default_voice)
        record = build_record(name, provider, model, voice, server, settings)
        logger.info("agent bench reference agent: {}", record)

        llm = provider.build(credential, model, voice, server.system_prompt, settings)
        register_tools(llm, server, trace)
        context = LLMContext(opening_messages(server.first_message), tools=build_tools(server))
        aggregators = LLMContextAggregatorPair(
            context,
            realtime_service_mode=True,
            user_params=user_aggregator_params(realtime=True),
        )
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
    task, tracer = create_task(pipeline, context, params, runner_args, transport, record)

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        # Every call opens from the agent, and this first context frame is also
        # what installs the tools on the realtime session.
        logger.info("caller connected")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("caller disconnected")
        # The tool trace is complete only once the call is over, and the call
        # record is immutable once posted, so it is attached here -- in the last
        # moment where both are true.
        if tracer is not None:
            try:
                tracer.set_custom_metadata({**record, **trace.as_metadata()})
            except Exception as exc:  # noqa: BLE001 -- never fail a call over a record
                logger.warning("could not attach the tool trace: {}", exc)
        logger.info("tools called: {}", trace.as_metadata()["tool_call_count"])
        await task.cancel()

    await PipelineRunner(handle_sigint=False).run(task)


def create_task(pipeline, context, params, runner_args, transport, record) -> tuple[PipelineTask, Any]:
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
        return PipelineTask(pipeline, params=params), None

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
        if record.get("cekura_mode") == "observe":
            return tracer.observe_and_create_task(
                pipeline, context, runner_args=runner_args, transport=transport,
                custom_metadata=metadata, params=params,
            ), tracer
        return tracer.track_and_create_task(
            pipeline, context, runner_args=runner_args, transport=transport,
            custom_metadata=metadata, params=params,
        ), tracer
    except Exception as exc:  # noqa: BLE001 -- observability must never fail a call
        logger.warning("Cekura tracing disabled: {}", exc)
        return PipelineTask(pipeline, params=params), None


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

    Nothing here touches the network. Opening a throwaway connection per provider
    does not speed up the real first one -- a TLS session does not resume across a
    separate context and no DNS result is cached in-process -- while every one of
    those handshakes delays the moment this worker can answer a call at all.
    """
    import importlib

    _commit()
    _version("pipecat-ai")
    _version("cekura")
    # Every SDK, not the one this process will use: the provider is chosen by
    # the session, so by the time the process knows which one it needs, the
    # caller is already on the line. The cost is a slower container start and a
    # larger resident process, both paid before any call, in exchange for
    # keeping an import out of the window a first response is timed in.
    #
    # A cascade loads three services, and the two it does not name are the ones
    # that decide when it starts speaking and how fast its voice begins.
    modules = sorted(
        {entry.module for entry in PROVIDERS.values()}
        | {entry.module for entry in TEXT_MODELS.values()}
        | {"pipecat.services.deepgram.flux.stt", "pipecat.services.elevenlabs.tts"}
    )
    # ``cekura.pipecat`` is imported by ``create_task``, which runs per call --
    # so left alone it is a third of a second of import inside the first call on
    # every worker, and worse on a container whose page cache is cold. It is
    # warmed unconditionally rather than only when tracing is configured: a
    # warm-up that depends on a credential is a warm-up that silently stops
    # working the day a credential is missing.
    #
    for module in [*modules, "cekura.pipecat"]:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 -- the builder will raise this again, in context
            logger.warning("could not pre-import {} for {}: {}", module, name, exc)

    # The speech detector loads a model into an inference session the first time
    # one is constructed, and one is constructed per call. Building a throwaway
    # here moves that load out of the first call and leaves the loaded model in
    # the process for the ones after it.
    try:
        BenchVAD()
    except Exception as exc:  # noqa: BLE001 -- run_bot builds the real one and will raise in context
        logger.warning("could not warm the speech detector: {}", exc)


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point used by the Pipecat runner and by Pipecat Cloud."""
    telephony = lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)  # noqa: E731
    transport = await create_transport(
        runner_args,
        {
            "twilio": telephony,
            "telnyx": telephony,
            # Daily takes its own parameter class, not the generic one. The generic
            # one constructs and connects, and then the transport reads a
            # Daily-only field off it mid-call and stops working -- so the fault
            # surfaces as a bot that joined nothing rather than as a bad argument.
            "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
            "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        },
    )
    await run_bot(transport, runner_args)


warm()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
