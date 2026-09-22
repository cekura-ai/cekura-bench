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

import contextvars
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    EndTaskFrame,
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    STTUsageMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
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
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies

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


# ── the call's own log ───────────────────────────────────────────────────────
#
# A run is debugged from its log, so every line carries [MM:SS] since the call
# was answered, and every line from the call's first moment is kept for the run
# record at whatever level it was logged -- including the framework's own DEBUG
# account of what it did with each turn. The stamp matches the one the calling
# side uses, so both halves of a call read on a single clock.

_CALL: contextvars.ContextVar[str | None] = contextvars.ContextVar("bench_call", default=None)
_CLOCK: contextvars.ContextVar[float | None] = contextvars.ContextVar("bench_clock", default=None)
# A worker answers one call at a time in the deployment, but a local runner may
# answer several in one process. The context variables keep them apart where a
# context is carried; the latest values stand in where one is not, which is the
# transport's own threads calling back into us.
_latest_call: str | None = None
_latest_clock: float | None = None


def start_call_clock(call_id: str) -> None:
    """Zero the clock every log line is stamped against, for the call just answered."""
    global _latest_call, _latest_clock
    _latest_call, _latest_clock = call_id, time.monotonic()
    _CALL.set(call_id)
    _CLOCK.set(_latest_clock)


def call_stamp(now: float | None = None) -> str:
    """``[MM:SS]`` since the call started, or ``[--:--]`` before any call has."""
    started = _CLOCK.get() or _latest_clock
    if started is None:
        return "[--:--]"
    elapsed = int((now if now is not None else time.monotonic()) - started)
    return f"[{elapsed // 60:02d}:{elapsed % 60:02d}]"


def _stamp(record: dict) -> None:
    """Put the call clock on every line, wherever it is going."""
    record["message"] = f"{call_stamp()} {record['message']}"
    record["extra"]["bench_call"] = _CALL.get() or _latest_call


logger.configure(patcher=_stamp)


class _ToLoguru(logging.Handler):
    """Carry the standard library's log records into the run record.

    The mock tool server says why each call resolved as it did, and it does so
    through the standard library so that it carries no logging dependency of its
    own. Those lines belong in the record beside everything else, so they are
    re-emitted here: they then pick up the call clock and the call id like any
    other line, and ship with the payload.
    """

    def emit(self, record: logging.LogRecord) -> None:
        level = logger.level(record.levelname).name if record.levelname in _LEVELS else "INFO"
        logger.bind(std_logger=record.name).opt(depth=6, exception=record.exc_info).log(
            level, record.getMessage()
        )


_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
logging.getLogger("mock_tools").handlers = [_ToLoguru()]
logging.getLogger("mock_tools").setLevel(logging.DEBUG)
logging.getLogger("mock_tools").propagate = False


class CallLog:
    """Every line logged while a call is up, for the run's record.

    The tracing SDK ships the log with the run, which is where a reader looks
    for it months later; it just keeps too little of it. This collects from the
    call's first line rather than from task creation, and at DEBUG rather than
    INFO, and then hands the collection to the SDK so the record carries it
    without a second exporter. The level is a knob because a campaign that has
    stopped being debugged can turn the volume down without a code change.

    Bounded, because a payload is not a log file: a line is cut at a fixed
    width and the collection stops at a fixed count, and the record says so
    when either happens, so a reader knows to go to the container.
    """

    LEVEL = os.getenv("BENCH_LOG_LEVEL", "DEBUG").upper()
    MAX_LINES = int(os.getenv("BENCH_LOG_MAX_LINES", "6000"))
    MAX_CHARS = 1000

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self.lines: list[dict[str, Any]] = []
        self.dropped = 0
        # ``format`` is minimal on purpose: the sink reads the record and
        # discards the rendered string, so rendering the full template would
        # be paid on every line of the call for nothing.
        self._sink_id: int | None = logger.add(self._sink, level=self.LEVEL, format="{message}")

    def _sink(self, message) -> None:
        record = message.record
        owner = record["extra"].get("bench_call")
        if owner is not None and owner != self.call_id:
            return
        if len(self.lines) >= self.MAX_LINES:
            if self.dropped == 0:
                self.lines.append({
                    "timestamp": record["time"].timestamp(),
                    "level": "WARNING",
                    "logger": __name__,
                    "message": f"{call_stamp()} log capture reached {self.MAX_LINES} lines; "
                               "the rest of this call is in the container log only",
                })
            self.dropped += 1
            return
        text = record["message"]
        if len(text) > self.MAX_CHARS:
            text = text[: self.MAX_CHARS] + f"… [{len(text) - self.MAX_CHARS} more chars]"
        self.lines.append({
            "timestamp": record["time"].timestamp(),
            "level": record["level"].name,
            "logger": record["name"],
            "message": text,
        })

    def hand_to(self, tracer: Any) -> None:
        """Make this the log the SDK ships, in place of its own narrower one.

        The SDK opened a sink of its own when the task was created; it is
        closed here, and the SDK is pointed at this collection and this sink
        instead, so that its own finalisation removes the sink and ships the
        lines exactly as it would have shipped its own.
        """
        own = getattr(tracer, "_log_sink_id", None)
        if own is not None:
            try:
                logger.remove(own)
            except ValueError:
                pass
        if hasattr(tracer, "_session_logs"):
            tracer._session_logs = self.lines
            tracer._log_sink_id = self._sink_id

    def close(self) -> None:
        """Always remove the sink, even after handing it over.

        The sink lives on the process-wide logger, so one left installed keeps
        this call's lines alive and goes on capturing the next call's. The SDK
        normally removes it during finalisation; removing it twice is free, and
        not removing it at all is not.
        """
        if self._sink_id is not None:
            try:
                logger.remove(self._sink_id)
            except ValueError:
                pass
            self._sink_id = None


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
    from pipecat.services.google.gemini_live.llm import (
        GeminiLiveLLMService,
        GeminiLiveLLMSettings,
        GeminiVADParams,
    )

    # Turned off deliberately, and the reply times on this row depend on it.
    #
    # Left on, the service wants an uninterrupted stream to run its own detector
    # over, so the whole call is sent -- every silence between turns included.
    # This session falls behind that stream: it answers the first turn in about
    # three seconds and each later turn several seconds further back, measured
    # from the caller's speech ending, while the audio leaves here at exactly
    # real time. The lag is cumulative and only an interruption clears it, so a
    # late-call turn can be answered half a minute after it was spoken -- long
    # enough that a scripted caller has moved on and the turn scores as silence.
    #
    # Turned off, the caller's turn is announced by the shared detector every
    # row is measured against, and audio is sent only inside that turn -- with a
    # short pre-roll the service keeps, so the onset is not clipped. What the
    # session has to hold then is the speech alone, and the reply time stops
    # growing: measured over eight turns, flat at one to three seconds.
    #
    # It is also what this row claims. ``turns="local"`` in the table below says
    # the boundary belongs to the shared detector, which is only true with the
    # service's own detector out of the way.
    return GeminiLiveLLMService(
        api_key=api_key,
        settings=GeminiLiveLLMSettings(
            model=model,
            system_instruction=instructions,
            voice=voice,
            vad=GeminiVADParams(disabled=True),
        ),
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
    # Where the caller's turn boundary comes from. Most of these services decide
    # it on their own server and announce it, and that announcement is what the
    # pipeline should follow -- provider endpointing is part of what a row
    # measures. Three of them announce nothing at all: their API exposes an
    # interruption event and no turn start or end, so a pipeline that keeps a
    # conversation context has to find the boundary itself. Those rows run the
    # framework's own recommended arrangement, a local detector deciding turns,
    # and the record says so, because a row whose turns were decided locally is
    # not measuring the same thing as one whose were not.
    turns: str = "provider"
    # Whether this service runs its own voice-activity detection on the audio it
    # is sent. Leaving it on means the whole call has to be streamed for it to
    # listen to, silences included; turning it off means the shared detector
    # announces the turn and only the speech inside it is sent. That is a real
    # difference in what the service is asked to do, so it is declared here and
    # published, not left implicit in a builder.
    service_vad: bool = True
    # Whether the caller's own words reach the record because this pipeline asked
    # for them, or because the service sends them unprompted. It changes nothing
    # about the conversation and everything about whether a row has a caller in
    # it, which makes it worth a published cell rather than a comment.
    caller_transcription: str = "asked"
    # Whether a caller talking over the agent is this pipeline's business.
    #
    # Usually it is: the agent is stopped here and the service is told the reply
    # was cut short. One of these services handles it inside the model instead,
    # announces that it broadcasts no interruption of its own, and asks the
    # pipeline not to broadcast one either -- so a client-side interruption
    # there cuts work the model was going to carry on with, and the row would be
    # reporting this pipeline's barge-in rather than the service's. A service
    # that asks for this is taken at its word.
    interruptions: bool = True


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
        turns="local", service_vad=False, caller_transcription="automatic",
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
        interruptions=False, caller_transcription="automatic",
    ),
    # Nova Sonic listens at 16 kHz and speaks at 24 kHz. The pipeline runs at the
    # input rate and the service resamples its own output.
    "nova-sonic": Provider(
        _nova_sonic, 16000, "amazon.nova-2-sonic-v1:0", "matthew", ("AWS_ACCESS_KEY_ID",),
        "pipecat.services.aws.nova_sonic.llm",
        discloses=lambda settings: {"aws_region": aws_region(settings)},
        turns="local", caller_transcription="automatic",
    ),
    # Qwen listens at 16 kHz and speaks at 24 kHz, and no framework service
    # exists for it -- see ``qwen_realtime``.
    "qwen-realtime": Provider(
        _qwen_realtime, 16000, "qwen3-omni-flash-realtime", "Ethan", ("DASHSCOPE_API_KEY",),
        "qwen_realtime",
        discloses=lambda settings: {"qwen_region": qwen_region(settings)},
        turns="local",
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


# A control token is a tokenizer artifact, not speech. One of these services
# streams them into the transcript of what it said -- a run of ``<ctrl46>`` and
# the blank lines around them -- and they reach the record as though the agent
# had uttered them. Left alone they are scored: a judge reads them as the agent
# saying something incoherent, and any word-level comparison counts them as
# words. Removing them takes nothing real away, because there is no audio behind
# them; the model never said them.
#
# Narrow on purpose. Only the documented control-token shape goes, and nothing
# else about the text is touched -- a filter on a benchmark's transcript is a
# filter on its evidence, and the moment it starts tidying prose it is editing
# what is being measured.
CONTROL_TOKEN = re.compile(r"<ctrl\d+>")


def _short(value: Any, width: int = 300) -> str:
    """One line of a value, for a log: enough to recognise it, not to reproduce it."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        text = repr(value)
    return text if len(text) <= width else text[:width] + "…"


class DropControlTokens(FrameProcessor):
    """Keep tokenizer artifacts out of the transcript of what the agent said.

    Every removal is logged. An empty agent turn has two very different causes
    -- the model produced nothing, or it produced only control tokens and this
    processor emptied it -- and from the transcript alone they are identical.
    One is a finding about the model and the other would look like our filter
    damaging the evidence, so the log has to say which happened.
    """

    def __init__(self) -> None:
        super().__init__()
        self.removed = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (TTSTextFrame, LLMTextFrame)) and "<ctrl" in frame.text:
            cleaned, found = CONTROL_TOKEN.subn("", frame.text)
            # Whitespace that only ever separated the tokens is not speech either.
            emptied = not cleaned.strip()
            self.removed += found
            # Text arrives in chunks, so this can fire many times in one turn.
            # Only an emptied turn changes how the transcript should be read, so
            # that is the case worth an INFO line.
            report = logger.info if emptied else logger.debug
            report(
                "control tokens removed from what the agent said ({} so far){}: {}",
                self.removed,
                "; the turn is now empty, so the model said nothing else" if emptied else "",
                _short(frame.text, 200),
            )
            frame.text = "" if emptied else cleaned
        await self.push_frame(frame, direction)


class RestatementIsNotSpeech(FrameProcessor):
    """A final transcript that restates the turn so far is a restatement, not new speech.

    Services disagree on what a *final* caller transcript is. Most send one per
    turn; some send several, each restating the whole turn to date. The
    aggregator appends every final, which is right for the first kind and
    records the caller's words two or three times over for the second.

    Only the part of a final that is new is passed on, so the aggregator's own
    appending reconstructs exactly the text the service last reported. A service
    that sends one final per turn is unaffected, which is the test of a
    correction like this: it must be a statement about transcripts in general
    rather than an adjustment aimed at one service.
    """

    # How much of the shorter version two finals must agree on before the later
    # one is read as a rewrite of the same words rather than a new sentence.
    # It has to tolerate a corrected word -- "book an" becomes "book a new" --
    # while refusing two genuinely different utterances.
    SAME_TURN = 0.6

    def __init__(self) -> None:
        super().__init__()
        self._said: list[str] = []
        self.restatements = 0

    @staticmethod
    def _compare(words: list[str]) -> list[str]:
        """The form two versions of the same words are compared in.

        Case and punctuation are not evidence of new speech: the same service
        rewrites "July 8th." as "July 8th if possible." on the next pass, and a
        byte comparison would call that a fresh sentence and append the lot.
        """
        return ["".join(ch for ch in word if ch.isalnum()).lower() for word in words]

    @staticmethod
    def _agree_on(said: list[str], new: list[str]) -> int:
        """How many words from the start the two versions agree on."""
        agreed = 0
        for left, right in zip(said, new):
            if left != right:
                break
            agreed += 1
        return agreed

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            # A new turn: nothing has been said in it yet. Every service on this
            # board announces this, which is what makes the reset reliable.
            self._said = []
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            words = frame.text.split()
            said, new = self._compare(self._said), self._compare(words)
            agreed = self._agree_on(said, new)
            shorter = min(len(said), len(new))
            if said and agreed >= 2 and (agreed == shorter or agreed >= self.SAME_TURN * shorter):
                # The same words again, carried on or corrected. Only what comes
                # after the part they agree on is new.
                self.restatements += 1
                rest = " ".join(words[agreed:])
                if len(new) >= len(said):
                    self._said = words
                if not rest:
                    logger.debug("the transcript restated the turn and added nothing; not passed on")
                    return
                logger.debug("the transcript restated the turn; passing on only {}", _short(rest, 120))
                frame.text = rest
            else:
                self._said = self._said + words
        await self.push_frame(frame, direction)


class UsageMeter(BaseObserver):
    """Counts what a call consumed, so the row it produces can carry a price.

    Cost is one of the few columns a benchmark is actually read for, and it is
    the one that cannot be reconstructed after the fact: the framework reports
    consumption per call and then the numbers are gone. They do reach the trace
    store, but that empties after thirty days, and a board is questioned months
    later -- so a figure that lives only there is a figure we cannot defend on
    the day someone asks. This records it where the run record keeps it.

    Consumption, not money. Prices change, differ by account and are a judgement
    about a vendor page on a date; a token count is a measurement. Keeping them
    apart means a published price can be corrected, and argued with, without
    re-running a single call.

    Three shapes, because the rows are billed three ways. A native speech model
    bills tokens, and splits them into audio and text, and again into fresh and
    cached -- the splits are not a detail, since audio costs multiples of text
    and a cached prompt a fraction of a fresh one, so a single total cannot be
    priced at all. A cascade bills its transcriber by audio seconds and its
    voice by characters. Every field is summed across the call and left alone
    otherwise; a provider that reports nothing simply contributes nothing, which
    is a fact about that row worth seeing rather than a gap to paper over.
    """

    # The token fields worth keeping apart, in the framework's own names.
    TOKEN_FIELDS = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
        "input_audio_tokens",
        "output_audio_tokens",
        "cache_read_input_audio_tokens",
    )

    def __init__(self) -> None:
        super().__init__()
        # Some rows are billed by the minute rather than by the token, so the
        # length of the call is part of what it consumed. Measured from here --
        # the pipeline is built immediately before the bot joins -- to the
        # moment the record is taken.
        self._opened = time.monotonic()
        self._tokens: dict[str, int] = {}
        self._stt_audio_seconds = 0.0
        self._tts_characters = 0
        # One provider bills its speech model by the second and reports it
        # nowhere a metrics frame can reach. See ``record_live_audio``.
        self._live_audio_seconds = 0.0
        self._reports = 0
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, MetricsFrame) or frame.id in self._seen:
            # A metrics frame is broadcast, so the same one arrives more than
            # once; counting it twice would double the bill.
            return
        self._seen.add(frame.id)
        for entry in frame.data:
            if isinstance(entry, LLMUsageMetricsData):
                self._reports += 1
                for field in self.TOKEN_FIELDS:
                    value = getattr(entry.value, field, None)
                    if value:
                        self._tokens[field] = self._tokens.get(field, 0) + value
            elif isinstance(entry, STTUsageMetricsData):
                self._stt_audio_seconds += entry.value.audio_seconds
            elif isinstance(entry, TTSUsageMetricsData):
                self._tts_characters += entry.value

    def record_live_audio(self, seconds: float) -> None:
        """Seconds of live audio, reported cumulatively, so the largest wins."""
        self._live_audio_seconds = max(self._live_audio_seconds, seconds)

    def as_metadata(self) -> dict[str, Any]:
        """What the call consumed, priced by nothing.

        ``usage_reports`` is here because zero is ambiguous otherwise: a row
        with no tokens may be a provider that does not report them or a call
        that never reached the model, and those are different findings.
        """
        usage: dict[str, Any] = {
            "call_seconds": round(time.monotonic() - self._opened, 3),
            "usage_reports": self._reports,
            **self._tokens,
        }
        if self._stt_audio_seconds:
            usage["stt_audio_seconds"] = round(self._stt_audio_seconds, 3)
        if self._tts_characters:
            usage["tts_characters"] = self._tts_characters
        if self._live_audio_seconds:
            usage["live_audio_seconds"] = round(self._live_audio_seconds, 3)
        return {"usage": usage}


class CallNarrator(BaseObserver):
    """Writes the call's story into its log, one line per thing that happened.

    The framework logs what each processor did to each frame, which is the
    right record for debugging the framework and the wrong one for reading a
    call: the question a result raises is "when did the caller stop, when did
    the agent start, what did it call, what came back", and the answer is
    scattered across a thousand DEBUG lines from a dozen modules. These lines
    put it in one place, at INFO, on the call clock.

    Speech boundaries are logged twice on purpose, once when the detector hears
    them and once when the pipeline adopts them as a turn: the gap between the
    two is the endpointing delay, and the rows on this board differ in where
    that decision is made.

    Two rules keep these counts meaning the same thing on every row.

    A broadcast frame is *two* frames: the framework constructs one for each
    direction and links them by ``broadcast_sibling_id``, so counting by frame
    identity alone counts every turn twice. Which frames are broadcast differs
    by service, so the error would not even be a constant factor across rows.

    And an interruption frame does not mean the caller interrupted. In realtime
    mode the aggregator broadcasts one at the start of every caller turn,
    whether or not the agent was speaking. A barge-in is an interruption that
    arrives while the agent is actually speaking; the rest are ordinary turn
    starts and stay at DEBUG.
    """

    # Enough to catch a broadcast pair, which arrives back to back, without
    # holding every frame of a ten-minute call.
    MEMORY = 512

    def __init__(self, clock: "AudioClock | None" = None) -> None:
        super().__init__()
        # Handed in rather than reached for, so this narrator reports the call it
        # was built for and nothing a previous one left behind.
        self._clock = clock if clock is not None else AudioClock()
        self._seen: dict[int, None] = {}
        self.caller_turns = 0
        self.agent_turns = 0
        self.barge_ins = 0
        self.errors = 0
        self._agent_speaking = False
        self.replies: list[float] = []
        self.endpointing: list[float] = []
        self._heard_stop: float | None = None
        self._turn_closed: float | None = None

    def timing(self) -> dict[str, Any]:
        """How long the agent took to answer, and how much of that was endpointing.

        ``reply`` runs from the moment our own detector hears the caller stop to
        the first audible agent audio. The detector is the same one on every row,
        which is what makes the figure comparable: a service that decides a turn
        is over on its own schedule is measured from the same instant as one whose
        turns this pipeline decides.

        ``endpointing`` is the part of that interval spent waiting for the turn to
        be declared over -- detector stop to turn close. A service can be quick to
        generate and slow to commit, and those are different findings, so it is
        reported beside the total rather than folded into it.

        Read it against ``turn_source``, which names whose endpointing this is:
        the realtime service where it announces its own turns, this pipeline's
        detector where it does not, or the speech-to-text service on a cascade
        row. Rows from those three groups are not comparable on this column, and
        the column exists so the difference is visible rather than buried in the
        reply figure.

        Both are detector-derived and belong to this bench only. A figure anchored
        on authored audio, where the caller's last sample is known exactly rather
        than detected, is a different and better instrument.
        """
        def spread(values: list[float]) -> dict[str, float] | None:
            if not values:
                return None
            ordered = sorted(values)
            def at(q: float) -> float:
                return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
            return {
                "count": len(ordered),
                "p50_ms": round(at(0.5) * 1000),
                "p90_ms": round(at(0.9) * 1000),
                "max_ms": round(ordered[-1] * 1000),
            }

        timing: dict[str, Any] = {}
        reply, endpointing = spread(self.replies), spread(self.endpointing)
        if reply:
            timing["reply"] = reply
        if endpointing:
            timing["endpointing"] = endpointing
        return timing

    # What a healthy call looks like, as numbers rather than as judgement.
    #
    # These are deliberately loose. They are not a quality bar -- a slow model is
    # not a broken row -- they are the shape of a pipeline that has stopped
    # keeping up with the call, which is a different thing and one that no
    # transcript reveals. A row that trips one of these is not scored badly; it
    # is not scored at all until someone has looked.
    DRIFTING_REPLIES_MS = 2000
    STARVED_AUDIO_MS = 1000
    ANSWERED_SHARE = 0.6

    def integrity(self) -> dict[str, Any]:
        """Whether this call was delivered and answered as a call, not just scored.

        Every row is checked, every time, whatever the result looked like. That
        order matters more than the thresholds: a fault is only ever found in the
        row someone thought to examine, and the row nobody examines is the one
        that looks fine. Checking on the way past removes the choice.

        ``reply_drift_ms`` is the second half of a call's replies against the
        first half. A model is slow evenly; a session falling behind the audio it
        is being sent gets slower as the call goes on, and the gap between halves
        is what separates the two. ``audio_in_drift_ms`` is the caller audio the
        pipeline received against the time it ran, which should be zero on a live
        call. ``answered`` is how many caller turns drew a reply: a call the agent
        is too far behind to answer still produces turns, and they score as
        silence.
        """
        report: dict[str, Any] = {}
        failed: list[str] = []

        if len(self.replies) >= 4:
            half = len(self.replies) // 2
            def middle(values: list[float]) -> float:
                return sorted(values)[len(values) // 2]
            drift = middle(self.replies[half:]) - middle(self.replies[:half])
            report["reply_drift_ms"] = round(drift * 1000)
            if drift * 1000 > self.DRIFTING_REPLIES_MS:
                failed.append("replies_drifting")

        drift = self._clock.drift()
        if drift is not None:
            report["audio_in_drift_ms"] = round(drift * 1000)
            if abs(drift) * 1000 > self.STARVED_AUDIO_MS:
                failed.append("audio_in_starved")

        if self.caller_turns:
            report["answered"] = f"{self.agent_turns}/{self.caller_turns}"
            if self.agent_turns < self.caller_turns * self.ANSWERED_SHARE:
                failed.append("turns_unanswered")
        elif self.agent_turns == 0:
            # Neither side said anything. The row exists and holds no call.
            failed.append("silent_call")

        # Named rather than counted, because the name is the whole finding: a
        # reader who sees one of these needs to know which invariant broke, and a
        # reader who sees none needs to know the checks ran.
        report["checks"] = failed or ["ok"]
        return report

    def _first_sighting(self, frame: Frame) -> bool:
        """False for a frame already narrated, or for the sibling of one."""
        if frame.id in self._seen:
            return False
        self._seen[frame.id] = None
        sibling = getattr(frame, "broadcast_sibling_id", None)
        if sibling is not None:
            self._seen[sibling] = None
        while len(self._seen) > self.MEMORY:
            self._seen.pop(next(iter(self._seen)))
        return True

    # Everything this narrates. Audio frames outnumber these by orders of
    # magnitude, so they are turned away before anything else happens -- which
    # also keeps the dedupe window holding only frames that were narrated,
    # rather than being flushed by audio within the second.
    NARRATED = (
        VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
        UserStartedSpeakingFrame, UserStoppedSpeakingFrame, TranscriptionFrame,
        LLMFullResponseStartFrame, LLMFullResponseEndFrame,
        BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterruptionFrame,
        FunctionCallInProgressFrame, FunctionCallResultFrame,
        ErrorFrame, EndFrame, CancelFrame, MetricsFrame,
    )

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, self.NARRATED) or not self._first_sighting(frame):
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            logger.debug("detector: caller speech starts")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # The anchor. A later stop replaces an earlier one, so a caller who
            # pauses mid-turn is measured from when they actually finished.
            self._heard_stop = time.monotonic()
            self._turn_closed = None
            logger.debug("detector: caller speech stops")
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.caller_turns += 1
            logger.info("caller turn {} starts", self.caller_turns)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._heard_stop is not None:
                self._turn_closed = time.monotonic()
                waited = self._turn_closed - self._heard_stop
                self.endpointing.append(waited)
                logger.info("caller turn {} ends ({:.0f} ms after the detector heard it stop)",
                            self.caller_turns, waited * 1000)
            else:
                logger.info("caller turn {} ends", self.caller_turns)
        elif isinstance(frame, TranscriptionFrame):
            logger.info("caller transcript: {}", _short(frame.text))
        elif isinstance(frame, LLMFullResponseStartFrame):
            self.agent_turns += 1
            logger.info("agent response {} starts", self.agent_turns)
        elif isinstance(frame, LLMFullResponseEndFrame):
            logger.info("agent response {} ends", self.agent_turns)
        elif isinstance(frame, BotStartedSpeakingFrame):
            # Only the first audio of a reply is a reply: once the agent is
            # speaking, later starts belong to the same turn. A greeting has no
            # caller stop before it and is not timed.
            if not self._agent_speaking and self._heard_stop is not None:
                answered = time.monotonic() - self._heard_stop
                self.replies.append(answered)
                self._heard_stop = None
                logger.info("agent audio starts ({:.0f} ms after the caller stopped)", answered * 1000)
            else:
                logger.info("agent audio starts")
            self._agent_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._agent_speaking = False
            logger.info("agent audio stops")
        elif isinstance(frame, InterruptionFrame):
            if self._agent_speaking:
                self.barge_ins += 1
                logger.info("barge-in {}: the caller spoke over the agent", self.barge_ins)
            else:
                logger.debug("interruption while the agent was silent (an ordinary turn start)")
        # Only the timing: these two bracket how long the tool took on the call
        # clock. What was asked and what came back is on the handler's own line,
        # which also knows the resolution, so it is not repeated here.
        elif isinstance(frame, FunctionCallInProgressFrame):
            logger.info("tool {} requested", frame.function_name)
        elif isinstance(frame, FunctionCallResultFrame):
            logger.info("tool {} answered", frame.function_name)
        elif isinstance(frame, ErrorFrame):
            self.errors += 1
            logger.warning("error frame{}: {}", " (fatal)" if frame.fatal else "", frame.error)
        elif isinstance(frame, EndFrame):
            logger.info("pipeline ending: the agent closed the call")
        elif isinstance(frame, CancelFrame):
            logger.info("pipeline cancelled: the call was torn down")
        elif isinstance(frame, MetricsFrame):
            for entry in frame.data:
                if isinstance(entry, TTFBMetricsData):
                    logger.debug("ttfb {} {:.0f} ms", entry.processor, entry.value * 1000)
                elif isinstance(entry, ProcessingMetricsData):
                    logger.debug("processing {} {:.0f} ms", entry.processor, entry.value * 1000)


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
        # Called after every recorded call. The run's record is assembled from
        # several pieces and there is no reliable last moment to assemble it in,
        # so it is rewritten whenever a piece changes. See ``run_bot``.
        self.on_change: Any = None

    def _offset(self) -> float:
        now = time.monotonic()
        if self._origin is None:
            self._origin = now
        return round((now - self._origin) * 1000, 1)

    def record(self, name: str, arguments: dict, matched: bool, output: Any, requested_ms: float,
               resolution: str = "none") -> None:
        self._calls.append(
            {
                "name": name,
                "arguments": arguments,
                "matched": matched,
                # Three outcomes, not two. A record found after speech bent an
                # argument is the interesting middle case, and a bare matched
                # flag hides it from anything reading the payload.
                "resolution": resolution,
                "output": output,
                "requested_ms": requested_ms,
                "answered_ms": self._offset(),
            }
        )
        if self.on_change is not None:
            self.on_change()

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
        record = server.calls[-1]
        trace.record(params.function_name, arguments, record.matched, result, requested, record.resolution)
        # Three outcomes, not two: a row found outright, the nearest row found
        # after speech bent an argument, and nothing found. Reading a call log
        # afterwards, the middle one is the interesting case and a bare
        # hit-or-miss hides it. The arguments and the answer are on the same
        # line, because a miss is only explicable next to what was asked.
        logger.info(
            "tool {} -> {} for {} => {}",
            params.function_name, record.resolution, _short(arguments), _short(result),
        )
        await params.result_callback(result)

    async def end_call(params: FunctionCallParams) -> None:
        requested = trace._offset()
        logger.info("end_call -- closing the call")
        result = {"status": "ending_call"}
        # Recorded like any other call: whether the agent terminated the call
        # appropriately is scored, so the row has to say whether it tried.
        trace.record("end_call", params.arguments or {}, True, result, requested, "exact")
        await params.result_callback(result)
        await params.llm.push_frame(EndTaskFrame())

    async def transfer_call(params: FunctionCallParams) -> None:
        requested = trace._offset()
        logger.info("transfer_call -- mock handover, closing the call")
        result = {"status": "transferred"}
        trace.record("transfer_call", params.arguments or {}, True, result, requested, "exact")
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
        # Who decided where the caller's turns ended. Three of these services
        # announce no turn boundary at all, so those rows run a local detector
        # instead -- a real configuration difference, and one a reader comparing
        # two rows has to be able to see.
        "turn_source": provider.turns,
        # Every place this row is not arranged like the others, in one block.
        #
        # One pipeline answers every row, but the session opened at the top of it
        # is not identical, because these services do not offer the same
        # contract. Those differences are the part of a result that is ours
        # rather than the model's, and a reader comparing two rows cannot weigh
        # them without seeing them. Keeping them here, beside the score, is what
        # makes the unit being compared "this service under this arrangement"
        # rather than an unqualified provider name.
        "divergences": {
            # Whose detector decided the caller had finished, and so which
            # service's endpointing the reply figure includes.
            "endpointing": provider.turns,
            # Whether the service ran its own detector over the audio as well.
            "service_vad": "on" if provider.service_vad else "off",
            # Who stops the agent when the caller talks over it.
            "interruptions": "pipeline" if provider.interruptions else "model",
            # Whether the caller's words had to be asked for.
            "caller_transcription": provider.caller_transcription,
            # The rate the session was opened at. These services do not resample,
            # so this is a requirement rather than a setting, but a wrong one
            # reads exactly like a bad model and belongs on the record.
            "input_rate": provider.input_rate,
        },
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
        # Who decided where the caller's turns ended -- a third answer, and the
        # reason this field is not just native/local. These rows follow the
        # strategies the speech-to-text service recommends, so the endpointing
        # figure beside their reply time belongs to that service, named above,
        # and not to the text model the row is otherwise about.
        "turn_source": "stt",
        # The same block a native row carries, answered for a cascade. These
        # rows diverge in one direction only: the speech path is fixed and
        # identical across all of them, so the row is the text model and nothing
        # else. Saying that explicitly is what lets a cascade row sit on a board
        # beside a native one without the two being read as the same measurement.
        "divergences": {
            "endpointing": "stt",
            "service_vad": "off",
            "interruptions": "pipeline",
            "caller_transcription": "asked",
            "input_rate": CASCADE_RATE,
        },
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


class AudioClock:
    """How much caller audio the pipeline received, against how long it ran.

    A live call delivers audio at one second per second. When these two numbers
    part company the pipeline is no longer being handed the call as it happens,
    and every figure measured from the caller's speech is measured against a
    clock that has slipped. That is a fault no transcript shows and no score
    explains, so it is counted on every row rather than looked for after a row
    disappoints.

    One call at a time runs in a worker, so one accumulator is enough; it is
    reset when a call starts.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._began: float | None = None
        self._audio = 0.0

    def add(self, seconds: float) -> None:
        if self._began is None:
            self._began = time.monotonic()
        self._audio += seconds

    def drift(self) -> float | None:
        """Audio received minus wall time elapsed, in seconds. None before audio."""
        if self._began is None:
            return None
        return self._audio - (time.monotonic() - self._began)


AUDIO_CLOCK = AudioClock()


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
        # Counted here because this is the one place every row's caller audio
        # passes exactly once, before any provider has touched it.
        AUDIO_CLOCK.add(len(buffer) / 2 / self._pipeline_rate)
        if self._pipeline_rate != VAD_RATE:
            buffer = await self._resampler.resample(buffer, self._pipeline_rate, VAD_RATE)
        return await super().analyze_audio(buffer)


def user_aggregator_params(
    realtime: bool, turns: str = "provider", interruptions: bool = True
) -> LLMUserAggregatorParams:
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
        # Three cases, not two.
        #
        # A realtime service that announces its turn boundary: follow it, so the
        # provider's own endpointing is the authority, which is what the row is
        # measuring.
        #
        # A realtime service that announces nothing -- its API has an
        # interruption event and no turn start or end -- cannot be followed. The
        # framework's own guidance for a pipeline like this one, which keeps a
        # conversation context, is a local detector driving the default
        # strategies, so that is what those rows run. It is a real difference
        # between rows and the record names it rather than hiding it.
        #
        # Cascade rows leave this unset on purpose. Their speech-to-text service
        # recommends its own strategies when it announces itself, and naming
        # strategies here would override that recommendation.
        # Naming strategies here discards whatever the service recommends, so
        # what it recommends has to be carried rather than lost: a service that
        # handles being talked over inside the model asks for no interruption to
        # be broadcast, and gets that here.
        user_turn_strategies=(
            None if not realtime
            else ExternalUserTurnStrategies(enable_interruptions=interruptions)
            if turns == "provider"
            else UserTurnStrategies()
        ),
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    # What is being measured is decided here, by the session that started this
    # call, and not by the image -- one deployment answers for every row.
    global _calls_answered
    _calls_answered += 1
    # The log starts here, on this call's clock, before anything else is done:
    # a failure in the lines below is exactly what a reader will want the log
    # for, and the session id is the key the platform files the run under.
    session_id = str(getattr(runner_args, "session_id", "") or uuid.uuid4().hex[:12])
    start_call_clock(session_id)
    AUDIO_CLOCK.reset()
    call_log = CallLog(session_id)
    body = getattr(runner_args, "body", None)
    logger.info(
        "call {} answered by worker {} (call {} on it); session keys: {}",
        session_id, INSTANCE, _calls_answered,
        sorted(body.keys()) if isinstance(body, dict) else "none",
    )
    settings = Settings(body)
    server = load_agent(settings)
    trace = ToolTrace()
    restatements = RestatementIsNotSpeech()
    name = settings.get("s2s_provider", "openai-realtime")
    cascade = TEXT_MODELS.get(name)

    if cascade is not None:
        model = settings.get("s2s_model", cascade.default_model)
        credential = _credential(cascade.credential_env, name)
        record = cascade_record(name, cascade, model, server, settings)
        # Ahead of the build on purpose: a build that fails is then the next
        # line in the log after the thing it was building.
        logger.info("building {} on {}", name, model)

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
        # ``restatements`` sits between the transcription source and the context,
        # which here means straight after the speech-to-text service: a cascade
        # transcript travels *downstream*, so anything after the aggregator would
        # see it only once the aggregator had already appended it.
        stages = [transport.input(), stt, restatements, aggregators.user(), llm, tts, transport.output(),
                  DropControlTokens(), aggregators.assistant()]
        rate = CASCADE_RATE
        realtime = False
    else:
        if name not in PROVIDERS:
            known = sorted([*PROVIDERS, *TEXT_MODELS])
            raise ValueError(f"unknown S2S_PROVIDER {name!r}; expected one of {known}")
        provider = PROVIDERS[name]
        credential = _credential(provider.credential_env, name)
        model = settings.get("s2s_model", provider.default_model)
        voice = settings.get("s2s_voice", provider.default_voice)
        record = build_record(name, provider, model, voice, server, settings)
        # See the note in the cascade branch above.
        logger.info("building {} on {}", name, model)

        llm = provider.build(credential, model, voice, server.system_prompt, settings)
        register_tools(llm, server, trace)
        context = LLMContext(opening_messages(server.first_message), tools=build_tools(server))
        aggregators = LLMContextAggregatorPair(
            context,
            realtime_service_mode=True,
            user_params=user_aggregator_params(
                realtime=True, turns=provider.turns, interruptions=provider.interruptions
            ),
        )
        # No separate speech-to-text or text-to-speech: the realtime model is the
        # whole agent, so the pipeline is the transport, the context and the model.
        # ``restatements`` again sits between the transcription source and the
        # context, which here is the other side of the aggregator: a caller
        # transcript travels *upstream* from the realtime service, so it must be
        # trimmed on the service side to reach the filter before the aggregator.
        stages = [transport.input(), aggregators.user(), restatements, llm, transport.output(),
                  DropControlTokens(), aggregators.assistant()]
        rate = provider.input_rate
        realtime = True

    pipeline = Pipeline(stages)
    params = PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
        audio_in_sample_rate=rate,
        audio_out_sample_rate=rate,
    )
    meter = UsageMeter()
    capture_live_audio(llm, meter)
    narrator = CallNarrator(AUDIO_CLOCK)
    task, tracer = create_task(pipeline, context, params, runner_args, transport, record, [meter, narrator])
    if tracer is not None:
        call_log.hand_to(tracer)
        if realtime:
            capture_caller_turns(tracer, aggregators.user())
    logger.info("agent bench reference agent: {}", record)

    # What each side said, in the log as well as the transcript: the log is
    # where a reader reconstructs a call, and a transcript row without the
    # turn events around it says what was said but not when or why.
    @aggregators.user().event_handler("on_user_turn_message_added")
    async def _caller_said(_aggregator, message):
        logger.info("caller said: {}", _short(message.content, 500))

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def _agent_said(_aggregator, message):
        logger.info(
            "agent said{}: {}", " (interrupted)" if message.interrupted else "", _short(message.content, 500)
        )

    def publish() -> None:
        """Rewrite the run's record with everything known so far.

        The record is assembled from three pieces -- the configuration, the tool
        trace and what the call consumed -- and only the first is complete when
        the run starts. It is rewritten after every tool call, and once more at
        the moment the SDK takes its snapshot (see ``finish_record``), so
        whatever ends the call, the record holds everything that happened.
        """
        if tracer is None:
            return
        try:
            timing = narrator.timing()
            tracer.set_custom_metadata({
                **record, **trace.as_metadata(), **meter.as_metadata(),
                **({"timing": timing} if timing else {}),
                "integrity": narrator.integrity(),
            })
        except Exception as exc:  # noqa: BLE001 -- never fail a call over a record
            logger.warning("could not update the run record: {}", exc)

    trace.on_change = publish

    def summarise() -> None:
        """The one line to read first when a result looks wrong."""
        usage = meter.as_metadata()["usage"]
        tools = trace.as_metadata()
        reply = narrator.timing().get("reply")
        checks = narrator.integrity()
        logger.info(
            "call summary: {:.0f}s, caller turns {}, agent responses {}, barge-ins {}, "
            "tools {} ({} exact), errors {}, usage reports {}, log lines {}{}{}",
            usage["call_seconds"], narrator.caller_turns, narrator.agent_turns, narrator.barge_ins,
            tools["tool_call_count"], tools["tool_calls_matched"], narrator.errors, usage["usage_reports"],
            len(call_log.lines), f" (+{call_log.dropped} dropped)" if call_log.dropped else "",
            f", reply p50 {reply['p50_ms']} ms / p90 {reply['p90_ms']} ms over {reply['count']}" if reply else "",
        )
        # On its own line and at WARNING when something broke, because this is
        # the line that says whether the one above can be believed.
        failed = [name for name in checks["checks"] if name != "ok"]
        report = ", ".join(f"{key} {value}" for key, value in checks.items() if key != "checks")
        if failed:
            logger.warning("call integrity: {} -- {}", ", ".join(failed), report or "no measurements")
        else:
            logger.info("call integrity: ok{}", f" -- {report}" if report else "")

    if tracer is not None:
        finish_record(tracer, context, publish, summarise)

    _on_transport_event(transport, "on_joined", lambda *_: logger.info("transport: joined"))
    _on_transport_event(transport, "on_left", lambda *_: logger.info("transport: left"))
    _on_transport_event(transport, "on_error", lambda _t, error: logger.warning("transport error: {}", error))
    _on_transport_event(
        transport, "on_participant_joined",
        lambda _t, participant: logger.info("transport: participant joined {}", _participant_id(participant)),
    )
    _on_transport_event(
        transport, "on_participant_left",
        lambda _t, participant, reason=None: logger.info(
            "transport: participant left {} ({})", _participant_id(participant), reason
        ),
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        # Every call opens from the agent, and this first context frame is also
        # what installs the tools on the realtime session.
        logger.info("caller connected; opening the call")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("caller disconnected; tearing the call down")
        publish()
        await task.cancel()

    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        if tracer is None:
            # Nobody shipped the log; say where the call ended up anyway.
            summarise()
        call_log.close()
        logger.info("call {} finished", session_id)


def _participant_id(participant: Any) -> str:
    if isinstance(participant, dict):
        return str(participant.get("id") or participant.get("info", {}).get("userName") or "?")
    return str(getattr(participant, "identity", None) or participant)


def _on_transport_event(transport: BaseTransport, event: str, handler: Callable) -> None:
    """Log a transport event when this transport has it; not every transport does."""
    async def _handle(*args, **kwargs):
        handler(*args, **kwargs)

    # Asked first rather than tried: the framework answers an unknown event
    # with a warning in the log, and five of those on every telephony call would
    # bury the lines this exists to add.
    if event in getattr(transport, "_event_handlers", {}):
        transport.add_event_handler(event, _handle)


def finish_record(tracer, context: LLMContext, publish: Callable[[], None], summarise: Callable[[], None]) -> None:
    """Complete the transcript and the record at the moment the SDK reads them.

    Two things are otherwise lost. A tool call after the agent's last spoken
    turn -- ``end_call`` and ``transfer_call`` are always that, and a scenario is
    scored on whether they were made -- is written to the context but never to
    the exported transcript, because the exporter copies the context only when
    an assistant turn ends and no turn ends after the call is over. And the
    record's usage figures are whatever the last tool call left them at.

    The SDK snapshots the transcript inside its own finalisation, and every path
    that ends a call -- the agent hanging up, the caller hanging up, the
    pipeline ending -- goes through that one method. So the snapshot is wrapped:
    first the rows still in the context are copied over, then the record is
    rewritten, then the snapshot proceeds. That is the reliable last moment
    this file previously said did not exist.
    """
    capture = getattr(tracer, "_transcript_capture", None)
    if capture is None:
        return
    original = capture.to_dict

    def to_dict():
        try:
            sweep_transcript(capture, context)
            publish()
            summarise()
        except Exception as exc:  # noqa: BLE001 -- never lose the snapshot over its trimmings
            logger.warning("could not finish the record: {}", exc)
        return original()

    capture.to_dict = to_dict


def sweep_transcript(capture, context: LLMContext) -> None:
    """Copy context rows the exporter has not seen yet, the way it copies them itself."""
    messages = context.messages
    start = getattr(capture, "last_processed_index", len(messages))
    now = datetime.now(timezone.utc).isoformat()
    added = 0
    for message in messages[start:]:
        if not isinstance(message, dict) or message.get("role") == "user":
            continue
        capture.session_transcript.append({
            "started_at": now,
            "ended_at": now,
            "_cekura_timing_role": "assistant",
            "_cekura_turn_index": getattr(capture, "_assistant_turn_count", 0),
            **message,
        })
        added += 1
    capture.last_processed_index = len(messages)
    if added:
        logger.info("transcript: {} row(s) written after the last agent turn were swept into the export", added)


def capture_live_audio(llm: LLMService, meter: UsageMeter) -> None:
    """Record the seconds a provider bills for, where the framework only logs them.

    One of these services does not bill its speech model by the token at all. It
    reports the session's audio duration in seconds, and the framework prints
    that to the log and emits no metric for it -- so the token counts that do
    reach the record are the *backend* model's, and they are the cheaper half.
    A row priced from those alone understates what the call cost by most of it,
    and it would sit on a board next to rows that are complete.

    There is no public surface for this, so the service's own reporting is
    wrapped and still called. A service that reports no seconds is untouched,
    which is every other row.
    """
    original = getattr(llm, "_report_usage", None)
    if original is None:
        return

    async def report(usage: Any):
        seconds = getattr(usage, "seconds", None)
        if seconds is not None:
            meter.record_live_audio(float(seconds))
        return await original(usage)

    llm._report_usage = report


def capture_caller_turns(tracer, user_aggregator) -> None:
    """Put the caller's words back into the transcript on a realtime row.

    In realtime mode the framework deliberately reports the end of a caller's
    turn with no text attached: the service may still be transcribing when the
    boundary is announced, so the finalized text is delivered later, on a
    separate event, once it has been written to the context. The tracing SDK
    subscribes only to the boundary and discards it when it carries no text --
    correct for a cascade, where the two coincide, and silently lossy for every
    native speech model. The caller is in the context and in the call; only the
    exported transcript is missing them, which is why the run still scored.

    So the later event is subscribed to as well, and its text handed to the same
    recorder. Cascade rows must not do this: there the two events coincide and
    every caller turn would be recorded twice.

    Reaching past the SDK's public surface is deliberate and is the narrower of
    the two options -- the alternative is assembling and posting our own
    transcript, which would put the benchmark in the business of maintaining a
    second exporter. Revisit when the SDK handles realtime services itself.
    """
    capture = getattr(tracer, "_transcript_capture", None)
    if capture is None:
        # No credentials, or a tracer that changed shape under us. Either way the
        # call proceeds unobserved rather than not at all.
        logger.warning("caller turns will not be exported: no transcript capture on the tracer")
        return

    from pipecat.processors.aggregators.llm_response_universal import UserTurnStoppedMessage

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def _on_caller_turn(aggregator, message):
        await capture.on_user_turn_stopped(
            UserTurnStoppedMessage(content=message.content, timestamp=message.timestamp)
        )


def create_task(pipeline, context, params, runner_args, transport, record, observers) -> tuple[PipelineTask, Any]:
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
        return PipelineTask(pipeline, params=params, observers=observers), None

    try:
        from cekura.pipecat import PipecatTracer

        tracer = PipecatTracer(
            api_key=api_key,
            agent_id=int(agent_id),
            host=os.getenv("CEKURA_HOST", "https://api.cekura.ai"),
            # Off for a local run against a local receiver, where the span
            # exporter would otherwise retry against a host it cannot reach for
            # the whole call and then hold the finalisation for its timeout.
            enable_otel_traces=os.getenv("CEKURA_OTEL_TRACES", "1").lower() not in ("0", "false", "no"),
        )
        metadata = dict(record)
        # "track" correlates a scenario run and captures transcripts and metadata;
        # "observe" additionally uploads the call audio and starts evaluation.
        # A benchmark run is dispatched with its own run id, so track is the
        # default and runner_args is passed through untouched to carry it.
        if record.get("cekura_mode") == "observe":
            return tracer.observe_and_create_task(
                pipeline, context, runner_args=runner_args, transport=transport,
                custom_metadata=metadata, params=params, observers=observers,
            ), tracer
        return tracer.track_and_create_task(
            pipeline, context, runner_args=runner_args, transport=transport,
            custom_metadata=metadata, params=params, observers=observers,
        ), tracer
    except Exception as exc:  # noqa: BLE001 -- observability must never fail a call
        logger.warning("Cekura tracing disabled: {}", exc)
        return PipelineTask(pipeline, params=params, observers=observers), None


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

    # The rows whose turns are decided locally load a turn model as well, and it
    # loads on the first call that needs it unless it is built here.
    try:
        UserTurnStrategies()
    except Exception as exc:  # noqa: BLE001 -- the row that needs it will raise in context
        logger.warning("could not warm the turn detector: {}", exc)


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
