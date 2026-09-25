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

import asyncio
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
from typing import Any, Awaitable, Callable

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
    CancelWorkerFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallCancelFrame,
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
    LLMAssistantAggregator,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallRunnerItem,
    LLMService,
)
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


def call_elapsed(now: float | None = None) -> float | None:
    """Seconds since the call started, or None before any call has."""
    started = _CLOCK.get() or _latest_clock
    if started is None:
        return None
    return (now if now is not None else time.monotonic()) - started


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
        Reasoning,
        SessionProperties,
    )
    from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService, OpenAIRealtimeLLMSettings

    return OpenAIRealtimeLLMService(
        api_key=api_key,
        settings=OpenAIRealtimeLLMSettings(
            model=model,
            system_instruction=instructions,
            session_properties=SessionProperties(
                # A named level, sent rather than left to the server, whose
                # default the record could not name.
                reasoning=Reasoning(effort=OPENAI_REASONING),
                audio=AudioConfiguration(
                    # Asked for, because this service transcribes the caller only
                    # when told to. See the caller-transcription note above the table.
                    input=AudioInput(transcription=InputAudioTranscription()),
                    output=AudioOutput(voice=voice),
                ),
            ),
        ),
    )


@lru_cache(maxsize=None)
def gemini_service_class():
    """The framework's Gemini service, minus one replay after a reconnect.

    The service drops its connection a few times an hour on a server-side
    error and resumes the same session from a handle, so the server keeps the
    conversation. Disconnecting also forgets which tool results have been
    delivered, so the next context frame would re-send every result into a
    session that already holds them, and a result in flight across the gap
    would go out under a placeholder name. Delivered ids and names are kept
    across a resume; a reconnect without a handle re-seeds the history itself
    and is left alone.

    The framework also connects before the tools arrive, then reconnects to
    apply them, and resumes if the server has already issued a handle. A
    resumed session keeps the setup it was opened with, tools included, so a
    reconnect made to change the configuration opens a new session instead.

    Imported lazily like the builders, so a row that never touches this
    provider does not load it.
    """
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

    class GeminiResumesQuietly(GeminiLiveLLMService):
        async def _handle_server_message(self, message):
            await super()._handle_server_message(message)
            cancellation = getattr(message, "tool_call_cancellation", None)
            ids = list(getattr(cancellation, "ids", None) or [])
            if ids:
                await self._handle_tool_call_cancellation(ids)

        async def _handle_tool_call_cancellation(self, ids: list[str]) -> None:
            """The service withdrew tool calls it is no longer waiting for.

            The framework has no branch for this message, so a withdrawn call
            went on looking like an open one: its result was still sent, and
            the record kept a call the model had itself abandoned and would
            ask for again. Each id is closed here so no result follows it, the
            pipeline is told the call was cancelled, and the trace marks it.

            The message says only *that* the service withdrew the call, so the
            log states no cause.
            """
            wanted = set(ids)
            cancelled = await self._cancel_function_call_tasks(
                lambda item: item.tool_call_id in wanted,
                reason="withdrawn by the service (toolCallCancellation)",
            )
            # A call still running is settled by the helper above. It never
            # reached the trace, so there is nothing on the record to mark --
            # which is why the two cases are logged apart rather than together.
            running = {item.tool_call_id: item.function_name for item in cancelled}
            self._completed_tool_calls.update(ids)
            settled = []
            for tool_call_id in ids:
                name = (self._tool_call_id_to_name.get(tool_call_id)
                        or running.get(tool_call_id, "unknown"))
                if tool_call_id in running:
                    logger.info(
                        "the service withdrew tool call {} ({}) while it was still "
                        "running; it never reached the record", name, tool_call_id,
                    )
                    continue
                logger.info(
                    "the service withdrew tool call {} ({}) after it had returned; its "
                    "result will not be sent", name, tool_call_id,
                )
                settled.append(await self._broadcast_function_call_cancelled(
                    FunctionCallRunnerItem(
                        registry_item=self._functions.get(name),
                        function_name=name,
                        tool_call_id=tool_call_id,
                        arguments={},
                        context=self._context,
                    )
                ))
            if settled:
                await self._call_event_handler("on_function_calls_cancelled", settled)

        # Set when the context that carries the tools arrives, consumed by the
        # one reconnect that applies them.
        _open_a_new_session: bool = False

        async def _handle_context(self, context):
            if not self._context:
                # The first context is the one that carries the tools, and the
                # reconnect it triggers exists to apply them. Resuming would
                # keep the setup the session was opened with, which has none.
                # The handle is dropped in the reconnect rather than here: the
                # framework reconnects on the first context only when there is
                # something to apply, and a handle discarded for a reconnect
                # that never happens leaves a later mid-call error with nothing
                # to resume from.
                self._open_a_new_session = True
            await super()._handle_context(context)

        async def _reconnect(self):
            if self._open_a_new_session:
                self._open_a_new_session = False
                logger.info(
                    "opening a new Gemini session to apply the tools (the service had "
                    "already issued a resumption handle: {})",
                    bool(self._session_resumption_handle),
                )
                self._session_resumption_handle = None
            resuming = bool(self._session_resumption_handle)
            delivered = set(self._completed_tool_calls)
            names = dict(self._tool_call_id_to_name)
            await super()._reconnect()
            if resuming:
                self._completed_tool_calls |= delivered
                self._tool_call_id_to_name = {**names, **self._tool_call_id_to_name}
                logger.info(
                    "resumed the Gemini session, carrying {} already-delivered tool "
                    "result(s) across the reconnect so none is sent twice", len(delivered),
                )

    return GeminiResumesQuietly


def _gemini(api_key: str, model: str, voice: str, instructions: str, settings: Settings) -> LLMService:
    from google.genai.types import ThinkingConfig
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMSettings, GeminiVADParams

    # Service VAD off. With it on, the whole call is streamed, silences
    # included, and the session falls cumulatively behind: about 3 s on the
    # first reply, up to half a minute late in a long call. Off, the shared
    # detector sends audio only inside the caller's turn (the service keeps a
    # short pre-roll) and replies stay at 1-3 s. It is also what
    # ``turns="local"`` below claims.
    return gemini_service_class()(
        api_key=api_key,
        settings=GeminiLiveLLMSettings(
            model=model,
            system_instruction=instructions,
            voice=voice,
            vad=GeminiVADParams(disabled=True),
            # The extended-thinking model reasons in the background between
            # output chunks, at a level the setup must name; the framework
            # would otherwise pick the lowest. Sent as the row's setting.
            thinking=ThinkingConfig(thinking_level=GEMINI_THINKING),
        ),
    )


def _grok(api_key: str, model: str, voice: str, instructions: str, settings: Settings) -> LLMService:
    from pipecat.services.xai.realtime.events import (
        AudioConfiguration,
        AudioInput,
        InputAudioTranscription,
        Reasoning,
        SessionProperties,
        TurnDetection,
    )
    from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService, GrokRealtimeLLMSettings

    return GrokRealtimeLLMService(
        api_key=api_key,
        settings=GrokRealtimeLLMSettings(
            model=model,
            system_instruction=instructions,
            session_properties=SessionProperties(
                voice=voice,
                # Both are the vendor's own defaults, sent explicitly so the
                # record can name them. This model reasons before it speaks
                # unless told not to, and that is a latency the row carries;
                # the detector's threshold decides how loud the caller must be.
                # Neither belongs to a default that can move between runs.
                reasoning=Reasoning(effort=GROK_REASONING),
                turn_detection=TurnDetection(type="server_vad", **GROK_VAD),
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
    """Qwen's realtime audio model, through the service in ``qwen_realtime``.

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


def _phonic(credential: str, model: str, voice: str, instructions: str, settings: Settings) -> LLMService:
    """Phonic's speech-to-speech model, through the service in ``phonic_realtime``.

    Pipecat has no service for this provider either. Every setting the row
    depends on is sent, not left to the server: omit the model and this
    account is served an older one, with nothing in the reply to say so.
    """
    from phonic_realtime import PhonicRealtimeLLMService

    return PhonicRealtimeLLMService(
        api_key=credential,
        model=model,
        voice=voice,
        instructions=instructions,
        settings={"intelligence_level": PHONIC_INTELLIGENCE, **PHONIC_TURNS},
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
    from pipecat.services.openai.responses.llm import (
        OpenAIResponsesLLMSettings,
        OpenAIResponsesReasoningConfig,
    )

    return OpenAILiveLLMService(
        api_key=api_key,
        settings=OpenAILiveLLMSettings(
            model=model, system_instruction=live_instructions(instructions), voice=voice
        ),
        delegation=OpenAILiveLLMService.ResponsesDelegation(
            # The backend is the half that does the task, so it gets the
            # agent's instructions whole. Left unset, it would work from the
            # tool schemas and the transcript alone -- an agent that had never
            # read its own prompt.
            # Its reasoning effort is named too: left unset, the framework turns
            # a backend's reasoning off to keep latency down, which is not the
            # configuration the row claims.
            settings=OpenAIResponsesLLMSettings(
                model=backend_model(settings),
                system_instruction=instructions,
                reasoning=OpenAIResponsesReasoningConfig(effort=backend_reasoning(settings)),
            ),
        ),
    )


# The live model cannot call a function itself; it hands work to the backend
# when its prompt says to, in the shape the vendor's prompting guide gives.
# Without this section a farewell is just conversation and the call is never
# ended. Only the mechanics live here -- which steps go to the backend and when;
# behaviour rules are in ``SHARED_RULES``, the same for every row. It names
# kinds of step, not tools, so it serves every agent definition, and it is
# disclosed on the record as ``prompt_addendum``.
DELEGATION_PROMPT = """\
# Working with the backend

Backchannel policy: Use moderate backchannels. Acknowledge naturally without \
competing with the main response.

Interruption policy: Stop speaking when the user interrupts. Listen to what \
they say.

You hold the conversation. A backend does every step the instructions above \
assign to a tool: checking, saving, recording, routing, transferring, and \
ending the call. You cannot do those yourself.

Backend tools: every tool the instructions above name, plus ending the call \
and transferring the call.

Delegate to the backend when:
- a step of the task needs something checked, saved, recorded or routed;
- the caller confirms or corrects details that a pending step depends on;
- the caller asks to be transferred, or a transfer has been arranged and announced;
- the caller says goodbye, asks to hang up, or the conversation has reached \
its natural end, so that the call can be ended.

Do not delegate to the backend when:
- you need a brief clarification before you can tell what the caller wants;
- a simple conversational reply is all the caller needs.

Delegate before giving an answer that depends on backend work.
"""


# Appended to the agent definition's prompt on every row, native and cascade
# alike, in the same words. Each rule is already implied by the definitions
# (report results only once a tool returns them; sign off, then end the call);
# stating them once for all rows means no row is the only one reminded.
# "Ending the call" is a tool step, so a row that reaches its tools through a
# backend reads it through the section above.
SHARED_RULES = """\
# Results and ending the call

- Never say that something is booked, cancelled, saved or confirmed, and never \
give a confirmation number, until the step that does it has returned its result.
- Saying goodbye and ending the call are one step: whenever you give your \
sign-off, end the call in the same turn (or transfer it, if a transfer has been \
arranged and announced).
- When the caller says "bye", "goodbye" or "that's all", end the call, even if \
you have already said goodbye.
"""


def agent_instructions(prompt: str) -> str:
    """What every row is told: the agent definition's prompt, then ``SHARED_RULES``."""
    return f"{(prompt or '').rstrip()}\n\n{SHARED_RULES}"


def live_instructions(instructions: str) -> str:
    """The agent's prompt as the live model receives it: unchanged, then the section above."""
    return f"{instructions.rstrip()}\n\n{DELEGATION_PROMPT}"


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

    # The vendor's recommended default, sent explicitly so the record can name
    # it: this is how long the service waits after the caller stops before it
    # answers, and the row's reply time carries all of it.
    nova_settings = AWSNovaSonicLLMSettings(
        model=model, system_instruction=instructions, voice=voice,
        endpointing_sensitivity=NOVA_ENDPOINTING,
    )
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
    two results incomparable without either of them looking wrong. The default
    is the model the vendor's own guide says to start from; a smaller backend
    is a cost row and is named as such by the session.
    """
    return settings.get("s2s_backend_model", "gpt-6-sol")


def backend_reasoning(settings: Settings) -> str:
    """How hard that backend reasons.

    Low: a larger backend at medium stalled calls -- a reply twenty seconds
    after a tool result, then a dropped connection.
    """
    return settings.get("s2s_backend_reasoning", "low")


def aws_region(settings: Settings) -> str:
    """The region Bedrock is called in, and the region the record names.

    One accessor because those two must be the same string. Credentials are
    scoped by region and a model is served in some regions only, so a record
    naming a different region than the call used would describe a run nobody made.

    The default is a region where the model is served. The credentials must
    also be granted there, and the two failures look identical from outside.
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
    # declaring it here is what stops a new provider being added without it.
    discloses: Callable[["Settings"], dict[str, str]] = lambda _settings: {}
    # Where the caller's turn boundary comes from. Most of these services decide
    # it on their own server and announce it, and that announcement is what the
    # pipeline should follow -- provider endpointing is part of what a row
    # measures. Rows marked local (both Gemini rows, with the service's
    # detector off, Nova Sonic and Qwen) run the framework's recommended
    # arrangement, a local detector deciding turns, and the record says so,
    # because a row whose turns were decided locally measures something else.
    turns: str = "provider"
    # When a tool's result is handed to the model, of which there are three
    # cases and the difference between them is the row's behaviour under a
    # barge-in. ``after_speech``: the framework waits for the agent to stop
    # speaking, which for a narrated tool is exactly the window in which a
    # caller talking over the agent makes the service withdraw the call -- but
    # these services also open a new response when the result lands, so a late
    # result is not orphaned and the wait costs only the narration it spares.
    # ``immediate``: the service's context handling merely forwards the result
    # and nothing prompts the model afterwards, so the result has to arrive
    # while the model is still waiting for it. ``service``: the service takes
    # the result off the frame itself and never reads it out of the context,
    # so this pipeline does not time the delivery at all.
    results: str = "after_speech"
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

    @property
    def answers(self) -> str:
        """Who decides *when the model answers*, which is not the turn question.

        A service that runs its own detector over the audio answers when that
        detector says the caller has finished, whatever the pipeline decided
        about the turn -- so a row whose turns are local can still carry the
        provider's endpointing in its reply time, and the record has to say
        which. Only a service whose own detector is switched off answers on
        the pipeline's word, which is what makes this derived rather than
        another thing to set correctly per row.
        """
        return "local" if not self.service_vad else "provider"
    # Whether a caller talking over the agent is this pipeline's business.
    #
    # Usually it is: the agent is stopped here and the service is told the reply
    # was cut short. One of these services handles it inside the model instead,
    # announces that it broadcasts no interruption of its own, and asks the
    # pipeline not to broadcast one either -- so a client-side interruption
    # there cuts work the model was going to carry on with, and the row would be
    # reporting this pipeline's barge-in rather than the service's. A service
    # that asks for this is taken at its word. Another decides on its own side
    # whether it was interrupted and reports it; that service broadcasts the
    # interruption itself, when it happens.
    interruptions: bool = True


# Whether the caller's own words reach the record is a per-provider decision,
# and it is not a detail: a scored run needs both halves of the conversation, and
# a transcript holding only the agent reads as a caller who never spoke. Two of
# these services transcribe the caller only when asked, and each asks
# differently; five do it themselves. Nothing warns about the difference,
# because a session without transcription is a working session.
#
#   openai-realtime   asked for  -- an input transcription config, default model
#   grok-realtime     asked for  -- same shape, but only under its own ASR model
#   qwen-realtime     automatic  -- documented for the audio model
#   gemini-live       automatic  -- the service configures both directions itself
#   gpt-live          automatic  -- the protocol is transcript-driven throughout
#   nova-sonic        automatic  -- the service emits caller transcripts natively
#   phonic            automatic  -- the service sends each finished caller turn
#
# The tool calls travel separately, through the context aggregator, which is why
# a run can show resolved tools and still carry no speech.
GROK_TRANSCRIBE_MODEL = "grok-transcribe"

# Vendor defaults, made explicit. Each is a setting the vendor documents as
# the one it applies when nothing is sent, so sending it changes nothing about
# the call and everything about whether the record can say what was measured.
#
# Grok reasons before it speaks at effort ``high`` unless told ``none``; the
# vendor says to turn it off when it is not needed, which is a judgement this
# bench does not make for a vendor. Its detector fires at 0.85 and keeps 333 ms
# before the onset. Nova Sonic waits 1.75 s of pause at ``MEDIUM`` before it
# answers (``HIGH`` is 1.5 s, ``LOW`` about 2 s) and calls ``MEDIUM`` the
# recommended default.
GROK_REASONING = "high"
GROK_VAD = {"threshold": 0.85, "prefix_padding_ms": 333}
NOVA_ENDPOINTING = "MEDIUM"

# Where a vendor offers a reasoning or thinking level, each row runs its
# model's strongest agent configuration at a named level, rather than whatever
# the server does when nothing is sent -- a default the record cannot name. A
# faster, lower setting is a different configuration and gets its own row.
OPENAI_REASONING = "high"
GEMINI_THINKING = "HIGH"
PHONIC_INTELLIGENCE = "high"

# Phonic's documented defaults for its detector and for when a caller has
# talked over the agent: 800 ms of silence ends a turn, and one word interrupts.
PHONIC_TURNS = {
    "vad_threshold": 0.38,
    "vad_min_silence_duration_ms": 800,
    "vad_min_speech_duration_ms": 50,
    "vad_prebuffer_duration_ms": 500,
    "min_words_to_interrupt": 1,
}


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
        discloses=lambda settings: {"openai_reasoning": OPENAI_REASONING},
    ),
    # The vendor's smaller tier, on the same service and settings. A row of its
    # own rather than a model override on the one above, because the row is
    # what the record, the report and the price table are keyed by: an override
    # would be reported, and priced, as the larger model.
    "openai-realtime-mini": Provider(
        _openai, 24000, "gpt-realtime-2.1-mini", "marin", ("OPENAI_API_KEY",),
        "pipecat.services.openai.realtime.llm",
        discloses=lambda settings: {"openai_reasoning": OPENAI_REASONING},
    ),
    # The vendor's current Live model in its extended-thinking form, which is a
    # separate model id rather than a setting: the plain model refuses a
    # thinking level at all. The preview before both is the one whose empty
    # control-token turns and mid-call drops are on the vendor's issue tracker.
    "gemini-live": Provider(
        _gemini, 16000, "models/gemini-3.8-live-extended-thinking", "Charon",
        ("GEMINI_API_KEY", "GEMINI_AUTHORIZATION"), "pipecat.services.google.gemini_live.llm",
        discloses=lambda settings: {"gemini_thinking_level": GEMINI_THINKING},
        turns="local", results="immediate", service_vad=False,
        caller_transcription="automatic",
    ),
    # The vendor's fast Live tier. Unlike the plain 3.8 model it takes a
    # thinking level, so it runs at the same named level as the row above, on the
    # same arrangement; a row of its own for the reason given at the mini row.
    "gemini-flash-live": Provider(
        _gemini, 16000, "models/gemini-3.1-flash-live-preview", "Charon",
        ("GEMINI_API_KEY", "GEMINI_AUTHORIZATION"), "pipecat.services.google.gemini_live.llm",
        discloses=lambda settings: {"gemini_thinking_level": GEMINI_THINKING},
        turns="local", results="immediate", service_vad=False,
        caller_transcription="automatic",
    ),
    # Pinned to the versioned name the vendor's ``latest`` alias resolves to:
    # an alias is not a configuration. 24 kHz is the vendor's recommended rate.
    "grok-realtime": Provider(
        _grok, 24000, "grok-voice-think-fast-2.0", "eve", ("XAI_API_KEY",),
        "pipecat.services.xai.realtime.llm",
        discloses=lambda settings: {
            "grok_reasoning": GROK_REASONING,
            "grok_vad": "server_vad, " + ", ".join(f"{k} {v}" for k, v in GROK_VAD.items())
            + ", silence_duration_ms server default",
        },
    ),
    # GPT-Live is the exception to the paragraph above: it resamples what it is
    # handed. The rate is still declared, so the record says what was sent.
    "gpt-live": Provider(
        _gpt_live, 24000, "gpt-live-1", "marin", ("OPENAI_API_KEY",),
        "pipecat.services.openai.live.llm",
        discloses=lambda settings: {
            "s2s_backend_model": backend_model(settings),
            "s2s_backend_reasoning": backend_reasoning(settings),
            # Named *and* hashed. The name alone would let the section be
            # rewritten without any digest on the record changing, and
            # ``system_prompt_sha256`` covers the agent prompt the row shares
            # with every other row, not the addendum this one adds to it.
            "prompt_addendum": "gpt-live-delegation",
            "prompt_addendum_sha256": _digest(DELEGATION_PROMPT),
        },
        results="service", interruptions=False, caller_transcription="automatic",
    ),
    # Nova Sonic listens at 16 kHz and speaks at 24 kHz. The pipeline runs at the
    # input rate and the service resamples its own output.
    "nova-sonic": Provider(
        _nova_sonic, 16000, "amazon.nova-2-sonic-v1:0", "matthew", ("AWS_ACCESS_KEY_ID",),
        "pipecat.services.aws.nova_sonic.llm",
        discloses=lambda settings: {
            "aws_region": aws_region(settings),
            "nova_endpointing": NOVA_ENDPOINTING,
        },
        turns="local", results="immediate", caller_transcription="automatic",
    ),
    # Phonic takes and returns 16 kHz, and no framework service exists for it
    # -- see ``phonic_realtime``. ``phonic_v1`` is the newest model this account
    # is served; the next is enabled per account on request. The service decides
    # when it has been talked over, and delivers a tool's result into the reply
    # it is already speaking, so the result is sent as soon as it exists.
    "phonic": Provider(
        _phonic, 16000, "phonic_v1", "sabrina", ("PHONIC_API_KEY",),
        "phonic_realtime",
        discloses=lambda settings: {
            "phonic_intelligence_level": PHONIC_INTELLIGENCE,
            "phonic_turns": ", ".join(f"{k} {v}" for k, v in PHONIC_TURNS.items()),
            "tool_schema": "strict; optional parameters nullable, null arguments dropped",
        },
        results="immediate", interruptions=False, caller_transcription="automatic",
    ),
    # Qwen listens at 16 kHz and speaks at 24 kHz, and no framework service
    # exists for it -- see ``qwen_realtime``. Its audio model is documented to
    # transcribe the caller on its own; confirm on a call before publishing it.
    "qwen-realtime": Provider(
        _qwen_realtime, 16000, "qwen-audio-3.0-realtime-plus", "longanqian", ("DASHSCOPE_API_KEY",),
        "qwen_realtime",
        discloses=lambda settings: {"qwen_region": qwen_region(settings)},
        turns="local", caller_transcription="automatic",
    ),
}


# ── the cascade, for comparison ──────────────────────────────────────────────
#
# "Is a native speech model better than the pipeline it replaces?" is only
# answerable if the two sides differ in one thing. So the cascade is this same file, this
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

    build: Callable[[str, str, str | None], LLMService]
    default_model: str
    credential_env: tuple[str, ...]
    module: str
    # The speech model this one is the counterpart to, or None for the neutral
    # baseline that belongs to no vendor.
    counterpart_to: str | None = None
    # Sent on every call and named on the record, for the same reason as the
    # native rows' levels. None only for a model with no reasoning step.
    reasoning: str | None = None


def _openai_text(api_key: str, model: str, reasoning: str | None) -> LLMService:
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(api_key=api_key, model=model)


def _openai_responses_text(api_key: str, model: str, reasoning: str | None) -> LLMService:
    # The Responses API, because chat completions refuses a reasoning effort
    # alongside function tools on the current models.
    from pipecat.services.openai.responses.llm import (
        OpenAIResponsesLLMService,
        OpenAIResponsesLLMSettings,
        OpenAIResponsesReasoningConfig,
    )

    return OpenAIResponsesLLMService(
        api_key=api_key,
        settings=OpenAIResponsesLLMSettings(
            model=model, reasoning=OpenAIResponsesReasoningConfig(effort=reasoning)
        ),
    )


def _google_text(api_key: str, model: str, reasoning: str | None) -> LLMService:
    from pipecat.services.google.llm import GoogleLLMService

    # Named, because unset is not the vendor's default here: the framework
    # drops a Gemini Flash model to the lowest thinking level it accepts.
    thinking = GoogleLLMService.ThinkingConfig(thinking_level=reasoning) if reasoning else None
    return GoogleLLMService(api_key=api_key, settings=GoogleLLMService.Settings(model=model, thinking=thinking))


def _grok_text(api_key: str, model: str, reasoning: str | None) -> LLMService:
    from pipecat.services.xai.llm import GrokLLMService

    extra = {"reasoning_effort": reasoning} if reasoning else {}
    return GrokLLMService(api_key=api_key, settings=GrokLLMService.Settings(model=model, extra=extra))


def _qwen_text(api_key: str, model: str, reasoning: str | None) -> LLMService:
    from pipecat.services.qwen.llm import QwenLLMService

    return QwenLLMService(api_key=api_key, model=model)


TEXT_MODELS: dict[str, TextModel] = {
    # The neutral baseline: the pipeline every cascade row shares, with a text
    # model chosen for being widely understood rather than for matching anyone.
    # It is also held still, so the baseline means the same thing run to run.
    "cascade-baseline": TextModel(
        _openai_text, "gpt-4.1", ("OPENAI_API_KEY",), "pipecat.services.openai.llm",
    ),
    # Each counterpart is the vendor's current text model at a named level. The
    # OpenAI row runs GPT-Live's backend model and effort (gpt-6-sol, low).
    "cascade-openai": TextModel(
        _openai_responses_text, "gpt-6-sol", ("OPENAI_API_KEY",), "pipecat.services.openai.responses.llm",
        counterpart_to="openai-realtime", reasoning="low",
    ),
    "cascade-google": TextModel(
        _google_text, "gemini-3.8-flash", ("GEMINI_API_KEY", "GEMINI_AUTHORIZATION"),
        "pipecat.services.google.llm", counterpart_to="gemini-live", reasoning="high",
    ),
    "cascade-grok": TextModel(
        _grok_text, "grok-4.7", ("XAI_API_KEY",), "pipecat.services.xai.llm",
        counterpart_to="grok-realtime", reasoning="high",
    ),
    # Not yet brought up to date: there is no credential to list the vendor's
    # models with, and this row cannot start without one either.
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
    llm = text.build(credential, model, text.reasoning)
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


# Tokenizer artifacts (a run of ``<ctrl46>`` and the blank lines around it)
# that one service streams into its own transcript. No audio lies behind them,
# but a judge would score them as speech. Only that exact shape is removed;
# anything broader would be editing the evidence.
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


class HangsUpOnceHeard(FrameProcessor):
    """Ends the call as soon as the agent's goodbye has been heard.

    A hang-up cancels rather than pushing an end frame: an end frame waits for
    every stage to drain, which left the agent deaf for 3-4 s after its goodbye
    (about 30 s on Gemini) while the caller was still talking. What makes that safe is waiting for the goodbye first, which
    is why this sits after the output transport, where the agent's audio is
    actually played. It hangs up once the agent has been quiet for
    ``QUIET_SECS`` -- long enough for a reply to the tool's own result to begin,
    and to ride out a pause inside one sentence -- or after ``MAX_WAIT_SECS``
    whatever the agent is still doing.

    Where the transport allows it the room is left before the cancel is pushed
    (see ``leaves_the_room``), because the cancel reaches the transport only
    after the realtime service in front of it has closed its own connection.
    """

    QUIET_SECS = 1.0
    MAX_WAIT_SECS = 10.0
    LEAVE_SECS = 2.0

    def __init__(self, leave: Callable[[], Awaitable[None]] | None = None) -> None:
        super().__init__()
        self._leave = leave
        self._quiet = asyncio.Event()
        self._quiet.set()
        self._talking = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.hung_up_at: float | None = None
        self.left_at: float | None = None
        self.reason: str | None = None

    def hang_up(self, reason: str) -> None:
        """Called by the tool that closes the call. A second close is the same close."""
        if self._task is None:
            self.reason = reason
            self._task = self.create_task(self._once_quiet(reason))

    @property
    def closed_by(self) -> str | None:
        """Who ended the call: the agent through its own tool, or the backstop after its goodbye."""
        if self.reason is None:
            return None
        return "agent" if self.reason in CALL_CONTROL else "harness"

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._quiet.clear()
            self._talking.set()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._talking.clear()
            self._quiet.set()
        await self.push_frame(frame, direction)

    async def _once_quiet(self, reason: str) -> None:
        try:
            async with asyncio.timeout(self.MAX_WAIT_SECS):
                while True:
                    await self._quiet.wait()
                    try:
                        async with asyncio.timeout(self.QUIET_SECS):
                            await self._talking.wait()
                    except TimeoutError:
                        break
            outcome = "the agent has finished speaking"
        except TimeoutError:
            outcome = f"the agent was still speaking {self.MAX_WAIT_SECS:.0f} s after {reason}"
        self.hung_up_at = time.monotonic()
        logger.info("hanging up ({}): {}", reason, outcome)
        if self._leave is not None:
            try:
                async with asyncio.timeout(self.LEAVE_SECS):
                    await self._leave()
                self.left_at = time.monotonic()
            except Exception as exc:  # noqa: BLE001 -- the cancel below still ends the call
                logger.warning("could not leave the room ahead of the teardown: {!r}", exc)
        await self.push_frame(CancelWorkerFrame(), FrameDirection.UPSTREAM)

    async def cleanup(self) -> None:
        await super().cleanup()
        if self._task is not None and not self._task.done():
            await self.cancel_task(self._task)


# A sign-off, read off the last sentence the agent said and nowhere else, so a
# greeting that thanks the caller for calling and then asks what they need is
# not one. "take care of" is a task, not a goodbye.
_FAREWELL = re.compile(
    r"\b(good\s?-?bye|bye(?:[\s-]bye|\s+now)?|take care(?!\s+of)|"
    r"have an? (?:great|good|nice|wonderful|lovely|safe) (?:day|one|afternoon|evening|morning|weekend|night)|"
    r"thanks?(?: you)? (?:so much )?for calling|enjoy (?:the rest of )?your (?:day|afternoon|evening|weekend))\b"
)
# What a caller says back to a goodbye: not a question, and made of closing
# words -- short when it is only an acknowledgement ("great, thanks"), longer
# when it says outright that the caller is done ("thank you, that's all I
# needed"). Anything carrying one of the ``_MORE`` words is the caller starting
# something new, and the agent owes it an answer.
_DONE = re.compile(r"\b(bye|goodbye|that's all|that's it|that is all|nothing else|all set)\b")
_CLOSING_WORDS = re.compile(
    r"\b(thanks|thank you|okay|ok|alright|all right|got it|you too|great|perfect|"
    r"sounds good|nope|no|cheers|take care|have a (?:good|great|nice) (?:day|one))\b"
)
_MORE = re.compile(r"\b(wait|actually|one more|also|question|but|another|what|when|where|how|why|can you|could you)\b")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z']+")


def is_farewell(text: str) -> bool:
    """The agent's turn ended on a sign-off rather than on a question."""
    last = _SENTENCE_END.split(text.strip())[-1].lower()
    return not last.endswith("?") and bool(_FAREWELL.search(last))


def is_closing_reply(text: str) -> bool:
    """The caller answered a goodbye with one, or with an acknowledgement."""
    said = text.strip().lower().replace("\u2019", "'")
    if not said or "?" in said or _MORE.search(said):
        return False
    words = len(_WORD.findall(said))
    return (words <= 8 and bool(_CLOSING_WORDS.search(said))) or (words <= 16 and bool(_DONE.search(said)))


class ClosesAfterAnUnfinishedGoodbye(BaseObserver):
    """Ends a call the agent said goodbye to and then left open.

    A model can sign off without hanging up, and the silence after is then
    scored as an agent that stopped answering. So the call is closed here, the
    same way for every row, only where closing cannot cut anything short: the
    agent's last turn ended on a sign-off, the caller has since said nothing or
    only a goodbye, nobody is speaking, no tool is running, and the line has
    been quiet long enough (``due``). The hang-up is ``end_call``'s own, and the
    record says who closed the call, so ``end_call`` stays absent from the
    trace of a model that never ends its calls.
    """

    REPLY_QUIET_SECS = 3.0
    SILENCE_SECS = 5.0
    # A backend asked for work after the goodbye may be about to hang up itself.
    DELEGATION_GRACE_SECS = 4.0
    POLL_SECS = 0.2
    # Audio frames outnumber these by orders of magnitude; they are turned away first.
    WATCHED = (
        BotStartedSpeakingFrame, BotStoppedSpeakingFrame, VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame, TranscriptionFrame, FunctionCallInProgressFrame,
        FunctionCallResultFrame, FunctionCallCancelFrame,
    )

    def __init__(self, hang_up: Callable[[str], None]) -> None:
        super().__init__()
        self._hang_up = hang_up
        self._armed = False
        self._replied = False
        self._agent_speaking = False
        self._caller_speaking = False
        self._agent_stopped: float | None = None
        self._caller_stopped: float | None = None
        self._delegated: float | None = None
        self._tools: set[str] = set()
        self._task: asyncio.Task | None = None

    # ── what the call reports ────────────────────────────────────────────
    def agent_said(self, text: str, interrupted: bool = False, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        farewell = not interrupted and is_farewell(text)
        if farewell and not self._armed:
            logger.debug("backstop: the agent signed off; watching for a hang-up")
        elif self._armed and not farewell:
            logger.debug("backstop: the agent spoke again after its goodbye; standing down")
        self._armed, self._replied, self._delegated = farewell, False, None
        self._agent_stopped = self._agent_stopped or now

    def caller_said(self, text: str) -> None:
        if not self._armed:
            return
        if is_closing_reply(text):
            self._replied = True
        else:
            self._armed = False
            logger.debug("backstop: the caller said more after the goodbye; standing down")

    def delegated(self, now: float | None = None) -> None:
        if self._armed:
            self._delegated = time.monotonic() if now is None else now
            logger.debug("backstop: a backend was asked for work after the goodbye; holding {:.0f} s",
                         self.DELEGATION_GRACE_SECS)

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, self.WATCHED):
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            self._agent_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._agent_speaking, self._agent_stopped = False, time.monotonic()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._caller_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._caller_speaking, self._caller_stopped = False, time.monotonic()
        elif isinstance(frame, TranscriptionFrame):
            self.caller_said(frame.text)
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._tools.add(frame.tool_call_id)
        else:
            self._tools.discard(frame.tool_call_id)

    # ── the decision ─────────────────────────────────────────────────────
    def due(self, now: float) -> bool:
        if (not self._armed or self._agent_speaking or self._caller_speaking or self._tools
                or self._agent_stopped is None):
            return False
        deadline = self._agent_stopped + self.SILENCE_SECS
        if self._caller_stopped is not None and self._caller_stopped > self._agent_stopped:
            # The caller spoke after the goodbye: time from when they finished.
            # Until their words arrive they are not known to be a goodbye, so a
            # reply still in transcription waits out the silence rule instead.
            quiet = self._caller_stopped + self.REPLY_QUIET_SECS
            deadline = quiet if self._replied else max(deadline, quiet)
        if self._delegated is not None:
            deadline = max(deadline, self._delegated + self.DELEGATION_GRACE_SECS)
        return now >= deadline

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._watch())

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(self.POLL_SECS)
            now = time.monotonic()
            if self.due(now):
                logger.info(
                    "backstop: the agent said goodbye and left the call open; closing it {:.1f} s after it ({})",
                    now - self._agent_stopped, "the caller said goodbye too" if self._replied else "silence",
                )
                self._hang_up("backstop")
                return


# Disclosed on every row, because it is the same for every row and a reader has
# to know the call can end without the model ending it.
HANGUP_BACKSTOP = (
    f"after the agent's goodbye: {ClosesAfterAnUnfinishedGoodbye.REPLY_QUIET_SECS:g} s after a closing "
    f"reply from the caller, or {ClosesAfterAnUnfinishedGoodbye.SILENCE_SECS:g} s of silence; "
    f"{ClosesAfterAnUnfinishedGoodbye.DELEGATION_GRACE_SECS:g} s more after a backend delegation"
)


def leaves_the_room(transport: BaseTransport) -> Callable[[], Awaitable[None]] | None:
    """How to leave a call ahead of the pipeline's own teardown, where the transport allows it.

    Both halves of a Daily transport hold the room, and it is left when the
    second lets go. On a cancel that is the output half, which the cancel
    reaches only after the realtime service in front of it has closed its own
    socket: 2.3 to 2.5 s on websocket services, measured, with the caller on a line the agent had already hung
    up. So both holds are let go here, and the halves' own releases, when the
    cancel reaches them, find nothing left to release.

    Any other transport ends with the pipeline.
    """
    try:
        from pipecat.transports.daily.transport import DailyTransport
    except Exception:  # noqa: BLE001 -- no Daily, nothing to leave early
        return None
    if not isinstance(transport, DailyTransport):
        return None
    client = transport._client

    async def let_go_of_both_halves() -> None:
        await client.leave()  # the input half's hold
        await client.leave()  # the output half's, the one that actually leaves

    return let_go_of_both_halves


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

    Recorded on the run because the framework reports consumption per call and
    the trace store keeps it for thirty days; a board is read for longer.

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
        # Another reports speech and text apart and the framework adds them
        # together. See ``record_speech_tokens``.
        self._speech_tokens: dict[str, int] = {}
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

    def record_speech_tokens(self, input_speech: int, output_speech: int) -> None:
        """The speech share of the call's tokens, reported cumulatively, so the largest wins.

        Stored under the same names the other rows use for their audio tokens,
        which already sit inside ``prompt_tokens`` and ``completion_tokens``
        there -- so text is the total less the audio on every row alike.
        """
        for field, value in (("input_audio_tokens", input_speech), ("output_audio_tokens", output_speech)):
            self._speech_tokens[field] = max(self._speech_tokens.get(field, 0), int(value or 0))

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
            **{field: value for field, value in self._speech_tokens.items() if value},
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

    def __init__(
        self,
        clock: "AudioClock | None" = None,
        hangup: HangsUpOnceHeard | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        # Handed in rather than reached for, so this narrator reports the call it
        # was built for and nothing a previous one left behind.
        self._clock = clock if clock is not None else AudioClock()
        self._hangup = hangup
        self._now = now
        self._seen: dict[int, None] = {}
        self.caller_turns = 0
        # Replies the service started, and replies the caller could hear. They
        # differ when a service begins a response and fails before any audio,
        # and only the second is an answer.
        self.agent_turns = 0
        self.audible_turns = 0
        self.barge_ins = 0
        self.errors = 0
        self._agent_speaking = False
        # The wait the agent owes the caller: open from the caller connecting
        # (the greeting) or stopping (a reply), settled by the agent's audio or
        # by the caller giving up and speaking again.
        self._owed_since: float | None = None
        self._owed_kind = ""
        self.stalls: list[dict[str, Any]] = []
        # Caller turns the service has not yet sent a transcript after.
        self._untranscribed_turns: list[float] = []
        # Errors while the call was live, apart from the ones a teardown throws.
        self._call_errors = 0
        self._first_error_at_s: float | None = None
        self._error_messages: dict[str, None] = {}
        self._teardown_errors = 0
        self._ended_at: float | None = None
        self.replies: list[float] = []
        self.endpointing: list[float] = []
        # The same figures turn by turn, each with the moment in the call the
        # caller stopped, so a later reading can line them up with the
        # platform's own per-turn latencies from the recording.
        self.reply_turns: list[dict[str, Any]] = []
        self.endpointing_turns: list[dict[str, Any]] = []
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
            timing["reply"] = {**reply, "turns": self.reply_turns}
        if endpointing:
            timing["endpointing"] = {**endpointing, "turns": self.endpointing_turns}
        return timing

    def _turn(self, seconds: float) -> dict[str, Any]:
        """One turn's figure, stamped with when in the call the caller stopped."""
        stopped = call_elapsed(self._heard_stop)
        return {"caller_stopped_at_s": round(stopped, 3) if stopped is not None else None,
                "ms": round(seconds * 1000)}

    def caller_connected(self) -> None:
        """The call opens from the agent, so from here it owes the greeting."""
        if self.audible_turns == 0 and self._owed_since is None:
            self._owe("greeting")

    def _owe(self, kind: str) -> None:
        self._owed_since, self._owed_kind = self._now(), kind

    def _settle(self, answered: bool) -> None:
        if self._owed_since is None:
            return
        stall = self._stall(self._now(), answered)
        if stall is not None:
            self.stalls.append(stall)
        self._owed_since = None

    def _stall(self, until: float, answered: bool) -> dict[str, Any] | None:
        waited = until - self._owed_since
        if waited * 1000 < self.STALLED_REPLY_MS:
            return None
        began = call_elapsed(self._owed_since)
        return {"owed": self._owed_kind, "at_s": round(began, 3) if began is not None else None,
                "ms": round(waited * 1000), "answered": answered}

    def _ended_by(self) -> float | None:
        """When the caller's side of the call ended: the hang-up or the teardown, whichever came first."""
        ends = [t for t in (self._hangup.hung_up_at if self._hangup else None, self._ended_at) if t is not None]
        return min(ends) if ends else None

    def _call_ended_at(self) -> float:
        ended = self._ended_by()
        return ended if ended is not None else self._now()

    def _note_error(self, frame: ErrorFrame) -> None:
        ended = self._ended_by()
        if ended is not None and self._now() >= ended:
            self._teardown_errors += 1
            return
        self._call_errors += 1
        if self._first_error_at_s is None:
            at = call_elapsed(self._now())
            self._first_error_at_s = round(at, 3) if at is not None else None
        if len(self._error_messages) < self.ERROR_MESSAGES:
            text = str(frame.error)
            self._error_messages[text if len(text) <= 200 else text[:200] + "…"] = None

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
    HELD_HANGUP_MS = 2000
    # The silence after which the platform's simulated caller asks whether
    # anyone is still there: a wait this long is a dead line to the caller,
    # however it ends. The slowest healthy tool turns measured here, where a
    # reply takes two model calls, start their audio in about eight seconds.
    STALLED_REPLY_MS = 10000
    # A service that transcribes the caller sends each transcript within a
    # second or two of the turn, and some send one for several turns at once,
    # so only turns this old at the end of the call are counted as unheard,
    # and one alone is not a finding.
    TRANSCRIPT_GRACE_S = 10.0
    UNHEARD_TURNS = 2
    # Distinct error messages kept on the record; a failing socket can raise
    # the same one hundreds of times.
    ERROR_MESSAGES = 3

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
        call. ``answered`` is how many caller turns drew a reply the caller
        could hear: a call the agent is too far behind to answer still produces
        turns, and they score as silence. A response the service started and
        abandoned before any audio is not an answer. ``hangup_tail_ms`` runs
        from the moment the agent's hang-up was acted on to the moment the
        caller was let go: a hang-up that is decided but does not end the call
        leaves the caller talking into a line nobody is listening on, and the
        transcript reads that as an agent that stopped answering.

        Three checks name a service that failed rather than a model that
        answered badly, so a board can count those runs apart. ``service_errors``
        are the errors raised while the call was live -- a rejected request, a
        model the vendor says is unavailable, a dropped socket -- with the ones a
        teardown throws after the hang-up kept separately. ``stalls`` are the
        waits of ``STALLED_REPLY_MS`` or more for the agent's audio, answered in
        the end or not, including the greeting. ``unheard_turns`` are caller
        turns the service never sent a transcript after: every row here is
        configured to transcribe the caller, so a session still accepting audio
        but no longer processing it shows as turns with no transcript, whatever
        the timing.
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
            report["answered"] = f"{self.audible_turns}/{self.caller_turns}"
            if self.audible_turns < self.caller_turns * self.ANSWERED_SHARE:
                failed.append("turns_unanswered")
        elif self.audible_turns == 0:
            # Neither side said anything. The row exists and holds no call.
            failed.append("silent_call")

        ended = self._call_ended_at()
        if self._call_errors:
            report["service_errors"] = {
                "count": self._call_errors,
                "first_at_s": self._first_error_at_s,
                "messages": list(self._error_messages),
            }
            failed.append("service_error")
        if self._teardown_errors:
            report["teardown_errors"] = self._teardown_errors

        stalls = list(self.stalls)
        if self._owed_since is not None:
            # Still owed when the call ended: the caller heard nothing to the end.
            stall = self._stall(ended, answered=False)
            if stall is not None:
                stalls.append(stall)
        if stalls:
            report["stalls"] = stalls
            failed.append("reply_stalled")

        unheard = sum(1 for started in self._untranscribed_turns if ended - started >= self.TRANSCRIPT_GRACE_S)
        if unheard >= self.UNHEARD_TURNS:
            report["unheard_turns"] = unheard
            failed.append("caller_audio_unacknowledged")

        hung_up_at = self._hangup.hung_up_at if self._hangup is not None else None
        if hung_up_at is not None:
            # Until the caller is let go: the room left, or else the call over.
            tail = (self._hangup.left_at or time.monotonic()) - hung_up_at
            report["hangup_tail_ms"] = round(tail * 1000)
            if tail * 1000 > self.HELD_HANGUP_MS:
                failed.append("hangup_held")
            # Not a failed check: a call the backstop closed was delivered as a
            # call. It is the model's miss, and it is named so it can be counted.
            report["closed_by"] = self._hangup.closed_by

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
        FunctionCallInProgressFrame, FunctionCallResultFrame, FunctionCallCancelFrame,
        ErrorFrame, EndFrame, CancelFrame, MetricsFrame,
    )

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, self.NARRATED) or not self._first_sighting(frame):
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # A caller who speaks again has stopped waiting for whatever was owed.
            self._settle(answered=False)
            logger.debug("detector: caller speech starts")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # The anchor. A later stop replaces an earlier one, so a caller who
            # pauses mid-turn is measured from when they actually finished.
            self._heard_stop = self._now()
            self._turn_closed = None
            if not self._agent_speaking:
                self._owe("reply")
            logger.debug("detector: caller speech stops")
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.caller_turns += 1
            self._untranscribed_turns.append(self._now())
            logger.info("caller turn {} starts", self.caller_turns)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._heard_stop is not None:
                self._turn_closed = self._now()
                waited = self._turn_closed - self._heard_stop
                self.endpointing.append(waited)
                self.endpointing_turns.append(self._turn(waited))
                logger.info("caller turn {} ends ({:.0f} ms after the detector heard it stop)",
                            self.caller_turns, waited * 1000)
            else:
                logger.info("caller turn {} ends", self.caller_turns)
        elif isinstance(frame, TranscriptionFrame):
            # One transcript can cover several turns, so it settles all of them.
            self._untranscribed_turns.clear()
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
            if not self._agent_speaking:
                self.audible_turns += 1
                self._settle(answered=True)
            if not self._agent_speaking and self._heard_stop is not None:
                answered = self._now() - self._heard_stop
                self.replies.append(answered)
                self.reply_turns.append(self._turn(answered))
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
        elif isinstance(frame, FunctionCallCancelFrame):
            logger.info("tool {} cancelled", frame.function_name)
        elif isinstance(frame, ErrorFrame):
            self.errors += 1
            self._note_error(frame)
            logger.warning("error frame{}: {}", " (fatal)" if frame.fatal else "", frame.error)
        elif isinstance(frame, EndFrame):
            self._ended_at = self._ended_at or self._now()
            logger.info("pipeline ending: the agent closed the call")
        elif isinstance(frame, CancelFrame):
            self._ended_at = self._ended_at or self._now()
            logger.info("pipeline cancelled: the call was torn down")
        elif isinstance(frame, MetricsFrame):
            for entry in frame.data:
                if isinstance(entry, TTFBMetricsData):
                    logger.debug("ttfb {} {:.0f} ms", entry.processor, entry.value * 1000)
                elif isinstance(entry, ProcessingMetricsData):
                    logger.debug("processing {} {:.0f} ms", entry.processor, entry.value * 1000)


class ToolTrace:
    """What the agent asked of its tools, recorded by us rather than the provider.

    The framework's spans carry tool arguments and results for some providers
    only, and truncate them, so a comparison across providers cannot rest on
    them.

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
        # Called after every recorded call, so the record is rewritten whenever a
        # piece changes; ``finish_record`` rewrites it once more at the end.
        self.on_change: Any = None

    def _offset(self) -> float:
        now = time.monotonic()
        if self._origin is None:
            self._origin = now
        return round((now - self._origin) * 1000, 1)

    def record(self, name: str, arguments: dict, matched: bool, output: Any, requested_ms: float,
               resolution: str = "none", tool_call_id: str | None = None) -> None:
        self._calls.append(
            {
                "name": name,
                "tool_call_id": tool_call_id,
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

    def cancel(self, tool_call_ids: list[str]) -> None:
        """The service withdrew these calls; it is no longer waiting for them.

        A call that had already returned stays on the record -- it was made,
        and the tool ran -- but marked, because the model will ask again and a
        reader counting duplicates has to be able to tell a withdrawn call
        from a repeated one. A call withdrawn while it was still running never
        reached the record, so there is nothing to mark and the count does not
        cover it; that asymmetry is logged where it happens.

        The batch is marked in one pass because rewriting the record is not a
        flag flip -- it re-derives every published field and walks every call
        -- and a service withdraws calls several at a time.
        """
        wanted = set(tool_call_ids)
        marked = 0
        for call in self._calls:
            if call.get("tool_call_id") in wanted:
                call["cancelled"] = True
                marked += 1
        if marked and self.on_change is not None:
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
            "tool_calls_cancelled": sum(1 for call in self._calls if call.get("cancelled")),
        }


def register_tools(
    llm: LLMService, server: MockToolServer, trace: ToolTrace, hang_up: Callable[[str], None]
) -> None:
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
        trace.record(params.function_name, arguments, record.matched, result, requested, record.resolution,
                     tool_call_id=params.tool_call_id)
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
        trace.record("end_call", params.arguments or {}, True, result, requested, "exact",
                     tool_call_id=params.tool_call_id)
        await params.result_callback(result)
        hang_up("end_call")

    async def transfer_call(params: FunctionCallParams) -> None:
        requested = trace._offset()
        logger.info("transfer_call -- mock handover, closing the call")
        result = {"status": "transferred"}
        trace.record("transfer_call", params.arguments or {}, True, result, requested, "exact",
                     tool_call_id=params.tool_call_id)
        await params.result_callback(result)
        hang_up("transfer_call")

    for name in server.tool_names:
        llm.register_function(name, handler)
    llm.register_function("end_call", end_call)
    llm.register_function("transfer_call", transfer_call)

    @llm.event_handler("on_function_calls_cancelled")
    async def _withdrawn(_llm, calls):
        trace.cancel([call.tool_call_id for call in calls])


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
    window a first response is timed in.

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
        # Who decided where the caller's turns ended: the service, or the local
        # detector -- a configuration difference a reader comparing rows must see.
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
            # Whose detector decided the caller had finished *for the model*,
            # and so whose endpointing the reply figure includes. Not the same
            # as ``turn_source`` above: a service that runs its own detector
            # answers on it even where the pipeline decided the turn.
            "endpointing": provider.answers,
            # Whether the service ran its own detector over the audio as well.
            "service_vad": "on" if provider.service_vad else "off",
            # Whether a tool's result reached the model at once or after the
            # agent finished speaking.
            "result_delivery": provider.results,
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


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _common_record(server: MockToolServer, settings: Settings) -> dict[str, Any]:
    """The fields that describe the task rather than the stack.

    Identical for a native row and a cascade row by construction, which is what
    lets the two be compared: if these digests differ, the two agents were not
    given the same job and no difference between them means anything.
    """
    return {
        "bench": "agent",
        "agent_commit": _commit(),
        "agent_definition": server.suite,
        "system_prompt_sha256": _digest(server.system_prompt or ""),
        "first_message_sha256": _digest(server.first_message or ""),
        # Every tool the model can call, the two call-control tools included: a
        # reader checking why a call did not hang up needs to see that it could
        # have. Read off the schema the model is actually sent, so the record
        # cannot claim a tool set that was never declared.
        "tools": ",".join(tool.name for tool in build_tools(server).standard_tools),
        "pipecat_version": _version("pipecat-ai"),
        "cekura_version": _version("cekura"),
        "cekura_mode": settings.get("cekura_mode", "track"),
        # Whether this call was configured by the session that started it or by
        # the image it started in. One deployment answers for every provider, so
        # a row that does not say which is a row nobody can place.
        "config_source": settings.source("s2s_provider"),
        "worker_instance": INSTANCE,
        "worker_call": _calls_answered,
        # The one way a call can end without the model ending it. The same on
        # every row; see ``ClosesAfterAnUnfinishedGoodbye``.
        "hangup_backstop": HANGUP_BACKSTOP,
        # Appended to the definition's prompt for every row, so it is part of
        # the job and hashed beside it.
        "shared_rules_sha256": _digest(SHARED_RULES),
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
        "llm_reasoning": text.reasoning or "none",
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

    The figure is taken as each buffer arrives rather than when it is read.
    Caller audio stops before a call finishes -- the agent is still speaking,
    tools are still completing, the transport is still closing -- so a reading
    taken at the end would count that ordinary tail as audio that never came.

    One call at a time runs in a worker, so one accumulator is enough; it is
    reset when a call starts.
    """

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self.reset()

    def reset(self) -> None:
        self._last: float | None = None
        self._drift: float | None = None

    def add(self, seconds: float) -> None:
        now = self._now()
        elapsed = 0.0 if self._last is None else now - self._last
        self._drift = (self._drift or 0.0) + seconds - elapsed
        self._last = now

    def drift(self) -> float | None:
        """Audio received minus wall time elapsed, as of the last buffer."""
        return self._drift


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


class DeliversToolResults(LLMAssistantAggregator):
    """Keeps a tool result from being lost to a barge-in.

    A realtime service is told what a tool returned by the context frame this
    aggregator pushes upstream, and by nothing else. That push waits while the
    agent is speaking -- which is when a tool the agent narrated is most likely
    to finish, because the result frame queues behind the narration's audio in
    the output transport -- and the stock class then forgets it if the caller
    begins talking over the agent. Dropped and never retried, so the model asks
    for the same tool again on the next turn, with the same arguments, and a
    scored call gains an unmatched duplicate for each barge-in.

    The forgetting happens in two places, and both have to be covered. An
    interruption first records the cut-off utterance, which pushes a context
    frame *downstream* for observers and clears the pending flag on the way;
    only then does it reset the aggregation state. So the pending flag has to
    survive a downstream push as well as a reset. It is delivered once the
    caller's turn is over.

    Realtime rows only. On a cascade row the user half re-runs inference from
    the context at turn end, which carries the result anyway; a second push
    here would answer the same turn twice.
    """

    def __init__(self, context: LLMContext, *, deliver_immediately: bool = False, **kwargs) -> None:
        super().__init__(context, **kwargs)
        self._deliver_immediately = deliver_immediately

    async def _maybe_push_context_after_function_result(self) -> None:
        # Where the row allows it, a result the agent is still narrating over
        # is delivered now rather than after the narration: the service is
        # waiting for it, and the wait is the window in which a barge-in makes
        # the service withdraw the call. Several results still travel together.
        if (
            self._deliver_immediately
            and self._realtime_service_mode
            and self._bot_speaking
            and not self.has_queued_frame(FunctionCallResultFrame)
        ):
            logger.debug("tool result delivered while the agent is still speaking")
            await self.push_context_frame(FrameDirection.UPSTREAM)
            return
        await super()._maybe_push_context_after_function_result()

    async def reset(self):
        # A result the model has not been told about is not aggregation state,
        # so it does not belong in the sweep an interruption performs.
        pending = self._push_context_on_bot_stopped_speaking
        await super().reset()
        if self._realtime_service_mode:
            self._push_context_on_bot_stopped_speaking = pending

    async def push_context_frame(self, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        # The model is upstream. A downstream push -- an interrupted utterance
        # being recorded -- tells it nothing, so it cannot count as the delivery.
        pending = self._push_context_on_bot_stopped_speaking
        await super().push_context_frame(direction)
        if direction is not FrameDirection.UPSTREAM and self._realtime_service_mode:
            self._push_context_on_bot_stopped_speaking = pending

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            self._push_context_on_bot_stopped_speaking
            and self._realtime_service_mode
            and isinstance(frame, UserStoppedSpeakingFrame)
        ):
            logger.debug(
                "tool result outlived the turn it arrived in; delivering a context of "
                "{} message(s) now the caller has stopped", len(self._context.get_messages())
            )
            await self.push_context_frame(FrameDirection.UPSTREAM)


class BenchAggregators(LLMContextAggregatorPair):
    """The framework's pair, with an assistant half that delivers tool results."""

    def __init__(self, context: LLMContext, *, deliver_immediately: bool = False, **kwargs) -> None:
        super().__init__(context, **kwargs)
        # Built from what the pair already resolved rather than from the raw
        # arguments, so any normalising it does is not quietly lost here.
        built = self._assistant
        self._assistant = DeliversToolResults(
            context,
            deliver_immediately=deliver_immediately,
            params=built._params,
            _realtime_service_mode=built._realtime_service_mode,
            _paired_user_aggregator=self._user,
        )


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
    hangup = HangsUpOnceHeard(leave=leaves_the_room(transport))
    instructions = agent_instructions(server.system_prompt)
    name = settings.get("s2s_provider", "openai-realtime")
    cascade = TEXT_MODELS.get(name)

    if cascade is not None:
        model = settings.get("s2s_model", cascade.default_model)
        credential = _credential(cascade.credential_env, name)
        record = cascade_record(name, cascade, model, server, settings)
        # Ahead of the build on purpose: a build that fails is then the next
        # line in the log after the thing it was building.
        logger.info("building {} on {}", name, model)

        stt, llm, tts = build_cascade(cascade, credential, model, instructions, settings)
        register_tools(llm, server, trace, hangup.hang_up)
        context = LLMContext(
            [{"role": "system", "content": instructions}, *opening_messages(server.first_message)],
            tools=build_tools(server),
        )
        aggregators = BenchAggregators(
            context, user_params=user_aggregator_params(realtime=False)
        )
        # Three services where the native path has one. Everything either side of
        # them -- transport, context, tools, greeting -- is the same code.
        # ``restatements`` sits between the transcription source and the context,
        # which here means straight after the speech-to-text service: a cascade
        # transcript travels *downstream*, so anything after the aggregator would
        # see it only once the aggregator had already appended it.
        stages = [transport.input(), stt, restatements, aggregators.user(), llm, tts, transport.output(),
                  hangup, DropControlTokens(), aggregators.assistant()]
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

        llm = provider.build(credential, model, voice, instructions, settings)
        register_tools(llm, server, trace, hangup.hang_up)
        context = LLMContext(opening_messages(server.first_message), tools=build_tools(server))
        aggregators = BenchAggregators(
            context,
            realtime_service_mode=True,
            deliver_immediately=provider.results == "immediate",
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
                  hangup, DropControlTokens(), aggregators.assistant()]
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
    capture_speech_tokens(llm, meter)
    narrator = CallNarrator(AUDIO_CLOCK, hangup)
    backstop = ClosesAfterAnUnfinishedGoodbye(hangup.hang_up)
    _on_event(llm, "on_delegation_created", lambda *_: backstop.delegated())
    observers = [meter, narrator, backstop]
    task, tracer = create_task(pipeline, context, params, runner_args, transport, record, observers)
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
        backstop.agent_said(str(message.content or ""), message.interrupted)

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
            "call summary: {:.0f}s, caller turns {}, agent responses {} ({} heard), barge-ins {}, "
            "tools {} ({} exact), errors {}, usage reports {}, log lines {}{}{}",
            usage["call_seconds"], narrator.caller_turns, narrator.agent_turns, narrator.audible_turns,
            narrator.barge_ins,
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

    _on_event(transport, "on_joined", lambda *_: logger.info("transport: joined"))
    _on_event(transport, "on_left", lambda *_: logger.info("transport: left"))
    _on_event(transport, "on_error", lambda _t, error: logger.warning("transport error: {}", error))
    _on_event(
        transport, "on_participant_joined",
        lambda _t, participant: logger.info("transport: participant joined {}", _participant_id(participant)),
    )
    _on_event(
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
        narrator.caller_connected()
        backstop.start()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("caller disconnected; tearing the call down")
        publish()
        await task.cancel()

    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        await backstop.stop()
        if tracer is None:
            # Nobody shipped the log; say where the call ended up anyway.
            summarise()
        call_log.close()
        logger.info("call {} finished", session_id)


def _participant_id(participant: Any) -> str:
    if isinstance(participant, dict):
        return str(participant.get("id") or participant.get("info", {}).get("userName") or "?")
    return str(getattr(participant, "identity", None) or participant)


def _on_event(source: Any, event: str, handler: Callable) -> None:
    """Handle an event where the transport or service has it; not every one does."""
    async def _handle(*args, **kwargs):
        handler(*args, **kwargs)

    # Asked first rather than tried: the framework answers an unknown event
    # with a warning in the log, and five of those on every telephony call would
    # bury the lines this exists to add.
    if event in getattr(source, "_event_handlers", {}):
        source.add_event_handler(event, _handle)


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
    rewritten, then the snapshot proceeds.
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


def capture_speech_tokens(llm: LLMService, meter: UsageMeter) -> None:
    """Keep the speech and text split a provider reports and the framework adds up.

    One of these services bills speech tokens at many times the rate of text
    tokens and reports the two apart, as running totals on every usage event.
    The framework turns each event into a single input and output count, so a
    record holding only those cannot be priced: nobody can say how much of it
    was speech. The running totals are read off the same event before it is
    handled, which is every other part of it untouched.

    As with the live-audio seconds there is no public surface, so the handler is
    wrapped and still called. A service without it is left alone.
    """
    original = getattr(llm, "_handle_usage_event", None)
    if original is None:
        return

    async def handle(event_json: dict):
        total = ((event_json.get("usageEvent") or {}).get("details") or {}).get("total") or {}
        if total:
            meter.record_speech_tokens(
                (total.get("input") or {}).get("speechTokens", 0),
                (total.get("output") or {}).get("speechTokens", 0),
            )
        return await original(event_json)

    llm._handle_usage_event = handle


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


def cekura_agent_id(definition: str | None) -> str | None:
    """The platform agent this call's record is filed under.

    Each agent definition is its own platform agent, because the platform files
    a traced session under the agent id the tracer sends and a run looks only
    under its own agent's id: a medicare call reported as the appointments
    agent would run, and then come back to its run with no transcript and no
    record. So the id follows the definition that was loaded --
    ``CEKURA_AGENT_ID_<DEFINITION>`` -- and ``CEKURA_AGENT_ID`` answers only
    for a deployment that serves one definition.
    """
    if definition:
        specific = os.getenv(f"CEKURA_AGENT_ID_{definition.upper().replace('-', '_')}")
        if specific:
            return specific
    return os.getenv("CEKURA_AGENT_ID")


def create_task(pipeline, context, params, runner_args, transport, record, observers) -> tuple[PipelineTask, Any]:
    """Wrap the pipeline in Cekura tracing when credentials are present.

    Tracing is what makes an agent-bench run inspectable afterwards: transcripts, tool
    calls, logs and spans land against the run rather than in a container's
    stdout. Without credentials the agent still runs and still answers the phone,
    it is simply not observed -- a missing key must not be the reason a benchmark
    call fails.
    """
    definition = record.get("agent_definition")
    api_key, agent_id = os.getenv("CEKURA_API_KEY"), cekura_agent_id(definition)
    if not (api_key and agent_id):
        missing = "CEKURA_API_KEY" if not api_key else f"the agent id for {definition}"
        logger.info("Cekura tracing off: {} unset", missing)
        return PipelineTask(pipeline, params=params, observers=observers), None

    try:
        agent_id = int(agent_id)
        from cekura.pipecat import PipecatTracer

        tracer = PipecatTracer(
            api_key=api_key,
            agent_id=agent_id,
            host=os.getenv("CEKURA_HOST", "https://api.cekura.ai"),
            # Off for a local run against a local receiver, where the span
            # exporter would otherwise retry against a host it cannot reach for
            # the whole call and then hold the finalisation for its timeout.
            enable_otel_traces=os.getenv("CEKURA_OTEL_TRACES", "1").lower() not in ("0", "false", "no"),
        )
        # On the record itself, which is sent again whole when the call ends.
        record["cekura_agent_id"] = agent_id
        metadata = dict(record)
        logger.info("Cekura tracing on: {} reports to platform agent {}", definition, agent_id)
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
    for module in [*modules, "cekura.pipecat"]:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 -- the builder will raise this again, in context
            logger.warning("could not pre-import {}: {}", module, exc)

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
