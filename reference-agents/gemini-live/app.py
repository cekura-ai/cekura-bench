"""Twilio Media Streams <-> Gemini Live adapter."""
import asyncio
import audioop
import base64
from collections import deque
import hashlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from google import genai
from google.genai import types
from google.genai import _common, _transformers
from twilio.request_validator import RequestValidator

logging.basicConfig(level=logging.INFO, format="%(message)s")
app = FastAPI()
ROOT = Path(__file__).parent / "definitions"
MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live-extended-thinking")
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "HIGH")
LIVE_API_VERSION = "v1alpha"
MOCK_AGENT_ID = 20909
CEKURA_BASE = "https://new-prod.cekura.ai"
CEKURA_TRANSCRIPT_URL = "https://api.cekura.ai/test_framework/custom-provider-transcript-webhook"
CEKURA_AGENT_IDS = {"appointments": 23484, "insurance": 23485}
FRAME_BYTES = 160
OUTPUT_PCM_MIME = re.compile(r"^audio/pcm;rate=(\d+)$", re.IGNORECASE)
# Keep hybrid VAD off pending a controlled audio-path test. Google's generic
# Live docs support audio_stream_end with automatic VAD; one failed experiment
# with several simultaneous changes does not establish model incompatibility.
HYBRID_VAD_ENABLED = os.environ.get("HYBRID_VAD_ENABLED", "false").lower() == "true"
VAD_SPEECH_RMS = int(os.environ.get("VAD_SPEECH_RMS", "100"))
VAD_SILENCE_FRAMES = int(os.environ.get("VAD_SILENCE_FRAMES", "40"))
VAD_PRE_ROLL_FRAMES = int(os.environ.get("VAD_PRE_ROLL_FRAMES", "10"))
AUDIO_STREAM_END = object()


def log(call_sid: str, stage: str, **details: Any) -> None:
    logging.info(json.dumps({"source": "gemini-live-cloud-run", "call_sid": call_sid, "stage": stage, "monotonic_ms": round(time.monotonic() * 1000), **details}, default=str))


def normalize_number(number: str | None) -> str:
    return "".join(c for c in number or "" if c.isdigit())


def config_hash(workflow: dict[str, Any]) -> str:
    raw = workflow["prompt"] + workflow["greeting"] + json.dumps(workflow["tools"], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def workflow_for(number: str | None) -> dict[str, Any]:
    key = {"19715717785": "appointments", "19713912400": "insurance"}.get(normalize_number(number))
    if not key:
        raise ValueError("unknown destination number")
    root = ROOT / key
    tools = json.loads((root / "tool-definitions.json").read_text())
    if len(tools) != 4 or any(not x.get("name") or not x.get("parameters") for x in tools):
        raise ValueError("invalid canonical tool configuration")
    prompt = (root / "system-prompt.txt").read_text()
    return {"key": key, "prompt": prompt, "greeting": (root / "first-message.txt").read_text().strip(), "tools": tools}


def declarations(tools: list[dict[str, Any]]) -> list[types.FunctionDeclaration]:
    return [types.FunctionDeclaration(name=x["name"], description=x["description"], parameters=x["parameters"], behavior="NON_BLOCKING") for x in tools]


def live_config(workflow: dict[str, Any]) -> types.LiveConnectConfig:
    """Unsupported SDK fields must fail loudly here and in readiness."""
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        thinking_config=types.ThinkingConfig(
            thinking_level=THINKING_LEVEL,
            include_thoughts=False,
        ),
        system_instruction=types.Content(parts=[types.Part(text=workflow["prompt"])]),
        tools=[types.Tool(function_declarations=declarations(workflow["tools"]))],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


def public_base() -> str:
    value = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    if not value.startswith("https://"):
        raise RuntimeError("PUBLIC_BASE_URL must be the fixed https Cloud Run service URL")
    return value


def twilio_validator() -> RequestValidator:
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not token:
        raise RuntimeError("TWILIO_AUTH_TOKEN is missing")
    return RequestValidator(token)


def verify_http(request: Request, form: dict[str, str]) -> None:
    signature = request.headers.get("X-Twilio-Signature", "")
    url = public_base() + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    if not signature or not twilio_validator().validate(url, form, signature):
        raise HTTPException(status_code=403, detail="invalid Twilio signature")


def verify_websocket(ws: WebSocket) -> bool:
    signature = ws.headers.get("x-twilio-signature", "")
    url = public_base().replace("https://", "wss://", 1) + ws.url.path
    if ws.url.query:
        url += "?" + ws.url.query
    return bool(signature and twilio_validator().validate(url, {}, signature))


def twiml(stream_url: str, caller: str, destination: str) -> str:
    def esc(value: str) -> str:
        return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="{esc(stream_url)}"><Parameter name="callerNumber" value="{esc(caller)}"/><Parameter name="destinationNumber" value="{esc(destination)}"/></Stream></Connect></Response>'


@app.get("/api/health")
async def health() -> dict[str, Any]:
    try:
        workflows = []
        for number in ("19715717785", "19713912400"):
            workflow = workflow_for(number)
            live_config(workflow).model_dump(exclude_none=True)
            workflows.append({"name": workflow["key"], "config_sha256": config_hash(workflow), "tools": [x["name"] for x in workflow["tools"]]})
        return {"ok": True, "model": MODEL, "thinking_level": THINKING_LEVEL, "api_version": LIVE_API_VERSION, "sdk_config_valid": True, "twilio_signature_validation": bool(os.environ.get("TWILIO_AUTH_TOKEN")), "workflows": workflows}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/twilio/voice")
async def voice(request: Request) -> Response:
    form = {key: value for key, value in (await request.form()).items()}
    verify_http(request, form)
    try:
        workflow_for(form.get("To"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stream_url = public_base().replace("https://", "wss://", 1) + "/api/gemini/stream"
    return Response(twiml(stream_url, form.get("From", ""), form.get("To", "")), media_type="text/xml")


@dataclass
class CallState:
    call_sid: str = "unknown"
    stream_sid: str = ""
    workflow: dict[str, Any] | None = None
    input_q: asyncio.Queue[Any] = field(default_factory=lambda: asyncio.Queue(maxsize=250))
    output_q: asyncio.Queue[tuple[int, dict[str, Any]]] = field(default_factory=lambda: asyncio.Queue(maxsize=500))
    generation: int = 0
    input_rate_state: Any = None
    output_rate_state: Any = None
    output_buffer: bytearray = field(default_factory=bytearray)
    received_tool_ids: set[str] = field(default_factory=set)
    executing_tool_ids: set[str] = field(default_factory=set)
    response_sent_tool_ids: set[str] = field(default_factory=set)
    failed_tool_ids: set[str] = field(default_factory=set)
    cancelled_tool_ids: set[str] = field(default_factory=set)
    pending_tool_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    tool_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    worker_failure: asyncio.Future[BaseException] | None = None
    media_frames_in: int = 0
    media_frames_out: int = 0
    last_media_timestamp: str | None = None
    last_media_monotonic: float | None = None
    input_rms_min: int | None = None
    input_rms_max: int = 0
    input_rms_total: int = 0
    input_rms_samples: int = 0
    closing: bool = False
    playback_mark_number: int = 0
    input_stream_open: bool = False
    input_silence_frames: int = 0
    input_pre_roll: deque[bytes] = field(default_factory=lambda: deque(maxlen=VAD_PRE_ROLL_FRAMES))
    started: bool = False
    session_id: str = field(default_factory=lambda: f"gemini-live-{uuid.uuid4()}")
    started_wall: datetime | None = None
    caller_number: str = ""
    destination_number: str = ""
    cekura_run_id: int | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def elapsed_seconds(self) -> float:
        if self.started_wall is None:
            return 0.0
        return round((datetime.now(timezone.utc) - self.started_wall).total_seconds(), 3)

    def transcript_entry(self, role: str, *, content: str | None = None, data: dict[str, Any] | None = None) -> None:
        timestamp = self.elapsed_seconds()
        # Output transcription is streamed in short fragments. Preserve the
        # actual utterance rather than making every partial fragment a turn.
        if content and content.strip().lower().replace(" ", "") in {"<nospeech>", "nospeech"}:
            return
        if content and data is None and self.transcript:
            previous = self.transcript[-1]
            if (previous.get("role") == role and "data" not in previous
                    and timestamp - float(previous.get("end_time", 0)) <= 1.5):
                previous["content"] = f"{previous.get('content', '').rstrip()} {content.lstrip()}".strip()
                previous["end_time"] = timestamp
                return
        entry: dict[str, Any] = {"role": role, "start_time": timestamp, "end_time": timestamp}
        if content:
            entry["content"] = content
        if data is not None:
            entry["data"] = data
        self.transcript.append(entry)

    async def queue_output(self, payload: dict[str, Any]) -> None:
        try:
            self.output_q.put_nowait((self.generation, payload))
        except asyncio.QueueFull:
            raise RuntimeError("outbound audio queue overflow")

    async def flush_output_audio(self, *, pad: bool) -> None:
        if not self.output_buffer:
            return
        if pad:
            self.output_buffer.extend(b"\xff" * (FRAME_BYTES - len(self.output_buffer)))
        if len(self.output_buffer) < FRAME_BYTES:
            return
        frame = bytes(self.output_buffer[:FRAME_BYTES])
        del self.output_buffer[:FRAME_BYTES]
        await self.queue_output({"event": "media", "streamSid": self.stream_sid, "media": {"payload": base64.b64encode(frame).decode()}})

    def add_input_energy(self, pcm16: bytes) -> int:
        rms = audioop.rms(pcm16, 2)
        self.input_rms_min = rms if self.input_rms_min is None else min(self.input_rms_min, rms)
        self.input_rms_max = max(self.input_rms_max, rms)
        self.input_rms_total += rms
        self.input_rms_samples += 1
        return rms

    def take_input_energy(self) -> dict[str, int | None]:
        samples = self.input_rms_samples
        values = {
            "input_rms_min": self.input_rms_min,
            "input_rms_max": self.input_rms_max if samples else None,
            "input_rms_mean": round(self.input_rms_total / samples) if samples else None,
        }
        self.input_rms_min, self.input_rms_max = None, 0
        self.input_rms_total, self.input_rms_samples = 0, 0
        return values

    def classify_input_frame(self, pcm16: bytes) -> tuple[list[bytes], bool, str | None]:
        """Retain pre-roll and send an ordered end boundary after 800ms of silence."""
        if not HYBRID_VAD_ENABLED:
            return [pcm16], False, None
        rms = audioop.rms(pcm16, 2)
        self.input_pre_roll.append(pcm16)
        if rms >= VAD_SPEECH_RMS:
            self.input_silence_frames = 0
            if not self.input_stream_open:
                self.input_stream_open = True
                chunks = list(self.input_pre_roll)
                self.input_pre_roll.clear()
                return chunks, False, "speech_start"
            return [pcm16], False, None
        if not self.input_stream_open:
            return [], False, None
        self.input_silence_frames += 1
        if self.input_silence_frames >= VAD_SILENCE_FRAMES:
            self.input_stream_open = False
            self.input_silence_frames = 0
            return [pcm16], True, "speech_end"
        return [pcm16], False, None


def serialize_tool_response(function_responses: list[types.FunctionResponse]) -> dict[str, Any]:
    """Return the exact Live SDK WebSocket envelope for auditable logging.

    This mirrors google.genai.live.AsyncSession.send_tool_response in the
    installed, pinned SDK before that method serializes the same envelope.
    """
    tool_response = _transformers.t_tool_response(function_responses)
    payload = _common.convert_to_dict(tool_response, convert_keys=True)
    for response in payload.get("functionResponses", []):
        if response.get("id") is None:
            raise ValueError("FunctionResponse requires a model-issued id")
    return {"tool_response": payload}


def supervise_task(state: CallState, task: asyncio.Task[Any], name: str) -> None:
    """Make worker failure visible to the WebSocket owner immediately."""
    def done(completed: asyncio.Task[Any]) -> None:
        if completed.cancelled():
            return
        try:
            exc = completed.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            stage = "worker_failed" if not state.closing else "worker_exit_during_shutdown"
            log(state.call_sid, stage, worker=name, error=repr(exc))
            if not state.closing and state.worker_failure is not None and not state.worker_failure.done():
                state.worker_failure.set_result(exc)
    task.add_done_callback(done)


async def stop_call_workers(state: CallState, workers: list[asyncio.Task[Any]]) -> None:
    """Stop all call workers while the Live connection is still usable."""
    state.closing = True
    tasks = [*workers, *state.tool_tasks]
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def call_tool(workflow: dict[str, Any], name: str, args: dict[str, Any]) -> Any:
    if name not in {tool["name"] for tool in workflow["tools"]}:
        raise ValueError("unconfigured tool")
    url = f"https://new-prod.cekura.ai/test_framework/v1/aiagents/{MOCK_AGENT_ID}/tool/{name}/"
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
        response = await client.post(url, json=args)
        response.raise_for_status()
        return response.json()


async def resolve_cekura_run(state: CallState) -> int | None:
    """Associate only a unique active Cekura caller-number run; never guess."""
    if not state.workflow or not state.caller_number:
        return None
    key = os.environ.get("CEKURA_DEMO_API_KEY", "")
    agent_id = CEKURA_AGENT_IDS[state.workflow["key"]]
    if not key:
        return None
    caller = normalize_number(state.caller_number)
    destination = normalize_number(state.destination_number)
    if not caller:
        return None
    url = f"{CEKURA_BASE}/test_framework/v1/runs/?agent_id={agent_id}&page_size=100"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
            response = await client.get(url, headers={"X-CEKURA-API-KEY": key})
            response.raise_for_status()
            rows = response.json().get("results", [])
        matches = [row for row in rows if isinstance(row, dict)
                   and str(row.get("status", "")).lower() in {"running", "in_progress"}
                   and normalize_number(row.get("inbound_number")) == caller
                   and (not destination or normalize_number(row.get("agent_number")) == destination)]
        ids = {int(row["id"]) for row in matches if str(row.get("id", "")).isdigit()}
        if len(ids) == 1:
            return ids.pop()
        if ids:
            log(state.call_sid, "cekura_run_association_ambiguous", candidates=sorted(ids))
    except Exception as exc:
        log(state.call_sid, "cekura_run_association_failed", error=repr(exc))
    return None


async def publish_cekura_transcript(state: CallState) -> None:
    """Best-effort post-call evidence sink; cannot affect call teardown."""
    if not state.workflow or not state.transcript:
        return
    key = os.environ.get("CEKURA_DEMO_API_KEY", "")
    if not key:
        log(state.call_sid, "cekura_transcript_publish_skipped", reason="missing_api_key")
        return
    now = datetime.now(timezone.utc)
    payload = build_cekura_transcript_payload(state, now)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
            response = await client.post(CEKURA_TRANSCRIPT_URL, headers={"X-CEKURA-API-KEY": key}, json=payload)
            response.raise_for_status()
        log(state.call_sid, "cekura_transcript_published", run_id=state.cekura_run_id,
            entries=len(state.transcript), status=response.status_code)
    except Exception as exc:
        log(state.call_sid, "cekura_transcript_publish_failed", run_id=state.cekura_run_id, error=repr(exc))


def build_cekura_transcript_payload(state: CallState, ended_at: datetime) -> dict[str, Any]:
    """Pure custom-provider-webhook transform; timestamps are seconds from start."""
    if not state.workflow:
        raise ValueError("workflow required for transcript payload")
    call: dict[str, Any] = {
        "id": state.session_id,
        "startedAt": (state.started_wall or ended_at).isoformat(),
        "endedAt": ended_at.isoformat(),
        "messages": sorted(state.transcript, key=lambda item: item["start_time"]),
        "from_phone_number": state.caller_number,
        "endedReason": "twilio_stream_ended",
        "metadata": {"source": "gemini-live-cloud-run", "twilio_call_sid": state.call_sid},
    }
    if state.cekura_run_id is not None:
        call["run_id"] = state.cekura_run_id
    return {"agent_id": CEKURA_AGENT_IDS[state.workflow["key"]], "calls": [call]}


async def invoke_tool(state: CallState, session: Any, fc: Any) -> None:
    tool_id, name, arguments = fc.id, fc.name, fc.args or {}
    if not tool_id or tool_id in state.received_tool_ids or tool_id in state.cancelled_tool_ids:
        log(state.call_sid, "gemini_tool_duplicate_or_missing_id", tool_id=tool_id, name=name)
        return
    state.received_tool_ids.add(tool_id)
    state.executing_tool_ids.add(tool_id)
    try:
        log(state.call_sid, "gemini_tool_call_received", tool_id=tool_id, name=name, arguments=arguments)
        state.transcript_entry("function_call", data={"id": tool_id, "name": name, "arguments": arguments})
        result = await call_tool(state.workflow, name, arguments)  # type: ignore[arg-type]
        log(state.call_sid, "canonical_tool_succeeded", tool_id=tool_id, name=name, result=result)
    except asyncio.CancelledError:
        log(state.call_sid, "canonical_tool_cancelled", tool_id=tool_id, name=name)
        raise
    except Exception as exc:
        state.failed_tool_ids.add(tool_id)
        result = {"error": str(exc)}
        log(state.call_sid, "canonical_tool_failed", tool_id=tool_id, name=name, error=str(exc))
    if tool_id in state.cancelled_tool_ids:
        log(state.call_sid, "gemini_tool_response_suppressed_cancelled", tool_id=tool_id, name=name)
        return
    try:
        # Extended Thinking supports NON_BLOCKING calls but expressly forbids
        # scheduling configurations. The response body remains opaque tool data.
        response_body = {"output": result}
        response_obj = types.FunctionResponse(id=tool_id, name=name, response=response_body)
        envelope = serialize_tool_response([response_obj])
        log(state.call_sid, "gemini_tool_response_envelope", tool_id=tool_id, name=name, envelope=envelope)
        await session.send_tool_response(function_responses=[response_obj])
        state.response_sent_tool_ids.add(tool_id)
        state.transcript_entry("function_call_result", data={"id": tool_id, "name": name, "result": result})
        log(state.call_sid, "gemini_tool_response_sent", tool_id=tool_id, name=name)
    except Exception as exc:
        log(state.call_sid, "gemini_tool_response_send_failed", tool_id=tool_id, name=name, error=repr(exc))
        raise
    finally:
        state.executing_tool_ids.discard(tool_id)


async def receive_gemini(state: CallState, session: Any) -> None:
    # The iterator ends at a turn boundary; re-enter it until the call ends.
    while True:
        async for message in session.receive():
            sc = message.server_content
            log(state.call_sid, "gemini_message_received",
                has_server_content=bool(sc), has_tool_call=bool(message.tool_call),
                has_tool_cancellation=bool(message.tool_call_cancellation),
                has_go_away=bool(message.go_away), has_resumption=bool(message.session_resumption_update),
                turn_complete=bool(sc and sc.turn_complete), generation_complete=bool(sc and sc.generation_complete),
                interaction_status=str(sc.interaction_status) if sc and sc.interaction_status else None)
            if message.go_away:
                raise RuntimeError("Gemini sent goAway")
            if message.tool_call_cancellation:
                for tool_id in message.tool_call_cancellation.ids or []:
                    state.cancelled_tool_ids.add(tool_id)
                    pending = state.pending_tool_tasks.get(tool_id)
                    if pending and not pending.done():
                        pending.cancel()
                    log(state.call_sid, "gemini_tool_cancelled", tool_id=tool_id)
            if sc and sc.interrupted:
                state.generation += 1
                state.output_buffer.clear()
                await state.queue_output({"event": "clear", "streamSid": state.stream_sid})
                log(state.call_sid, "gemini_interrupted_clear_queued", generation=state.generation)
            if sc and sc.interaction_status:
                log(state.call_sid, "gemini_interaction_status", status=str(sc.interaction_status))
            if sc and sc.input_transcription and sc.input_transcription.text:
                log(state.call_sid, "gemini_input_transcription", text=sc.input_transcription.text)
                state.transcript_entry("user", content=sc.input_transcription.text)
            if sc and sc.output_transcription and sc.output_transcription.text:
                log(state.call_sid, "gemini_output_transcription", text=sc.output_transcription.text)
                state.transcript_entry("bot", content=sc.output_transcription.text)
            if sc and sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data:
                        log(state.call_sid, "gemini_audio_received", bytes=len(part.inline_data.data), mime_type=part.inline_data.mime_type)
                        match = OUTPUT_PCM_MIME.fullmatch(part.inline_data.mime_type or "")
                        if not match:
                            raise RuntimeError(f"unsupported Gemini output audio MIME type: {part.inline_data.mime_type!r}")
                        source_rate = int(match.group(1))
                        pcm8, state.output_rate_state = audioop.ratecv(part.inline_data.data, 2, 1, source_rate, 8000, state.output_rate_state)
                        state.output_buffer.extend(audioop.lin2ulaw(pcm8, 2))
                        while len(state.output_buffer) >= FRAME_BYTES:
                            frame = bytes(state.output_buffer[:FRAME_BYTES])
                            del state.output_buffer[:FRAME_BYTES]
                            await state.queue_output({"event": "media", "streamSid": state.stream_sid, "media": {"payload": base64.b64encode(frame).decode()}})
            if sc and sc.turn_complete:
                await state.flush_output_audio(pad=True)
                state.playback_mark_number += 1
                await state.queue_output({"event": "mark", "streamSid": state.stream_sid, "mark": {"name": f"gemini-{state.generation}-{state.playback_mark_number}"}})
                log(state.call_sid, "twilio_playback_mark_queued", generation=state.generation, mark=f"gemini-{state.generation}-{state.playback_mark_number}")
            if message.tool_call:
                for fc in message.tool_call.function_calls:
                    task = asyncio.create_task(invoke_tool(state, session, fc))
                    state.tool_tasks.add(task)
                    supervise_task(state, task, f"tool:{fc.name}:{fc.id}")
                    if fc.id:
                        state.pending_tool_tasks[fc.id] = task
                    task.add_done_callback(state.tool_tasks.discard)
                    task.add_done_callback(lambda _, tool_id=fc.id: state.pending_tool_tasks.pop(tool_id, None))


async def send_gemini(state: CallState, session: Any) -> None:
    while True:
        chunk = await state.input_q.get()
        if chunk is None:
            return
        if chunk is AUDIO_STREAM_END:
            await session.send_realtime_input(audio_stream_end=True)
            log(state.call_sid, "gemini_audio_stream_end_sent")
            continue
        await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))
        if state.media_frames_in % 50 == 0:
            log(state.call_sid, "gemini_audio_sent", frames=state.media_frames_in, queue_depth=state.input_q.qsize(), bytes=len(chunk))


async def send_twilio(ws: WebSocket, state: CallState) -> None:
    while True:
        generation, payload = await state.output_q.get()
        if payload["event"] == "media" and generation != state.generation:
            log(state.call_sid, "twilio_stale_audio_dropped", generation=generation, current_generation=state.generation)
            continue
        await ws.send_text(json.dumps(payload))
        if payload["event"] == "clear":
            log(state.call_sid, "twilio_clear_sent", generation=generation)
        elif payload["event"] == "mark":
            log(state.call_sid, "twilio_playback_mark_sent", generation=generation, mark=payload["mark"]["name"])
        elif payload["event"] == "media":
            state.media_frames_out += 1
            if state.media_frames_out % 50 == 0:
                log(state.call_sid, "twilio_audio_sent", frames=state.media_frames_out, queue_depth=state.output_q.qsize())


@app.websocket("/api/gemini/stream")
async def stream(ws: WebSocket) -> None:
    state = CallState()
    workers: list[asyncio.Task[Any]] = []
    twilio_receive_task: asyncio.Task[str] | None = None
    try:
        if not verify_websocket(ws):
            await ws.close(code=1008)
            return
        await ws.accept()
        connected = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=15))
        if connected.get("event") != "connected":
            raise ValueError("expected Twilio connected event")
        start = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=15))
        if start.get("event") != "start":
            raise ValueError("expected Twilio start event")
        meta = start.get("start", {})
        state.call_sid = meta.get("callSid", "unknown")
        state.stream_sid = meta.get("streamSid", "")
        custom = meta.get("customParameters", {})
        state.caller_number = custom.get("callerNumber") or meta.get("from", "")
        state.destination_number = custom.get("destinationNumber") or meta.get("to", "")
        state.workflow = workflow_for(state.destination_number)
        if not state.stream_sid:
            raise ValueError("missing Twilio stream SID")
        state.started = True
        state.started_wall = datetime.now(timezone.utc)
        state.cekura_run_id = await resolve_cekura_run(state)
        log(state.call_sid, "twilio_start_accepted", workflow=state.workflow["key"], config_sha256=config_hash(state.workflow), model=MODEL, cekura_run_id=state.cekura_run_id)
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": LIVE_API_VERSION})
        async with client.aio.live.connect(model=MODEL, config=live_config(state.workflow)) as session:
            loop = asyncio.get_running_loop()
            state.worker_failure = loop.create_future()
            worker_specs = [
                ("gemini_sender", send_gemini(state, session)),
                ("gemini_receiver", receive_gemini(state, session)),
                ("twilio_sender", send_twilio(ws, state)),
            ]
            workers = [asyncio.create_task(coro, name=name) for name, coro in worker_specs]
            for task, (name, _) in zip(workers, worker_specs):
                supervise_task(state, task, name)
            await session.send_client_content(turns=[types.Content(role="user", parts=[types.Part(text=f"Begin the phone call. Say exactly: {state.workflow['greeting']}")])], turn_complete=True)
            while True:
                twilio_receive_task = asyncio.create_task(ws.receive_text(), name="twilio_receiver")
                failure_waiter = state.worker_failure
                done, pending = await asyncio.wait({twilio_receive_task, failure_waiter}, return_when=asyncio.FIRST_COMPLETED)
                if failure_waiter in done:
                    if not twilio_receive_task.done():
                        twilio_receive_task.cancel()
                    exc = failure_waiter.result()
                    await stop_call_workers(state, workers)
                    workers.clear()
                    raise RuntimeError(f"call worker failed: {exc!r}") from exc
                try:
                    data = json.loads(twilio_receive_task.result())
                except WebSocketDisconnect:
                    await stop_call_workers(state, workers)
                    workers.clear()
                    raise
                twilio_receive_task = None
                event = data.get("event")
                if event == "media":
                    mulaw = base64.b64decode(data["media"]["payload"], validate=True)
                    pcm8 = audioop.ulaw2lin(mulaw, 2)
                    pcm16, state.input_rate_state = audioop.ratecv(pcm8, 2, 1, 8000, 16000, state.input_rate_state)
                    state.media_frames_in += 1
                    state.add_input_energy(pcm16)
                    now = time.monotonic()
                    wall_gap_ms = round((now - state.last_media_monotonic) * 1000) if state.last_media_monotonic else None
                    state.last_media_monotonic = now
                    state.last_media_timestamp = data.get("media", {}).get("timestamp")
                    chunks, stream_end, boundary = state.classify_input_frame(pcm16)
                    try:
                        for chunk in chunks:
                            state.input_q.put_nowait(chunk)
                        if stream_end:
                            state.input_q.put_nowait(AUDIO_STREAM_END)
                    except asyncio.QueueFull:
                        raise RuntimeError("inbound audio queue overflow")
                    if boundary:
                        log(state.call_sid, "hybrid_vad_boundary", boundary=boundary, rms=audioop.rms(pcm16, 2), pre_roll_frames=len(chunks) if boundary == "speech_start" else None)
                    if state.media_frames_in % 50 == 0:
                        log(state.call_sid, "twilio_audio_received", frames=state.media_frames_in, timestamp=state.last_media_timestamp, queue_depth=state.input_q.qsize(), bytes=len(mulaw), wall_gap_ms=wall_gap_ms, **state.take_input_energy())
                elif event == "mark":
                    log(state.call_sid, "twilio_mark_received", mark=data.get("mark", {}).get("name"))
                elif event == "stop":
                    log(state.call_sid, "twilio_stop_received")
                    break
                elif event != "connected":
                    log(state.call_sid, "twilio_event_ignored", event=event)
            await state.input_q.put(None)
            # The Live socket is still open here. Await children before the SDK
            # context closes, so no tool response is orphaned during shutdown.
            await stop_call_workers(state, workers)
            workers.clear()
    except WebSocketDisconnect:
        log(state.call_sid, "twilio_disconnected")
    except Exception as exc:
        log(state.call_sid, "stream_error", error=str(exc))
    finally:
        state.closing = True
        if twilio_receive_task is not None:
            twilio_receive_task.cancel()
        await stop_call_workers(state, workers)
        if state.started:
            await publish_cekura_transcript(state)
        with suppress(Exception):
            await ws.close()
        log(state.call_sid, "stream_finalized", started=state.started, inbound_frames=state.media_frames_in, outbound_frames=state.media_frames_out)
