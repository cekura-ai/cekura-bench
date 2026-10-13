"""Amazon Nova 2 Sonic over Bedrock's bidirectional stream, as a service-bench adapter.

Implements ``service.adapters.base.RealtimeAdapter`` for the published
benchmark entry "Amazon Nova 2 Sonic": model ``amazon.nova-2-sonic-v1:0``,
voice ``matthew``, endpointing sensitivity ``MEDIUM`` and inference config
``maxTokens 1024 / topP 0.9 / temperature 0.7`` -- the values the reference
agent (``reference-agents/pipecat-s2s/bot.py``, pipecat-ai 1.11.0) sends. The
region defaults to ``AWS_REGION`` from the environment / ``turn-bench/.env``,
else ``us-west-2``; it must be a region that serves the model *and* is granted
to the credentials (the two failures look the same from outside).

The wire is ``InvokeModelWithBidirectionalStream`` through AWS's Bedrock
runtime SDK (``aws-sdk-bedrock-runtime``, the ``s2s-nova`` extra), the only
client for this operation; nothing else (no pipecat) sits in the measured path.
The SDK is imported lazily, so the module and its offline tests need only the
service bench. Credentials: ``api_key`` / ``AWS_ACCESS_KEY_ID``,
``AWS_SECRET_ACCESS_KEY``, optional ``AWS_SESSION_TOKEN`` and ``AWS_REGION``,
each from the constructor, the environment, or the dotenv file named by
``TURN_BENCH_ENV_FILE`` (default ``turn-bench/.env``).

Protocol quirks, each read off the reference implementation and live frames:

* **SigV4 only.** The operation is excluded from Bedrock API-key (bearer)
  authentication, so the adapter takes ``AWS_ACCESS_KEY_ID`` (as ``api_key``)
  plus ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN``. Credentials are read
  in-process and never logged; only the region is recorded.
* **Everything is a JSON event inside an event-stream chunk**, in a strict
  order: ``sessionStart`` -> ``promptStart`` (output audio format, voice,
  tools) -> one SYSTEM text content (the instructions) -> optional history text
  contents (the agent's opening line goes here as ASSISTANT history) -> one
  long-lived USER AUDIO content that every ``audioInput`` belongs to. Closing
  is the mirror image: ``contentEnd`` (audio) -> ``promptEnd`` ->
  ``sessionEnd`` -> close the input stream.
* **No configuration echo.** Bedrock never acknowledges the session config or
  the model id. ``session_ack`` records that the stream opened, the first
  server event (a ``usageEvent`` for the system prompt, ~1 s after setup) and
  the provider's ``sessionId``, which first appears on ``userSpeechStart`` /
  ``completionStart``; it says ``echoes_configuration: False``. The resolved
  model id is the requested one, recorded with the region, because a wrong id
  or an ungranted region refuses the stream outright instead of falling back.
  ``completionStart`` arrives once per session, only after the first caller
  turn; ``completionEnd`` only when the session ends.
* **Audio in at 16 kHz, out at 24 kHz, LPCM base64.** Caller audio is sent as
  continuous 20 ms ``audioInput`` events (640 bytes); a partial chunk handed
  in is held until it completes a 20 ms frame (flushed at close). Silence must
  keep flowing: the endpointer decides a turn has ended by *hearing* silence,
  and a stream with no audio is closed by the service.
* **VAD events exist on Nova 2 Sonic** (pipecat 1.4's handler predates them):
  ``userSpeechStart {inputAudioOffsetMs}`` and ``userSpeechEnd
  {inputAudioOffsetMs, inputAudioDetectionOffsetMs}``, offsets on the caller
  audio clock. They normalize to ``vad.speech.start`` / ``vad.speech.end``
  (``audio_ms``, ``detection_audio_ms``). Observed: speech end lands within
  ~50 ms of the authored turn end, detection ~160-400 ms of audio later, and
  the event arrives on the wire ~0.3-0.9 s after the audio it describes.
* **Replies** are built from sentences: per sentence an ASSISTANT TEXT content
  with ``generationStage: SPECULATIVE`` (the plan) then an AUDIO content, with
  AUDIO ``stopReason`` ``PARTIAL_TURN`` between sentences and ``END_TURN``
  after the last; ASSISTANT TEXT ``FINAL`` contents (what was actually spoken)
  follow the audio. The first SPECULATIVE content of a reply normalizes to
  ``response.start`` (the provider's commit to answer; ~0.1 s before the first
  audio frame); the AUDIO ``END_TURN`` to ``agent.audio.end`` +
  ``response.done``. FINAL text can land *after* ``END_TURN``, so a reply's
  transcript is flushed when the next reply starts or the session closes.
  Output arrives in 80 ms frames at ~1.7x realtime, so arrival timestamps are
  recorded and playout is modelled by the base ``AudioTimeline``.
* **Barge-in** is a FINAL ASSISTANT TEXT content whose text is ``{
  "interrupted" : true }`` and whose ``contentEnd`` has ``stopReason:
  INTERRUPTED``; the cut AUDIO content itself ends ``PARTIAL_TURN``. It is
  sent right after ``userSpeechStart`` and also when the reply had already
  fully *arrived* but was still playing out. Each normalizes to one
  ``agent.interrupted`` per reply (cancelling the reply if still open).
* **Transcripts.** USER TEXT (FINAL) contents are the service's ASR of the
  caller, one per detected caller utterance, lower-cased.
* **No manual commit.** Turn detection is native only; the one knob is
  ``turnDetectionConfiguration.endpointingSensitivity`` (HIGH ~1.5 s, MEDIUM
  ~1.75 s, LOW ~2 s of pause). ``silence_duration_ms`` maps onto those three
  values only; any other value, ``semantic_vad`` or a VAD threshold is declared
  unsupported up front.
* **Usage** arrives as frequent ``usageEvent`` frames with *cumulative* session
  totals (speech/text tokens, in/out). Each ``response.done`` carries the delta
  since the previous one in the framework's field names (``prompt_tokens``,
  ``input_audio_tokens``, ``completion_tokens``, ``output_audio_tokens``), so
  ``service.metrics.usage`` sums to the session total; tokens accrued after the
  last reply (streamed caller audio) are emitted at close as one more
  ``response.done`` with ``status: "session_usage"``.
* **Session lifetime** is capped at 8 minutes by the service; longer
  conversations need a fresh session (not done here).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from turn_bench.author import service_bridge

service_bridge._ensure_path()

from service import events as ev  # noqa: E402
from service.adapters.base import AdapterError, RealtimeAdapter, SessionClosed, SessionConfig  # noqa: E402

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REGION",
    "DEFAULT_VOICE",
    "NovaSonicAdapter",
    "RESPONSE_START",
    "Transport",
]

DEFAULT_MODEL = "amazon.nova-2-sonic-v1:0"
DEFAULT_VOICE = "matthew"
DEFAULT_REGION = "us-west-2"
DEFAULT_ENDPOINTING = "MEDIUM"
# The reference agent's inference settings (pipecat-ai 1.11.0 defaults, sent explicitly).
DEFAULT_INFERENCE = {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7}
# The vendor's documented pause lengths per sensitivity, for mapping silence_duration_ms.
ENDPOINTING_BY_SILENCE_MS = {1500: "HIGH", 1750: "MEDIUM", 2000: "LOW"}

CHUNK_MS = 20
CONNECT_TIMEOUT_S = 20.0
DRAIN_TIMEOUT_S = 1.5

# Normalized kind for "the provider has started a reply" (its turn-end decision).
# Not in service.events' vocabulary, which has no provider-neutral equivalent.
RESPONSE_START = "response.start"

_ENV_FILE = Path(service_bridge.REPO_ROOT) / "turn-bench" / ".env"


class Transport(Protocol):
    """The four things the adapter needs from a bidirectional stream.

    The live implementation wraps the Bedrock SDK; the offline tests plug in a
    scripted fake. Each sent/received item is one UTF-8 JSON event.
    """

    async def send(self, data: bytes) -> None: ...

    async def ready(self) -> None: ...

    async def receive(self) -> bytes | None: ...

    async def close(self) -> None: ...


class _StreamError(RuntimeError):
    pass


class _BedrockTransport:
    """``InvokeModelWithBidirectionalStream`` through aws-sdk-bedrock-runtime."""

    def __init__(self, *, model: str, region: str, access_key_id: str, secret_access_key: str, session_token: str | None):
        self._args = (model, region, access_key_id, secret_access_key, session_token)
        self._stream: Any = None
        self._output: Any = None

    async def open(self) -> None:
        try:
            from aws_sdk_bedrock_runtime.client import (
                BedrockRuntimeClient,
                InvokeModelWithBidirectionalStreamOperationInput,
            )
            from aws_sdk_bedrock_runtime.config import Config
            from smithy_aws_core.identity.static import StaticCredentialsResolver
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the extra
            raise AdapterError("nova-sonic needs the s2s-nova extra: uv sync --extra s2s-nova") from exc
        model, region, access_key_id, secret_access_key, session_token = self._args
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
            region=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
            aws_credentials_identity_resolver=StaticCredentialsResolver(),
        )
        client = BedrockRuntimeClient(config=config)
        self._stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=model)
        )

    async def send(self, data: bytes) -> None:
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )

        await self._stream.input_stream.send(
            InvokeModelWithBidirectionalStreamInputChunk(value=BidirectionalInputPayloadPart(bytes_=data))
        )

    async def ready(self) -> None:
        _, self._output = await self._stream.await_output()

    async def receive(self) -> bytes | None:
        from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamOutputChunk

        if self._output is None:
            await self.ready()
        result = await self._output.receive()
        if result is None:
            return None
        if isinstance(result, InvokeModelWithBidirectionalStreamOutputChunk):
            return (result.value.bytes_ if result.value else None) or b""
        detail = getattr(getattr(result, "value", None), "message", None) or repr(result)
        raise _StreamError(f"{type(result).__name__}: {detail}")

    async def close(self) -> None:
        if self._stream is not None:
            try:
                await self._stream.input_stream.close()
            except Exception:  # noqa: BLE001 - closing a stream the service already ended
                pass


def _dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _setting(name: str, explicit: str | None = None) -> str | None:
    """Explicit argument, then the environment, then the dotenv file
    (``TURN_BENCH_ENV_FILE`` if set, else ``turn-bench/.env``). Never logged."""
    if explicit:
        return explicit
    if os.environ.get(name):
        return os.environ[name]
    env_file = Path(os.environ["TURN_BENCH_ENV_FILE"]) if os.environ.get("TURN_BENCH_ENV_FILE") else _ENV_FILE
    return _dotenv(env_file).get(name) or None


class NovaSonicAdapter(RealtimeAdapter):
    name = "nova-sonic"
    input_rate = 16000
    output_rate = 24000
    supports_manual_commit = False
    supports_text_modality = True
    emits_vad_events = True

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
        region: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        endpointing_sensitivity: str | None = None,
        transport_factory: Callable[[], Awaitable[Transport]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.region = _setting("AWS_REGION", region) or DEFAULT_REGION
        self._secret = aws_secret_access_key
        self._session_token = aws_session_token
        self._endpointing_override = endpointing_sensitivity
        self._transport_factory = transport_factory
        self._transport: Transport | None = None
        self._prompt_name = str(uuid.uuid4())
        self._audio_content = str(uuid.uuid4())
        self._audio_open = False
        self._pending = bytearray()           # caller audio short of one 20 ms frame
        self._chunk_bytes = self.input_rate * CHUNK_MS // 1000 * 2
        self._content: dict[str, Any] | None = None   # the content block being received
        self._responding = False
        self._interrupted = False
        self._agent_final: list[str] = []
        self._usage_total: dict[str, int] = {}
        self._usage_reported: dict[str, int] = {}
        self._send_lock = asyncio.Lock()
        self.resolved_model: str | None = None

    # ── configuration ────────────────────────────────────────────────────────

    @classmethod
    def unsupported_reason(cls, config: SessionConfig) -> str | None:
        reason = super().unsupported_reason(config)
        if reason:
            return reason
        detection = config.turn_detection
        if detection.mode == "semantic_vad":
            return "nova-sonic has no semantic turn detection mode; it exposes endpointing sensitivity only"
        if detection.threshold is not None or detection.prefix_padding_ms is not None:
            return "nova-sonic exposes no VAD threshold or prefix padding"
        if detection.silence_duration_ms is not None and detection.silence_duration_ms not in ENDPOINTING_BY_SILENCE_MS:
            return (
                "nova-sonic endpointing is LOW/MEDIUM/HIGH (~2000/1750/1500 ms); "
                f"{detection.silence_duration_ms} ms has no equivalent"
            )
        return None

    def endpointing_sensitivity(self) -> str:
        if self._endpointing_override:
            return self._endpointing_override.upper()
        silence = self.config.turn_detection.silence_duration_ms
        return ENDPOINTING_BY_SILENCE_MS.get(silence, DEFAULT_ENDPOINTING) if silence else DEFAULT_ENDPOINTING

    def _session_start(self) -> dict[str, Any]:
        inference = dict(DEFAULT_INFERENCE)
        if self.config.temperature is not None:
            inference["temperature"] = self.config.temperature
        return {
            "sessionStart": {
                "inferenceConfiguration": inference,
                "turnDetectionConfiguration": {"endpointingSensitivity": self.endpointing_sensitivity()},
            }
        }

    def _prompt_start(self) -> dict[str, Any]:
        prompt: dict[str, Any] = {
            "promptName": self._prompt_name,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": self.output_rate,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": self.config.voice or DEFAULT_VOICE,
                "encoding": "base64",
                "audioType": "SPEECH",
            },
        }
        if self.config.tools:
            prompt["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
            prompt["toolConfiguration"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": {"json": json.dumps(t.parameters)},
                        }
                    }
                    for t in self.config.tools
                ]
            }
        return {"promptStart": prompt}

    def _audio_content_start(self) -> dict[str, Any]:
        return {
            "contentStart": {
                "promptName": self._prompt_name,
                "contentName": self._audio_content,
                "type": "AUDIO",
                "interactive": True,
                "role": "USER",
                "audioInputConfiguration": {
                    "mediaType": "audio/lpcm",
                    "sampleRateHertz": self.input_rate,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                    "audioType": "SPEECH",
                    "encoding": "base64",
                },
            }
        }

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def _open_transport(self) -> Transport:
        if self._transport_factory is not None:
            return await self._transport_factory()
        access_key = _setting("AWS_ACCESS_KEY_ID", self._api_key)
        secret = _setting("AWS_SECRET_ACCESS_KEY", self._secret)
        if not access_key or not secret:
            raise AdapterError("nova-sonic needs AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (SigV4; API keys cannot open this stream)")
        transport = _BedrockTransport(
            model=self.model,
            region=self.region,
            access_key_id=access_key,
            secret_access_key=secret,
            session_token=_setting("AWS_SESSION_TOKEN", self._session_token),
        )
        await transport.open()
        return transport

    async def connect(self) -> None:
        try:
            self._transport = await asyncio.wait_for(self._open_transport(), CONNECT_TIMEOUT_S)
        except AdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 -- the reason is what we publish
            raise AdapterError(f"nova-sonic connect failed: {exc}") from exc

        session_start = self._session_start()
        prompt_start = self._prompt_start()
        self.session_sent = {
            **session_start,
            **prompt_start,
            "system_prompt_chars": len(self.config.instructions or ""),
            "first_message": self.config.first_message,
            "region": self.region,
            "model_id": self.model,
        }
        try:
            await self._send_json(session_start)
            await self._send_json(prompt_start)
            if self.config.instructions:
                await self._send_text_content(self.config.instructions, role="SYSTEM", interactive=False)
            if self.config.first_message:
                await self._send_text_content(self.config.first_message, role="ASSISTANT", interactive=False)
            await self._send_json(self._audio_content_start())
            self._audio_open = True
            await asyncio.wait_for(self._transport.ready(), CONNECT_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            await self._abort()
            raise AdapterError(f"nova-sonic refused the session: {exc}") from exc

        # Bedrock resolves the id exactly or refuses the stream; it never echoes it.
        self.resolved_model = self.model
        self.session_ack = {"echoes_configuration": False, "stream_open": True, "first_event": None}
        self.log.emit(ev.SESSION_OPEN, session_id=None, model=self.model, region=self.region)
        self.log.emit(
            ev.SESSION_CONFIGURED,
            turn_detection=self.config.turn_detection.label,
            endpointing_sensitivity=self.endpointing_sensitivity(),
        )
        self._receiver = asyncio.create_task(self._receive_loop(), name="nova-sonic-recv")

    async def _abort(self) -> None:
        if self._transport is not None:
            try:
                await self._transport.close()
            except Exception:  # noqa: BLE001
                pass
        self.closed.set()

    async def close(self) -> None:
        if self._transport is not None and not self.closed.is_set():
            try:
                if self._pending and self._audio_open:
                    await self._send_audio_frame(bytes(self._pending))  # the tail of the caller's audio
                    self._pending.clear()
                if self._audio_open:
                    await self._send_json({"contentEnd": {"promptName": self._prompt_name, "contentName": self._audio_content}})
                    self._audio_open = False
                await self._send_json({"promptEnd": {"promptName": self._prompt_name}})
                await self._send_json({"sessionEnd": {}})
            except Exception as exc:  # noqa: BLE001 -- the service may already have ended it
                self.log.emit(ev.SESSION_ERROR, error=f"close: {exc!r}")
            try:
                await self._transport.close()
            except Exception:  # noqa: BLE001
                pass
        if self._receiver is not None:
            try:
                # Let the trailing usage / completionEnd events land before cancelling.
                await asyncio.wait_for(asyncio.shield(self._receiver), DRAIN_TIMEOUT_S)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._receiver.cancel()
        self._flush_agent_transcript()
        remainder = self._usage_delta()
        if any(remainder.values()):
            self.log.emit(ev.RESPONSE_DONE, status="session_usage", usage=remainder)
        self.closed.set()
        self.log.emit(ev.SESSION_CLOSED, by="harness")

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        self.log.raw(payload, direction="out")
        await self._send_event(payload)

    async def _send_event(self, payload: dict[str, Any]) -> None:
        if self._transport is None:
            raise AdapterError("not connected")
        async with self._send_lock:
            await self._transport.send(json.dumps({"event": payload}).encode("utf-8"))

    async def _send_text_content(self, text: str, *, role: str, interactive: bool) -> None:
        name = str(uuid.uuid4())
        await self._send_json(
            {
                "contentStart": {
                    "promptName": self._prompt_name,
                    "contentName": name,
                    "type": "TEXT",
                    "interactive": interactive,
                    "role": role,
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        )
        await self._send_json({"textInput": {"promptName": self._prompt_name, "contentName": name, "content": text}})
        await self._send_json({"contentEnd": {"promptName": self._prompt_name, "contentName": name}})

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        if self.closed.is_set():
            raise SessionClosed("nova-sonic session is closed")
        self._pending.extend(pcm)
        size = self._chunk_bytes
        while len(self._pending) >= size:
            frame = bytes(self._pending[:size])
            del self._pending[:size]
            await self._send_audio_frame(frame)

    async def _send_audio_frame(self, frame: bytes) -> None:
        # Not raw-logged: the caller wav and timeline already hold the audio.
        await self._send_event(
            {
                "audioInput": {
                    "promptName": self._prompt_name,
                    "contentName": self._audio_content,
                    "content": base64.b64encode(frame).decode("ascii"),
                }
            }
        )

    async def _commit(self) -> None:
        raise AdapterError("nova-sonic has no manual commit")

    async def _send_text(self, text: str) -> None:
        await self._send_text_content(text, role="USER", interactive=True)

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        name = str(uuid.uuid4())
        await self._send_json(
            {
                "contentStart": {
                    "promptName": self._prompt_name,
                    "contentName": name,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": call_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    },
                }
            }
        )
        content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        await self._send_json({"toolResult": {"promptName": self._prompt_name, "contentName": name, "content": content}})
        await self._send_json({"contentEnd": {"promptName": self._prompt_name, "contentName": name}})

    # ── receiving ────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        assert self._transport is not None
        try:
            while True:
                data = await self._transport.receive()
                if data is None:
                    self.log.emit(ev.SESSION_CLOSED, by="provider")
                    break
                if not data:
                    continue
                self._handle(json.loads(data.decode("utf-8")))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a refusal mid-session is evidence
            self.log.emit(ev.SESSION_ERROR, error=str(exc))
        finally:
            self.closed.set()

    def _handle(self, payload: dict[str, Any]) -> None:
        event = payload.get("event") or {}
        if "audioOutput" in event:
            self._on_agent_audio(base64.b64decode(event["audioOutput"].get("content", "")))
            return
        self.log.raw(payload)

        kind = next(iter(event), None)
        body = event.get(kind) if kind else None
        if isinstance(body, dict):
            if self.session_id is None and body.get("sessionId"):
                self.session_id = body["sessionId"]
                if self.session_ack is not None:
                    self.session_ack["provider_session_id"] = self.session_id
            if self.session_ack is not None and self.session_ack.get("first_event") is None:
                self.session_ack["first_event"] = {"kind": kind, "t": round(self.log.clock.now(), 6)}

        if "userSpeechStart" in event:
            self._on_caller_speech_started(audio_ms=event["userSpeechStart"].get("inputAudioOffsetMs"))
        elif "userSpeechEnd" in event:
            end = event["userSpeechEnd"]
            self.log.emit(
                ev.VAD_SPEECH_END,
                audio_ms=end.get("inputAudioOffsetMs"),
                detection_audio_ms=end.get("inputAudioDetectionOffsetMs"),
            )
        elif "completionStart" in event:
            pass  # one completion per session, opened at the first caller turn; raw log only
        elif "contentStart" in event:
            self._on_content_start(event["contentStart"])
        elif "textOutput" in event:
            self._on_text(event["textOutput"])
        elif "toolUse" in event:
            use = event["toolUse"]
            record = {
                "name": use.get("toolName", ""),
                "call_id": use.get("toolUseId", ""),
                "arguments": _parse_json(use.get("content")),
            }
            self.tool_calls.append(record)
            self.log.emit(ev.TOOL_CALL, **record)
        elif "contentEnd" in event:
            self._on_content_end(event["contentEnd"])
        elif "usageEvent" in event:
            self._on_usage(event["usageEvent"])
        elif "completionEnd" in event:
            pass  # recorded in the raw log; per-reply ends come from contentEnd

    def _on_content_start(self, start: dict[str, Any]) -> None:
        stage = _parse_json(start.get("additionalModelFields")).get("generationStage")
        self._content = {"type": start.get("type"), "role": start.get("role"), "stage": stage, "text": ""}
        if start.get("role") == "ASSISTANT" and start.get("type") == "TEXT" and stage == "SPECULATIVE" and not self._responding:
            self._flush_agent_transcript()
            self._responding = True
            self._interrupted = False
            self.log.emit(RESPONSE_START)

    def _on_text(self, output: dict[str, Any]) -> None:
        text = output.get("content", "")
        if self._content is None:
            return
        if self._content["role"] == "ASSISTANT" and _is_interruption_marker(text):
            self._interrupt()
            return
        self._content["text"] += text

    def _on_content_end(self, end: dict[str, Any]) -> None:
        content, self._content = self._content, None
        if content is None:
            return
        reason = end.get("stopReason")
        role, kind = content["role"], content["type"]
        if role == "ASSISTANT":
            if reason == "INTERRUPTED":
                self._interrupt()
                return
            if kind == "TEXT" and content["stage"] == "FINAL" and content["text"]:
                self._agent_final.append(content["text"])
            elif kind == "AUDIO" and reason == "END_TURN":
                self._finish_response("completed", reason="end_turn")
        elif role == "USER" and kind == "TEXT" and content["text"].strip():
            text = content["text"].strip()
            self.caller_text.append(text)
            self.log.emit(ev.CALLER_TRANSCRIPT, text=text)

    def _interrupt(self) -> None:
        if self._interrupted:
            return
        self._interrupted = True
        self._on_agent_interrupted()
        if self._responding:
            self._finish_response("cancelled", reason="cancelled")

    def _finish_response(self, status: str, *, reason: str) -> None:
        self._responding = False
        self._on_agent_audio_done(reason=reason)
        self.log.emit(ev.RESPONSE_DONE, status=status, usage=self._usage_delta())

    def _on_usage(self, usage: dict[str, Any]) -> None:
        details = usage.get("details") or {}
        total = details.get("total")
        if isinstance(total, dict):
            self._usage_total = _usage_fields(total)
        elif isinstance(details.get("delta"), dict):
            delta = _usage_fields(details["delta"])
            self._usage_total = {k: self._usage_total.get(k, 0) + v for k, v in delta.items()}

    def _usage_delta(self) -> dict[str, int]:
        delta = {k: v - self._usage_reported.get(k, 0) for k, v in self._usage_total.items()}
        self._usage_reported = dict(self._usage_total)
        return delta

    def _flush_agent_transcript(self) -> None:
        if self._agent_final:
            text = " ".join(t.strip() for t in self._agent_final if t.strip())
            self._agent_final.clear()
            self.agent_text.append(text)
            self.log.emit(ev.AGENT_TRANSCRIPT, text=text)

    def state(self) -> dict[str, Any]:
        record = super().state()
        record["nova_sonic"] = {
            "region": self.region,
            "requested_model": self.model,
            "resolved_model": self.resolved_model,
            "endpointing_sensitivity": self.endpointing_sensitivity(),
            "usage_total": dict(self._usage_total),
        }
        return record


def _parse_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"__unparsed__": raw}
    return value if isinstance(value, dict) else {"__value__": value}


def _is_interruption_marker(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    return _parse_json(stripped).get("interrupted") is True


def _usage_fields(block: dict[str, Any]) -> dict[str, int]:
    """Nova's ``{input|output: {speechTokens, textTokens}}`` in the framework's names."""
    def n(side: str, key: str) -> int:
        return int((block.get(side) or {}).get(key) or 0)

    in_audio, in_text = n("input", "speechTokens"), n("input", "textTokens")
    out_audio, out_text = n("output", "speechTokens"), n("output", "textTokens")
    return {
        "prompt_tokens": in_audio + in_text,
        "input_audio_tokens": in_audio,
        "completion_tokens": out_audio + out_text,
        "output_audio_tokens": out_audio,
    }
