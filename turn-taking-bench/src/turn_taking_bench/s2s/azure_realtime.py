"""Azure Realtime (``azure-realtime`` on Azure Voice Live) as a service-bench adapter.

Implements ``service.adapters.base.RealtimeAdapter`` for Microsoft's own
speech-to-speech model, reached through the Voice Live socket the reference
agent uses (``reference-agents/pipecat-s2s/azure_realtime.py``), with the same
session: API version ``2026-07-15``, ``azure-realtime-native`` voice ``ava``,
24 kHz PCM16 both ways, Azure's own caller transcription (``azure-speech``,
en-US) and, as native turn detection, ``azure_semantic_vad`` at the values the
service reports when asked for it with nothing else (threshold 0.5, 420 ms
prefix padding, 500 ms silence, create and interrupt response on).

Protocol quirks (read off the reference agent and the live smoke):

* Voice Live speaks the **pre-GA, flat** OpenAI Realtime shape, not the GA
  nested one: ``input_audio_format`` / ``turn_detection`` /
  ``input_audio_transcription`` sit at the top of the session, ``modalities``
  instead of ``output_modalities``, and replies arrive as
  ``response.audio.delta`` / ``response.audio.done``. The OpenAI adapter's event
  handling already maps those pre-GA names, so this adapter inherits it and
  overrides only the URL, the auth header and the session document.
* Auth is an ``api-key`` header (not ``Authorization: Bearer``); the URL is the
  resource's own host (``AZURE_VOICE_ENDPOINT``), ``/voice-live/realtime`` with
  ``api-version`` and ``model`` query parameters.
* The first frame is ``session.created`` (the session id); the configuration is
  acknowledged by ``session.updated``, whose ``session`` is recorded as
  ``session_ack`` -- it reports the resolved detector values and voice.
* A reply's transcript leads its audio by up to about a second, so
  ``agent.transcript`` timing is not a speech-onset proxy; onsets come from the
  audio only.
* A caller barge-in arrives as ``input_audio_buffer.speech_started`` followed by
  ``response.done`` with ``status: cancelled`` (interrupt_response on), which
  the inherited handler turns into ``vad.speech.start`` + ``agent.interrupted``.
* ``response.created`` is logged as ``response.start`` (not in the service
  vocabulary; additive) so a reply's lifecycle is visible before its audio.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Awaitable, Callable

from turn_taking_bench.assets import TURN_TAKING_BENCH_ROOT
from turn_taking_bench.author import service_bridge

service_bridge._ensure_path()

from service import events as ev  # noqa: E402
from service.adapters.base import AdapterError  # noqa: E402
from service.adapters.openai_realtime import OpenAIRealtimeAdapter  # noqa: E402

API_VERSION = "2026-07-15"
SAMPLE_RATE = 24000
CONNECT_TIMEOUT_S = 20.0
DEFAULT_MODEL = "azure-realtime"
DEFAULT_VOICE = "ava"
ENDPOINT_ENV = "AZURE_VOICE_ENDPOINT"
CREDENTIAL_ENV = "AZURE_VOICE_API_KEY"

#: Normalized event for ``response.created``; additive to ``service.events``.
RESPONSE_START = "response.start"

#: The reference agent's native detector (``bot.py::AZURE_TURN_DETECTION``).
NATIVE_TURN_DETECTION: dict[str, Any] = {
    "type": "azure_semantic_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 420,
    "silence_duration_ms": 500,
    "create_response": True,
    "interrupt_response": True,
}
TRANSCRIPTION: dict[str, Any] = {"model": "azure-speech", "language": "en-US"}

Connector = Callable[[str, dict[str, str]], Awaitable[Any]]


def realtime_url(endpoint: str, model: str) -> str:
    """The session URL for a resource endpoint given as ``https://`` or ``wss://``."""
    host = endpoint.strip().rstrip("/")
    for scheme in ("https://", "wss://"):
        if host.startswith(scheme):
            host = host[len(scheme):]
    return f"wss://{host}/voice-live/realtime?api-version={API_VERSION}&model={model}"


def _setting(name: str) -> str | None:
    """Environment first, then turn-taking-bench/.env, then the repository root's .env.
    The value is returned to the caller only; nothing is logged."""
    if os.environ.get(name):
        return os.environ[name]
    for path in (TURN_TAKING_BENCH_ROOT / ".env", TURN_TAKING_BENCH_ROOT.parent / ".env"):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    return value
    return None


async def _websocket_connect(url: str, headers: dict[str, str]) -> Any:
    import websockets  # lazy: offline tests inject a fake connector

    return await websockets.connect(url, additional_headers=headers, open_timeout=CONNECT_TIMEOUT_S, max_size=None)


class AzureRealtimeAdapter(OpenAIRealtimeAdapter):
    name = "azure-realtime"
    input_rate = SAMPLE_RATE
    output_rate = SAMPLE_RATE
    supports_manual_commit = True   # turn_detection null, as the pre-GA protocol; not exercised live
    supports_text_modality = True
    emits_vad_events = True

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str | None = None,
        connector: Connector | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._endpoint = endpoint
        self._connector = connector or _websocket_connect
        self.resolved_model: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        endpoint = self._endpoint or _setting(ENDPOINT_ENV)
        if not endpoint:
            raise AdapterError(f"{self.name}: {ENDPOINT_ENV} is not set")
        try:
            self._ws = await self._connector(realtime_url(endpoint, self.model), {"api-key": self._api_key})
        except Exception as exc:  # noqa: BLE001 -- the reason is what we want to publish
            raise AdapterError(f"{self.name} connect failed: {type(exc).__name__}: {exc}") from exc

        first = json.loads(await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT_S))
        self.log.raw(first)
        if first.get("type") != "session.created":
            raise AdapterError(f"{self.name} refused the session: {first}")
        session = first.get("session") or {}
        self.session_id = session.get("id")
        self.resolved_model = session.get("model")
        self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model, resolved_model=self.resolved_model)

        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        self.session_sent = self._session_payload()
        await self._send_json({"type": "session.update", "session": self.session_sent})
        try:
            await asyncio.wait_for(self._configured.wait(), CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise AdapterError(f"{self.name} never acknowledged session.update") from exc
        if self.config.first_message:
            # Pre-GA content type for an assistant item is "text".
            await self._send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": self.config.first_message}],
                    },
                }
            )

    def _session_payload(self) -> dict[str, Any]:
        config = self.config
        session: dict[str, Any] = {
            "modalities": ["text", "audio"] if config.modality == "audio" else ["text"],
            "instructions": config.instructions,
            "voice": {"type": "azure-realtime-native", "name": config.voice or DEFAULT_VOICE},
            "input_audio_format": "pcm16",
            "input_audio_sampling_rate": self.input_rate,
            "output_audio_format": "pcm16",
            "turn_detection": self._turn_detection_payload(),
        }
        if config.transcribe_input:
            session["input_audio_transcription"] = dict(TRANSCRIPTION)
        if config.temperature is not None:
            session["temperature"] = config.temperature
        if config.tools:
            session["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in config.tools
            ]
            session["tool_choice"] = "auto"
        return session

    def _turn_detection_payload(self) -> dict[str, Any] | None:
        """``server_vad`` with no parameters and ``semantic_vad`` both mean the
        service's native detector (``azure_semantic_vad``, the published row's);
        ``server_vad`` with an explicit parameter is Voice Live's plain energy
        VAD with that parameter. Any explicit parameter overrides the default."""
        detection = self.config.turn_detection
        if detection.is_manual:
            return None
        explicit = {
            key: value
            for key, value in (
                ("silence_duration_ms", detection.silence_duration_ms),
                ("threshold", detection.threshold),
                ("prefix_padding_ms", detection.prefix_padding_ms),
            )
            if value is not None
        }
        if detection.mode == "server_vad" and explicit:
            return {"type": "server_vad", "create_response": True, "interrupt_response": True, **explicit}
        return {**NATIVE_TURN_DETECTION, **explicit}

    def state(self) -> dict[str, Any]:
        state = super().state()
        state["resolved_model"] = self.resolved_model
        state["api_version"] = API_VERSION
        return state

    # ── receiving ────────────────────────────────────────────────────────────

    def _handle(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type", "")
        if kind == "response.created":
            response = payload.get("response") or {}
            self.log.raw(payload)
            self.log.emit(RESPONSE_START, response_id=response.get("id"))
            return
        if kind == "session.updated":
            model = (payload.get("session") or {}).get("model")
            if model:
                self.resolved_model = model
        super()._handle(payload)


def provider_entry():
    """A ``service.registry.ProviderEntry`` for this adapter (not registered;
    a coordinator adds it to whichever registry the run reads)."""
    from service.registry import ProviderEntry

    return ProviderEntry(
        key="azure-realtime",
        adapter=AzureRealtimeAdapter,
        default_model=DEFAULT_MODEL,
        credential_env=CREDENTIAL_ENV,
        default_voice=DEFAULT_VOICE,
        discloses=(
            f"Azure Voice Live api-version {API_VERSION}; native turn detection azure_semantic_vad "
            "(threshold 0.5, prefix 420 ms, silence 500 ms); caller transcription azure-speech en-US",
        ),
    )
