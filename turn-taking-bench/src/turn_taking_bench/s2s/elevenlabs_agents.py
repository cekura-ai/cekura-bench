"""ElevenLabs Agents (conversational AI platform) as a service-bench adapter.

ElevenLabs does not sell a speech-to-speech model; what it benchmarks is an
*agent*: speech recognition hears the caller, an LLM writes the reply and an
ElevenLabs voice speaks it, with the platform's own turn-taking in between. The
agent's configuration is read from a local file (``load_agent_config``); turn
bench sets only what it sets for every model: our system prompt, and the caller
speaking first.

The agent lives in the ElevenLabs account (``ELEVENLABS_AGENT_ID``), created
once with ``scripts/elevenlabs_agent.py`` from ``agent_payload()``. Each
session overrides only the prompt (Track 1 and Track 2 differ) and the first
message (empty: the caller opens the call), which the agent permits.

Protocol: https://elevenlabs.io/docs/agents-platform/api-reference/agents-platform/websocket

* **Auth.** A private agent is reached through a signed URL
  (``GET /v1/convai/conversation/get-signed-url``, ``xi-api-key``), fetched per
  session; the socket itself carries no key.
* **Handshake.** The client sends ``conversation_initiation_client_data`` with
  the overrides; the server answers ``conversation_initiation_metadata`` (the
  conversation id and both audio formats), recorded as ``session_ack``. Audio
  is sent only after it.
* **Audio.** In: ``{"user_audio_chunk": <base64 pcm_16000>}``. Out: ``audio``
  events, faster than real time and only while replying, so arrival is not
  playout: the base adapter's playout model places them. A reply has no end
  event; it is closed once no audio has arrived for ``REPLY_GAP_S``.
* **Barge-in is the platform's decision**, signalled by ``interruption``; that
  is mapped to ``agent.interrupted`` (the client drops what it holds).
* **No endpointer events.** Only a continuous ``vad_score``, so
  ``emits_vad_events`` is False and turns are read from the reply onsets.
* **Keep-alive.** Every ``ping`` is answered with a ``pong`` of the same id,
  or the server closes the socket.
* **Hang-up.** If the configuration gives the agent an ``end_call`` tool, a
  session it ends early closes from the provider side, and the calls it cut
  short are unscored like any other broken session.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable

from turn_taking_bench.author import service_bridge
from turn_taking_bench.assets import TURN_TAKING_BENCH_ROOT
from turn_taking_bench.s2s._vendor_env import setting

service_bridge._ensure_path()

from service import events as ev  # noqa: E402
from service.adapters.base import AdapterError, RealtimeAdapter, SessionClosed  # noqa: E402

API = "https://api.elevenlabs.io"
SIGNED_URL = f"{API}/v1/convai/conversation/get-signed-url"
CREDENTIAL_ENV = "ELEVENLABS_BENCH_API_KEY"  # a workspace whose plan allows the agent's LLM
AGENT_ENV = "ELEVENLABS_AGENT_ID"
AGENT_NAME = "Cekura turn bench - ElevenLabs agent"
MODEL = "elevenlabs-agents"
SAMPLE_RATE = 16000
AUDIO_FORMAT = "pcm_16000"
CONNECT_TIMEOUT_S = 20.0
READY_TIMEOUT_S = 15.0
#: A reply is over once its audio has stopped arriving for this long. Audio comes
#: faster than real time, so a gap this size is the synthesis finishing, not a pause.
REPLY_GAP_S = 0.5

#: The agent's configuration (a create-agent body without name or prompt), kept out of
#: the repository: ELEVENLABS_AGENT_CONFIG, else turn-taking-bench/elevenlabs-agent.local.json.
CONFIG_ENV = "ELEVENLABS_AGENT_CONFIG"
DEFAULT_CONFIG_PATH = TURN_TAKING_BENCH_ROOT / "elevenlabs-agent.local.json"


def load_agent_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or setting(CONFIG_ENV) or DEFAULT_CONFIG_PATH)
    if not path.is_file():
        raise AdapterError(f"no ElevenLabs agent configuration at {path} (set {CONFIG_ENV})")
    return json.loads(path.read_text(encoding="utf-8"))


def agent_payload(prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    """The create-agent body: ``config`` plus what every turn-bench model shares
    (our prompt, the caller speaking first, 16 kHz out like in) and the two
    per-session overrides the adapter needs."""
    body = copy.deepcopy(config)
    conversation = body["conversation_config"]
    conversation["agent"]["prompt"]["prompt"] = prompt
    conversation["agent"]["first_message"] = ""
    conversation["tts"]["agent_output_audio_format"] = AUDIO_FORMAT
    body["platform_settings"]["overrides"] = {
        "conversation_config_override": {"agent": {"prompt": {"prompt": True}, "first_message": True}}
    }
    body["name"] = AGENT_NAME
    return body


Connector = Callable[..., Awaitable[Any]]
Signer = Callable[[str, str], Awaitable[str]]


async def _websocket_connect(url: str) -> Any:
    import websockets

    return await websockets.connect(url, open_timeout=CONNECT_TIMEOUT_S, max_size=None)


async def _signed_url(api_key: str, agent_id: str) -> str:
    def fetch() -> str:
        request = urllib.request.Request(
            f"{SIGNED_URL}?{urllib.parse.urlencode({'agent_id': agent_id})}", headers={"xi-api-key": api_key}
        )
        with urllib.request.urlopen(request, timeout=CONNECT_TIMEOUT_S) as response:
            return json.load(response)["signed_url"]

    return await asyncio.to_thread(fetch)


class ElevenLabsAgentsAdapter(RealtimeAdapter):
    name = "elevenlabs-agents"
    input_rate = SAMPLE_RATE
    output_rate = SAMPLE_RATE
    supports_manual_commit = False
    supports_text_modality = False
    emits_vad_events = False

    def __init__(
        self,
        *,
        model: str = MODEL,
        api_key: str = "",
        agent_id: str | None = None,
        connector: Connector | None = None,
        signer: Signer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._agent_id = agent_id or setting(AGENT_ENV)
        self._connector = connector or _websocket_connect
        self._signer = signer or _signed_url
        self._ws: Any = None
        self._ready = asyncio.Event()
        self._reply_end: asyncio.TimerHandle | None = None
        self._reply_text: list[str] = []

    # ── session ──────────────────────────────────────────────────────────────

    def initiation_payload(self) -> dict[str, Any]:
        config = self.config
        instructions = config.instructions
        if config.first_message:
            instructions = f"{instructions}\n\nYou have already greeted the caller with: \"{config.first_message}\"".strip()
        return {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {"agent": {"prompt": {"prompt": instructions}, "first_message": ""}},
        }

    async def connect(self) -> None:
        if not self._api_key:
            raise AdapterError(f"{self.name} has no API key")
        if not self._agent_id:
            raise AdapterError(f"{self.name} has no agent: set {AGENT_ENV} (scripts/elevenlabs_agent.py creates it)")
        try:
            url = await self._signer(self._api_key, self._agent_id)
            self._ws = await self._connector(url)
        except Exception as exc:  # noqa: BLE001 -- the reason is what we publish
            raise AdapterError(f"{self.name} connect failed: {exc}") from exc
        self.session_ack = {}
        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        self.session_sent = {"agent_id": self._agent_id, **self.initiation_payload()}
        await self._send_json(self.initiation_payload())
        ready = asyncio.ensure_future(self._ready.wait())
        closed = asyncio.ensure_future(self.closed.wait())
        _done, pending = await asyncio.wait({ready, closed}, timeout=READY_TIMEOUT_S,
                                            return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if not self._ready.is_set():
            errors = [e.data for e in self.log.of_kind(ev.SESSION_ERROR)]
            await self.close()
            raise AdapterError(f"{self.name} never became ready: {errors or 'timeout'}")
        formats = self.session_ack.get("conversation_initiation_metadata_event", {})
        for key in ("user_input_audio_format", "agent_output_audio_format"):
            if formats.get(key, AUDIO_FORMAT) != AUDIO_FORMAT:
                await self.close()
                raise AdapterError(f"{self.name} agent uses {key}={formats[key]}, expected {AUDIO_FORMAT}")
        self.log.emit(ev.SESSION_CONFIGURED, turn_detection=self.config.turn_detection.label, model=self.model)

    async def close(self) -> None:
        if self._reply_end:
            self._reply_end.cancel()
        if self._receiver:
            self._receiver.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        self._end_reply("closed")
        if not self.closed.is_set():
            self.closed.set()
            self.log.emit(ev.SESSION_CLOSED, by="harness")

    def state(self) -> dict[str, Any]:
        state = super().state()
        state["agent_id"] = self._agent_id
        state["reply_gap_s"] = REPLY_GAP_S
        return state

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        self.log.raw(payload, direction="out")
        await self._ws.send(json.dumps(payload))

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        if self._ws is None:
            raise AdapterError("not connected")
        if self.closed.is_set():
            raise SessionClosed(f"{self.name} session closed")
        await self._ws.send(json.dumps({"user_audio_chunk": base64.b64encode(pcm).decode()}))

    async def _commit(self) -> None:
        raise AdapterError(f"{self.name} has no manual commit")

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        await self._send_json({"type": "client_tool_result", "tool_call_id": call_id, "result": json.dumps(output),
                               "is_error": False})

    # ── receiving ────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        ws = self._ws
        try:
            async for message in ws:
                try:
                    payload = json.loads(message)
                except (TypeError, json.JSONDecodeError):
                    self.log.emit(ev.SESSION_ERROR, error="non-JSON frame")
                    continue
                await self._handle(payload)
            self.log.emit(ev.SESSION_CLOSED, by="provider", code=getattr(ws, "close_code", None),
                          reason=getattr(ws, "close_reason", None))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- includes websockets.ConnectionClosed
            self.log.emit(ev.SESSION_CLOSED, by="provider", code=getattr(ws, "close_code", None),
                          reason=getattr(ws, "close_reason", None), error=repr(exc))
        self._end_reply("closed")
        self.closed.set()

    def _end_reply(self, status: str) -> None:
        if self._reply_end:
            self._reply_end.cancel()
            self._reply_end = None
        if not self._speaking:
            return
        self._on_agent_audio_done(reason=status)
        self.log.emit(ev.RESPONSE_DONE, status=status, usage=None)

    async def _handle(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type", "")
        if kind == "audio":
            event = payload.get("audio_event") or {}
            self.log.raw({"type": kind, "event_id": event.get("event_id")})  # audio itself is in the wav
            audio = base64.b64decode(event.get("audio_base_64") or "")
            if audio:
                self._on_agent_audio(audio)
                if self._reply_end:
                    self._reply_end.cancel()
                self._reply_end = asyncio.get_running_loop().call_later(REPLY_GAP_S, self._end_reply, "completed")
            return
        if kind == "vad_score":
            return  # continuous, every frame; not an endpointer decision
        self.log.raw(payload)
        if kind == "ping":
            event = payload.get("ping_event") or {}
            await self._send_json({"type": "pong", "event_id": event.get("event_id")})
        elif kind == "conversation_initiation_metadata":
            event = payload.get("conversation_initiation_metadata_event") or {}
            self.session_id = event.get("conversation_id")
            self.session_ack["conversation_initiation_metadata_event"] = event
            self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model)
            self._ready.set()
        elif kind == "agent_response":
            text = (payload.get("agent_response_event") or {}).get("agent_response", "")
            if text:
                self.agent_text.append(text)
                self.log.emit(ev.AGENT_TRANSCRIPT, text=text)
        elif kind == "agent_response_correction":
            event = payload.get("agent_response_correction_event") or {}
            self.log.emit(ev.AGENT_TRANSCRIPT, text=event.get("corrected_agent_response", ""), corrected=True)
        elif kind == "user_transcript":
            text = (payload.get("user_transcription_event") or {}).get("user_transcript", "")
            self.caller_text.append(text)
            self.log.emit(ev.CALLER_TRANSCRIPT, text=text)
        elif kind == "interruption":
            replying = self._speaking
            if self._reply_end:
                self._reply_end.cancel()
                self._reply_end = None
            self._on_agent_interrupted()
            if replying:
                self.log.emit(ev.RESPONSE_DONE, status="cancelled", usage=None)
        elif kind == "client_tool_call":
            event = payload.get("client_tool_call") or {}
            call = {"name": event.get("tool_name", ""), "call_id": event.get("tool_call_id", ""),
                    "arguments": event.get("parameters") or {}}
            self.tool_calls.append(call)
            self.log.emit(ev.TOOL_CALL, **call)
        elif kind == "error":
            self.log.emit(ev.SESSION_ERROR, error=payload)


def provider_entry():
    """A ``service.registry.ProviderEntry`` for this adapter."""
    from service.registry import ProviderEntry

    return ProviderEntry(
        key="elevenlabs-agents",
        adapter=ElevenLabsAgentsAdapter,
        default_model=MODEL,
        credential_env=CREDENTIAL_ENV,
        default_voice=None,
        discloses=(
            "an ElevenLabs agent (speech recognition, LLM and voice) in the configuration it was created with, "
            "not a single speech-to-speech model",
            "turn-bench caller voices are ElevenLabs-generated",
        ),
    )
