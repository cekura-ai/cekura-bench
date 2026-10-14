"""Offline tests for the Azure Realtime and GPT-Live adapters: a scripted fake
socket stands in for the provider, so the session documents, the event
normalisation and the reply gate are checked with no network."""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

pytest.importorskip("websockets")  # the service bench's OpenAI adapter imports it

from turn_taking_bench.author import service_bridge  # noqa: E402

service_bridge._ensure_path()

from service import events as ev  # noqa: E402
from service.adapters.base import RealtimeAdapter, SessionConfig, TurnDetection  # noqa: E402

from turn_taking_bench.s2s import azure_realtime, gpt_live  # noqa: E402

RATE = 24000


class FakeSocket:
    """Records what the adapter sends; replies through ``script(sent) -> [events]``."""

    def __init__(self, script, first=None):
        self.sent: list[dict] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.script = script
        self.close_code = None
        self.close_reason = None
        self.closed = False
        if first is not None:
            self.inbox.put_nowait(first)

    async def send(self, text):
        payload = json.loads(text)
        self.sent.append(payload)
        for reply in self.script(payload) or ():
            await self.inbox.put(reply)

    async def recv(self):
        return json.dumps(await self.inbox.get())

    def push(self, payload):
        self.inbox.put_nowait(payload)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return json.dumps(item)

    async def close(self):
        self.closed = True
        self.inbox.put_nowait(None)


def connector_for(sock, seen):
    async def connect(url, headers):
        seen["url"], seen["headers"] = url, headers
        return sock

    return connect


def tone(ms, amp=8000):
    n = RATE * ms // 1000
    return (amp * np.sin(2 * np.pi * 220 * np.arange(n) / RATE)).astype("<i2").tobytes()


def silence(ms):
    return b"\x00\x00" * (RATE * ms // 1000)


def b64(pcm):
    return base64.b64encode(pcm).decode()


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


def kinds(log):
    return [e.kind for e in log.events]


# ── Azure ────────────────────────────────────────────────────────────────────


def azure_script(payload):
    if payload["type"] == "session.update":
        return [{"type": "session.updated", "session": {**payload["session"], "model": "azure-realtime-2026"}}]
    return []


def test_azure_conforms_and_runs_a_turn():
    assert issubclass(azure_realtime.AzureRealtimeAdapter, RealtimeAdapter)

    async def scenario():
        sock = FakeSocket(azure_script, first={"type": "session.created", "session": {"id": "sess_1", "model": "azure-realtime"}})
        seen: dict = {}
        log = ev.EventLog(ev.Clock())
        adapter = azure_realtime.AzureRealtimeAdapter(
            api_key="k", log=log, endpoint="https://res.example.com/",
            config=SessionConfig(instructions="be brief", voice=None),
            connector=connector_for(sock, seen),
        )
        await adapter.connect()
        await adapter.send_audio(silence(20))
        sock.push({"type": "input_audio_buffer.speech_started", "audio_start_ms": 100})
        sock.push({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 900})
        sock.push({"type": "response.created", "response": {"id": "r1"}})
        sock.push({"type": "response.audio.delta", "response_id": "r1", "delta": b64(tone(40))})
        sock.push({"type": "response.audio_transcript.done", "transcript": "Sure."})
        sock.push({"type": "response.audio.done"})
        sock.push({"type": "response.done", "response": {"id": "r1", "status": "completed", "usage": {"total_tokens": 9}}})
        # A barge-in cancels the next reply.
        sock.push({"type": "response.created", "response": {"id": "r2"}})
        sock.push({"type": "response.audio.delta", "response_id": "r2", "delta": b64(tone(40))})
        sock.push({"type": "input_audio_buffer.speech_started", "audio_start_ms": 2000})
        sock.push({"type": "response.done", "response": {"id": "r2", "status": "cancelled"}})
        await settle()
        await adapter.close()
        return adapter, sock, seen, log

    adapter, sock, seen, log = asyncio.run(scenario())
    assert seen["url"] == (
        "wss://res.example.com/voice-live/realtime?api-version=2026-07-15&model=azure-realtime"
    )
    assert seen["headers"] == {"api-key": "k"}
    update = sock.sent[0]
    assert update["type"] == "session.update"
    session = update["session"]
    assert session["input_audio_format"] == "pcm16" and session["input_audio_sampling_rate"] == 24000
    assert session["voice"] == {"type": "azure-realtime-native", "name": "ava"}
    assert session["turn_detection"]["type"] == "azure_semantic_vad"
    assert session["turn_detection"]["silence_duration_ms"] == 500
    assert session["input_audio_transcription"] == {"model": "azure-speech", "language": "en-US"}
    assert any(s["type"] == "input_audio_buffer.append" for s in sock.sent)

    assert adapter.session_id == "sess_1"
    assert adapter.session_ack["model"] == "azure-realtime-2026"
    assert adapter.resolved_model == "azure-realtime-2026"
    k = kinds(log)
    for kind in (ev.SESSION_OPEN, ev.SESSION_CONFIGURED, ev.VAD_SPEECH_START, ev.VAD_SPEECH_END,
                 azure_realtime.RESPONSE_START, ev.AGENT_AUDIO_START, ev.AGENT_AUDIO_END,
                 ev.AGENT_INTERRUPTED, ev.RESPONSE_DONE):
        assert kind in k, kind
    assert k.count(ev.AGENT_AUDIO_START) == 2
    assert adapter.agent_text == ["Sure."]
    assert adapter.agent_timeline.n_samples == 2 * RATE * 40 // 1000
    assert adapter.state()["session"]["acknowledged"]["model"] == "azure-realtime-2026"


def test_azure_turn_detection_mapping_and_refusal():
    def payload(detection):
        adapter = azure_realtime.AzureRealtimeAdapter(
            api_key="k", log=ev.EventLog(ev.Clock()), endpoint="x",
            config=SessionConfig(turn_detection=detection),
        )
        return adapter._turn_detection_payload()

    assert payload(TurnDetection("semantic_vad"))["type"] == "azure_semantic_vad"
    plain = payload(TurnDetection("server_vad", silence_duration_ms=300))
    assert plain["type"] == "server_vad" and plain["silence_duration_ms"] == 300
    assert payload(TurnDetection("manual")) is None

    async def refused():
        sock = FakeSocket(lambda p: [], first={"type": "error", "error": {"message": "bad key"}})
        adapter = azure_realtime.AzureRealtimeAdapter(
            api_key="k", log=ev.EventLog(ev.Clock()), endpoint="x", connector=connector_for(sock, {}),
        )
        with pytest.raises(azure_realtime.AdapterError):
            await adapter.connect()

    asyncio.run(refused())


# ── GPT-Live ─────────────────────────────────────────────────────────────────


def live_script(payload):
    if payload["type"] == "session.start":
        return [{"type": "session.started", "session": {"id": "live_1", "model": "gpt-live-1-2026-09", "status": "active",
                                                         "delegation": payload["session"]["delegation"]}}]
    if payload["type"] == "session.close":
        return [{"type": "session.closed", "reason": "client", "usage": {"seconds": 12.5}}]
    return []


def test_gpt_live_session_and_reply_gate():
    assert issubclass(gpt_live.GPTLiveAdapter, RealtimeAdapter)

    async def scenario():
        sock = FakeSocket(live_script)
        seen: dict = {}
        log = ev.EventLog(ev.Clock())
        adapter = gpt_live.GPTLiveAdapter(
            api_key="k", log=log, config=SessionConfig(instructions="be brief"), connector=connector_for(sock, seen),
        )
        await adapter.connect()
        await adapter.send_audio(silence(20))
        # Continuous output: silence, a reply, silence long enough to end it, a second reply cut by close.
        sock.push({"type": "session.output_audio.delta", "delta": b64(silence(200))})
        sock.push({"type": "session.output_transcript.delta", "delta": "Hi ", "start_ms": 0, "end_ms": 200})
        sock.push({"type": "session.output_audio.delta", "delta": b64(silence(15) + tone(300))})
        sock.push({"type": "session.output_transcript.delta", "delta": "there."})
        sock.push({"type": "session.output_audio.delta", "delta": b64(silence(700))})
        sock.push({"type": "session.input_transcript.delta", "delta": "hello", "start_ms": 0, "end_ms": 400})
        sock.push({"type": "response.event", "delegation_id": "d1",
                   "event": {"type": "response.completed", "response": {"usage": {"input_tokens": 10, "output_tokens": 3}}}})
        sock.push({"type": "session.output_audio.delta", "delta": b64(tone(100))})
        await settle()
        await adapter.close()
        return adapter, sock, seen, log

    adapter, sock, seen, log = asyncio.run(scenario())
    assert seen["url"] == "wss://api.openai.com/v1/live/sessions"
    assert seen["headers"] == {"Authorization": "Bearer k"}
    start = sock.sent[0]
    assert start["type"] == "session.start"
    session = start["session"]
    assert session["model"] == "gpt-live-1"
    assert session["delegation"] == {
        "type": "responses",
        "responses": {"model": "gpt-6-sol", "reasoning": {"effort": "low"}, "instructions": "be brief"},
    }
    assert session["audio"] == {"output": {"voice": "marin"}, "format": {"type": "audio/pcm", "rate": 24000}}
    assert "input" not in session  # unset fields are omitted, never null
    assert sock.sent[1]["type"] == "session.input_audio.append"
    assert sock.sent[-1]["type"] == "session.close"

    assert adapter.resolved_model == "gpt-live-1-2026-09"
    assert adapter.session_ack["id"] == "live_1"
    starts = [e for e in log.events if e.kind == ev.AGENT_AUDIO_START]
    ends = [e for e in log.events if e.kind == ev.AGENT_AUDIO_END]
    assert len(starts) == 2 and len(ends) == 2
    # The first reply's onset is at its first voiced 10 ms frame: 200 ms + 10 ms
    # of leading silence (the 15 ms pad straddles a frame), not at the stream start.
    assert starts[0].data["sample"] == RATE * 210 // 1000
    assert ends[0].data["reason"] == "silence"
    assert ends[0].data["sample"] == RATE * 520 // 1000
    assert ends[1].data["reason"] == "session_end"
    assert log.first(gpt_live.RESPONSE_START) is not None
    assert adapter.agent_text[0] == "Hi there."
    assert adapter.caller_text == ["hello"]
    assert adapter.agent_timeline.n_samples == RATE * (200 + 15 + 300 + 700 + 100) // 1000
    state = adapter.state()
    assert state["usage"]["live_audio_seconds"] == 12.5
    assert state["usage"]["backend"] == [{"input_tokens": 10, "output_tokens": 3}]
    assert state["backend"] == {"model": "gpt-6-sol", "reasoning": "low"}


def test_gpt_live_refusals():
    assert gpt_live.GPTLiveAdapter.unsupported_reason(SessionConfig(turn_detection=TurnDetection("manual")))
    assert gpt_live.GPTLiveAdapter.unsupported_reason(SessionConfig(modality="text"))
    assert gpt_live.GPTLiveAdapter.unsupported_reason(SessionConfig()) is None

    async def startup_error():
        sock = FakeSocket(lambda p: [{"type": "error", "error": {"code": "invalid_model"}}, None]
                          if p["type"] == "session.start" else [])
        adapter = gpt_live.GPTLiveAdapter(api_key="k", log=ev.EventLog(ev.Clock()), connector=connector_for(sock, {}))
        with pytest.raises(gpt_live.AdapterError, match="invalid_model"):
            await adapter.connect()

    asyncio.run(startup_error())
