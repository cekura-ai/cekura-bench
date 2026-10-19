"""Offline tests for the ElevenLabs Agents S2S adapter, over a fake websocket."""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

from turn_taking_bench.author import service_bridge

service_bridge._ensure_path()

from service import events as ev  # noqa: E402
from service.adapters.base import AdapterError, SessionConfig  # noqa: E402

from turn_taking_bench.s2s import SYSTEM_PROMPT  # noqa: E402
from turn_taking_bench.s2s.elevenlabs_agents import (  # noqa: E402
    REPLY_GAP_S, ElevenLabsAgentsAdapter, agent_payload, load_agent_config,
)
from turn_taking_bench.s2s.model import provider_entry  # noqa: E402



class FakeSocket:
    """A websocket stand-in: ``feed`` queues a server frame, ``sent`` records client frames."""

    def __init__(self) -> None:
        self.sent: list[object] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.on_send = None

    async def send(self, data: object) -> None:
        self.sent.append(data)
        if self.on_send:
            self.on_send(data)

    def feed(self, data: object) -> None:
        self.inbox.put_nowait(data)

    async def close(self) -> None:
        self.inbox.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return item


def make_log() -> ev.EventLog:
    return ev.EventLog(ev.Clock())


def kinds(log: ev.EventLog) -> list[str]:
    return [e.kind for e in log.events]


def tone(ms: int, rate: int = 16000) -> bytes:
    n = rate * ms // 1000
    return (8000 * np.sin(2 * np.pi * 220 * np.arange(n) / rate)).astype("<i2").tobytes()


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def metadata(output_format: str = "pcm_16000") -> str:
    return json.dumps({"type": "conversation_initiation_metadata", "conversation_initiation_metadata_event": {
        "conversation_id": "conv_1", "agent_output_audio_format": output_format,
        "user_input_audio_format": "pcm_16000"}})


def adapter_with(socket: FakeSocket, output_format: str = "pcm_16000", **config) -> ElevenLabsAgentsAdapter:
    async def signer(key, agent_id):
        assert (key, agent_id) == ("k", "agent_1")
        return "wss://signed"

    async def connector(url):
        assert url == "wss://signed"
        return socket

    def reply(data):
        if json.loads(data).get("type") == "conversation_initiation_client_data":
            socket.feed(metadata(output_format))

    socket.on_send = reply
    return ElevenLabsAgentsAdapter(api_key="k", agent_id="agent_1", log=make_log(), config=SessionConfig(**config),
                                   connector=connector, signer=signer)


def audio(pcm: bytes, event_id: int = 1) -> str:
    return json.dumps({"type": "audio", "audio_event": {"audio_base_64": base64.b64encode(pcm).decode(),
                                                        "event_id": event_id}})


CONFIG = {
    "conversation_config": {"agent": {"language": "en", "prompt": {"llm": "some-llm", "temperature": 0}},
                            "tts": {"voice_id": "v1"}, "turn": {"soft_timeout_config": {"timeout_seconds": 2.5}}},
    "platform_settings": {"summary_language": "en"},
}


def test_agent_payload_keeps_config_and_sets_ours(tmp_path):
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(CONFIG))
    config = load_agent_config(path)
    body = agent_payload(SYSTEM_PROMPT, config)
    agent = body["conversation_config"]["agent"]
    assert agent["prompt"]["prompt"] == SYSTEM_PROMPT and agent["first_message"] == ""
    assert agent["prompt"]["llm"] == "some-llm"
    assert body["conversation_config"]["turn"] == CONFIG["conversation_config"]["turn"]
    assert body["conversation_config"]["tts"]["agent_output_audio_format"] == "pcm_16000"
    assert body["platform_settings"]["overrides"]["conversation_config_override"]["agent"]["prompt"]["prompt"]
    assert "prompt" not in config["conversation_config"]["agent"]["prompt"]  # the loaded config is untouched
    with pytest.raises(AdapterError, match="ELEVENLABS_AGENT_CONFIG"):
        load_agent_config(tmp_path / "missing.json")
    assert provider_entry("elevenlabs-agents").adapter is ElevenLabsAgentsAdapter


def test_session_reply_ping_and_interruption():
    async def run():
        socket = FakeSocket()
        adapter = adapter_with(socket, instructions="Be brief.")
        await adapter.connect()
        sent = json.loads(socket.sent[0])
        assert sent["conversation_config_override"]["agent"] == {"prompt": {"prompt": "Be brief."}, "first_message": ""}
        assert adapter.session_id == "conv_1"

        await adapter.send_audio(tone(20))
        assert "user_audio_chunk" in json.loads(socket.sent[-1])

        socket.feed(json.dumps({"type": "ping", "ping_event": {"event_id": 7, "ping_ms": 40}}))
        socket.feed(json.dumps({"type": "user_transcript", "user_transcription_event": {"user_transcript": "Hi"}}))
        socket.feed(json.dumps({"type": "agent_response", "agent_response_event": {"agent_response": "Hello."}}))
        socket.feed(audio(tone(200)))
        await settle()
        assert json.loads(socket.sent[-1]) == {"type": "pong", "event_id": 7}
        assert adapter.agent_speaking and adapter.caller_text == ["Hi"] and adapter.agent_text == ["Hello."]

        await asyncio.sleep(REPLY_GAP_S + 0.1)  # no audio for the gap: the reply is over
        assert kinds(adapter.log).count(ev.AGENT_AUDIO_END) == 1

        socket.feed(audio(tone(2000), 2))
        socket.feed(json.dumps({"type": "interruption", "interruption_event": {"event_id": 2}}))
        await settle()
        assert ev.AGENT_INTERRUPTED in kinds(adapter.log)
        assert [e.data["status"] for e in adapter.log.of_kind(ev.RESPONSE_DONE)] == ["completed", "cancelled"]
        assert not adapter.agent_speaking  # buffered playout was dropped at the interruption
        await adapter.close()

    asyncio.run(run())


def test_refuses_wrong_audio_format_and_missing_agent():
    async def run():
        with pytest.raises(AdapterError, match="agent_output_audio_format"):
            await adapter_with(FakeSocket(), output_format="pcm_44100").connect()
        missing = ElevenLabsAgentsAdapter(api_key="k", agent_id="", log=make_log())
        missing._agent_id = None
        with pytest.raises(AdapterError, match="ELEVENLABS_AGENT_ID"):
            await missing.connect()

    asyncio.run(run())
