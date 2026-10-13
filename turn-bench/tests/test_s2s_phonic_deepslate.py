"""Offline tests for the Phonic and Deepslate S2S adapters, over a fake websocket.

No network: each test scripts the provider's frames and checks what the adapter
sent, what it recorded and which normalized events it emitted.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

from turn_bench.author import service_bridge

service_bridge._ensure_path()

from service import events as ev  # noqa: E402
from service.adapters.base import AdapterError, SessionConfig, ToolSpec  # noqa: E402

from turn_bench.s2s.phonic import PhonicAdapter, strict_parameters  # noqa: E402


class FakeSocket:
    """A websocket stand-in: ``feed`` queues a server frame, ``sent`` records client frames."""

    def __init__(self) -> None:
        self.sent: list[object] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.on_send = None
        self.close_code = None
        self.close_reason = None

    async def send(self, data: object) -> None:
        self.sent.append(data)
        if self.on_send:
            self.on_send(data)

    def feed(self, data: object) -> None:
        self.inbox.put_nowait(data)

    def end(self) -> None:
        self.inbox.put_nowait(None)

    async def close(self) -> None:
        self.end()

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


# ── Phonic ───────────────────────────────────────────────────────────────────


def phonic_with(socket: FakeSocket, **config) -> PhonicAdapter:
    async def connector(url, *, headers):
        assert headers == {"Authorization": "Bearer k"}
        return socket

    def reply_to_config(data):
        if json.loads(data).get("type") == "config":
            socket.feed(json.dumps({"type": "conversation_created", "conversation_id": "conv_1"}))
            socket.feed(json.dumps({"type": "ready_to_start_conversation"}))

    socket.on_send = reply_to_config
    return PhonicAdapter(api_key="k", log=make_log(), config=SessionConfig(**config), connector=connector)


def audio_frame(pcm: bytes, text: str = "") -> str:
    return json.dumps({"type": "audio_chunk", "audio": base64.b64encode(pcm).decode(), "text": text})


def test_phonic_session_reply_and_barge_in():
    async def run():
        socket = FakeSocket()
        adapter = phonic_with(socket, instructions="Be brief.")
        await adapter.connect()
        config = json.loads(socket.sent[0])
        assert config["phonic_model"] == "phonic_v1" and config["voice_id"] == "sabrina"
        assert config["input_format"] == config["output_format"] == "pcm_16000"
        assert config["vad_min_silence_duration_ms"] == 800 and config["min_words_to_interrupt"] == 1
        assert adapter.session_id == "conv_1"
        assert set(adapter.session_ack) == {"conversation_created", "ready_to_start_conversation"}

        await adapter.send_audio(tone(20))
        assert json.loads(socket.sent[-1])["type"] == "audio_chunk"
        assert adapter.caller_timeline.n_samples == 320

        socket.feed(audio_frame(bytes(640)))  # between-reply silence: dropped
        socket.feed(json.dumps({"type": "user_started_speaking"}))
        socket.feed(json.dumps({"type": "user_finished_speaking"}))
        socket.feed(json.dumps({"type": "input_text", "text": "Hi there"}))
        socket.feed(json.dumps({"type": "assistant_started_speaking"}))
        socket.feed(audio_frame(tone(20), "Hello, "))
        socket.feed(audio_frame(tone(20), "how can I help?"))
        socket.feed(json.dumps({"type": "assistant_finished_speaking"}))
        socket.feed(audio_frame(bytes(640)))  # silence again: dropped
        # A second reply, talked over.
        socket.feed(audio_frame(tone(20), "Sure"))  # text before started_speaking opens the reply
        socket.feed(json.dumps({"type": "interrupted_response", "text": "Sure"}))
        socket.feed(json.dumps({"type": "assistant_finished_speaking"}))
        await settle()

        assert adapter.agent_timeline.n_samples == 3 * 320
        assert adapter.caller_text == ["Hi there"]
        assert adapter.agent_text == ["Hello, how can I help?", "Sure"]
        k = kinds(adapter.log)
        assert k.count(ev.AGENT_AUDIO_START) == 2 and k.count(ev.AGENT_AUDIO_END) == 2
        assert k.count(ev.AGENT_INTERRUPTED) == 1
        assert ev.VAD_SPEECH_START in k and ev.VAD_SPEECH_END in k
        statuses = [e.data["status"] for e in adapter.log.of_kind(ev.RESPONSE_DONE)]
        assert statuses == ["completed", "cancelled"]
        await adapter.close()
        assert adapter.closed.is_set()

    asyncio.run(run())


def test_phonic_refusal_and_capabilities():
    async def run():
        socket = FakeSocket()

        async def connector(url, *, headers):
            return socket

        socket.on_send = lambda data: (
            socket.feed(json.dumps({"type": "error", "error": {"message": "bad key"}})), socket.end()
        )
        adapter = PhonicAdapter(api_key="k", log=make_log(), connector=connector)
        with pytest.raises(AdapterError, match="never became ready"):
            await adapter.connect()

    asyncio.run(run())
    assert PhonicAdapter.unsupported_reason(SessionConfig(turn_detection=type(SessionConfig().turn_detection)(mode="manual")))
    assert PhonicAdapter.input_rate == PhonicAdapter.output_rate == 16000


def test_phonic_strict_tools_and_null_arguments():
    schema = {"type": "object", "properties": {"zip": {"type": "string"}, "note": {"type": "string"}}, "required": ["zip"]}
    strict = strict_parameters(schema)
    assert strict["required"] == ["zip", "note"] and strict["properties"]["note"]["type"] == ["string", "null"]

    async def run():
        socket = FakeSocket()
        adapter = phonic_with(socket, tools=(ToolSpec("lookup", "Look up.", schema),))
        await adapter.connect()
        socket.feed(json.dumps({"type": "tool_call", "tool_call_id": "c1", "tool_name": "lookup",
                                "parameters": {"zip": "null", "note": "null"}}))
        await settle()
        assert adapter.tool_calls == [{"name": "lookup", "call_id": "c1", "arguments": {"zip": "null"}}]
        await adapter.send_tool_result("c1", {"ok": True})
        assert json.loads(socket.sent[-1]) == {"type": "tool_call_output", "tool_call_id": "c1", "output": {"ok": True}}
        await adapter.close()

    asyncio.run(run())


# ── Deepslate ────────────────────────────────────────────────────────────────

try:
    from deepslate.core.proto import realtime_pb2 as proto
except ImportError:  # the adapter's one extra dependency, deepslate-core==0.1.20
    proto = None

needs_deepslate = pytest.mark.skipif(proto is None, reason="deepslate-core not installed")


def ds_frame(**payload) -> bytes:
    return proto.ClientBoundMessage(**payload).SerializeToString()


def deepslate_with(socket: FakeSocket):
    from turn_bench.s2s.deepslate import DeepslateAdapter

    seen = {}

    async def connector(url, *, headers):
        seen["url"] = url
        return socket

    def reply_to_init(data):
        message = proto.ServiceBoundMessage()
        message.ParseFromString(data)
        if message.WhichOneof("payload") == "update_tool_definitions_request":
            socket.feed(ds_frame(session_ready=proto.SessionReady()))

    socket.on_send = reply_to_init
    adapter = DeepslateAdapter(api_key="k", vendor_id="v", organization_id="o", elevenlabs_api_key="secret-el",
                               log=make_log(), config=SessionConfig(instructions="Be brief."), connector=connector)
    return adapter, seen


@needs_deepslate
def test_deepslate_session_reply_and_barge_in():
    async def run():
        socket = FakeSocket()
        adapter, seen = deepslate_with(socket)
        await adapter.connect()
        assert seen["url"].endswith("/vendors/v/organizations/o/realtime")
        init = proto.ServiceBoundMessage()
        init.ParseFromString(socket.sent[0])
        request = init.initialize_session_request
        assert request.input_audio_line.sample_rate == request.output_audio_line.sample_rate == 16000
        assert request.vad_configuration.stop_duration.nanos == 390_000_000
        assert request.tts_configuration.eleven_labs.model_id == "eleven_flash_v2_5"
        # The ElevenLabs key goes on the wire but never into the record.
        assert "secret-el" not in json.dumps(adapter.state())
        assert adapter.session_ack["session_ready"] == {}

        await adapter.send_audio(tone(20))
        sent = proto.ServiceBoundMessage()
        sent.ParseFromString(socket.sent[-1])
        assert sent.user_input.packet_id == 1 and len(sent.user_input.audio_data.data) == 640

        S = proto.VadState
        socket.feed(ds_frame(vad_state_event=proto.VadStateEvent(from_state=S.SPEECH_STARTING, to_state=S.SPEECH)))
        socket.feed(ds_frame(vad_state_event=proto.VadStateEvent(from_state=S.SPEECH_ENDING, to_state=S.SILENCE)))
        socket.feed(ds_frame(user_transcription_result=proto.UserTranscriptionResult(text="Hi there")))
        socket.feed(ds_frame(playback_clear_buffer=proto.PlaybackClearBuffer()))  # nothing playing: not a cut
        socket.feed(ds_frame(response_begin=proto.ResponseBegin(turn_id=0)))
        socket.feed(ds_frame(model_text_fragment=proto.ModelTextFragment(text="Hello.", turn_id=0)))
        socket.feed(ds_frame(model_audio_chunk=proto.ModelAudioChunk(audio=proto.AudioData(data=tone(20)), turn_id=0)))
        socket.feed(ds_frame(response_end=proto.ResponseEnd(turn_id=0)))
        socket.feed(ds_frame(response_begin=proto.ResponseBegin(turn_id=1)))
        socket.feed(ds_frame(model_audio_chunk=proto.ModelAudioChunk(audio=proto.AudioData(data=tone(20)), turn_id=1)))
        socket.feed(ds_frame(playback_clear_buffer=proto.PlaybackClearBuffer()))  # a real barge-in
        socket.feed(ds_frame(model_audio_chunk=proto.ModelAudioChunk(audio=proto.AudioData(data=tone(20)), turn_id=1)))
        socket.feed(ds_frame(response_end=proto.ResponseEnd(turn_id=1)))
        # A reply shipped in one burst: generation ends while it is still playing,
        # so a clear-buffer after response_end still cuts it.
        socket.feed(ds_frame(response_begin=proto.ResponseBegin(turn_id=3)))
        socket.feed(ds_frame(model_audio_chunk=proto.ModelAudioChunk(audio=proto.AudioData(data=tone(2000)), turn_id=3)))
        socket.feed(ds_frame(response_end=proto.ResponseEnd(turn_id=3)))
        socket.feed(ds_frame(playback_clear_buffer=proto.PlaybackClearBuffer()))
        await settle()

        assert adapter.agent_timeline.n_samples == 2 * 320 + 32000  # the cut turn's late chunk is dropped
        assert adapter.agent_timeline.playout_end() < adapter.agent_timeline.chunks[-1].t_wall + 1.0
        assert adapter.caller_text == ["Hi there"] and adapter.agent_text == ["Hello."]
        k = kinds(adapter.log)
        assert k.count(ev.AGENT_AUDIO_START) == 3 and k.count(ev.AGENT_INTERRUPTED) == 2
        assert ev.VAD_SPEECH_START in k and ev.VAD_SPEECH_END in k
        statuses = [e.data["status"] for e in adapter.log.of_kind(ev.RESPONSE_DONE)]
        assert statuses == ["completed", "cancelled", "completed"]
        await adapter.close()

    asyncio.run(run())


@needs_deepslate
def test_deepslate_tool_call_numbers_and_missing_account():
    from google.protobuf.struct_pb2 import Struct

    from turn_bench.s2s.deepslate import DeepslateAdapter

    async def run():
        socket = FakeSocket()
        adapter, _ = deepslate_with(socket)
        await adapter.connect()
        params = Struct()
        params.update({"zip": 90210, "name": "x"})
        socket.feed(ds_frame(tool_call_request=proto.ToolCallRequest(id="c1", name="lookup", parameters=params)))
        await settle()
        assert adapter.tool_calls == [{"name": "lookup", "call_id": "c1", "arguments": {"zip": 90210, "name": "x"}}]
        await adapter.close()

        missing = DeepslateAdapter(api_key="k", vendor_id="", organization_id="o", elevenlabs_api_key="e", log=make_log())
        missing._vendor_id = None
        with pytest.raises(AdapterError, match="DEEPSLATE_VENDOR_ID"):
            await missing.connect()

    asyncio.run(run())
