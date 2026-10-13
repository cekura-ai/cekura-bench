"""Nova Sonic adapter against a scripted bidirectional stream (no network, no SDK)."""

import asyncio
import base64
import json

import pytest

from turn_taking_bench.s2s import nova_sonic
from turn_taking_bench.s2s.nova_sonic import RESPONSE_START, NovaSonicAdapter

from service import events as ev
from service.adapters.base import AdapterError, SessionConfig, TurnDetection


class FakeStream:
    """Records every client event; replays server events pushed by the test."""

    def __init__(self, refuse: str | None = None):
        self.sent: list[dict] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.refuse = refuse
        self.closed = False

    async def send(self, data: bytes) -> None:
        self.sent.append(json.loads(data)["event"])

    async def ready(self) -> None:
        if self.refuse:
            raise RuntimeError(self.refuse)

    async def receive(self):
        item = await self.inbox.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True
        await self.inbox.put(None)

    def push(self, event: dict) -> None:
        self.inbox.put_nowait(json.dumps({"event": event}).encode())

    def kinds(self) -> list[str]:
        return [next(iter(e)) for e in self.sent]


def make(stream: FakeStream, **config) -> NovaSonicAdapter:
    async def factory():
        return stream

    log = ev.EventLog(ev.Clock())
    return NovaSonicAdapter(
        api_key="unused", region="us-west-2", log=log, config=SessionConfig(**config), transport_factory=factory
    )


def usage(in_speech: int, out_speech: int, out_text: int) -> dict:
    return {
        "usageEvent": {
            "details": {
                "total": {
                    "input": {"speechTokens": in_speech, "textTokens": 10},
                    "output": {"speechTokens": out_speech, "textTokens": out_text},
                }
            }
        }
    }


def content_start(role: str, kind: str, stage: str | None = None) -> dict:
    start = {"role": role, "type": kind}
    if stage:
        start["additionalModelFields"] = json.dumps({"generationStage": stage})
    return {"contentStart": start}


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


def kinds(adapter: NovaSonicAdapter) -> list[str]:
    return [e.kind for e in adapter.log.events]


def test_session_setup_audio_framing_reply_and_close():
    async def scenario():
        stream = FakeStream()
        adapter = make(stream, instructions="Be brief.", first_message="Hello, how can I help?")
        await adapter.connect()

        # Required opening order: session, prompt, system text, history text, audio content.
        assert stream.kinds() == [
            "sessionStart", "promptStart",
            "contentStart", "textInput", "contentEnd",
            "contentStart", "textInput", "contentEnd",
            "contentStart",
        ]
        assert stream.sent[0]["sessionStart"]["turnDetectionConfiguration"] == {"endpointingSensitivity": "MEDIUM"}
        assert stream.sent[1]["promptStart"]["audioOutputConfiguration"]["voiceId"] == "matthew"
        assert [stream.sent[i]["contentStart"]["role"] for i in (2, 5)] == ["SYSTEM", "ASSISTANT"]
        assert stream.sent[8]["contentStart"]["audioInputConfiguration"]["sampleRateHertz"] == 16000

        # 50 ms of caller audio at 16 kHz -> two 20 ms frames now, 10 ms held back.
        pcm = (b"\x01\x00" * 800)
        await adapter.send_audio(pcm)
        frames = [e["audioInput"] for e in stream.sent if "audioInput" in e]
        assert [len(base64.b64decode(f["content"])) for f in frames] == [640, 640]
        assert adapter.caller_timeline.n_samples == 800

        stream.push(usage(100, 0, 0))
        stream.push({"userSpeechStart": {"inputAudioOffsetMs": 3800, "sessionId": "sess-1"}})
        stream.push({"userSpeechEnd": {"inputAudioOffsetMs": 6680, "inputAudioDetectionOffsetMs": 7080, "sessionId": "sess-1"}})
        stream.push({"completionStart": {"sessionId": "sess-1", "promptName": "p"}})
        stream.push(content_start("USER", "TEXT"))
        stream.push({"textOutput": {"content": "I need to reschedule."}})
        stream.push({"contentEnd": {"stopReason": "END_TURN"}})
        stream.push(content_start("ASSISTANT", "TEXT", "SPECULATIVE"))
        stream.push({"textOutput": {"content": "Sure, which day?"}})
        stream.push({"contentEnd": {"stopReason": "PARTIAL_TURN"}})
        stream.push(content_start("ASSISTANT", "AUDIO"))
        reply = b"\x00\x10" * 2400  # 100 ms at 24 kHz
        stream.push({"audioOutput": {"content": base64.b64encode(reply).decode()}})
        stream.push({"audioOutput": {"content": base64.b64encode(reply).decode()}})
        stream.push(usage(150, 40, 5))
        stream.push({"contentEnd": {"stopReason": "END_TURN"}})
        stream.push(content_start("ASSISTANT", "TEXT", "FINAL"))
        stream.push({"textOutput": {"content": "Sure, which day?"}})
        stream.push({"contentEnd": {"stopReason": "END_TURN"}})
        stream.push(usage(220, 40, 5))
        await settle()

        assert adapter.session_id == "sess-1"
        assert adapter.session_ack["first_event"]["kind"] == "usageEvent"
        assert adapter.session_ack["provider_session_id"] == "sess-1"
        assert adapter.log.first(ev.VAD_SPEECH_START).data["audio_ms"] == 3800
        assert adapter.log.first(ev.VAD_SPEECH_END).data == {"audio_ms": 6680, "detection_audio_ms": 7080}
        assert len(adapter.agent_pcm) == 2 * len(reply)
        assert adapter.caller_text == ["I need to reschedule."]

        await adapter.close()
        # The held 10 ms is flushed, then the mirror-image teardown.
        tail = stream.kinds()[-4:]
        assert tail == ["audioInput", "contentEnd", "promptEnd", "sessionEnd"]
        assert len(base64.b64decode(stream.sent[-4]["audioInput"]["content"])) == 320
        assert stream.closed
        return adapter

    adapter = asyncio.run(scenario())
    order = [k for k in kinds(adapter) if k in (RESPONSE_START, ev.AGENT_AUDIO_START, ev.AGENT_AUDIO_END, ev.RESPONSE_DONE)]
    assert order == [RESPONSE_START, ev.AGENT_AUDIO_START, ev.AGENT_AUDIO_END, ev.RESPONSE_DONE, ev.RESPONSE_DONE]
    done = list(adapter.log.of_kind(ev.RESPONSE_DONE))
    assert done[0].data == {
        "status": "completed",
        "usage": {"prompt_tokens": 160, "input_audio_tokens": 150, "completion_tokens": 45, "output_audio_tokens": 40},
    }
    # Caller audio streamed after the reply is billed too: the remainder lands at close.
    assert done[1].data["status"] == "session_usage"
    assert done[1].data["usage"]["input_audio_tokens"] == 70
    assert adapter.agent_text == ["Sure, which day?"]
    assert adapter.state()["nova_sonic"]["resolved_model"] == "amazon.nova-2-sonic-v1:0"
    assert adapter.closed.is_set()


def test_barge_in_markers_normalize_to_one_interruption():
    async def scenario():
        stream = FakeStream()
        adapter = make(stream)
        await adapter.connect()
        stream.push(content_start("ASSISTANT", "TEXT", "SPECULATIVE"))
        stream.push({"contentEnd": {"stopReason": "PARTIAL_TURN"}})
        stream.push(content_start("ASSISTANT", "AUDIO"))
        stream.push({"audioOutput": {"content": base64.b64encode(b"\x00\x10" * 4800).decode()}})
        stream.push(content_start("ASSISTANT", "TEXT", "FINAL"))
        stream.push({"textOutput": {"content": '{ "interrupted" : true }'}})
        stream.push({"contentEnd": {"stopReason": "INTERRUPTED"}})
        await settle()
        await adapter.close()
        return adapter

    adapter = asyncio.run(scenario())
    assert sum(1 for _ in adapter.log.of_kind(ev.AGENT_INTERRUPTED)) == 1
    assert [e.data["status"] for e in adapter.log.of_kind(ev.RESPONSE_DONE)] == ["cancelled"]
    assert adapter.log.first(ev.AGENT_AUDIO_END).data["reason"] == "cancelled"
    assert adapter.agent_text == []  # the interruption marker is not speech


def test_refusal_and_unsupported_configs():
    async def refused():
        adapter = make(FakeStream(refuse="AccessDeniedException: model not granted"))
        with pytest.raises(AdapterError, match="AccessDenied"):
            await adapter.connect()

    asyncio.run(refused())

    async def dies_mid_session():
        stream = FakeStream()
        adapter = make(stream)
        await adapter.connect()
        stream.inbox.put_nowait(RuntimeError("ValidationException: timed out waiting for input"))
        await settle()
        assert adapter.closed.is_set()
        assert "timed out" in adapter.log.first(ev.SESSION_ERROR).data["error"]

    asyncio.run(dies_mid_session())

    reason = NovaSonicAdapter.unsupported_reason
    assert reason(SessionConfig()) is None
    assert reason(SessionConfig(turn_detection=TurnDetection(mode="manual"))) == "nova-sonic has no manual commit"
    assert "semantic" in reason(SessionConfig(turn_detection=TurnDetection(mode="semantic_vad")))
    assert reason(SessionConfig(turn_detection=TurnDetection(silence_duration_ms=1500))) is None
    assert "no equivalent" in reason(SessionConfig(turn_detection=TurnDetection(silence_duration_ms=500)))
    high = make(FakeStream(), turn_detection=TurnDetection(silence_duration_ms=1500))
    assert high.endpointing_sensitivity() == "HIGH"
    assert nova_sonic.NovaSonicAdapter.input_rate == 16000 and nova_sonic.NovaSonicAdapter.output_rate == 24000
