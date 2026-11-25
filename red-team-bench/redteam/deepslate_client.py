"""Deepslate Opal in voice, turn by turn, set up as the speech-to-speech benchmark's bot sets it up
(cekura-bench's reference-agents/pipecat-s2s/deepslate_realtime.py and bot.py's _deepslate): the session configured once at 16 kHz
before any audio, the bot's detector settings and temperature, replies voiced by ElevenLabs with the cascade rows'
voice and model, and the tools sent right after the configuration. The wire format is protobuf; only the vendor
SDK's generated classes are used, never its session layer, which reconnects on its own into a session that remembers
nothing.

The protocol has no way to seed earlier turns (text input would be heard as the caller speaking), so each turn opens
a session whose system prompt carries the call so far as a note, and a refused turn is erased by leaving it out. The
greeting is already in that note and the opening rule is left out, so the agent does not greet again. The newest
caller turn is streamed at speaking pace and followed by an open line's silence; the service's detector decides
when the caller has finished. Tool calls go to the conversation and their results come back into the same session.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

from deepslate.core.proto import realtime_pb2 as proto
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from . import harness_config as C
from .voice_clients import (RESPONSE_SECONDS, VAPI_QUIET_AFTER_TOOL_S, VAPI_QUIET_S, VAPI_SILENT_S, ElevenAgentVoice,
                       Speech, VapiVoice, _Lag, _answer, _inlined, _spoken_turn, _split, _to16k, _voiced)

URL = "wss://app.deepslate.eu/api/v1/vendors/{vendor}/organizations/{organization}/realtime"
RATE, FRAME = 16000, 3200  # 16 kHz both ways; 3200 bytes: 100 ms
PER_MINUTE = 0.10  # no published rate: an estimate, to be checked against the bill
# Sessions opened too fast are refused (HTTP 429): at most this many open in a process, and a refused one waits
DEEPSLATE_LINES, DEEPSLATE_BUSY_WAITS = 9, (2, 5, 10, 20, 30, None)
# bot.py's settings: the vendor SDK's detector defaults (390 ms of silence ends a turn), temperature 0.3, and the
# cascade rows' ElevenLabs voice and model
VAD = {"confidence_threshold": 0.4, "min_volume": 0.0, "start_ms": 150, "stop_ms": 390, "backbuffer_ms": 1000}
TEMPERATURE, TTS_VOICE, TTS_MODEL = 0.3, "21m00Tcm4TlvDq8ikWAM", "eleven_flash_v2_5"


def _duration(ms: int) -> proto.Duration:
    return proto.Duration(seconds=ms // 1000, nanos=(ms % 1000) * 1_000_000)


def _whole(value):
    """A Struct carries one number type: whole floats back to the integers the model wrote."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _whole(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_whole(v) for v in value]
    return value


class DeepslateVoice(VapiVoice):
    _slots: asyncio.Semaphore | None = None  # sessions open in this process

    def __init__(self, model: str, purpose: str, speech: Speech):
        self._model, self._purpose, self._speech = model, purpose, speech
        self._env = C.load_env(C.ENV_FILE)
        self._open = self._waiting = None  # (socket, state, turns it has seen, tasks, queue); (call ids, future)
        self._lag = _Lag(purpose)
        self.chat = self.completions = self

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        told, turns = _split(messages, opening=False)
        try:
            async with asyncio.timeout(RESPONSE_SECONDS):
                waiting = self._waiting
                after_tool = self._resume(turns)
                if after_tool:
                    ws, state = self._open[:2]
                    state.reply_at = None  # the reply is what it says once the results are back
                    for r in waiting[1].result():  # the service needs an answer to every call, as a string
                        await ws.send(proto.ServiceBoundMessage(tool_call_response=proto.ToolCallResponse(
                            id=r["toolCallId"], result=r["result"])).SerializeToString())
                else:
                    await self._close()
                    await self._start(told, turns, tools)
                text, calls = await self._listen(VAPI_QUIET_AFTER_TOOL_S if after_tool else VAPI_QUIET_S)
        except BaseException:
            await self._close()
            raise
        if calls:
            self._open = (*self._open[:2], turns, *self._open[3:])
        else:
            await self._close()
        return _answer(text.strip(), calls, SimpleNamespace(prompt_tokens=0, total_tokens=0))

    async def _start(self, told: str, turns: list[dict], tools: list[dict] | None) -> None:
        import websockets
        k = _spoken_turn(turns)
        note = ElevenAgentVoice._notes((turns if k is None else turns[:k])[1:])  # the greeting and what followed
        prompt = f"{told}\n\nThe call so far, oldest first. Carry on from here, without greeting again:\n{note}" if note else told
        DeepslateVoice._slots = DeepslateVoice._slots or asyncio.Semaphore(DEEPSLATE_LINES)
        await DeepslateVoice._slots.acquire()
        self._slot = True
        for wait in DEEPSLATE_BUSY_WAITS:
            try:
                ws = await websockets.connect(
                    URL.format(vendor=self._env["DEEPSLATE_VENDOR_ID"], organization=self._env["DEEPSLATE_ORGANIZATION_ID"]),
                    additional_headers={"Authorization": f"Bearer {self._env['DEEPSLATE_API_KEY']}"}, max_size=None)
                break
            except websockets.InvalidStatus as exc:
                if exc.response.status_code != 429 or wait is None:
                    raise
                await asyncio.sleep(wait)
        line = proto.AudioLineConfiguration(sample_rate=RATE, channel_count=1, sample_format=proto.SampleFormat.SIGNED_16_BIT)
        await ws.send(proto.ServiceBoundMessage(initialize_session_request=proto.InitializeSessionRequest(
            input_audio_line=line, output_audio_line=line,
            vad_configuration=proto.VadConfiguration(
                confidence_threshold=VAD["confidence_threshold"], min_volume=VAD["min_volume"],
                start_duration=_duration(VAD["start_ms"]), stop_duration=_duration(VAD["stop_ms"]),
                backbuffer_duration=_duration(VAD["backbuffer_ms"])),
            inference_configuration=proto.InferenceConfiguration(system_prompt=prompt, temperature=TEMPERATURE),
            tts_configuration=proto.TtsConfiguration(eleven_labs=proto.ElevenLabsTtsConfiguration(
                api_key=self._env["ELEVENLABS_API_KEY"], voice_id=TTS_VOICE, model_id=TTS_MODEL,
                location=proto.ElevenLabsLocation.Value("US"))),
            supports_playback_reporting=False)).SerializeToString())
        definitions = []
        for f in (_inlined(t)["function"] for t in tools or []):
            parameters = Struct()
            json_format.ParseDict(f.get("parameters") or {"type": "object", "properties": {}}, parameters)
            definitions.append(proto.ToolDefinition(name=f["name"], description=f.get("description", ""),
                                                    parameters=parameters))
        # the tools follow the configuration directly, so the session never answers without them
        await ws.send(proto.ServiceBoundMessage(update_tool_definitions_request=proto.UpdateToolDefinitionsRequest(
            tool_definitions=definitions)).SerializeToString())
        async with asyncio.timeout(10):
            while True:
                m = proto.ClientBoundMessage()
                m.ParseFromString(await ws.recv())
                if m.WhichOneof("payload") == "session_ready":
                    break
                if m.WhichOneof("payload") == "error":
                    raise RuntimeError(f"deepslate start: {m.error.message}")
        # turn: the reply being delivered; cut: replies the caller talked over (their words are not the reply)
        state = SimpleNamespace(spoken_at=None, play_end=0.0, voice_at=None, reply_at=None, turn=None, ended=set(),
                                cut=set(), t0=time.time(), closing=False)
        queue = asyncio.Queue()

        async def read():  # the reply's words and audio, its tool calls, and the socket's end
            try:
                async for raw in ws:
                    m = proto.ClientBoundMessage()
                    m.ParseFromString(raw)
                    kind, now = m.WhichOneof("payload"), time.time()
                    if kind == "model_audio_chunk":  # it comes faster than it plays: playback is tracked by length
                        chunk = m.model_audio_chunk
                        turn = chunk.turn_id if chunk.HasField("turn_id") else state.turn
                        state.play_end = max(state.play_end, now) + len(chunk.audio.data) / (2 * RATE)
                        if _voiced(chunk.audio.data) and turn not in state.cut:
                            await queue.put(("voice", now))
                    elif kind == "model_text_fragment":
                        part = m.model_text_fragment
                        await queue.put(("words", (part.turn_id if part.HasField("turn_id") else state.turn, part.text)))
                    elif kind == "response_begin":
                        state.turn = m.response_begin.turn_id
                    elif kind == "response_end":
                        state.ended.add(m.response_end.turn_id)
                    elif kind == "playback_clear_buffer":  # the caller spoke: a reply still being delivered is cut
                        if state.turn is not None and (state.turn not in state.ended or now < state.play_end):
                            state.cut.add(state.turn)
                    elif kind == "tool_call_request":
                        await queue.put(("call", m.tool_call_request))
                    elif kind == "error":
                        await queue.put(("error", m.error.message))
            except websockets.ConnectionClosed:
                pass
            await queue.put(("closed", (ws.close_code, ws.close_reason)))
        spoken = b"" if k is None else _to16k(await self._speech(turns[k]["content"]))

        async def speak():  # the line at speaking pace, then an open line's silence
            packet = 0

            async def send(chunk: bytes) -> None:
                nonlocal packet
                packet += 1
                await ws.send(proto.ServiceBoundMessage(user_input=proto.UserInput(
                    packet_id=packet, mode=proto.InferenceTriggerMode.IMMEDIATE,
                    audio_data=proto.AudioData(data=chunk))).SerializeToString())
                await asyncio.sleep(FRAME / (2 * RATE))
            for i in range(0, len(spoken), FRAME):
                await send(spoken[i:i + FRAME].ljust(FRAME, b"\0"))
            state.spoken_at = time.time()
            if spoken:
                self._lag.spoke()
            while True:
                await send(b"\0" * FRAME)
        self._open = (ws, state, turns, [asyncio.create_task(read()), asyncio.create_task(speak())], queue)

    async def _listen(self, quiet: float) -> tuple[str, list]:
        """The agent's reply once the caller's line has been spoken, until it has played out and the line has been
        quiet a while; or the tools it calls (several can come together). Replies the caller talked over are not it."""
        _, state, _, _, queue = self._open
        words, calls = [], []  # words: (turn, text)
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), 0.3)
            except TimeoutError:
                now, said = time.time(), "".join(t for turn, t in words if turn not in state.cut).strip()
                if calls:  # the round's calls are all in
                    self._lag.tool_round()
                    self._waiting = ([c.id for c in calls], asyncio.get_running_loop().create_future())
                    return said, [(c.id, c.name, json.dumps(_whole(json_format.MessageToDict(c.parameters))
                                                            if c.HasField("parameters") else {})) for c in calls]
                if (said and state.spoken_at and state.turn in state.ended and state.voice_at
                        and now > state.play_end + quiet):
                    self._lag.answered(state.reply_at)
                    return said, []
                if state.spoken_at and not said and now - max(state.spoken_at, state.play_end) > VAPI_SILENT_S:
                    return "", []
                continue
            if kind == "voice":
                if state.spoken_at:
                    self._lag.sound(item)
                    state.voice_at, state.reply_at = item, state.reply_at or item
            elif kind == "words":
                words.append(item)
            elif kind == "call":
                calls.append(item)
            elif kind == "error":
                raise RuntimeError(f"deepslate error: {item}")
            elif kind == "closed":  # a session the far end dropped would not know the call: never the agent's silence
                raise RuntimeError(f"deepslate closed the session: {item}")

    async def _close(self) -> None:
        self._waiting = None
        if getattr(self, "_slot", False):  # create() closes on any failure, so the slot is always given back
            DeepslateVoice._slots.release()
            self._slot = False
        if self._open:
            (ws, state, _, tasks, _), self._open = self._open, None
            state.closing = True
            for task in tasks:
                task.cancel()
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - a socket already closed
                pass
            C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "deepslate", "model": self._model,
                                      "purpose": self._purpose, "in": 0, "out": 0,
                                      "usd": round((time.time() - state.t0) / 60 * PER_MINUTE, 6)})
