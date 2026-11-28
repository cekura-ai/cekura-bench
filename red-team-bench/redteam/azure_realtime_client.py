"""Azure Realtime, Microsoft's own speech-to-speech model, on Azure Voice Live, turn by turn for the red-team cells.

Set up as the speech-to-speech benchmark's bot sets it up (cekura-bench's reference-agents/pipecat-s2s/azure_realtime.py): model
azure-realtime, voice Ava. Voice Live speaks the earlier, flat shape of OpenAI's realtime protocol (settings at the top
of the session, response.audio.delta), so the socket is spoken directly rather than through the OpenAI SDK.

Each caller turn opens a session seeded with the conversation kept so far, speaks the caller's newest line into the
input buffer and asks for the reply; a refused turn is erased by leaving it out of the next session. Turn detection
is off, so the model answers only once the whole line is in. A tool round stays in the session that made it: the
results go back into it and the reply is asked for again, as on a call. A response that fails or comes back empty,
such as a rate limit, is asked for again after a pause, never taken as the agent's silence.

The model now and then speaks an answer as JSON ('{"cabin": "business"}'): the airline prompt every agent gets says
"Always make sure you generate valid JSON only", and this model takes it literally. Its words are recorded as said.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from types import SimpleNamespace

from .voice_clients import (C, RESPONSE_SECONDS, Speech, VapiVoice, _answer, _filter_log, _Lag, _spoken_turn, _split,
                       _unwrapped, _voiced)

API_VERSION = "2026-07-15"  # the bot's pinned generally available version
VOICE = "ava"
PER_MINUTE = 0.08  # Voice Live by the session minute, an estimate
RETRY_WAITS = (0, 10, 30, 60)  # seconds before each request of a reply the service answered with nothing
CHUNK = 48000  # one second of 24 kHz PCM16 per append


class AzureRealtimeVoice(VapiVoice):
    """Azure Realtime behind target_server's chat-completions call. Only VapiVoice's _resume is used: it matches the
    conversation's tool results to the calls the open session is waiting on."""

    def __init__(self, model: str, purpose: str, speech: Speech):
        env = C.load_env(C.ENV_FILE)
        host = env["AZURE_VOICE_ENDPOINT"].split("//")[-1].split("/")[0]
        self._url = f"wss://{host}/voice-live/realtime?api-version={API_VERSION}&model={model}"
        self._key, self._model, self._purpose, self._speech = env["AZURE_VOICE_API_KEY"], model, purpose, speech
        self._lag = _Lag(purpose)
        self._open = None  # (socket, when it opened, turns it has seen)
        self._waiting = None  # (tool call ids, the future their results are set on)
        self.chat = self.completions = self

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        told, turns = _split(messages)
        try:
            async with asyncio.timeout(RESPONSE_SECONDS + sum(RETRY_WAITS)):
                waiting = self._waiting
                if self._resume(turns):
                    for r in waiting[1].result():
                        await self._send({"type": "conversation.item.create", "item": {
                            "type": "function_call_output", "call_id": r["toolCallId"], "output": r["result"]}})
                else:
                    await self._close()
                    await self._start(told, turns, tools)
                for wait in RETRY_WAITS:
                    await asyncio.sleep(wait)
                    await self._send({"type": "response.create"})
                    out = await self._listen()
                    if out.text.strip() or out.calls:
                        break
                    _filter_log(self._purpose, "unanswered", out.status)
                else:
                    raise RuntimeError(f"azure-realtime: no answer after {len(RETRY_WAITS)} tries: {out.status}")
        except BaseException:
            await self._close()
            raise
        if out.calls:
            self._lag.tool_round()
            self._waiting = ([c[0] for c in out.calls], asyncio.get_running_loop().create_future())
            self._open = (*self._open[:2], turns)
        else:
            self._lag.answered(out.onset)
            await self._close()
        # the shared prompt asks for "valid JSON only" and it sometimes speaks a reply as JSON: unwrapped as GPT Realtime's
        return _answer(_unwrapped(out.text.strip()), out.calls, SimpleNamespace(prompt_tokens=0, total_tokens=0))

    async def _send(self, event: dict) -> None:
        await self._open[0].send(json.dumps({"event_id": f"event_{uuid.uuid4().hex[:24]}", **event}))

    async def _until(self, kind: str) -> None:
        async for raw in self._open[0]:
            event = json.loads(raw)
            if event.get("type") == "error":
                raise RuntimeError(f"azure-realtime error: {event.get('error')}")
            if event.get("type") == kind:
                return
        raise RuntimeError("azure-realtime: the session closed")

    async def _start(self, told: str, turns: list[dict], tools: list[dict] | None) -> None:
        import websockets
        k = _spoken_turn(turns)
        spoken = await self._speech(turns[k]["content"]) if k is not None else b""
        ws = await websockets.connect(self._url, additional_headers={"api-key": self._key}, max_size=None)
        self._open = (ws, time.time(), turns)
        await self._send({"type": "session.update", "session": {
            "modalities": ["text", "audio"], "instructions": told,
            "voice": {"type": "azure-realtime-native", "name": VOICE},
            "input_audio_format": "pcm16", "input_audio_sampling_rate": 24000, "output_audio_format": "pcm16",
            # turns are committed by hand, so no detector; Azure's transcription needs one, and the model hears the
            # audio, not that transcript (the caller's words are known)
            "turn_detection": None,
            "tools": [{"type": "function", "name": f["name"], "description": f.get("description", ""),
                       "parameters": f.get("parameters") or {"type": "object", "properties": {}}}
                      for f in (t["function"] for t in tools or [])],
            "tool_choice": "auto"}})
        await self._until("session.updated")
        for item in self._items(turns if k is None else turns[:k]):
            await self._send({"type": "conversation.item.create", "item": item})
        if spoken:  # as a call delivers it: the input buffer, committed as one turn
            for i in range(0, len(spoken), CHUNK):
                await self._send({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(spoken[i:i + CHUNK]).decode()})
            await self._send({"type": "input_audio_buffer.commit"})
            self._lag.spoke()
            await self._until("input_audio_buffer.committed")

    async def _listen(self) -> SimpleNamespace:
        """One response: its words, its tool calls, and when its speech started."""
        out = SimpleNamespace(text="", calls=[], onset=None, status="no response")
        async for raw in self._open[0]:
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "response.audio.delta" and _voiced(base64.b64decode(event.get("delta") or "")):
                out.onset = out.onset or time.time()  # the reply is the last response's speech, after any tools
                self._lag.sound(out.onset)
            elif kind == "error":
                out.status = json.dumps(event.get("error"))[:300]
                if "rate" not in out.status.lower() and "server" not in out.status.lower():
                    raise RuntimeError(f"azure-realtime error: {out.status}")
                return out
            elif kind == "response.done":
                r = event.get("response") or {}
                out.status = f"{r.get('status')} {json.dumps(r.get('status_details'))}"[:300]
                for item in r.get("output") or []:
                    if item.get("type") == "message":
                        out.text += "".join(c.get("transcript") or c.get("text") or "" for c in item.get("content") or [])
                    elif item.get("type") == "function_call":
                        out.calls.append((item["call_id"], item["name"], item.get("arguments") or "{}"))
                return out
        raise RuntimeError("azure-realtime: the session closed mid-response")

    async def _close(self) -> None:
        self._waiting = None
        if self._open:
            (ws, opened, _), self._open = self._open, None
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - a session the service already ended
                pass
            C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "azure", "model": self._model,
                                      "purpose": self._purpose, "in": 0, "out": 0,
                                      "usd": round((time.time() - opened) / 60 * PER_MINUTE, 6)})

    @staticmethod
    def _items(turns: list[dict]) -> list[dict]:
        """Earlier turns as items, as RealtimeText replays them, in the flat shape: the agent's words are text content,
        not output_text."""
        items = []
        for m in turns:
            if m["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
                continue
            if m.get("content"):
                kind = "input_text" if m["role"] == "user" else "text"
                items.append({"type": "message", "role": m["role"], "content": [{"type": kind, "text": m["content"]}]})
            items += [{"type": "function_call", "call_id": c["id"], "name": c["function"]["name"],
                       "arguments": c["function"]["arguments"]} for c in m.get("tool_calls") or []]
        return items
