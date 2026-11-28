"""Phonic-1.1 in voice, turn by turn, spoken directly over Phonic's speech-to-speech socket, configured as the
speech-to-speech benchmark's bot configures it (the speech-to-speech benchmark's Pipecat bot).

Each turn is a new Phonic conversation. Phonic takes no history, so the call so far (from the greeting on) goes into
the system prompt as a note, and a refused turn is erased by leaving it out; no reply is asked for at the start, so
the agent does not greet again and answers the caller's line. Its audio streams on at real time, silence between
replies, so a reply is the speech between its assistant_started_speaking and assistant_finished_speaking events.
Tools are strict inline tools: every field is sent, so an optional one accepts null, and nulls are dropped from the
arguments before the conversation runs the tool. A result goes back into the same conversation, which carries on."""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import re
import time
from types import SimpleNamespace

from . import harness_config as C
from .voice_clients import (RESPONSE_SECONDS, VAPI_QUIET_AFTER_TOOL_S, VAPI_QUIET_S, VAPI_SILENT_S, ElevenAgentVoice, Speech,
                       VapiVoice, _Lag, _answer, _backchannel, _inlined, _split, _spoken_turn, _to16k, _voiced)

PHONIC_URL, PHONIC_RATE, PHONIC_FRAME = "wss://api.phonic.ai/v1/sts/ws", 16000, 1280  # 1280 bytes: 40 ms, its longest chunk
PHONIC_VOICE, PHONIC_PER_MINUTE = "sabrina", 0.07  # the bot's voice; an estimated rate
# The bot's settings: its strongest intelligence level, and Phonic's documented detector defaults (800 ms of silence
# ends a turn, one word interrupts)
PHONIC_SETTINGS = {"intelligence_level": "high", "vad_threshold": 0.38, "vad_min_silence_duration_ms": 800,
                   "vad_min_speech_duration_ms": 50, "vad_prebuffer_duration_ms": 500, "min_words_to_interrupt": 1}
PHONIC_TOOL_TIMEOUT_MS = 60000  # its default (5 s) is shorter than a turn of this harness can take
# A part of the agent's speech it started over the caller's line counts as the reply only past this many words, and
# after a reply of backchannels alone the turn waits this long for the answer (as GPT-Live's turns do)
PHONIC_OVER_LINE_WORDS, PHONIC_HOLD_S = 6, 10.0
NULL_MEANS_OMITTED = ("Every argument must be sent: for an optional argument with no real, relevant value, send null, "
                      "which leaves it out.")


def _phrase(text: str) -> re.Pattern[str]:
    return re.compile(r"\s+".join(map(re.escape, text.split())))


# The prompt's rule for optional arguments, restated for strict tools (phonic_realtime.PROMPT_RULES)
PROMPT_RULES = (
    (_phrase("Omit an optional argument entirely when its value is unknown, unavailable, or irrelevant."),
     "When an optional argument's value is unknown, unavailable, or irrelevant, send null: every argument must be "
     "sent, and null leaves it out."),
    (_phrase("Never send null, None, an empty placeholder, or an invented ID as a tool argument. A key set to null is "
             "invalid and is not equivalent to omitting that key."),
     "Never send None, an empty placeholder, or an invented ID as a tool argument."),
)


def _strict(schema: dict) -> dict:
    """A tool's parameters in the strict form Phonic requires: every property required, every object closed, an
    optional property accepting null. A choice of shapes (anyOf) keeps its structured one: strict mode refuses an
    open object."""
    schema = copy.deepcopy(schema)
    if "anyOf" in schema:
        kept = [o for o in schema.pop("anyOf") if o.get("properties")] or [{"type": schema.get("type", "object")}]
        schema = {**kept[0], **{k: v for k, v in schema.items() if k == "description"}}
    if schema.get("type") == "object":
        props, required = schema.setdefault("properties", {}), set(schema.get("required") or ())
        for name, prop in props.items():
            props[name] = _strict(prop)
            if name not in required:
                kind = props[name].get("type")
                props[name]["type"] = [kind, "null"] if isinstance(kind, str) else kind
                props[name]["description"] = f"{props[name].get('description', '').rstrip()} Optional: send null to leave it out.".strip()
        schema["required"], schema["additionalProperties"] = list(props), False
    elif schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        schema["items"] = _strict(schema["items"])
    return schema


class _Dropped(Exception):
    """Phonic closed the conversation mid-turn."""


# The plan refuses conversations past its concurrency limit ("Concurrency limit reached"; 10 at once were accepted)
PHONIC_LINES, PHONIC_BUSY_WAITS = 10, (2, 5, 10, 20, 30, None)


class PhonicVoice(VapiVoice):
    _slots: asyncio.Semaphore | None = None  # conversations open in this process
    """Phonic-1.1 in voice, turn by turn (module docstring)."""

    def __init__(self, model: str, purpose: str, speech: Speech):
        self._model, self._purpose, self._speech, self._lag = model, purpose, speech, _Lag(purpose)
        self._key = C.load_env(C.ENV_FILE)["PHONIC_API_KEY"]
        self._open = self._waiting = None  # (socket, state, turns it has seen, tasks, queue); (call ids, future)
        self._optional: dict[str, set[str]] = {}  # each tool's optional fields, whose nulls are dropped
        self.hung_up = False  # the service ended the conversation itself: the call ends as on a transfer
        self.chat = self.completions = self

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        told, turns = _split(messages, opening=False)  # no opening rule: no reply is asked for at the start
        for attempt in range(2):  # a conversation dropped on a fresh turn runs once more
            after_tool = False
            try:
                async with asyncio.timeout(RESPONSE_SECONDS):
                    waiting = self._waiting
                    after_tool = self._resume(turns)
                    if after_tool:
                        self._open[1].reply_at = None  # the reply is what it says once the results are back
                        for r in waiting[1].result():
                            await self._open[0].send(json.dumps({"type": "tool_call_output", "tool_call_id": r["toolCallId"],
                                                                 "output": r["result"]}))
                    else:
                        await self._close()
                        await self._start(told, turns, tools)
                    text, calls = await self._listen(VAPI_QUIET_AFTER_TOOL_S if after_tool else VAPI_QUIET_S)
                break
            except _Dropped:
                await self._close()
                if attempt or after_tool:  # mid tool round, a rerun would run the tools again
                    raise RuntimeError("phonic: the server closed the conversation") from None
            except BaseException:
                await self._close()  # a failed turn's conversation would hold its line
                raise
        if calls:
            self._open = (*self._open[:2], turns, *self._open[3:])
        else:
            await self._close()
        return _answer(text.strip(), calls, SimpleNamespace(prompt_tokens=0, total_tokens=0))

    def _config(self, told: str, said: list[dict], tools: list[dict] | None) -> dict:
        prompt = told
        for pattern, restated in PROMPT_RULES:
            prompt = pattern.sub(restated, prompt)
        if said:  # the call so far; the greeting is in it, so the agent does not open again
            prompt += "\n\nThe call so far, oldest first. Carry on from here:\n" + ElevenAgentVoice._notes(said)
        encoded = []
        for f in (_inlined(t)["function"] for t in tools or []):
            params = f.get("parameters") or {"type": "object", "properties": {}}
            self._optional[f["name"]] = set(params.get("properties", {})) - set(params.get("required") or ())
            encoded.append({"type": "custom_websocket", "tool_call_output_timeout_ms": PHONIC_TOOL_TIMEOUT_MS,
                            "tool_schema": {"type": "function", "function": {
                                "name": f["name"], "description": f"{f.get('description', '').rstrip()} {NULL_MEANS_OMITTED}",
                                "parameters": _strict(params), "strict": True}}})
        return {"type": "config", "model": self._model, "system_prompt": prompt, "voice_id": PHONIC_VOICE,
                "input_format": "pcm_16000", "output_format": "pcm_16000", "tools": encoded, **PHONIC_SETTINGS}

    async def _start(self, told: str, turns: list[dict], tools: list[dict] | None) -> None:
        import websockets
        k = _spoken_turn(turns)
        said = [m for m in (turns if k is None else turns[:k])[1:]]  # after the "connected" turn
        PhonicVoice._slots = PhonicVoice._slots or asyncio.Semaphore(PHONIC_LINES)
        await PhonicVoice._slots.acquire()
        self._slot = True
        try:
            for wait in PHONIC_BUSY_WAITS:
                ws = await websockets.connect(PHONIC_URL, additional_headers={"Authorization": f"Bearer {self._key}"},
                                              max_size=None)
                await ws.send(json.dumps(self._config(told, said, tools)))
                async with asyncio.timeout(15):
                    while (event := json.loads(await ws.recv())).get("type") not in ("ready_to_start_conversation", "error"):
                        pass
                if event.get("type") == "ready_to_start_conversation":
                    break
                await ws.close()
                busy = "concurrency limit" in json.dumps(event.get("error")).lower()
                if not busy or wait is None:
                    raise RuntimeError(f"phonic config: {event.get('error')} {event.get('param_errors') or ''}")
                await asyncio.sleep(wait)  # a conversation another process just closed is released a little later
        except BaseException:
            self._release()
            raise
        state = SimpleNamespace(spoken_at=None, speaking=False, reply_at=None, t0=time.time(), closing=False)
        queue = asyncio.Queue()

        async def read():  # its speech, words and tool calls; the socket's end
            try:
                async for raw in ws:
                    e = json.loads(raw)
                    t = e.get("type")
                    if t == "audio_chunk":  # paced to real time, silence between replies: speech is what is voiced
                        now, text = time.time(), e.get("text") or ""
                        if text or (state.speaking and _voiced(base64.b64decode(e.get("audio") or ""))):
                            await queue.put(("voice", now))
                        if text:
                            await queue.put(("words", text))
                    elif t == "assistant_started_speaking":
                        state.speaking = True
                    elif t in ("assistant_finished_speaking", "interrupted_response"):
                        state.speaking = False
                        await queue.put(("finished", time.time()))
                    elif t == "tool_call":
                        await queue.put(("call", e))
                    elif t == "assistant_ended_conversation":  # no hang-up tool is configured: the service's own end
                        await queue.put(("ended", None))
                    elif t == "error":
                        await queue.put(("error", f"{e.get('error')} {e.get('param_errors') or ''}"))
            except websockets.ConnectionClosed:
                pass
            await queue.put(("closed", (ws.close_code, ws.close_reason)))
        spoken = b"" if k is None else _to16k(await self._speech(turns[k]["content"]))

        async def speak():  # the caller's line at speaking pace, then an open line's silence
            for i in range(0, len(spoken), PHONIC_FRAME):
                await ws.send(json.dumps({"type": "audio_chunk",
                                          "audio": base64.b64encode(spoken[i:i + PHONIC_FRAME].ljust(PHONIC_FRAME, b"\0")).decode()}))
                await asyncio.sleep(PHONIC_FRAME / (2 * PHONIC_RATE))
            state.spoken_at = time.time()
            if spoken:
                self._lag.spoke()
            silence = base64.b64encode(b"\0" * PHONIC_FRAME).decode()
            while True:
                await ws.send(json.dumps({"type": "audio_chunk", "audio": silence}))
                await asyncio.sleep(PHONIC_FRAME / (2 * PHONIC_RATE))
        self._open = (ws, state, turns, [asyncio.create_task(read()), asyncio.create_task(speak())], queue)

    async def _listen(self, quiet: float) -> tuple[str, list]:
        """The agent's reply once the caller's line has been spoken: its speech until it has finished speaking and
        stayed quiet a while; or the tools it calls; or a dropped conversation, an error. Talk over the line and a
        reply of backchannels alone are handled as GPT-Live's are."""
        _, state, _, _, queue = self._open
        parts, calls, quiet_since = [], [], None
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), 0.3)
            except TimeoutError:
                now = time.time()
                if calls:  # the round's calls are all in
                    self._lag.tool_round()
                    self._waiting = ([c["tool_call_id"] for c in calls], asyncio.get_running_loop().create_future())
                    return self._words(parts), [(c["tool_call_id"], c["tool_name"], json.dumps(self._args(c))) for c in calls]
                said = [p for p in parts if (not p[2] or len("".join(p[1]).split()) > PHONIC_OVER_LINE_WORDS)
                        and "".join(p[1]).strip()]
                real = [p for p in said if not _backchannel("".join(p[1]))]
                if state.spoken_at and not state.speaking and quiet_since:
                    if real and now - quiet_since > quiet or said and now - quiet_since > PHONIC_HOLD_S:
                        self._lag.answered(state.reply_at)
                        return self._words(real or said), []
                if state.spoken_at and not said and not state.speaking and now - state.spoken_at > VAPI_SILENT_S:
                    return "", []
                continue
            if kind == "voice":  # parts: [onset, words, started over the caller's line, "open" or "done"]
                if not parts or parts[-1][-1] == "done":
                    parts.append([item, [], state.spoken_at is None, "open"])
                if state.spoken_at:
                    self._lag.sound(item)
                    state.reply_at = state.reply_at or item
                quiet_since = None
            elif kind == "words":
                if not parts or parts[-1][-1] == "done":
                    parts.append([time.time(), [], state.spoken_at is None, "open"])
                parts[-1][1].append(item)
            elif kind == "finished":
                if parts:
                    parts[-1][-1] = "done"
                quiet_since = item
            elif kind == "call":
                calls.append(item)
            elif kind == "error":
                raise RuntimeError(f"phonic error: {item}")
            elif kind == "ended":
                self.hung_up, state.closing = True, True
                self._lag.answered(state.reply_at)
                return self._words([p for p in parts if p[-1] == "done"] or parts), []
            elif kind == "closed":
                if not state.closing:
                    raise _Dropped()

    @staticmethod
    def _words(parts: list) -> str:
        return " ".join("".join(p[1]).strip() for p in parts if not p[2] or len("".join(p[1]).split()) > PHONIC_OVER_LINE_WORDS).strip()

    def _args(self, call: dict) -> dict:
        """The call's arguments without the nulls strict mode made it send for fields it left out (it writes some as
        the string "null", which on an optional field means left out too)."""
        optional = self._optional.get(call.get("tool_name"), set())
        return {k: v for k, v in (call.get("parameters") or {}).items() if v is not None and not (k in optional and v == "null")}

    def _release(self) -> None:
        if getattr(self, "_slot", False):
            PhonicVoice._slots.release()
            self._slot = False

    async def _close(self) -> None:
        self._waiting = None
        self._release()
        if self._open:
            (ws, state, _, tasks, _), self._open = self._open, None
            state.closing = True
            for task in tasks:
                task.cancel()
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - a socket the server already closed
                pass
            C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "phonic", "model": self._model,
                                      "purpose": self._purpose, "in": 0, "out": 0,
                                      "usd": round((time.time() - state.t0) / 60 * PHONIC_PER_MINUTE, 6)})
