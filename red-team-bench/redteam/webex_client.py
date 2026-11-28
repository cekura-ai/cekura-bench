"""Webex AI Agent (Cisco), turn by turn, over the phone: the airline agent built in Webex AI Agent Studio on the Webex
AI Pro-US 2.0 engine (Cisco's own speech recognition, LLM and voice), reached through a Webex Contact Center entry
point whose flow hands the call to it (agent-configurations/webex-ai-agent.json).

Each turn is a new phone call, placed from Twilio, one call at a time: as a call connects, the flow claims the
oldest waiting session from the tool server (GET /webex/<token>/claim), which gives it the session's code and the call
so far, and hands that to the agent as its starting data ({{history}} in its instructions), so a refused turn is
erased by leaving it out. Once claimed, the next call is placed; calls then run side by side. The agent greets, the caller's newest line is played into the call over Twilio's
bidirectional media stream, and the agent's reply is the speech it starts after the line, until it has been quiet a
while; its words are transcribed on our side (Webex gives no live transcript to a caller). The airline's 14 tools are
the agent's actions, each handled in the flow ("Manage in the source flow"): the flow posts the action's name and
arguments to the tool server (POST /webex/<token>/tool?code=&name=, the arguments as the body), which hands them to the turn waiting on that code, and sends
the result back into the agent, which carries on in the same call."""
from __future__ import annotations

import asyncio
import audioop
import base64
import io
import json
import os
import random
import re
import time
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace

from . import harness_config as C
from .voice_clients import (PCM_RATE, RESPONSE_SECONDS, TOOL_TOKEN, ElevenAgentVoice, Speech, VapiVoice, _answer,
                            _backchannel, _Lag, _platform_calls, _split, _spoken_turn, _tool_server)

WEBEX_NUMBER = os.environ.get("WEBEX_NUMBER", "")  # the entry point's number
# The caller ids the calls come from: each open call takes its own number, and a number rests before its next call
# (first tried against Webex's call drops, which turned out to be an org-wide limit: see CALLS_PER_MIN)
TWILIO_FROM = [n for n in os.environ.get("TWILIO_FROM", "").split(",") if n]  # Twilio numbers, comma-separated
NUMBER_REST_S = float(os.environ.get("WEBEX_NUMBER_REST_S", "45"))
NUMBER_BENCH_S = 600.0  # a number Webex dropped before the greeting rests this long
TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}"
RATE, FRAME = 8000, 160  # the phone line: 8 kHz mu-law, 20 ms frames
VOICED_RMS = 400  # a 20 ms frame of the agent's audio above this is speech
GREETING_QUIET_S, GREETING_MAX_S = 0.9, 25.0  # the greeting is over after this much quiet
QUIET_S, QUIET_AFTER_TOOL_S, SILENT_S, HOLD_S = 1.8, 2.5, 30.0, 10.0
TRANSCRIBE_MODEL, TRANSCRIBE_PER_MINUTE = "gpt-4o-transcribe", 0.006
FILLER = re.compile(r"^\W*(one moment|just a moment|let me (check|look|see|pull)|please (hold|wait)|bear with me|"
                    r"give me a (moment|second)|i'?ll (check|look)|checking)\b", re.I)

_sessions: dict[str, dict] = {}  # session code -> {"history": text, "queue": the turn's queue, "ws": media socket}
_unclaimed: list[str] = []  # codes of calls placed but not yet claimed by the flow, oldest first
_placing = asyncio.Lock()  # one call at a time until the flow has claimed it
CLAIM_S, PLACING_S = 40.0, 600.0
# The sandbox org stopped starting AI agent sessions after about 600 in an hour at 12 calls a minute, and had not
# recovered hours later: calls are paced to this many a minute
CALLS_PER_MIN = float(os.environ.get("WEBEX_CALLS_PER_MIN", "6"))
_last_placed = [0.0]
TOOL_LOG = C.RUNS / "webex-tool-requests.jsonl"  # what the flow sends, as it arrives


def _keys() -> dict:
    """Twilio's keys: from the file TWILIO_ENV names (provider-keys.env), else the harness's keys file."""
    path = Path(os.environ.get("TWILIO_ENV") or C.ENV_FILE)
    pairs = re.findall(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", path.read_text(), re.M)  # KEY=value, spaces allowed
    return {k: v.strip('"').strip("'") for k, v in pairs}


def _ulaw(pcm24k: bytes) -> bytes:
    """The caller's speech (24 kHz PCM16) as the phone line carries it."""
    pcm8k, _ = audioop.ratecv(pcm24k, 2, 1, PCM_RATE, RATE, None)
    return audioop.lin2ulaw(pcm8k, 2)


def _wav(pcm8k: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes(pcm8k)
    return out.getvalue()


async def claim_request(request):
    """GET /webex/<token>/claim: the oldest call waiting to be claimed, as the flow's starting data."""
    from aiohttp import web
    if request.match_info["token"] != TOOL_TOKEN:
        return web.Response(status=404)
    while _unclaimed:
        code = _unclaimed.pop(0)
        if code in _sessions:
            _sessions[code]["claimed"].set()
            return web.json_response({"code": code, "history": _sessions[code]["history"]})
    return web.json_response({"code": "", "history": ""})


async def session_request(request):
    """GET /webex/<token>/session/<code>: the call so far, for the flow to hand to the agent."""
    from aiohttp import web
    if request.match_info["token"] != TOOL_TOKEN:
        return web.Response(status=404)
    s = _sessions.get(request.match_info["code"])
    return web.json_response({"found": s is not None, "history": s["history"] if s else ""})


async def tool_request(request):
    """POST /webex/<token>/tool?code=&name= with the action's arguments as the body: one action the agent took, run by
    the turn waiting on that code."""
    from aiohttp import web
    from .voice_clients import _ran
    if request.match_info["token"] != TOOL_TOKEN:
        return web.Response(status=404)
    code, name, raw = request.query.get("code", ""), request.query.get("name", ""), await request.text()
    C.append_jsonl(TOOL_LOG, {"t": round(time.time()), "code": code, "name": name, "body": raw[:4000]})
    s = _sessions.get(code)
    if s is None or not name:
        return web.json_response({"result": "error: call ended"})
    try:
        args = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    call = {"id": f"call_{uuid.uuid4().hex[:16]}", "function": {"name": name, "arguments": _typed(name, args)}}
    results = await _ran(s["queue"], [call])
    return web.json_response({"result": results[0]["result"] if results else "error: no result"})


def _typed(name: str, args: dict) -> dict:
    """An action's arguments as the tool's schema types them. Webex's input fields are flat text and numbers, so a list
    field arrives as JSON text and an integer as a number or numeric text (webex-ai-agent.json)."""
    from .airline_agent import TOOL_DEFINITIONS
    props = next((t["parameters"].get("properties", {}) for t in TOOL_DEFINITIONS if t["name"] == name), {})
    out = {}
    for key, value in args.items():
        spec = props.get(key, {})
        kinds = {spec.get("type")} | {x.get("type") for x in spec.get("anyOf", [])}
        if isinstance(value, str) and kinds & {"array", "object"}:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        elif isinstance(value, (str, float)) and "integer" in kinds:
            try:
                value = int(float(value))
            except ValueError:
                pass
        elif isinstance(value, str) and "number" in kinds:
            try:
                value = float(value)
            except ValueError:
                pass
        out[key] = value
    return out


async def media_socket(request):
    """Twilio's bidirectional media stream for one call: the agent's audio in, the caller's line out."""
    from aiohttp import web
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    s = _sessions.get(request.match_info["code"]) if request.match_info["token"] == TOOL_TOKEN else None
    if s is None:
        await ws.close()
        return ws
    s["ws"] = ws
    async for msg in ws:
        if msg.type != 1:  # text frames only
            continue
        event = json.loads(msg.data)
        kind = event.get("event")
        if kind == "start":
            s["stream"] = event["start"]["streamSid"]
            await s["queue"].put(("started", None))
        elif kind == "media" and event["media"].get("track", "inbound") == "inbound":
            await s["queue"].put(("audio", (time.time(), audioop.ulaw2lin(base64.b64decode(event["media"]["payload"]), 2))))
        elif kind == "mark":
            await s["queue"].put(("mark", (time.time(), event["mark"]["name"])))
        elif kind == "stop":
            break
    await s["queue"].put(("stopped", None))
    return ws


class WebexVoice(VapiVoice):
    _numbers: asyncio.Queue | None = None  # caller ids free to call from; one open call each

    def __init__(self, model: str, purpose: str, speech: Speech):  # VapiVoice's turn and tool-round logic, not its setup
        if not WEBEX_NUMBER or not TWILIO_FROM:
            raise RuntimeError("webex: set WEBEX_NUMBER (the entry point's number) and TWILIO_FROM (the caller ids)")
        self._model, self._purpose, self._speech, self._lag = model, purpose, speech, _Lag(purpose)
        self._open = self._waiting = self._spoken_at = None
        self._billing = set()
        self.chat = self.completions = self
        keys = _keys()
        self._sid, self._token = keys["TWILIO_ACCOUNT_SID"], keys["TWILIO_AUTH_TOKEN"]
        self._host = re.sub(r"^https?://", "", os.environ["TOOL_URL"].rstrip("/"))
        self.hung_up = False

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        told, turns = _split(messages, opening=False)
        await _tool_server()
        try:
            after_tool = self._resume(turns)
            if not after_tool:
                await self._close()
                if WebexVoice._numbers is None:
                    WebexVoice._numbers = asyncio.Queue()
                    for n in TWILIO_FROM:
                        WebexVoice._numbers.put_nowait(n)
                self._from = await WebexVoice._numbers.get()
                self._line = True
            async with asyncio.timeout(RESPONSE_SECONDS + PLACING_S):  # waiting for a call's turn to be placed is not the agent's time
                if not after_tool:
                    await self._start(told, turns, tools)
                text, calls = await self._listen(QUIET_AFTER_TOOL_S if after_tool else QUIET_S)
        except BaseException:
            await self._close()
            raise
        if calls:
            self._open = (*self._open[:2], turns, *self._open[3:])
        else:
            self._lag.answered(getattr(self, "_began", None))
            await self._close()
        return _answer(text.strip(), calls, SimpleNamespace(prompt_tokens=0, total_tokens=0))

    async def _twilio(self, method: str, path: str, data: dict | None = None) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=30, auth=(self._sid, self._token)) as http:
            r = await http.request(method, TWILIO_API.format(sid=self._sid) + path, data=data)
        if r.status_code >= 300:
            raise RuntimeError(f"twilio {path}: {r.status_code} {r.text[:300]}")
        return r.json()

    async def _start(self, told: str, turns: list[dict], tools: list[dict] | None) -> None:
        k = _spoken_turn(turns)
        said = (turns if k is None else turns[:k])[1:]  # the call so far, after the "connected" turn
        code = next(c for c in (f"{random.randrange(10**5, 10**6)}" for _ in range(100)) if c not in _sessions)
        queue = asyncio.Queue()
        _sessions[code] = {"history": ElevenAgentVoice._notes(said), "queue": queue, "claimed": asyncio.Event()}
        _platform_calls[code] = queue
        speech = b"" if k is None else _ulaw(await self._speech(turns[k]["content"]))
        twiml = f'<Response><Connect><Stream url="wss://{self._host}/twilio/{TOOL_TOKEN}/{code}"/></Connect></Response>'
        self._open = (None, code, turns, [], queue)
        self._spoken_at, self._line_sent, self.hung_up = None, None, False
        async with _placing:  # the flow claims the oldest waiting call: place the next only once this one is claimed
            await asyncio.sleep(max(0.0, _last_placed[0] + 60 / CALLS_PER_MIN - time.time()))
            _last_placed[0] = time.time()
            _unclaimed.append(code)
            call = await self._twilio("POST", "/Calls.json", {"To": WEBEX_NUMBER, "From": self._from, "Twiml": twiml})
            self._call_sid = call["sid"]
            try:
                await asyncio.wait_for(_sessions[code]["claimed"].wait(), CLAIM_S)
            except TimeoutError:
                raise RuntimeError("webex: the flow never claimed the call") from None
        await self._greeted(queue)
        if speech:
            await self._say(code, speech)

    async def _greeted(self, queue: asyncio.Queue) -> None:
        """Wait out the agent's greeting: its first speech, then a short quiet."""
        t0, heard, last = time.time(), False, None
        while time.time() - t0 < GREETING_MAX_S:
            try:
                kind, item = await asyncio.wait_for(queue.get(), 0.5)
            except TimeoutError:
                if heard and last and time.time() - last > GREETING_QUIET_S:
                    return
                continue
            if kind == "stopped":
                self._dropped = True
                raise RuntimeError("webex: the call ended before the agent greeted")
            if kind == "audio" and audioop.rms(item[1], 2) > VOICED_RMS:
                heard, last = True, item[0]
            elif kind == "audio" and heard and last and item[0] - last > GREETING_QUIET_S:
                return
        if not heard:
            raise RuntimeError("webex: no greeting from the agent")

    async def _say(self, code: str, speech: bytes) -> None:
        """Play the caller's line at speaking pace; a mark tells when its last word has gone out."""
        s = _sessions[code]
        for i in range(0, len(speech), FRAME):
            await s["ws"].send_str(json.dumps({"event": "media", "streamSid": s["stream"],
                                               "media": {"payload": base64.b64encode(speech[i:i + FRAME]).decode()}}))
            await asyncio.sleep(FRAME / RATE)
        await s["ws"].send_str(json.dumps({"event": "mark", "streamSid": s["stream"], "mark": {"name": "line"}}))
        self._line_sent = time.time()  # if Twilio's mark never comes back, the line counts as played soon after

    async def _listen(self, quiet: float) -> tuple[str, list]:
        """The agent's reply: the speech it starts after the caller's line (once Twilio has played it out), until it
        has been quiet a while, or the actions it takes. A filler ("one moment") before an action is not the reply."""
        queue = self._open[4]
        audio, last, held = bytearray(), None, 0.0
        self._began = None
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), 0.5)
            except TimeoutError:
                kind, item = None, None
            now = time.time()
            if not self._spoken_at and getattr(self, "_line_sent", None) and now - self._line_sent > 3.0:
                self._spoken_at = self._line_sent
                self._lag.spoke()
            if kind == "mark" and not self._spoken_at:
                self._spoken_at = item[0]
                self._lag.spoke()
            elif kind == "stopped":
                self.hung_up = True
                return await self._words(audio), []
            elif kind == "tools":
                calls, results = item
                self._lag.tool_round()
                self._waiting = ([c["id"] for c in calls], results)
                return await self._words(audio), [(c["id"], c["function"]["name"], json.dumps(c["function"].get("arguments") or {}))
                                                  for c in calls]
            elif kind == "audio" and (self._spoken_at or not self._has_line()):
                t, pcm = item
                if audioop.rms(pcm, 2) > VOICED_RMS:
                    self._lag.sound(t)
                    self._began = self._began or t
                    last = t
                if self._began:
                    audio += pcm
            if last and now - last > quiet + held:
                text = await self._words(audio)
                if (FILLER.match(text) or _backchannel(text)) and held < HOLD_S:
                    held = HOLD_S  # an answer may follow after an action or a pause
                    continue
                return text, []
            if not self._began and self._spoken_at and now - self._spoken_at > SILENT_S:
                return "", []

    def _has_line(self) -> bool:
        return _spoken_turn(self._open[2]) is not None

    async def _words(self, pcm: bytes) -> str:
        """The agent's speech, transcribed."""
        if len(pcm) < RATE * 2 * 0.3:
            return ""
        r = await C.clients()["openai"].audio.transcriptions.create(model=TRANSCRIBE_MODEL,
                                                                     file=("reply.wav", _wav(bytes(pcm)), "audio/wav"))
        C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "openai", "model": TRANSCRIBE_MODEL,
                                  "purpose": "redteam:webex-transcribe", "in": len(pcm), "out": len(r.text),
                                  "usd": round(len(pcm) / (RATE * 2) / 60 * TRANSCRIBE_PER_MINUTE, 6)})
        return r.text.strip()

    async def _close(self) -> None:
        if getattr(self, "_line", False):
            number, self._line = self._from, False
            wait = NUMBER_BENCH_S if getattr(self, "_dropped", False) else NUMBER_REST_S
            self._dropped = False

            async def rest():  # the number's Webex session needs a while to close before it calls again
                await asyncio.sleep(wait)
                WebexVoice._numbers.put_nowait(number)
            task = asyncio.create_task(rest())
            self._billing.add(task)
            task.add_done_callback(self._billing.discard)
        if self._waiting:
            if not self._waiting[1].done():
                self._waiting[1].set_result([{"toolCallId": i, "result": "error: call ended"} for i in self._waiting[0]])
            self._waiting = None
        if self._open:
            (_, code, _, _, _), self._open = self._open, None
            s = _sessions.pop(code, None)
            _platform_calls.pop(code, None)
            try:
                await self._twilio("POST", f"/Calls/{self._call_sid}.json", {"Status": "completed"})
            except Exception:  # noqa: BLE001 - a call already over
                pass
            if s and s.get("ws") is not None:
                await s["ws"].close()
            task = asyncio.create_task(self._bill(self._call_sid))
            self._billing.add(task)
            task.add_done_callback(self._billing.discard)

    async def _bill(self, call_sid: str) -> None:
        """The call's Twilio price once it has settled."""
        await asyncio.sleep(60)
        try:
            call = await self._twilio("GET", f"/Calls/{call_sid}.json")
        except Exception:  # noqa: BLE001
            return
        C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "twilio", "model": self._model, "purpose": self._purpose,
                                  "in": int(call.get("duration") or 0), "out": 0, "usd": abs(float(call.get("price") or 0))})
