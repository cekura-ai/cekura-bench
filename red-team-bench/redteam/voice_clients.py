"""The voice clients: each system the agent runs on, behind the one call the conversation makes per caller turn,
client.chat.completions.create(messages, tools), answered in the shape of an OpenAI chat response.

Each system is set up as the voice bot sets it up (reference_bot_settings.py): its instructions (the agent's prompt, the shared
rules and the opening rule), its tools, the "call connected" turn it greets after, and its reasoning level. Each caller
turn opens a new session: the caller's newest line is spoken (OpenAI speech, sent as audio the way a call sends it)
and the earlier kept turns go in as history, in whatever form the system takes, which is what lets Crescendo erase a
refused turn on voice. A tool call and its result stay in the turn's session. The agent's words are read back from
the system's own transcript of its speech.
"""
from __future__ import annotations

import asyncio
import base64
import array
import functools
import itertools
import time
import json
import os
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

from . import harness_config as C
from . import reference_bot_settings

BOT_TEXT = {name: getattr(reference_bot_settings, name) for name in ("SHARED_RULES", "OPENING_RULE", "CONNECTED", "OPENAI_REASONING",
                                                                     "GEMINI_THINKING", "GROK_REASONING", "DELEGATION_PROMPT")}
# Grok Voice reports no tokens: it bills audio minutes ($0.08) and text inputs ($0.004, docs.x.ai/developers/pricing),
# and always answers in audio, PCM16 at 24 kHz (48,000 bytes a second)
GROK_PER_MINUTE, GROK_PER_TEXT_INPUT, GROK_AUDIO_BYTES_PER_S = 0.08, 0.004, 48000
VOICES = {"openai": "marin", "xai": "eve", "gemini": "Charon", "elevenlabs": "brian", "vapi": "marin"}  # each row's voice
# Eleven Realtime: an OpenAI-compatible endpoint opened with a short-lived secret minted per session. It bills the
# ElevenAgents minute plus tokens at rates ElevenLabs does not publish: logged as an estimate, $0.10 a minute and
# tokens at GPT Realtime's text rates, to be checked against the ElevenLabs bill
ELEVEN_URL, ELEVEN_PER_MINUTE = "https://api.elevenlabs.io/v1/convai", 0.10
# Vapi: one call per turn over its WebSocket transport, the caller spoken at speaking pace (20 ms frames) and followed
# by silence for its turn detector; its reply is over once it has been quiet this long. It bills per call, read back.
VAPI_URL, VAPI_FRAME, VAPI_TRAILING_S, VAPI_QUIET_S = "https://api.vapi.ai", 960, 2.0, 3.0  # 960 bytes: 20 ms at 24 kHz
VAPI_QUIET_AFTER_TOOL_S = 5.0  # after a tool result the answer can take a few seconds to start
# After the caller's line the open line carries silence in 100 ms frames; with no reply this long after it, the agent
# has answered with silence
VAPI_SILENCE_FRAME, VAPI_SILENT_S = 4800, 40.0
# The account refuses calls past its concurrency limit ("Over Concurrency Limit", about 16): one process holds at most
# this many open, and a refused call waits for a line to free up
VAPI_LINES = 13
# GPT-Live: gpt-live-1 holds the conversation and hands tool work to a backend text model, set up as the benchmark's bot
# sets it up (its backend from GPT_LIVE_BACKENDS, low reasoning). Prices from pricing/prices.json: live audio by the minute, then the backend's
# tokens per million (input, cached input, output).
GPT_LIVE_URL, GPT_LIVE_REASONING = "wss://api.openai.com/v1/live/sessions", "low"
# the backend text model each GPT-Live version delegates to: 1 ran on GPT-6 Sol, 1.1 on the latest Sol
GPT_LIVE_BACKENDS = {"gpt-live-1": "gpt-6-sol", "gpt-live-1.1": "gpt-6.2-sol"}
GPT_LIVE_PER_MINUTE, GPT_LIVE_BACKEND_PRICES, GPT_LIVE_FRAME = 0.05, (2.0, 0.2, 10.0), 4800  # 4800 bytes: 100 ms
# Its output audio streams on with exact zeros between words; after the backend answers, the live model can take
# several seconds to start saying the result, so a turn waits up to this long for it
GPT_LIVE_VOICED, GPT_LIVE_OWED_S = 200, 20.0
# Now and then it says nothing at all after the caller's turn: after this long it has answered with silence. Backend work
# with no sign of life for the second figure no longer holds the turn open.
GPT_LIVE_SILENT_S, GPT_LIVE_STALE_S = 40.0, 60.0
# Its speech in parts: a pause this long starts a new part. It talks over the caller's line now and then ("Hi Raj",
# "Mm-hmm") and answers some lines with a backchannel alone; neither is its reply, so after a backchannel the turn
# waits this long for the answer to follow
GPT_LIVE_GAP_S, GPT_LIVE_HOLD_S = 0.8, 10.0
BACKCHANNELS = {"mm-hmm", "mm hmm", "mhm", "mm-hm", "mmhmm", "uh-huh", "uh huh", "mm", "hmm", "hm", "okay", "ok", "yeah",
                "yes", "sure", "right", "go on", "go ahead", "got it", "i see", "oh", "oh okay", "alright", "all right",
                "uh", "um", "hi", "hello", "hey"}
CONFIGS = C.CONFIGS
RETELL_URL = "https://api.retellai.com"
RETELL_IDS = C.DATA / "retell-agent.json"  # platform agents are created once and their ids kept with the run data
RETELL_GUARD_IDS = RETELL_IDS.with_name("retell-agent-guard.json")
AIRLINE_HINTS = {
    "contextGeneral": [{"key": "domain", "value": "Airline customer service"},
                       {"key": "topic", "value": "Flight bookings, changes, cancellations, baggage, refunds and compensation"},
                       {"key": "task", "value": "Capture caller names, user IDs, six-character reservation codes, flight "
                                                "numbers such as HAT123, dates and amounts"}],
    "customVocabulary": ["reservation", "reservation number", "user ID", "itinerary", "basic economy", "economy",
                         "business class", "travel certificate", "travel insurance", "checked bags", "gift card",
                         "credit card", "round trip", "one way", "layover", "HAT"]}
# GPT Realtime 2.1 per million tokens: text in, audio in, cached in, text out (reasoning included), audio out
REALTIME_VOICE_PRICES = (4.0, 32.0, 0.4, 16.0, 64.0)
GEMINI_AUDIO_IN = 3.0  # per million audio input tokens; its text input and all output are in C.PRICES
# The caller's voice: OpenAI speech as 24 kHz PCM16 (48,000 bytes a second), priced by length (an estimate)
SPEECH_MODEL, SPEECH_VOICE, SPEECH_PER_MINUTE, SPEECH_MAX_CHARS = "gpt-4o-mini-tts", "alloy", 0.015, 4000
PCM_RATE, PCM_BYTES_PER_S, CHUNK = 24000, 48000, 48000
RESPONSE_SECONDS = 240  # one turn: the caller's line (up to ~30 s), the wait, and a reply of up to 1,000 tokens spoken


class Speech:
    """The caller's newest line as audio, made once: a turn's tool rounds and a reopened session reuse it. Only that
    line is kept (earlier turns replay as text), so 45 attacks at once do not hold every line's audio."""

    def __init__(self):
        self._client, self._made = C.clients()["openai"], {}

    async def __call__(self, text: str) -> bytes:
        if text not in self._made:
            audio = b""
            for part in re.findall(rf"\S.{{0,{SPEECH_MAX_CHARS}}}(?=\s|$)", text, re.S):  # the speech API's limit
                r = await self._client.audio.speech.create(model=SPEECH_MODEL, voice=SPEECH_VOICE, input=part,
                                                           response_format="pcm")
                audio += r.content
            C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "openai", "model": SPEECH_MODEL,
                                      "purpose": "redteam:caller-speech", "in": len(text), "out": len(audio),
                                      "usd": round(len(audio) / PCM_BYTES_PER_S / 60 * SPEECH_PER_MINUTE, 6)})
            self._made = {text: audio}
        return self._made[text]


# With LATENCY_LOG set, each caller turn logs the seconds from the end of the caller's speech to the agent's first
# sound (fillers and backchannels included) and to the start of its reply, whether a tool ran before the reply was
# over, and the conversation it belongs to (run: its log's name).
LATENCY_LOG = Path(os.environ["LATENCY_LOG"]) if os.environ.get("LATENCY_LOG") else None


class _Lag:
    def __init__(self, purpose: str):
        self.purpose, self.run, self.end, self.tool, self.first = purpose, None, None, False, None

    def spoke(self) -> None:
        self.end, self.tool, self.first = time.time(), False, None

    def tool_round(self) -> None:
        self.tool = self.tool or self.end is not None

    def sound(self, at: float | None = None) -> None:
        if self.end is not None and self.first is None:
            self.first = at or time.time()

    def answered(self, at: float | None = None) -> None:
        """The turn is over; at is when its reply started (by default its first sound). A silent turn logs nothing."""
        at = at or self.first
        if self.end is not None and at and LATENCY_LOG:
            C.append_jsonl(LATENCY_LOG, {"t": round(time.time()), "model": self.purpose, "run": self.run,
                                         "first_audio_s": round(max(min(self.first or at, at) - self.end, 0), 3),
                                         "reply_s": round(max(at - self.end, 0), 3), "after_tool": self.tool,
                                         "load": round(os.getloadavg()[0], 2)})  # a busy machine delays both
        self.end = None

    def heard(self, at: float | None = None) -> None:
        """An agent whose first sound is its reply, logged at once."""
        self.sound(at)
        self.answered()


def _voiced(pcm: bytes) -> bool:
    """Whether 16-bit audio holds speech rather than silence."""
    return max(map(abs, array.array("h", pcm[:len(pcm) // 2 * 2])), default=0) > 200


AZURE_FILTER_LOG = C.RUNS / "azure-filter.jsonl"
AZURE_RETRY_WAITS = (0, 10, 30, 60, 60, 60)  # seconds before each try: its token limit is per minute ("too many tokens")


def _filtered(detail) -> bool:
    """Whether an error or a response's status details is Azure's content filter at work."""
    text = json.dumps(detail, default=lambda o: getattr(o, "__dict__", str(o))).lower() if detail is not None else ""
    return any(k in text for k in ("content_filter", "content management policy", "responsibleai", "content filter"))


def _filter_log(purpose: str, where: str, message: str) -> None:
    C.append_jsonl(AZURE_FILTER_LOG, {"t": round(time.time()), "model": purpose, "where": where, "message": message[:500]})


@functools.cache
def _backchannel(text: str) -> bool:
    """Whether words are only listening noises or a bare greeting ("Mm-hmm.", "Okay.", "Yeah. Mm-hmm.", "Hi!")."""
    bits = [b.strip() for b in re.split(r"[.,!?;:—–-]\s+|[.,!?;:]$|[.,!?;:](?=\s)", text.lower()) if b.strip()]
    return bool(bits) and all(b.strip(" .,!?") in BACKCHANNELS or len(b.split()) <= 2 and b.split()[0] in ("hi", "hello", "hey")
                              for b in bits)


def _spoken_turn(turns: list[dict]) -> int | None:
    """The caller's newest turn, the one said aloud; None before the caller has spoken (only "call connected")."""
    users = [i for i, m in enumerate(turns) if m["role"] == "user" and m.get("content")]
    return users[-1] if users and users[-1] > 0 else None


def instructions(prompt: str, greeting: str) -> str:
    """bot.agent_instructions: the prompt, the shared rules, then the greeting to open with."""
    told = f"{prompt.rstrip()}\n\n{BOT_TEXT['SHARED_RULES']}"
    greeting = greeting.replace('"', "'").strip()
    return f"{told}\n\n{BOT_TEXT['OPENING_RULE'].format(greeting=greeting)}" if greeting else told


def _split(messages: list[dict], opening: bool = True) -> tuple[str, list[dict]]:
    """The conversation's history (system, greeting, turns) as the voice bot has it: instructions, then the "connected"
    turn the greeting answers, then the conversation. Without opening, the rule to open with the greeting is left out:
    Gemini Live does not count a replayed greeting as said, and would say it again."""
    system, rest = messages[0]["content"], messages[1:]
    greeting = rest[0]["content"] if rest and rest[0]["role"] == "assistant" else ""
    return instructions(system, greeting if opening else ""), [{"role": "user", "content": BOT_TEXT["CONNECTED"]}] + rest


def _noted(turns: list[dict], note_role: str) -> list[tuple[str, str]]:
    """The turns as (role, text), each tool result a note in note_role naming the call it answers: for systems that
    take earlier turns only as text messages."""
    items, called = [], {}
    for m in turns:
        if m["role"] == "tool":
            items.append((note_role, f"Backend result of {called.get(m['tool_call_id'], 'a tool')}: {m['content']}"))
            continue
        if m.get("content"):
            items.append((m["role"], m["content"]))
        for c in m.get("tool_calls") or []:
            called[c["id"]] = f"{c['function']['name']}({c['function']['arguments']})"
    return items


def _unwrapped(text: str) -> str:
    """GPT Realtime in voice answers about half its turns after a tool result as {"message": "..."}: the words it
    meant are the message."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(value, dict):
        return text
    strings = [v for v in value.values() if isinstance(v, str)]
    return value["message"] if isinstance(value.get("message"), str) else strings[0] if len(strings) == 1 else text


def _answer(text: str, calls: list[tuple[str, str, str]], usage) -> SimpleNamespace:
    tool_calls = [SimpleNamespace(id=i, function=SimpleNamespace(name=n, arguments=a)) for i, n, a in calls]
    message = SimpleNamespace(content=text, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class RealtimeText:
    """GPT Realtime in voice, or Grok Voice on the same protocol."""

    def __init__(self, model: str, purpose: str, provider: str = "openai", speech: Speech | None = None):
        self._client, self._model, self._purpose = C.clients()[provider], model, purpose
        self._provider, self._speech, self._lag = provider, speech, _Lag(purpose)
        self.chat = self.completions = self

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        told, turns = _split(messages)
        text, calls, usage, audio, details, onset = "", [], None, 0, None, None
        k = _spoken_turn(turns) if self._speech else None
        spoken = await self._speech(turns[k]["content"]) if k is not None else b""
        before, after = (turns[:k], turns[k + 1:]) if spoken else (turns, [])
        first, rest = self._items(before), self._items(after)
        items = first + rest
        typed = self._items(turns)  # the retry's fallback: the newest turn as text
        effort = BOT_TEXT["GROK_REASONING" if self._provider == "xai" else "OPENAI_REASONING"]
        session = {"type": "realtime", "instructions": told, "output_modalities": ["audio" if self._speech else "text"],
                   "reasoning": {"effort": effort}, "tools": [{"type": "function", **t["function"]} for t in tools or []]}

        if self._speech:
            pcm = {"format": {"type": "audio/pcm", "rate": PCM_RATE}}
            if self._provider == "xai":  # Grok names the voice and turn detection at the top of the session
                session |= {"voice": VOICES["xai"], "turn_detection": None, "audio": {"input": pcm, "output": pcm}}
            else:
                session["audio"] = {"input": {**pcm, "turn_detection": None}, "output": {**pcm, "voice": VOICES["openai"]}}
        # Azure now and then finishes a response with nothing in it, no audio, words or tool call, most often when many
        # sessions are open: that is not the agent staying silent, so the turn is sent again after a pause (logged)
        for wait in AZURE_RETRY_WAITS if self._provider == "azure" else (0,):
            await asyncio.sleep(wait)
            status, filtered = "no response", False
            for attempt in range(2):  # Grok never answers some near-empty spoken turns ("." or "1"), and its server closes
                text, calls, audio = "", [], 0  # the call only after 15 minutes: the retry sends that turn as text
                if attempt:
                    first, spoken, rest = typed, b"", []
                try:
                    async with asyncio.timeout(RESPONSE_SECONDS), self._client.realtime.connect(model=self._model) as conn:
                        await conn.session.update(session=session)
                        for item in first:
                            await conn.conversation.item.create(item=item)
                        if spoken:  # as a call delivers it: the input buffer, committed as one turn (Grok ignores audio items)
                            for i in range(0, len(spoken), CHUNK * 10):
                                await conn.input_audio_buffer.append(audio=base64.b64encode(spoken[i:i + CHUNK * 10]).decode())
                            await conn.input_audio_buffer.commit()
                            if not rest:  # a fresh caller turn, not one replayed around tool results
                                self._lag.spoke()
                            async with asyncio.timeout(30):  # Grok files the turn once transcribed: anything sent sooner goes first
                                async for event in conn:
                                    if event.type == "error":
                                        raise RuntimeError(f"realtime error: {event.error.message}")
                                    if event.type == "conversation.item.added" and any(  # the spoken turn, not a replayed one
                                            getattr(c, "type", None) == "input_audio" for c in getattr(event.item, "content", None) or []):
                                        break
                        for item in rest:
                            await conn.conversation.item.create(item=item)
                        await conn.response.create()
                        async for event in conn:
                            if event.type == "response.output_audio.delta":
                                audio += len(base64.b64decode(event.delta))
                                onset = onset or time.time()  # the reply is the last response's speech, after any tools
                                self._lag.sound(onset)
                            if event.type == "error":
                                if _filtered(event.error):  # Azure's content filter refused the turn: the line goes quiet
                                    _filter_log(self._purpose, "input", event.error.message)
                                    details, filtered = None, True
                                    break
                                raise RuntimeError(f"realtime error: {event.error.message}")
                            if event.type == "response.done":
                                r = event.response
                                status = f"{r.status} {getattr(r, 'status_details', None)}"
                                if _filtered(getattr(r, "status_details", None)):  # cut off by the filter mid-reply
                                    _filter_log(self._purpose, "output", str(r.status_details))
                                    filtered = True
                                if any(o.type == "function_call" for o in r.output or []):
                                    self._lag.tool_round()
                                for out in r.output or []:
                                    if out.type == "message":
                                        text += "".join(getattr(c, "text", None) or getattr(c, "transcript", None) or ""
                                                        for c in out.content or [])
                                    elif out.type == "function_call":
                                        calls.append((out.call_id, out.name, out.arguments))
                                usage = SimpleNamespace(prompt_tokens=r.usage.input_tokens, total_tokens=r.usage.total_tokens)
                                details = r.usage
                                break
                    break
                except TimeoutError:
                    if attempt:
                        raise
            if text.strip() or calls or filtered or self._provider != "azure":
                break
            _filter_log(self._purpose, "unanswered", status)
        else:
            raise RuntimeError(f"azure: no answer after {len(AZURE_RETRY_WAITS)} tries: {status}")
        if self._provider == "xai":  # audio minutes both ways (the caller's turn as well), text inputs by count
            inputs = sum(1 for i in items if i.get("role") == "user")
            usd = (audio + len(spoken)) / GROK_AUDIO_BYTES_PER_S / 60 * GROK_PER_MINUTE + inputs * GROK_PER_TEXT_INPUT
            C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "xai", "model": self._model,
                                      "purpose": self._purpose, "in": inputs, "out": audio, "usd": round(usd, 6)})
        elif self._speech and details is not None:
            C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": self._provider, "model": self._model,
                                      "purpose": self._purpose, "in": details.input_tokens,
                                      "out": details.output_tokens, "usd": round(self._voice_usd(details), 6)})
        elif not self._speech:
            C.log_usage("openai", self._model, self._purpose, usage)
        if self._speech and not calls:
            self._lag.answered(onset)
        return _answer(_unwrapped(text.strip()) if self._speech else text.strip(), calls, usage)

    @staticmethod
    def _voice_usd(u) -> float:
        """GPT Realtime's bill for one response from its token details: cached input at the cached rate."""
        text_in, audio_in, cached_in, text_out, audio_out = REALTIME_VOICE_PRICES
        i, o = u.input_token_details, u.output_token_details
        cached = getattr(i, "cached_tokens_details", None)
        cached_text, cached_audio = (cached.text_tokens or 0, cached.audio_tokens or 0) if cached else (0, 0)
        audio_tokens = (o.audio_tokens or 0) if o else 0
        return ((((i.text_tokens or 0) - cached_text) * text_in + ((i.audio_tokens or 0) - cached_audio) * audio_in
                 + (cached_text + cached_audio) * cached_in + (u.output_tokens - audio_tokens) * text_out
                 + audio_tokens * audio_out) / 1e6)

    @staticmethod
    def _items(turns: list[dict]) -> list[dict]:
        items = []
        for m in turns:
            if m["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
                continue
            if m.get("content"):
                kind = "input_text" if m["role"] == "user" else "output_text"
                items.append({"type": "message", "role": m["role"], "content": [{"type": kind, "text": m["content"]}]})
            items += [{"type": "function_call", "call_id": c["id"], "name": c["function"]["name"],
                       "arguments": c["function"]["arguments"]} for c in m.get("tool_calls") or []]
        return items


class ElevenVoice(RealtimeText):
    """Eleven Realtime in voice, on its OpenAI-compatible endpoint. It keeps no tool call a client adds to a session
    (nor a system note), so a turn's tool rounds stay in the session that made them, and a new turn's session replays
    the conversation's messages, with the earlier tool results as a note in its instructions. Its transcript marks
    emotion ("[happy]"), which is dropped."""

    def __init__(self, model: str, purpose: str, speech: Speech):
        self._model, self._purpose, self._provider, self._speech = model, purpose, "elevenlabs", speech
        self._lag = _Lag(purpose)
        self._open = None  # (context manager, connection, turns it has seen, pending call ids)
        self.chat = self.completions = self

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        told, turns = _split(messages)
        for attempt in range(2):  # a session that stops answering is opened again once, the newest turn typed
            started = time.time()
            try:
                async with asyncio.timeout(RESPONSE_SECONDS):
                    conn = await self._resume(turns)
                    if conn is None:
                        await self._close()
                        conn = await self._start(told, turns, tools, typed=bool(attempt))
                    await conn.response.create()
                    text, calls, details, audio = await self._reply(conn)
                break
            except TimeoutError:
                await self._close()
                if attempt:
                    raise
        usd = (time.time() - started) / 60 * ELEVEN_PER_MINUTE + self._voice_usd(details)
        C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "elevenlabs", "model": self._model,
                                  "purpose": self._purpose, "in": details.input_tokens, "out": details.output_tokens,
                                  "usd": round(usd, 6)})
        if calls:
            self._open = (*self._open[:2], turns, [c[0] for c in calls])
        else:
            await self._close()
        usage = SimpleNamespace(prompt_tokens=details.input_tokens, total_tokens=details.total_tokens)
        return _answer(re.sub(r"\s*\[[a-z][a-z ]{1,20}\]", "", text).strip(), calls, usage)

    async def _resume(self, turns: list[dict]):
        """The open session, after sending it the results of the tool calls it made; None when a new one is needed."""
        if not self._open:
            return None
        _, conn, seen, pending = self._open
        new = turns[len(seen):]
        if (turns[:len(seen)] != seen or not new or [c["id"] for c in new[0].get("tool_calls") or []] != pending
                or any(m["role"] != "tool" for m in new[1:])):
            return None
        for m in new[1:]:
            await conn.conversation.item.create(item={"type": "function_call_output", "call_id": m["tool_call_id"],
                                                      "output": m["content"]})
        return conn

    async def _start(self, told: str, turns: list[dict], tools: list[dict] | None, typed: bool):
        import httpx
        from openai import AsyncOpenAI
        key = C.load_env(C.ENV_FILE)["ELEVENLABS_API_KEY"]
        async with httpx.AsyncClient(timeout=30) as http:  # a short-lived secret, minted with the account key
            r = await http.post(f"{ELEVEN_URL}/realtime/client_secrets", headers={"xi-api-key": key},
                                json={"session": {"type": "realtime", "model": self._model}})
            r.raise_for_status()
        cm = AsyncOpenAI(base_url=ELEVEN_URL, api_key=r.json()["value"]).realtime.connect(model=self._model)
        conn = await cm.__aenter__()
        self._open = (cm, conn, turns, [])
        k = None if typed else _spoken_turn(turns)
        results = [text for role, text in _noted(turns if k is None else turns[:k], "note") if role == "note"]
        if results:  # the session drops replayed tool rounds and system notes: earlier results go in the instructions
            told += "\n\nTool results from earlier in this call, oldest first:\n" + "\n".join(results)
        pcm = {"format": {"type": "audio/pcm", "rate": PCM_RATE}}
        await conn.session.update(session={
            "type": "realtime", "instructions": told, "output_modalities": ["audio"],
            "audio": {"input": {**pcm, "turn_detection": None}, "output": {**pcm, "voice": VOICES["elevenlabs"]}},
            "tools": [{"type": "function", **t["function"]} for t in tools or []]})
        said = [m for m in (turns if k is None else turns[:k]) if m["role"] != "tool" and m.get("content")]
        for item in self._items([{"role": m["role"], "content": m["content"]} for m in said]):
            await conn.conversation.item.create(item=item)
        if k is not None:  # the newest turn spoken; it answers what it transcribed, so wait for the transcript
            spoken = await self._speech(turns[k]["content"])
            for i in range(0, len(spoken), CHUNK * 10):
                await conn.input_audio_buffer.append(audio=base64.b64encode(spoken[i:i + CHUNK * 10]).decode())
            await conn.input_audio_buffer.commit()
            self._lag.spoke()
            async for event in conn:
                if event.type == "error":
                    raise RuntimeError(f"realtime error: {event.error.message}")
                if event.type == "conversation.item.input_audio_transcription.completed":
                    break
        return conn

    async def _reply(self, conn):
        text, calls, audio = "", [], 0
        async for event in conn:
            if event.type == "response.output_audio.delta":
                audio += len(base64.b64decode(event.delta))
                self._lag.heard()
            if event.type == "error":
                raise RuntimeError(f"realtime error: {event.error.message}")
            if event.type == "response.done":
                for out in event.response.output or []:
                    if out.type == "message":
                        text += "".join(getattr(c, "text", None) or getattr(c, "transcript", None) or ""
                                        for c in out.content or [])
                    elif out.type == "function_call":
                        calls.append((out.call_id, out.name, out.arguments))
                if calls and not audio:
                    self._lag.tool_round()
                return text, calls, event.response.usage, audio

    async def _close(self) -> None:
        if self._open:
            cm, self._open = self._open[0], None
            try:
                await cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 - a session the server already closed
                pass


def _inlined(tool: dict) -> dict:
    """The tool with its schema's $defs written in place and its titles dropped: Vapi refuses either."""
    params = tool["function"].get("parameters") or {}
    defs = params.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                return walk(defs[node["$ref"].split("/")[-1]])
            out = {k: walk(v) for k, v in node.items() if k != "$defs" and not (k == "title" and isinstance(v, str))}
            kinds = {o.get("type") for o in out.get("anyOf", []) if isinstance(o, dict)}
            if "anyOf" in out and "type" not in out and len(kinds) == 1 and None not in kinds:
                out["type"] = kinds.pop()  # Vapi wants a type beside alternatives that share one
            return out
        return [walk(v) for v in node] if isinstance(node, list) else node
    return {**tool, "function": {**tool["function"], "parameters": walk(params)}}


# Vapi and Retell call our tools at a public address, a tunnel to TOOL_PORT given as TOOL_URL. One server per process
# hands each call's tool request to the turn waiting on that call, and answers the platform once the conversation has
# run it.
# TOOL_TOKEN may be fixed (an address a platform keeps, such as the Webex flow's); otherwise a new one per process
TOOL_PORT, TOOL_TOKEN = 8799, os.environ.get("TOOL_TOKEN") or uuid.uuid4().hex
_platform_calls: dict[str, asyncio.Queue] = {}  # call id -> its turn's queue of call events and tool requests
_tool_runner = None


async def _ran(queue: asyncio.Queue, calls: list[dict]) -> list[dict] | None:
    """The results of the tool calls, once the turn waiting on this call has run them; None if it never does."""
    results = asyncio.get_running_loop().create_future()
    await queue.put(("tools", (calls, results)))
    try:
        return await asyncio.wait_for(asyncio.shield(results), 60)
    except TimeoutError:
        return None


async def _tool_server() -> None:
    global _tool_runner
    if _tool_runner:
        return
    from aiohttp import web

    async def vapi_request(request):
        if request.match_info["token"] != TOOL_TOKEN:
            return web.Response(status=404)
        msg = (await request.json()).get("message") or {}
        queue = _platform_calls.get((msg.get("call") or {}).get("id"))
        calls = msg.get("toolCallList") or []
        if msg.get("type") != "tool-calls" or queue is None:
            return web.json_response({})
        results = await _ran(queue, calls)
        return web.json_response({"results": results or [{"toolCallId": c["id"], "result": "error: no result"}
                                                         for c in calls]})

    async def retell_request(request):  # one custom-function call: its name, its arguments and the call
        if request.match_info["token"] != TOOL_TOKEN:
            return web.Response(status=404)
        body = await request.json()
        queue = _platform_calls.get((body.get("call") or {}).get("call_id"))
        if queue is None:
            return web.Response(text="error: call ended")
        call = {"id": f"call_{uuid.uuid4().hex[:16]}", "function": {"name": body.get("name"), "arguments": body.get("args") or {}}}
        results = await _ran(queue, [call])
        return web.Response(text=results[0]["result"] if results else "error: no result")

    app = web.Application()
    app.router.add_post("/vapi/{token}", vapi_request)
    app.router.add_post("/retell/{token}", retell_request)
    from . import webex_client as webex  # the Webex flow's calls, and Twilio's media stream for its phone calls
    app.router.add_get("/webex/{token}/claim", webex.claim_request)
    app.router.add_get("/webex/{token}/session/{code}", webex.session_request)
    app.router.add_post("/webex/{token}/tool", webex.tool_request)
    app.router.add_get("/twilio/{token}/{code}", webex.media_socket)
    _tool_runner = web.AppRunner(app)
    await _tool_runner.setup()
    await web.TCPSite(_tool_runner, "127.0.0.1", TOOL_PORT).start()


class VapiVoice:
    """A speech-to-speech model as Vapi hosts it (OpenAI realtime), turn by turn. Each turn is a Vapi call on its
    WebSocket transport, created with the conversation so far as the assistant's starting messages: Vapi hands those
    to the model, while turns added mid-call are recorded but never reach it. So a refused turn is erased by leaving it
    out of the next call. The newest caller turn is streamed as speech at speaking pace and Vapi decides when it ends.
    Tools are Vapi's usual kind, called at our address: a request waits there while the conversation runs the tool,
    and is answered when the next turn of this class brings the result. Earlier tool rounds go into the starting
    messages as system notes, in order, as starting messages carry no tool turns. Each call's bill is read back from
    Vapi."""

    def __init__(self, model: str, purpose: str, speech: Speech, config: Path | None = None):
        self._model, self._purpose, self._speech = model, purpose, speech
        self._config = json.loads(config.read_text()) if config else None  # a cascaded assistant; else realtime
        self._key = C.load_env(C.ENV_FILE)["VAPI_PRIVATE_KEY"]
        self._tool_url = f"{os.environ['TOOL_URL'].rstrip('/')}/vapi/{TOOL_TOKEN}"
        self._open = None  # (websocket, call id, turns it has seen, reader task, queue)
        self._waiting = None  # (tool call ids, the future Vapi's request is waiting on)
        self._billing = set()  # cost read-backs still running
        self._lag = _Lag(purpose)
        self._spoken_at = None  # when the caller's line finished going out on the open call
        self.chat = self.completions = self

    _calls: asyncio.Semaphore | None = None  # VapiVoice's own open calls in this process (VAPI_LINES)

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        told, turns = _split(messages)
        await _tool_server()
        try:
            after_tool = self._resume(turns)
            if not after_tool:
                await self._close()
                if type(self) is VapiVoice:  # waiting for a line is not the turn's time
                    VapiVoice._calls = VapiVoice._calls or asyncio.Semaphore(VAPI_LINES)
                    await VapiVoice._calls.acquire()
                    self._line = True
            async with asyncio.timeout(RESPONSE_SECONDS):
                if not after_tool:
                    await self._start(told, turns, tools)
                text, calls = await self._listen(VAPI_QUIET_AFTER_TOOL_S if after_tool else VAPI_QUIET_S)
        except BaseException:
            await self._close()  # a failed turn's call would hold its line
            raise
        if calls:
            self._open = (*self._open[:2], turns, *self._open[3:])
        else:
            self._lag.answered(getattr(self, "_began", None))
            await self._close()
        return _answer(text.strip(), calls, SimpleNamespace(prompt_tokens=0, total_tokens=0))

    def _headers(self) -> dict:  # Vapi's edge refuses Python's default user agent
        return {"Authorization": f"Bearer {self._key}", "User-Agent": "curl/8.5.0"}

    async def _start(self, told: str, turns: list[dict], tools: list[dict] | None) -> None:
        import httpx
        import websockets
        k = _spoken_turn(turns)
        said = [{"role": role, "content": text} for role, text in _noted(turns if k is None else turns[:k], "system")]
        model = {"provider": "openai", "model": self._model,
                 "tools": [{**_inlined(t), "async": False, "server": {"url": self._tool_url}} for t in tools or []],
                 "messages": [{"role": "system", "content": told}] + said}
        if self._config:  # the provider configuration, with the airline's prompt, tools and transcription hints
            assistant = json.loads(json.dumps(self._config))
            assistant["model"] = {**assistant["model"], **model, "promptCacheKey": "vapi-airline-redteam"}
            assistant["transcriber"] = {**assistant["transcriber"], **AIRLINE_HINTS}
        else:
            assistant = {"model": model, "voice": {"provider": "openai", "voiceId": VOICES["vapi"]}}
        assistant |= {"firstMessageMode": "assistant-waits-for-user",
                      "clientMessages": ["transcript", "speech-update", "status-update"]}
        async with httpx.AsyncClient(timeout=30) as http:
            # Vapi's edge now and then refuses a call (403, 429), fails (5xx) or is over the account's concurrency
            for wait in (2, 5, 10, 20, 30, 45, 60, 60, 60, None):
                r = await http.post(f"{VAPI_URL}/call", headers=self._headers(), json={"assistant": assistant, "transport": {
                    "provider": "vapi.websocket", "audioFormat": {"format": "pcm_s16le", "container": "raw", "sampleRate": PCM_RATE}}})
                busy = r.status_code == 400 and "concurrency" in r.text.lower()
                if r.status_code < 300 or not busy and r.status_code not in (403, 429) and r.status_code < 500 or wait is None:
                    break
                await asyncio.sleep(wait)
        if r.status_code >= 300:
            raise RuntimeError(f"vapi call: {r.status_code} {r.text[:400]}")
        call = r.json()
        queue = asyncio.Queue()
        _platform_calls[call["id"]] = queue
        ws = await websockets.connect(call["transport"]["websocketCallUrl"], max_size=None)

        async def read():  # the socket's messages, in the same queue as the call's tool requests
            async for raw in ws:
                if not isinstance(raw, bytes):  # bytes are the model's audio; its words come as transcripts
                    await queue.put(("socket", json.loads(raw)))
                elif _voiced(raw):
                    self._lag.sound()
        speech = b"" if k is None else await self._speech(turns[k]["content"])
        self._spoken_at = None

        async def speak():  # at speaking pace, then the silence of an open line until the call is closed
            for i in range(0, len(speech), VAPI_FRAME):
                await ws.send(speech[i:i + VAPI_FRAME])
                await asyncio.sleep(VAPI_FRAME / PCM_BYTES_PER_S)
            self._spoken_at = time.time()  # the caller's last word has gone out
            if speech:
                self._lag.spoke()
            while True:
                await ws.send(b"\0" * VAPI_SILENCE_FRAME)
                await asyncio.sleep(VAPI_SILENCE_FRAME / PCM_BYTES_PER_S)
        self._open = (ws, call["id"], turns, [asyncio.create_task(read()), asyncio.create_task(speak())], queue)

    def _resume(self, turns: list[dict]) -> bool:
        """Answer the tool request the open call is waiting on, from the results now in the conversation."""
        if not self._open or not self._waiting:
            return False
        ids, results = self._waiting
        seen = self._open[2]
        new = turns[len(seen):]
        if (turns[:len(seen)] != seen or not new or [c["id"] for c in new[0].get("tool_calls") or []] != ids
                or any(m["role"] != "tool" for m in new[1:])):
            return False
        if not results.done():  # Vapi may have stopped waiting
            results.set_result([{"toolCallId": m["tool_call_id"], "result": m["content"]} for m in new[1:]])
        self._waiting = None
        return True

    async def _listen(self, quiet: float) -> tuple[str, list]:
        """What the model says once the caller's line has been spoken, until it has been quiet a while, or the tool
        calls it asks us to run. Words it starts over the caller's line (cut off as the caller goes on, "How can I—")
        are not the reply: the reply is the speech it starts after the line."""
        queue, text, quiet_since, replying = self._open[4], [], None, False
        self._began = None  # when the reply's speech started: after the line and after any tool round
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), 0.5)
            except TimeoutError:
                now = time.time()
                if replying and quiet_since and text and now - quiet_since > quiet:
                    return " ".join(text), []
                if not replying and self._spoken_at and now - self._spoken_at > VAPI_SILENT_S:
                    return "", []
                continue
            if kind == "tools":
                calls, results = item
                self._lag.tool_round()
                self._waiting = ([c["id"] for c in calls], results)
                return " ".join(text), [(c["id"], c["function"]["name"], json.dumps(c["function"].get("arguments") or {}))
                                        for c in calls]
            t = item.get("type")
            after = self._spoken_at is not None
            if t == "transcript" and item.get("role") == "assistant" and item.get("transcriptType") == "final":
                replying = replying or after and time.time() - self._spoken_at > VAPI_TRAILING_S  # no speech-update came
                if replying and (not text or text[-1] != item.get("transcript", "")):  # a final can come twice
                    text.append(item.get("transcript", ""))
            elif t == "speech-update" and item.get("role") == "assistant":
                replying = replying or after and item.get("status") == "started"
                if replying and self._began is None:
                    self._began = time.time()
                quiet_since = time.time() if item.get("status") == "stopped" else None
            elif t == "status-update" and item.get("status") == "ended":
                return " ".join(text), []

    async def _close(self) -> None:
        if getattr(self, "_line", False):
            VapiVoice._calls.release()
            self._line = False
        if self._waiting:  # a request nobody will answer now: release Vapi
            if not self._waiting[1].done():
                self._waiting[1].set_result([{"toolCallId": i, "result": "error: call ended"} for i in self._waiting[0]])
            self._waiting = None
        if self._open:
            (ws, call_id, _, tasks, _), self._open = self._open, None
            _platform_calls.pop(call_id, None)
            for task in tasks:
                task.cancel()
            try:
                await self._hang_up(ws)
            except Exception:  # noqa: BLE001 - a call the platform already ended
                pass
            task = asyncio.create_task(self._bill(call_id))
            self._billing.add(task)
            task.add_done_callback(self._billing.discard)

    @staticmethod
    async def _hang_up(ws) -> None:
        await ws.send(json.dumps({"type": "hangup"}))
        await ws.close()

    async def _bill(self, call_id: str) -> None:
        """The call's cost as Vapi reports it once the call has settled."""
        import httpx
        await asyncio.sleep(20)
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.get(f"{VAPI_URL}/call/{call_id}", headers=self._headers())
        cost = (r.json().get("cost") or 0) if r.status_code == 200 else 0
        C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "vapi", "model": self._model,
                                  "purpose": self._purpose, "in": 0, "out": 0, "usd": round(cost, 6)})


@functools.cache
def _caller_track():
    """aiortc's audio track class for the caller: queued speech in 20 ms frames at speaking pace, else silence."""
    import fractions
    import av
    from aiortc import MediaStreamTrack

    class CallerTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self.pcm, self._t0, self._pts, self.drained = bytearray(), None, 0, None  # drained: called when speech runs out

        async def recv(self):
            samples = VAPI_FRAME // 2
            self._t0 = self._t0 or time.time()
            await asyncio.sleep(max(0.0, self._t0 + self._pts / PCM_RATE - time.time()))
            chunk = bytes(self.pcm[:VAPI_FRAME]).ljust(VAPI_FRAME, b"\0")
            had = bool(self.pcm)
            del self.pcm[:VAPI_FRAME]
            if had and not self.pcm and self.drained:
                self.drained()
            frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
            frame.planes[0].update(chunk)
            frame.sample_rate, frame.pts, frame.time_base = PCM_RATE, self._pts, fractions.Fraction(1, PCM_RATE)
            self._pts += samples
            return frame
    return CallerTrack


class RetellVoice(VapiVoice):
    """The cascaded Retell agent (Retell's LLM, transcriber and voice), turn by turn. Each turn is a Retell web call,
    joined as its web SDK joins it (WebRTC through Retell's gateway); the newest caller turn is spoken into it and the
    agent's words come back as Retell's live transcript (its monitor socket). A web call cannot start with earlier
    turns, so they go in as text: an injected context note (update-live-call) before the caller speaks, or with
    RETELL_HISTORY=prompt a variable in the agent's prompt. Either way a refused turn is erased by leaving it out. Tools
    and bills work as in VapiVoice. The agent is set up once per process from the provider configuration, with the
    airline's prompt, tools and keywords; its ids are kept in RETELL_IDS. The guarded agent is a second agent, the
    same with Retell's AI-disclosure and scope-boundary handbook prompts on (RETELL_GUARD_IDS)."""

    _ready: dict[bool, tuple[str, str]] = {}  # guarded or not -> (the prompt it was set up with, agent id)
    _setting_up = asyncio.Lock()
    _lines = asyncio.Semaphore(18)  # calls open at once, both agents together: the account allows 20

    async def create(self, **kw) -> SimpleNamespace:
        """A turn, once a line is free; the line stays held while the call waits on a tool result."""
        if not self._holding:
            await RetellVoice._lines.acquire()
            self._holding = True
        try:
            return await super().create(**kw)
        except BaseException:
            await self._close()  # a failed turn's call would hold its line
            raise
        finally:
            if self._open is None and self._holding:
                RetellVoice._lines.release()
                self._holding = False

    def __init__(self, model: str, purpose: str, speech: Speech, config: Path, guard: bool = False):
        self._model, self._purpose, self._speech, self._guard = model, purpose, speech, guard
        self._config = json.loads(config.read_text())
        self._key = C.load_env(C.ENV_FILE)["RETELL_API_KEY"]
        self._tool_url = f"{os.environ['TOOL_URL'].rstrip('/')}/retell/{TOOL_TOKEN}"
        self._inject = os.environ.get("RETELL_HISTORY", "inject") == "inject"
        self._open = None  # (connection, call id, turns it has seen, tasks, queue)
        self._waiting = None
        self._heard = {}  # this call's utterances by id, as Retell's transcript last gave them
        self._said = ""  # the agent's words in this call already returned (before a tool round)
        self._holding = False  # one of the account's call lines
        self._billing = set()
        self._lag = _Lag(purpose)
        self.chat = self.completions = self

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"}

    async def _api(self, http, method: str, path: str, body: dict | None = None) -> dict:
        for wait in (2, 5, 15, None):  # rate limits and server errors pass
            r = await http.request(method, f"{RETELL_URL}{path}", headers=self._headers(), json=body)
            if r.status_code < 300 or r.status_code != 429 and r.status_code < 500 or wait is None:
                break
            await asyncio.sleep(wait)
        if r.status_code >= 300:
            raise RuntimeError(f"retell {path}: {r.status_code} {r.text[:400]}")
        return r.json() if r.content else {}

    async def _agent(self, http, told: str, tools: list[dict] | None) -> str:
        """The agent's id, after setting up its LLM (prompt, tools at this process's address) and the agent itself."""
        async with RetellVoice._setting_up:
            path = RETELL_GUARD_IDS if self._guard else RETELL_IDS
            if self._guard not in RetellVoice._ready:
                ids = json.loads(path.read_text()) if path.exists() else {}
                # the config asks for the fast tier (model_high_priority); tell-1 offers only the standard one
                llm = {"model_config": {"type": "cascading", "primary": {"model": self._model, "tier": "standard"}},
                       "general_prompt": f"{told}\n\n{{{{history}}}}", "start_speaker": "user", "begin_message": "",
                       "general_tools": [{"type": "custom", "name": f["name"], "description": f.get("description", ""),
                                          "url": self._tool_url, "method": "POST", "timeout_ms": 60000,
                                          "parameters": f.get("parameters") or {"type": "object", "properties": {}},
                                          "speak_during_execution": False, "speak_after_execution": True}
                                         for f in (_inlined(t)["function"] for t in tools or [])]}
                ids["llm_id"] = (await self._api(http, "PATCH", f"/update-retell-llm/{ids['llm_id']}", llm) if "llm_id" in ids
                                 else await self._api(http, "POST", "/create-retell-llm", llm))["llm_id"]
                skip = {"agent_id", "channel", "last_modification_timestamp", "version", "is_published", "retellLlmData"}
                agent = {k: v for k, v in self._config.items() if k not in skip} | {
                    "agent_name": "airline-redteam-guard" if self._guard else "airline-redteam",
                    "boosted_keywords": AIRLINE_HINTS["customVocabulary"],
                    "response_engine": {"type": "retell-llm", "llm_id": ids["llm_id"]},
                    "handbook_config": self._config["handbook_config"] | (
                        {"ai_disclosure": True, "scope_boundaries": True} if self._guard else {})}
                if "agent_id" not in ids:
                    ids["agent_id"] = (await self._api(http, "POST", "/create-agent", agent))["agent_id"]
                else:  # the handbook is what tells the two agents apart: set it on every start
                    await self._api(http, "PATCH", f"/update-agent/{ids['agent_id']}",
                                    {"handbook_config": agent["handbook_config"]})
                path.write_text(json.dumps(ids))
                RetellVoice._ready[self._guard] = (told, ids["agent_id"])
            if RetellVoice._ready[self._guard][0] != told:
                raise RuntimeError("the agent's prompt differs between conversations; one Retell LLM serves them all")
            return RetellVoice._ready[self._guard][1]

    async def _start(self, told: str, turns: list[dict], tools: list[dict] | None) -> None:
        import httpx
        import websockets
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
        from aiortc.contrib.media import MediaBlackhole
        k = _spoken_turn(turns)
        history = ElevenAgentVoice._notes((turns if k is None else turns[:k])[1:])  # with its tool rounds
        note = f"The call so far, oldest first. Carry on from here:\n{history}" if history else ""
        queue = asyncio.Queue()
        async with httpx.AsyncClient(timeout=30) as http:
            agent_id = await self._agent(http, told, tools)
            call = await self._api(http, "POST", "/v3/create-web-call", {
                "agent_id": agent_id, "retell_llm_dynamic_variables": {"history": "" if self._inject else note}})
            _platform_calls[call["call_id"]] = queue
            # UDP does not leave this machine, so media goes through Retell's relay over TLS on port 443
            pc = RTCPeerConnection(RTCConfiguration([
                RTCIceServer(urls=s["urls"], username=s.get("username"), credential=s.get("credential"))
                for s in call["ice_servers"] if s["urls"].startswith("turns:")]))
            conn = SimpleNamespace(pc=pc, sink=MediaBlackhole(), caller=_caller_track()(), session=None, monitor=None,
                                   auth={"Authorization": f"Bearer {call['access_token']}"})
            self._open, self._heard, self._said = (conn, call["call_id"], turns, [], queue), {}, ""
            control, opened = pc.createDataChannel("control"), asyncio.Event()
            control.on("open", opened.set)
            control.on("message", lambda m: json.loads(m).get("state") in ("ended", "replaced")
                       and queue.put_nowait(("ended", None)))
            conn.caller.drained = self._lag.spoke

            def heard(track):  # the agent's audio, read for when it starts; its words come as the transcript
                async def read():
                    try:
                        while True:
                            frame = await track.recv()
                            if self._lag.end is not None and _voiced(frame.to_ndarray().tobytes()):
                                self._lag.heard()
                    except Exception:  # noqa: BLE001 - the call ended
                        pass
                self._open[3].append(asyncio.create_task(read()))
            pc.on("track", heard)
            pc.addTrack(conn.caller)
            await pc.setLocalDescription(await pc.createOffer())
            sessions = f"{RETELL_URL}/webrtc-proxy/{call['call_id']}/v1/webrtc/sessions"
            r = await http.post(sessions, headers=conn.auth, json={"identity": "client", "sdp": pc.localDescription.sdp})
            if r.status_code >= 300:
                raise RuntimeError(f"retell webrtc session: {r.status_code} {r.text[:300]}")
            conn.session = f"{sessions}/{r.json()['session_id']}"
            await pc.setRemoteDescription(RTCSessionDescription(sdp=r.json()["sdp"], type="answer"))
            await conn.sink.start()
            await asyncio.wait_for(opened.wait(), 20)
            monitor = f"wss://{RETELL_URL.split('//')[1]}/v2/monitor-call/{call['call_id']}"
            conn.monitor = await websockets.connect(monitor, subprotocols=["bearer", self._key])
            if note and self._inject:
                await self._api(http, "PATCH", f"/v2/update-live-call/{call['call_id']}",
                                {"call_control": {"additional_context": note}})
        if k is not None:
            conn.caller.pcm += await self._speech(turns[k]["content"])

        async def watch():  # the live transcript: utterances by id, each growing as it is spoken
            for attempt in range(8):  # "call not live" (4004) passes; Retell's web SDK reconnects the same way
                try:
                    async for raw in conn.monitor:
                        m = json.loads(raw)
                        if m.get("type") == "call_ended":
                            await queue.put(("ended", None))
                        for u in m.get("transcripts") or []:  # a reconnect's snapshot repeats them: merged by id
                            await queue.put(("utterance", u))
                    await queue.put(("ended", None))  # closed normally: the call is over
                    return
                except websockets.ConnectionClosedError as e:
                    if not (e.rcvd and e.rcvd.code == 4004):
                        raise
                await asyncio.sleep(0.5 * 2 ** attempt)
                try:
                    conn.monitor = await websockets.connect(monitor, subprotocols=["bearer", self._key])
                except (OSError, websockets.WebSocketException):
                    pass  # the closed socket raises 4004 again on the next pass
            raise RuntimeError("retell monitor: call not live")
        self._open[3].append(asyncio.create_task(watch()))

    async def _listen(self, quiet: float) -> tuple[str, list]:
        """What the agent says until the caller's line has been spoken and the transcript (either side) has stopped
        growing a while, or the tool call it asks us to run. The agent can start answering in a pause within the
        caller's line and go on after it, so its words are all of its utterances in the call; words returned before
        a tool round are not returned again."""
        conn, queue, quiet_since = self._open[0], self._open[4], time.time()
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), 0.5)
            except TimeoutError:
                if (not conn.caller.pcm and time.time() - quiet_since > quiet and self._words() != self._said):
                    return self._new(), []
                continue
            if kind == "tools":
                calls, results = item
                self._lag.tool_round()
                self._waiting = ([c["id"] for c in calls], results)
                text, self._said = self._new(), self._words()
                return text, [(c["id"], c["function"]["name"], json.dumps(c["function"]["arguments"])) for c in calls]
            if kind == "utterance":
                self._heard[item["id"]] = item
                quiet_since = time.time()
            elif kind == "ended":
                return self._new(), []

    def _words(self) -> str:
        """The agent's words in this call, in the order said (the earlier turns are not in it: they went in as text)."""
        heard = sorted(self._heard.values(), key=lambda u: u.get("time_sec") or 0)
        return " ".join((u.get("content") or "").strip() for u in heard if u.get("role") == "agent").strip()

    def _new(self) -> str:
        words = self._words()
        return words[len(self._said):].strip() if words.startswith(self._said) else words

    @staticmethod
    async def _hang_up(conn) -> None:
        import httpx
        if conn.session:
            async with httpx.AsyncClient(timeout=10) as http:
                await http.delete(conn.session, headers=conn.auth)
        if conn.monitor:
            await conn.monitor.close()
        await conn.sink.stop()
        await conn.pc.close()

    async def _bill(self, call_id: str) -> None:
        """The call's cost as Retell reports it (in cents) once the call has settled."""
        import httpx
        await asyncio.sleep(30)
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.get(f"{RETELL_URL}/v2/get-call/{call_id}", headers=self._headers())
        cents = ((r.json().get("call_cost") or {}).get("combined_cost") or 0) if r.status_code == 200 else 0
        C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "retell", "model": self._model,
                                  "purpose": self._purpose, "in": 0, "out": 0, "usd": round(cents / 100, 6)})


class _Dropped(Exception):
    """GPT-Live's server closed the session mid-turn."""


class GptLiveVoice(VapiVoice):
    """GPT-Live, turn by turn. Each turn opens a Live session seeded with the conversation kept so far (the API takes
    history only at session start, and only as text messages, so earlier tool rounds are seeded as developer notes),
    then streams the caller's newest turn as speech at speaking pace, followed by silence as an open line would carry.
    The reply is the live model's transcript, over once its audio has been silent a while with no backend work under way
    and no backend answer still to be said. The
    backend's tool calls are handed to the conversation and answered in the same session on the next turn of this
    class. The bill is the session's live audio plus the backend's tokens."""

    def __init__(self, model: str, purpose: str, speech: Speech):
        self._model, self._purpose, self._speech, self._lag = model, purpose, speech, _Lag(purpose)
        self._key = C.load_env(C.ENV_FILE)["OPENAI_API_KEY"]
        self._open = None  # (websocket, session state, turns it has seen, [reader, speaker], queue)
        self._waiting = None  # (tool call ids, the future their results are set on)
        self.chat = self.completions = self

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        told, turns = _split(messages)
        for attempt in range(2):  # the server now and then drops a session mid-turn: a fresh turn runs once more
            after_tool = False
            try:
                async with asyncio.timeout(RESPONSE_SECONDS):
                    waiting = self._waiting
                    after_tool = self._resume(turns)
                    if after_tool:
                        ws = self._open[0]
                        for r in waiting[1].result():
                            await ws.send(json.dumps({"type": "response.item.create", "item": {
                                "type": "function_call_output", "call_id": r["toolCallId"], "output": r["result"]}}))
                        await ws.send(json.dumps({"type": "response.create"}))
                    else:
                        await self._close()
                        await self._start(told, turns, tools)
                    text, calls = await self._listen(VAPI_QUIET_AFTER_TOOL_S if after_tool else VAPI_QUIET_S)
                break
            except _Dropped:
                await self._close()
                if attempt or after_tool:  # mid tool round, a rerun would run the tools again
                    raise RuntimeError("gpt-live: the server dropped the session") from None
        if calls:
            self._open = (*self._open[:2], turns, *self._open[3:])
        else:
            await self._close()
        return _answer(text.strip(), calls, SimpleNamespace(prompt_tokens=0, total_tokens=0))

    @staticmethod
    def _history(turns: list[dict]) -> list[dict]:
        return [{"type": "message", "role": role, "content": [
            {"type": "output_text" if role == "assistant" else "input_text", "text": text}]}
            for role, text in _noted(turns, "developer")]

    async def _start(self, told: str, turns: list[dict], tools: list[dict] | None) -> None:
        import websockets
        k = _spoken_turn(turns)
        backend = {"model": GPT_LIVE_BACKENDS[self._model], "instructions": told, "reasoning": {"effort": GPT_LIVE_REASONING},
                   "tools": [{"type": "function", "name": f["name"], "description": f.get("description", ""),
                              "parameters": f.get("parameters") or {"type": "object", "properties": {}},
                              "strict": False}  # as the bot sends them: optional parameters stay optional
                             for f in (t["function"] for t in tools or [])]}
        session = {"model": self._model, "instructions": f"{told.rstrip()}\n\n{BOT_TEXT['DELEGATION_PROMPT']}",
                   "audio": {"output": {"voice": VOICES["openai"]}, "format": {"type": "audio/pcm", "rate": PCM_RATE}},
                   "delegation": {"type": "responses", "responses": backend}}
        history = self._history(turns if k is None else turns[:k])
        if history:
            session["input"] = history
        ws = await websockets.connect(GPT_LIVE_URL, additional_headers={"Authorization": f"Bearer {self._key}"},
                                      max_size=None)
        await ws.send(json.dumps({"type": "session.start", "session": session}))
        async with asyncio.timeout(30):
            while (event := json.loads(await ws.recv()))["type"] != "session.started":
                if event["type"] == "error":
                    await ws.close()
                    raise RuntimeError(f"gpt-live start: {event.get('error')}")
        # busy: delegations under way; owed: when the backend last answered, until the live model starts saying it
        state = SimpleNamespace(busy=0, owed=None, seconds=0.0, tokens=[0, 0, 0], t0=time.time(),
                                spoken_at=None, backend_at=time.time(),  # caller's line sent; last backend event
                                delegated=False, answered=None, voiced_after=None)  # backend work for the caller's line
        queue = asyncio.Queue()

        async def read():  # the live model's words, the backend's tool calls, and the session's usage
            try:
                await read_events()
            except websockets.ConnectionClosed:
                pass
            await queue.put(("dropped", None))  # reached only when the socket ends before session.closed

        async def read_events():
            calls = []  # the backend's tool calls in the response under way
            async for raw in ws:
                event = json.loads(raw)
                t = event.get("type")
                if t == "session.output_transcript.delta":
                    await queue.put(("words", event.get("delta", "")))
                elif t == "session.output_audio.delta":  # it streams on, silence included: only speech counts
                    if max(map(abs, array.array("h", base64.b64decode(event.get("delta", "")))), default=0) > GPT_LIVE_VOICED:
                        await queue.put(("voice", time.time()))
                elif t == "session.delegation.created":
                    state.busy, state.backend_at = state.busy + 1, time.time()
                    state.delegated = state.delegated or state.spoken_at is not None
                elif t == "response.event":
                    state.backend_at = time.time()
                    inner = event.get("event") or {}
                    item = inner.get("item") or {}
                    if inner.get("type") == "response.output_item.done" and item.get("type") == "function_call":
                        calls.append((item["call_id"], item["name"], item.get("arguments") or "{}"))
                    elif inner.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                        usage = (inner.get("response") or {}).get("usage") or {}
                        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
                        for i, n in enumerate((usage.get("input_tokens") or 0, cached, usage.get("output_tokens") or 0)):
                            state.tokens[i] += n
                        if calls:  # the delegation goes on once their results are in
                            self._lag.tool_round()
                            await queue.put(("calls", calls))
                            calls = []
                        else:  # done: the live model says its result next
                            state.busy, state.owed = max(0, state.busy - 1), time.time()
                            state.answered = state.owed
                elif t in ("session.usage.updated", "session.closed"):
                    state.seconds = (event.get("usage") or {}).get("seconds") or state.seconds
                    if t == "session.closed":
                        await queue.put(("closed", None))
                        return
                elif t == "error":
                    await queue.put(("error", event.get("error")))
        spoken = b"" if k is None else await self._speech(turns[k]["content"])

        async def speak():  # the caller's line at speaking pace, then silence until the turn is over
            for i in itertools.count(0, GPT_LIVE_FRAME):
                frame = spoken[i:i + GPT_LIVE_FRAME] or b"\0" * GPT_LIVE_FRAME
                if i >= len(spoken) and state.spoken_at is None:
                    state.spoken_at = time.time()
                    self._lag.spoke()
                await ws.send(json.dumps({"type": "session.input_audio.append", "audio": base64.b64encode(frame).decode()}))
                await asyncio.sleep(GPT_LIVE_FRAME / PCM_BYTES_PER_S)
        self._open = (ws, state, turns, [asyncio.create_task(read()), asyncio.create_task(speak())], queue)

    async def _listen(self, quiet: float) -> tuple[str, list]:
        """The live model's reply, or the backend's tool calls. The reply is what it says once the caller's line has
        been spoken, until its audio has been silent a while with no backend work under way and no backend answer
        still to be said. A part it started over the caller's line is not the reply unless it carries on into one
        (more than a few words); a reply of backchannels alone ("Mm-hmm.") waits GPT_LIVE_HOLD_S for the answer."""
        _, state, _, _, queue = self._open
        parts, heard_at = [], time.time()  # parts: [onset, words, started over the caller's line]
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), 0.5)
            except TimeoutError:
                now = time.time()
                owed = state.owed is not None and now - state.owed < GPT_LIVE_OWED_S
                busy = state.busy and now - state.backend_at < GPT_LIVE_STALE_S
                said = [p for p in parts if (not p[2] or len("".join(p[1]).split()) > 6) and "".join(p[1]).strip()]
                real = [p for p in said if not _backchannel("".join(p[1]))]
                if state.spoken_at and not busy and not owed:
                    if real and now - heard_at > quiet or said and now - heard_at > GPT_LIVE_HOLD_S:
                        # the reply starts with the backend's answer when there was one, else with its first real part
                        self._lag.answered(state.voiced_after if state.delegated else (real or said)[0][0])
                        return "".join(w for p in (real or said) for w in p[1]), []
                    if not said and now - max(heard_at, state.spoken_at) > GPT_LIVE_SILENT_S:
                        return "", []
                continue
            if kind in ("words", "voice"):
                now = item if kind == "voice" else time.time()
                if not parts or now - heard_at > (GPT_LIVE_GAP_S if kind == "voice" else 2.0):  # a new part
                    parts.append([now, [], state.spoken_at is None])
                if kind == "words":
                    parts[-1][1].append(item)
                else:
                    self._lag.sound(item)
                    if state.answered is not None and state.voiced_after is None and item >= state.answered:
                        state.voiced_after = item
                    if state.owed is not None and item - state.owed > 0.5:
                        state.owed = None  # it has started saying the backend's answer
                heard_at = now
            elif kind == "calls":
                self._waiting = ([c[0] for c in item], asyncio.get_running_loop().create_future())
                return "".join(w for p in parts if not p[2] for w in p[1]), item
            elif kind == "closed":
                return "".join(w for p in parts if not p[2] for w in p[1]), []
            elif kind == "dropped":
                raise _Dropped()
            else:
                raise RuntimeError(f"gpt-live error: {item}")

    async def _close(self) -> None:
        self._waiting = None
        if self._open:
            (ws, state, _, (reader, speaker), _), self._open = self._open, None
            speaker.cancel()
            try:  # the final usage comes with session.closed
                async with asyncio.timeout(5):
                    await ws.send(json.dumps({"type": "session.close"}))
                    await reader
            except Exception:  # noqa: BLE001 - a session the server already ended
                pass
            reader.cancel()
            await ws.close()
            i, cached, o = state.tokens
            per_in, per_cached, per_out = GPT_LIVE_BACKEND_PRICES
            usd = ((state.seconds or time.time() - state.t0) / 60 * GPT_LIVE_PER_MINUTE  # wall time if never reported
                   + ((i - cached) * per_in + cached * per_cached + o * per_out) / 1e6)
            C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "openai", "model": self._model,
                                      "purpose": self._purpose, "in": i, "out": o, "usd": round(usd, 6)})


# ElevenLabs Agents, cascaded: ElevenLabs' recommended agent configuration with its red-team overlay merged on (its
# voice stack, LLM and blocking guardrails). Audio is 16 kHz PCM both ways. It bills by the minute: logged at an
# estimated rate, to be checked against the bill.
ELEVEN_AGENTS_URL, ELEVEN_AGENTS_WS = "https://api.elevenlabs.io/v1/convai", "wss://api.elevenlabs.io/v1/convai/conversation"
ELEVEN_AGENT_IDS = RETELL_IDS.with_name("eleven-agent.json")
ELEVEN_AGENT_RATE, ELEVEN_AGENT_FRAME, ELEVEN_AGENT_PER_MINUTE = 16000, 3200, 0.10  # 3200 bytes: 100 ms
ELEVEN_AGENT_OVERLAY = CONFIGS / "elevenlabs-redteam-overlay.json"
ELEVEN_QUIET_EVENTS = {"user_transcript", "tentative_user_transcript", "vad_score", "internal_tentative_agent_response",
                       "interruption", "agent_tool_response", "agent_chat_response_part", "internal_turn_probability",
                       "agent_response_metadata", "conversation_initiation_metadata"}


def _merged(base: dict, overlay: dict) -> dict:
    """The overlay written onto base, nested settings merged key by key, its notes (_comment) left out."""
    out = dict(base)
    for k, v in overlay.items():
        if not k.startswith("_"):
            out[k] = _merged(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _to16k(pcm: bytes) -> bytes:
    """24 kHz PCM16 as 16 kHz, by linear interpolation."""
    import numpy as np
    x = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32)
    return np.interp(np.arange(0, len(x), 1.5), np.arange(len(x)), x).astype(np.int16).tobytes() if len(x) else b""


def _eleven_params(node):
    """A tool's JSON schema as ElevenLabs takes it: one type per field (optional ones lose their null), every
    property described, no titles, defaults or additionalProperties."""
    if isinstance(node, list):
        return [_eleven_params(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {k: _eleven_params(v) for k, v in node.items() if k not in ("title", "default", "additionalProperties")}
    if "anyOf" in out:
        kept = [o for o in out.pop("anyOf") if o.get("type") != "null"]
        out = {**(kept[0] if kept else {}), **{k: v for k, v in out.items() if k != "type"}}
    if isinstance(out.get("properties"), dict):
        out["properties"] = {name: {"description": name.replace("_", " "), **p} for name, p in out["properties"].items()}
    return out


class ElevenAgentVoice(VapiVoice):
    """The cascaded ElevenLabs agent (ElevenLabs' transcriber, LLM, voice and guardrails), turn by turn. The agent is
    set up once per process from the provider configuration with ElevenLabs' red-team overlay merged on, and the
    airline's prompt, greeting and tools; the tools are client tools, called over the conversation's socket and run by
    the conversation, so no public address is needed. Each turn is a conversation on the agent's socket: the first
    hears the greeting and then the caller; a later one opens without the greeting, with the call so far as a
    contextual update, so a refused turn is erased by leaving it out. The caller's line is streamed at speaking pace
    and followed by the silence of an open line. A guardrail ends the call by closing the socket: the conversation then
    ends (hung_up) as on a transfer, and words the caller never heard are not the reply."""

    _ready: tuple[str, str] | None = None  # (the prompt it was set up with, agent id)
    _setting_up = asyncio.Lock()

    def __init__(self, model: str, purpose: str, speech: Speech, config: Path):
        self._model, self._purpose, self._speech = model, purpose, speech
        self._config = _merged(json.loads(config.read_text()), json.loads(ELEVEN_AGENT_OVERLAY.read_text()))
        self._key = C.load_env(C.ENV_FILE)["ELEVENLABS_API_KEY"]
        self._open = self._waiting = None  # (socket, state, turns it has seen, tasks, queue); (call ids, future)
        self._lag, self.hung_up, self.hangups = _Lag(purpose), False, []
        self.chat = self.completions = self

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        told, turns = _split(messages, opening=False)  # the greeting is the agent's first message, not a rule
        async with asyncio.timeout(RESPONSE_SECONDS):
            waiting = self._waiting
            after_tool = self._resume(turns)
            if after_tool:
                self._open[1].reply_at = None  # the reply is what it says once the results are back
                for r in waiting[1].result():
                    await self._open[0].send(json.dumps({"type": "client_tool_result", "tool_call_id": r["toolCallId"],
                                                         "result": r["result"], "is_error": False}))
            else:
                await self._close()
                await self._start(told, turns, tools)
            text, calls = await self._listen(VAPI_QUIET_AFTER_TOOL_S if after_tool else VAPI_QUIET_S)
        if calls:
            self._open = (*self._open[:2], turns, *self._open[3:])
        else:
            await self._close()
        return _answer(text.strip(), calls, SimpleNamespace(prompt_tokens=0, total_tokens=0))

    async def _api(self, http, method: str, path: str, body: dict | None = None) -> dict:
        for wait in (2, 5, 15, None):  # rate limits and server errors pass
            r = await http.request(method, f"{ELEVEN_AGENTS_URL}{path}", headers={"xi-api-key": self._key}, json=body)
            if r.status_code < 300 or r.status_code != 429 and r.status_code < 500 or wait is None:
                break
            await asyncio.sleep(wait)
        if r.status_code >= 300:
            raise RuntimeError(f"elevenlabs {path}: {r.status_code} {r.text[:800]}")
        return r.json() if r.content else {}

    async def _agent(self, http, told: str, greeting: str, tools: list[dict] | None) -> str:
        """The agent's id, after writing its configuration (created on first use, ids kept in ELEVEN_AGENT_IDS)."""
        async with ElevenAgentVoice._setting_up:
            if ElevenAgentVoice._ready is None:
                ids = json.loads(ELEVEN_AGENT_IDS.read_text()) if ELEVEN_AGENT_IDS.exists() else {}
                conf = json.loads(json.dumps(self._config))
                agent = conf["conversation_config"]["agent"]
                agent["prompt"].pop("built_in_tools", None)  # the clinic's end-call tool: no other agent has one
                agent["prompt"] |= {"prompt": told, "tools": [
                    {"type": "client", "name": f["name"], "description": f.get("description", ""),
                     "expects_response": True, "response_timeout_secs": 30,
                     "parameters": _eleven_params(f.get("parameters") or {"type": "object", "properties": {}})}
                    for f in (_inlined(t)["function"] for t in tools or [])]}
                agent["first_message"] = greeting
                conf["conversation_config"]["asr"]["keywords"] = AIRLINE_HINTS["customVocabulary"]
                conf["platform_settings"]["overrides"] = {
                    "conversation_config_override": {"agent": {"first_message": True}}}  # later turns skip the greeting
                body = {"name": "airline-redteam", **conf}
                if "agent_id" in ids:
                    await self._api(http, "PATCH", f"/agents/{ids['agent_id']}", body)
                else:
                    ids["agent_id"] = (await self._api(http, "POST", "/agents/create", body))["agent_id"]
                ELEVEN_AGENT_IDS.write_text(json.dumps(ids))
                ElevenAgentVoice._ready = (told, ids["agent_id"])
            if ElevenAgentVoice._ready[0] != told:
                raise RuntimeError("the agent's prompt differs between conversations; one ElevenLabs agent serves them all")
            return ElevenAgentVoice._ready[1]

    @staticmethod
    def _notes(turns: list[dict]) -> str:
        lines, called = [], {}
        for m in turns:
            if m["role"] == "tool":
                lines.append(f"[{called.get(m['tool_call_id'], 'A tool')} returned: {m['content']}]")
                continue
            if m.get("content"):
                lines.append(f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}")
            for c in m.get("tool_calls") or []:
                called[c["id"]] = f"{c['function']['name']}({c['function']['arguments']})"
                lines.append(f"[Agent called {called[c['id']]}]")
        return "\n".join(lines)

    async def _start(self, told: str, turns: list[dict], tools: list[dict] | None) -> None:
        import httpx
        import websockets
        k = _spoken_turn(turns)
        said = (turns if k is None else turns[:k])[1:]  # after the "connected" turn
        greeting = said[0]["content"] if said and said[0]["role"] == "assistant" and not said[0].get("tool_calls") else ""
        later = len(said) > (1 if greeting else 0)  # the call has gone on past the greeting
        async with httpx.AsyncClient(timeout=30) as http:
            agent_id = await self._agent(http, told, greeting, tools)
        ws = await websockets.connect(f"{ELEVEN_AGENTS_WS}?agent_id={agent_id}", additional_headers={"xi-api-key": self._key},
                                      max_size=None)
        start = {"type": "conversation_initiation_client_data"}
        if later:
            start["conversation_config_override"] = {"agent": {"first_message": ""}}
        await ws.send(json.dumps(start))
        async with asyncio.timeout(20):
            while (event := json.loads(await ws.recv())).get("type") != "conversation_initiation_metadata":
                if event.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong", "event_id": event["ping_event"]["event_id"]}))
        if later:
            await ws.send(json.dumps({"type": "contextual_update",
                                      "text": "The call so far, oldest first. Carry on from here:\n" + self._notes(said)}))
        state = SimpleNamespace(conversation=(event.get("conversation_initiation_metadata_event") or {}).get("conversation_id"),
                                spoken_at=None, play_end=0.0, voice_at=None, reply_at=None, t0=time.time(),
                                closing=False, other=[])
        queue = asyncio.Queue()

        async def read():  # the agent's audio and words, its tool calls, and the socket's end
            try:
                async for raw in ws:
                    e = json.loads(raw)
                    t = e.get("type")
                    if t == "ping":
                        await ws.send(json.dumps({"type": "pong", "event_id": e["ping_event"]["event_id"]}))
                    elif t == "audio":  # it comes faster than it plays: playback is tracked by its length
                        pcm, now = base64.b64decode(e["audio_event"]["audio_base_64"]), time.time()
                        state.play_end = max(state.play_end, now) + len(pcm) / (2 * ELEVEN_AGENT_RATE)
                        if _voiced(pcm):
                            await queue.put(("voice", now))
                    elif t == "agent_response":  # expressive mode marks delivery in brackets ("[happy]"): not words said
                        said = (e.get("agent_response_event") or {}).get("agent_response") or ""
                        await queue.put(("words", (time.time(), re.sub(r"\[[\w\s-]{1,30}\]\s*", "", said))))
                    elif t == "agent_response_correction":
                        await queue.put(("correction", e.get("agent_response_correction_event") or {}))
                    elif t == "client_tool_call":
                        await queue.put(("call", e["client_tool_call"]))
                    elif t not in ELEVEN_QUIET_EVENTS:
                        state.other.append(e)  # guardrail notices and the like, kept for the record
            except websockets.ConnectionClosed:
                pass
            await queue.put(("closed", (ws.close_code, ws.close_reason)))
        spoken = b"" if k is None else _to16k(await self._speech(turns[k]["content"]))

        async def speak():  # after the greeting has played: the line at speaking pace, then an open line's silence
            silence = b"\0" * ELEVEN_AGENT_FRAME

            async def send(chunk: bytes) -> None:
                await ws.send(json.dumps({"user_audio_chunk": base64.b64encode(chunk).decode()}))
                await asyncio.sleep(ELEVEN_AGENT_FRAME / (2 * ELEVEN_AGENT_RATE))
            waited = time.time()
            while greeting and not later and time.time() - waited < 15 and not (
                    state.play_end and time.time() > state.play_end + 0.7):
                await send(silence)
            for i in range(0, len(spoken), ELEVEN_AGENT_FRAME):
                await send(spoken[i:i + ELEVEN_AGENT_FRAME].ljust(ELEVEN_AGENT_FRAME, b"\0"))
            state.spoken_at = time.time()
            if spoken:
                self._lag.spoke()
            while True:
                await send(silence)
        self._open = (ws, state, turns, [asyncio.create_task(read()), asyncio.create_task(speak())], queue)

    async def _listen(self, quiet: float) -> tuple[str, list]:
        """The agent's reply once the caller's line has been spoken, until its speech has played out and the line has
        been quiet a while; or the tools it calls (several can come together); or the end of the call."""
        _, state, _, _, queue = self._open
        words, calls = [], []  # words: [when it came, text, played]
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), 0.3)
            except TimeoutError:
                now = time.time()
                if calls:  # the round's calls are all in
                    self._lag.tool_round()
                    self._waiting = ([c["tool_call_id"] for c in calls], asyncio.get_running_loop().create_future())
                    return " ".join(w[1] for w in words), [
                        (c["tool_call_id"], c["tool_name"], json.dumps(c.get("parameters") or {})) for c in calls]
                if words and state.voice_at and now > max(state.play_end, words[-1][0] + 1.0) + quiet:
                    self._lag.answered(state.reply_at)
                    return " ".join(w[1] for w in words), []
                if state.spoken_at and not words and now - max(state.spoken_at, state.play_end) > VAPI_SILENT_S:
                    return "", []
                continue
            if kind == "voice":
                if state.spoken_at:
                    self._lag.sound(item)
                    state.voice_at, state.reply_at = item, state.reply_at or item
                    for w in words:
                        w[2] = w[2] or w[0] <= item
            elif kind == "words":
                at, text = item
                if state.spoken_at and at >= state.spoken_at and text.strip():  # not the greeting or talk over the line
                    words.append([at, text.strip(), False])
            elif kind == "correction":  # cut off by the caller: what was actually said
                if words and words[-1][1] == (item.get("original_agent_response") or "").strip():
                    words[-1][1] = (item.get("corrected_agent_response") or "").strip()
            elif kind == "call":
                calls.append(item)
            elif kind == "closed":
                if not state.closing:  # the platform hung up: a guardrail
                    self.hung_up = True
                    self.hangups.append({"close": item, "events": state.other[-5:]})
                    words = [w for w in words if w[2]]
                self._lag.answered()
                return " ".join(w[1] for w in words), []

    async def _close(self) -> None:
        self._waiting = None
        if self._open:
            (ws, state, _, tasks, _), self._open = self._open, None
            state.closing = True
            for task in tasks:
                task.cancel()
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - a socket the platform already closed
                pass
            C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "elevenlabs", "model": self._model,
                                      "purpose": self._purpose, "in": 0, "out": 0,
                                      "usd": round((time.time() - state.t0) / 60 * ELEVEN_AGENT_PER_MINUTE, 6)})


class LiveText:
    """Gemini Live (native audio) with typed caller turns; its reply is the transcript of what it said.

    As in the voice bot (Pipecat's Gemini service), one session carries the whole conversation: the opening is seeded
    as client content and each caller turn arrives as realtime input (Gemini 3 runs no inference on seeded content
    alone); tool results go back as tool responses. A new session, seeded with the history, is opened only when the
    history no longer extends what the session has seen (a backtrack) or the session has dropped."""

    TURN_SECONDS = 90  # the extended-thinking model reasons in the background between spoken chunks

    def __init__(self, model: str, purpose: str, speech: Speech | None = None):
        self._lag = _Lag(purpose)
        from google import genai
        self._client = genai.Client(api_key=C.load_env(C.ENV_FILE)["GEMINI_API_KEY"])
        self._model, self._purpose, self._speech = model, purpose, speech
        self._open = None  # (context manager, session, turns it has seen, pending call ids)
        self.chat = self.completions = self

    async def _say(self, session, text: str, types) -> None:
        """A caller turn: typed, or in voice spoken as one turn the caller starts and ends (as a local turn
        detector marks it), the audio sent as fast as it is taken."""
        if not self._speech:
            await session.send_realtime_input(text=text)
            return
        spoken = await self._speech(text)
        await session.send_realtime_input(activity_start=types.ActivityStart())
        for i in range(0, len(spoken), CHUNK):
            await session.send_realtime_input(audio=types.Blob(data=spoken[i:i + CHUNK],
                                                               mime_type=f"audio/pcm;rate={PCM_RATE}"))
        await session.send_realtime_input(activity_end=types.ActivityEnd())
        self._lag.spoke()

    async def create(self, *, messages: list[dict], tools: list[dict] | None = None, **_) -> SimpleNamespace:
        from google.genai import types
        told, turns = _split(messages, opening=False)
        session = await self._resume(turns, types)
        if session is None:
            await self._close()
            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO], system_instruction=told,
                thinking_config=types.ThinkingConfig(thinking_level=BOT_TEXT["GEMINI_THINKING"]),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon"))),
                tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                    name=t["function"]["name"], description=t["function"].get("description", ""),
                    parameters_json_schema=t["function"].get("parameters")) for t in tools])] if tools else None,
                realtime_input_config=types.RealtimeInputConfig(automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True)) if self._speech else None)
            cm = self._client.aio.live.connect(model=self._model, config=config)
            session = await cm.__aenter__()
            self._open = (cm, session, turns, [])
            last = turns[-1]
            if last["role"] == "user" and last.get("content"):
                if turns[:-1]:
                    await session.send_client_content(turns=self._contents(turns[:-1], types), turn_complete=False)
                await self._say(session, last["content"], types)
            else:  # a conversation that ends on tool results: seed it all and prompt an answer
                await session.send_client_content(turns=self._contents(turns, types), turn_complete=True)
                await session.send_realtime_input(text=" ")
        text, calls, prompt_tokens, total_tokens, audio_in, interrupted = "", [], 0, 0, 0, False
        first_audio = None
        try:
            async with asyncio.timeout(self.TURN_SECONDS):
                done = False
                while not done:  # receive() may end at an in-progress turn end; the turn is over only when idle
                    async for msg in session.receive():
                        if msg.usage_metadata:
                            prompt_tokens += msg.usage_metadata.prompt_token_count or 0
                            total_tokens += msg.usage_metadata.total_token_count or 0
                            audio_in += sum(d.token_count or 0 for d in msg.usage_metadata.prompt_tokens_details or []
                                            if d.modality == types.MediaModality.AUDIO)
                        if msg.tool_call:
                            self._lag.tool_round()
                            calls = [(f.id or f"call_{uuid.uuid4().hex[:12]}", f.name, json.dumps(f.args or {}))
                                     for f in msg.tool_call.function_calls or []]
                            done = True
                            break
                        content = msg.server_content
                        if content and content.model_turn and any(p.inline_data for p in content.model_turn.parts or []):
                            first_audio = first_audio or time.time()
                        if content and content.output_transcription and content.output_transcription.text:
                            text += content.output_transcription.text
                        if content and content.interrupted:  # it began on the replayed history before the caller
                            text, interrupted = "", True     # spoke: the answer to the caller's turn comes next
                            first_audio = None
                        if (content and content.turn_complete
                                and content.interaction_status == types.InteractionStatus.IDLE):
                            if interrupted:
                                interrupted = False
                                continue
                            done = True
                            break
        except TimeoutError:
            pass
        if first_audio:
            self._lag.heard(first_audio)
        seen = turns if calls else turns + [{"role": "assistant", "content": text.strip()}]
        self._open = (self._open[0], session, seen, [c[0] for c in calls])
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=total_tokens)
        C.log_usage("gemini", self._model, self._purpose, usage)
        if audio_in:  # the caller's audio, billed above at the text rate: add the difference
            C.append_jsonl(C.LEDGER, {"t": round(time.time()), "provider": "gemini", "model": self._model,
                                      "purpose": self._purpose, "in": audio_in, "out": 0,
                                      "usd": round(audio_in * (GEMINI_AUDIO_IN - C.PRICES[self._model][0]) / 1e6, 6)})
        return _answer(text.strip(), calls, usage)

    async def _resume(self, turns: list[dict], types):
        """The open session, after sending it what is new (the tool results it asked for, or the caller's next turn),
        when the history extends what it has seen; None when a new session is needed."""
        if not self._open:
            return None
        _, session, seen, pending = self._open
        new = turns[len(seen):]
        if turns[:len(seen)] != seen or not new:
            return None
        try:
            if pending:
                if ([c["id"] for c in new[0].get("tool_calls") or []] != pending
                        or any(m["role"] != "tool" for m in new[1:])):
                    return None
                names = {c["id"]: c["function"]["name"] for c in new[0]["tool_calls"]}
                await session.send_tool_response(function_responses=[types.FunctionResponse(
                    id=m["tool_call_id"], name=names.get(m["tool_call_id"], ""), response=self._result(m["content"]))
                    for m in new[1:]])
            elif len(new) == 1 and new[0]["role"] == "user" and new[0].get("content"):
                await self._say(session, new[0]["content"], types)
            else:
                return None
        except Exception:  # noqa: BLE001 - the server dropped the session: reopen and seed the history
            return None
        return session

    async def _close(self) -> None:
        if self._open:
            cm, self._open = self._open[0], None
            try:
                await cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 - a session already dropped by the server
                pass

    @staticmethod
    def _result(content: str) -> dict:
        try:
            result = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            result = content
        return result if isinstance(result, dict) else {"result": result}

    @staticmethod
    def _contents(turns: list[dict], types) -> list:
        names = {c["id"]: c["function"]["name"] for m in turns for c in m.get("tool_calls") or []}
        out = []
        for m in turns:
            if m["role"] == "tool":
                out.append(types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                    id=m["tool_call_id"], name=names.get(m["tool_call_id"], ""),
                    response=LiveText._result(m["content"])))]))
                continue
            parts = [types.Part(text=m["content"])] if m.get("content") else []
            parts += [types.Part(function_call=types.FunctionCall(
                id=c["id"], name=c["function"]["name"], args=json.loads(c["function"]["arguments"] or "{}")))
                for c in m.get("tool_calls") or []]
            if parts:
                out.append(types.Content(role="user" if m["role"] == "user" else "model", parts=parts))
        return out


def client(spec: dict):
    purpose = f"redteam:target:{spec['model']}"
    speech = Speech() if spec.get("voice") else None
    if spec["provider"] == "gemini":
        return LiveText(spec["model"], purpose, speech)
    if spec["provider"] == "elevenlabs":
        return ElevenVoice(spec["model"], purpose, speech)
    if spec["provider"] == "vapi":
        return VapiVoice(spec["model"], purpose, speech, spec.get("config"))
    if spec["provider"] in ("retell", "retell-guard"):
        return RetellVoice(spec["model"], purpose, speech, spec["config"], spec["provider"] == "retell-guard")
    if spec["provider"] == "openai-live":
        return GptLiveVoice(spec["model"], purpose, speech)
    if spec["provider"] == "eleven-agent":
        return ElevenAgentVoice(spec["model"], purpose, speech, spec["config"])
    if spec["provider"] == "azure-realtime":
        from .azure_realtime_client import AzureRealtimeVoice
        return AzureRealtimeVoice(spec["model"], purpose, speech)
    if spec["provider"] == "deepslate":
        from .deepslate_client import DeepslateVoice
        return DeepslateVoice(spec["model"], purpose, speech)
    if spec["provider"] == "phonic":
        from .phonic_client import PhonicVoice
        return PhonicVoice(spec["model"], purpose, speech)
    if spec["provider"] == "webex":
        from .webex_client import WebexVoice
        return WebexVoice(spec["model"], purpose, speech)
    return RealtimeText(spec["model"], purpose, spec["provider"], speech)
