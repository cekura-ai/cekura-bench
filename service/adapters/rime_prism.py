"""Rime Prism over ``/v1/realtime``.

Prism speaks the event names and message shapes of OpenAI's Realtime **beta**
interface, so this adapter inherits the OpenAI event handling (which already
keeps the beta aliases ``response.audio.*`` and ``response.text.done``) and
overrides only what differs from the GA protocol:

* **Session.** The session document is flat and short: instructions, voice,
  tools and an optional ``turn_detection`` holding only ``eagerness`` and
  ``prism_end_of_turn_timeout_ms``. Prism refuses the whole update over any GA
  field (``type``, ``audio``, ``output_modalities``, transcription, VAD
  thresholds), so none is sent. The caller is transcribed without being asked.
* **Audio.** 16 kHz PCM16 in, 24 kHz PCM16 out.
* **Turns.** Prism has one built-in detector and no commit: the turn boundary is
  always the provider's, so the manual arm and every VAD parameter are declared
  exclusions. ``eagerness`` and the end-of-turn timeout are fixed at the first
  update; both stay unsent unless set, so a session at the provider's defaults
  shows them absent from what was sent.
* **Playback reports.** Prism decides what an interrupted reply leaves in the
  history (and whether the next reply acknowledges the interruption) from
  ``conversation.item.truncate``, the client's account of how much of the reply
  was heard. The reference client sends it at the barge-in cut, measured on the
  playout model, and sends ``prism.playback.drained`` when a reply plays out
  uncut, so the provider's idle and turn decisions follow the same player the
  measurements assume rather than its own padded estimate.
* **Replies the client asks for.** ``response.create`` must name its cause. A
  tool continuation is sent once the parent reply's ``response.done`` and the
  acknowledgement of every one of its results have both arrived; earlier, Prism
  refuses it. The text arm is a typed turn, accepted only while Prism is idle.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import websockets

from service import events as ev
from service.adapters.base import AdapterError, SessionConfig
from service.adapters.openai_realtime import CONNECT_TIMEOUT_S, OpenAIRealtimeAdapter

URL = "wss://prism.api.rime.ai/v1/realtime"
# Points the adapter at another Prism endpoint (Rime's beta host, for instance).
# Read from the environment only; the record names the one used.
URL_ENV = "RIME_PRISM_URL"
MODEL = "prism"  # the endpoint serves one model and takes no model parameter
EAGERNESS = ("low", "medium", "high", "auto")
END_OF_TURN_TIMEOUT_MS = (1000, 10000)
# A typed turn is refused unless Prism is idle, and Prism is idle only once the
# previous reply has played out.
IDLE_TIMEOUT_S = 30.0


class RimePrismAdapter(OpenAIRealtimeAdapter):
    name = "rime-prism"
    input_rate = 16000   # the service does not resample; 24 kHz input is heard slow
    output_rate = 24000
    supports_manual_commit = False   # no input_audio_buffer.commit
    supports_text_modality = True    # typed turns: text in, audio and text out
    emits_vad_events = True

    url = URL

    def __init__(
        self,
        *,
        model: str = MODEL,
        eagerness: str | None = None,
        end_of_turn_timeout_ms: int | None = None,
        **kwargs: Any,
    ) -> None:
        if eagerness is not None and eagerness not in EAGERNESS:
            raise AdapterError(f"rime-prism eagerness must be one of {', '.join(EAGERNESS)}")
        low, high = END_OF_TURN_TIMEOUT_MS
        if end_of_turn_timeout_ms is not None and not low <= end_of_turn_timeout_ms <= high:
            raise AdapterError(f"rime-prism end_of_turn_timeout_ms must be {low} to {high}")
        super().__init__(model=model, **kwargs)
        self.url = os.environ.get(URL_ENV) or URL
        self.eagerness = eagerness
        self.end_of_turn_timeout_ms = end_of_turn_timeout_ms
        self._idle = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        # (item_id, index of its first chunk on agent_timeline), in arrival order.
        self._audio_items: list[tuple[str, int]] = []
        self._latest_response: str | None = None
        self._calls: dict[str, set[str]] = {}     # response id -> its tool call ids
        self._acked: set[str] = set()             # call ids whose result Prism stored
        self._done: set[str] = set()              # response ids that have ended
        self._continued: set[str] = set()
        self._cuts_at_response_start = 0
        self._typed_turns = 0

    @classmethod
    def unsupported_reason(cls, config: SessionConfig) -> str | None:
        reason = super().unsupported_reason(config)
        if reason:
            return reason
        detection = config.turn_detection
        if detection.mode != "server_vad":
            return f"rime-prism has one built-in turn detector; {detection.mode} is not selectable"
        # The default 500 ms cell runs at Prism's own timing (disclosed in the registry);
        # any other VAD setting would be a label Prism never applied.
        if (
            detection.threshold is not None
            or detection.prefix_padding_ms is not None
            or detection.silence_duration_ms not in (None, 500)
        ):
            return "rime-prism takes no VAD parameters; its turn timing is eagerness and an end-of-turn timeout"
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        try:
            self._ws = await websockets.connect(
                self.url,  # Prism reads no query parameters
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                open_timeout=CONNECT_TIMEOUT_S,
                max_size=None,
            )
        except Exception as exc:  # noqa: BLE001 -- the reason is what we want to publish
            raise AdapterError(f"{self.name} connect failed: {exc}") from exc

        # session.created always comes first, before Prism reads anything.
        first = json.loads(await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT_S))
        self.log.raw(first)
        if first.get("type") != "session.created":
            raise AdapterError(f"{self.name} refused the session: {first}")
        self.session_id = first["session"].get("id")
        self.log.emit(ev.SESSION_OPEN, session_id=self.session_id, model=self.model)

        self._receiver = asyncio.create_task(self._receive_loop(), name=f"{self.name}-recv")
        self.session_sent = self._session_payload()
        await self._send_json({"type": "session.update", "event_id": "setup", "session": self.session_sent})
        try:
            await asyncio.wait_for(self._configured.wait(), CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise AdapterError(f"{self.name} never acknowledged session.update") from exc
        if self.config.first_message:
            await self._send_json(
                {
                    "type": "conversation.item.create",
                    "event_id": "first_message",
                    "item": {"type": "message", "role": "assistant", "text": self.config.first_message},
                }
            )

    def _session_payload(self) -> dict[str, Any]:
        config = self.config
        session: dict[str, Any] = {"instructions": config.instructions}
        if config.voice:
            session["voice"] = config.voice
        if config.tools:
            session["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in config.tools
            ]
        turn: dict[str, Any] = {}
        if self.eagerness is not None:
            turn["eagerness"] = self.eagerness
        if self.end_of_turn_timeout_ms is not None:
            turn["prism_end_of_turn_timeout_ms"] = self.end_of_turn_timeout_ms
        if turn:
            session["turn_detection"] = turn
        return session

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await super().close()

    def state(self) -> dict[str, Any]:
        state = super().state()
        # None is the provider's default, left unsent.
        state["endpoint"] = self.url
        state["turn_timing"] = {"eagerness": self.eagerness, "end_of_turn_timeout_ms": self.end_of_turn_timeout_ms}
        return state

    # ── sending ──────────────────────────────────────────────────────────────

    async def _send_text(self, text: str) -> None:
        try:
            await asyncio.wait_for(self._idle.wait(), IDLE_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise AdapterError(f"{self.name} never became idle for a typed turn") from exc
        self._idle.clear()
        self._typed_turns += 1
        await self._send_json(
            {
                "type": "response.create",
                "event_id": f"typed_{self._typed_turns}",
                "response": {"prism_input_text": text, "metadata": {"prism_cause": "user_text"}},
            }
        )

    async def _send_tool_result(self, call_id: str, output: Any) -> None:
        # The result only; the continuation follows from _maybe_continue once
        # Prism has acknowledged it and the parent reply has ended.
        await self._send_json(
            {
                "type": "conversation.item.create",
                "event_id": f"result_{call_id}",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output if isinstance(output, str) else json.dumps(output),
                },
            }
        )

    def _spawn_send(self, payload: dict[str, Any], delay_s: float = 0.0, *, unless_cut: bool = False) -> None:
        """Send from the receive path without blocking it.

        ``unless_cut`` drops the send if playback was cut while it waited: a
        reply the caller talked over did not play out.
        """
        cuts = len(self.agent_timeline.cuts)

        async def send() -> None:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if unless_cut and len(self.agent_timeline.cuts) != cuts:
                return
            try:
                await self._send_json(payload)
            except Exception as exc:  # noqa: BLE001 -- recorded, never raised into the receive loop
                self.log.emit(ev.SESSION_ERROR, error=f"{payload['type']} not sent: {exc!r}")

        task = asyncio.get_running_loop().create_task(send(), name=f"{self.name}-{payload['type']}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ── receiving ────────────────────────────────────────────────────────────

    def _handle(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type", "")
        if kind == "response.audio.delta":
            item = payload.get("item_id") or ""
            if not self._audio_items or self._audio_items[-1][0] != item:
                self._audio_items.append((item, len(self.agent_timeline.chunks)))
        elif kind == "response.text.done" and not payload.get("text"):
            # A reply that only called tools and said nothing: no words to record.
            self.log.raw(payload)
            return
        elif kind == "prism.typed_input.ready":
            self._idle.set()
        elif kind == "response.created":
            self._idle.clear()
            self._latest_response = (payload.get("response") or {}).get("id")
            self._cuts_at_response_start = len(self.agent_timeline.cuts)
        elif kind == "response.function_call_arguments.done":
            self._calls.setdefault(payload.get("response_id", ""), set()).add(payload.get("call_id", ""))
        elif kind == "conversation.item.created":
            item = payload.get("item") or {}
            if item.get("type") == "function_call_output":
                self._acked.add(item.get("call_id", ""))
                self._maybe_continue()
        elif kind == "input_audio_buffer.speech_started":
            self._idle.clear()

        super()._handle(payload)

        if kind == "response.done":
            response = payload.get("response") or {}
            if response.get("id"):
                self._done.add(response["id"])
            self._maybe_continue()
            token = (response.get("metadata") or {}).get("prism_drain_token")
            uncut = len(self.agent_timeline.cuts) == self._cuts_at_response_start
            if token and response.get("status") == "completed" and uncut:
                self._report_drained(token)

    def _on_caller_speech_started(self, **data: Any) -> None:
        cuts = len(self.agent_timeline.cuts)
        super()._on_caller_speech_started(**data)
        if len(self.agent_timeline.cuts) == cuts:
            return  # nothing was still playing, so nothing was cut short
        heard = self._heard_at_cut(self.agent_timeline.cuts[-1])
        if heard is None:
            return
        item_id, audio_end_ms = heard
        self._spawn_send(
            {
                "type": "conversation.item.truncate",
                "event_id": f"truncate_{item_id}",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": audio_end_ms,
            }
        )

    def _heard_at_cut(self, at: float) -> tuple[str, int] | None:
        """The message that was playing at the cut, and how much of it was heard.

        Measured on the playout model after the cut, from the start of that
        message's audio: the sum of its chunks' played spans. A message queued
        behind it was never heard and gets no report of its own.
        """
        timeline = self.agent_timeline
        bounds = [first for _item, first in self._audio_items[1:]] + [len(timeline.chunks)]
        for (item_id, first), stop in zip(reversed(self._audio_items), reversed(bounds)):
            spans = [timeline.playout_span(i) for i in range(first, stop)]
            if any(start < at and end >= at for start, end in spans):
                heard_s = sum(end - start for start, end in spans)
                return item_id, int(round(heard_s * 1000.0))
        return None

    def _report_drained(self, token: str) -> None:
        """Tell Prism the reply played out, at the instant the modelled player finishes it."""
        end = self.agent_timeline.playout_end()
        delay = 0.0 if end is None else max(0.0, end - self.log.clock.now())
        self._spawn_send({"type": "prism.playback.drained", "drain_token": token}, delay, unless_cut=True)

    def _maybe_continue(self) -> None:
        """Ask for the reply to a tool round once Prism will accept it.

        Prism refuses a continuation that arrives before the parent reply's
        ``response.done`` or before every result's acknowledgement, and keeps
        nothing it refused. A round superseded by a newer reply is not continued.
        """
        parent = self._latest_response
        if parent is None or parent in self._continued or parent not in self._done:
            return
        calls = self._calls.get(parent)
        if not calls or not calls <= self._acked:
            return
        self._continued.add(parent)
        self._spawn_send(
            {
                "type": "response.create",
                "event_id": f"continue_{parent}",
                "response": {
                    "metadata": {"prism_cause": "tool_continuation", "prism_parent_response_id": parent}
                },
            }
        )
