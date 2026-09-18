"""A scripted agent with controlled reply timing, for testing the harness itself.

The caller's state machine and every metric are exercised against this before a
provider is ever involved. That ordering matters: if the first time a barge-in
anchor runs is against a live model, a harness bug and a model behaviour look
identical, and a harness that has never been checked against known timing has to
be taken on trust.

It answers on a timer rather than by understanding anything -- the point is that
reply onset, duration and endpointing delay are *known*, so a measurement can be
checked against ground truth instead of against a plausible-looking number.
"""

from __future__ import annotations

import asyncio
from typing import Any

from lane_a import events as ev
from lane_a.adapters.base import RealtimeAdapter
from lane_a.audio import SAMPLE_WIDTH, to_pcm

import numpy as np


class FakeAdapter(RealtimeAdapter):
    """Replies ``reply_ms`` of tone, ``endpoint_ms`` after the caller goes quiet."""

    name = "fake"
    input_rate = 24000
    output_rate = 24000

    def __init__(
        self,
        *,
        model: str = "scripted",
        api_key: str = "",
        reply_ms: float = 1200.0,
        endpoint_ms: float = 500.0,
        chunk_ms: float = 40.0,
        yields_to_barge_in: bool = True,
        silence_threshold: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.reply_ms = reply_ms
        self.endpoint_ms = endpoint_ms
        self.chunk_ms = chunk_ms
        self.yields_to_barge_in = yields_to_barge_in
        self.silence_threshold = silence_threshold
        self._last_caller_speech: float | None = None
        self._speaker: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self._responses = 0

    async def connect(self) -> None:
        self.log.emit(ev.SESSION_OPEN, session_id="fake", model=self.model)
        self.log.emit(ev.SESSION_CONFIGURED, turn_detection=self.config.turn_detection.label)
        self._watchdog = asyncio.create_task(self._endpointer(), name="fake-endpointer")

    async def close(self) -> None:
        for task in (self._speaker, self._watchdog):
            if task:
                task.cancel()
        self.closed.set()
        self.log.emit(ev.SESSION_CLOSED)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        self.log.raw(payload, direction="out")

    async def _send_audio_chunk(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype="<i2")
        if samples.size and np.abs(samples).max() > self.silence_threshold:
            self._last_caller_speech = self.log.clock.now()
            if self._speaker and not self._speaker.done() and self.yields_to_barge_in:
                self._speaker.cancel()
                self._speaker = None
                self._on_agent_audio_done(reason="cancelled")
                self.log.emit(ev.AGENT_INTERRUPTED)

    async def _commit(self) -> None:
        self._start_reply()

    async def send_tool_result(self, call_id: str, output: Any) -> None:
        self.log.emit(ev.TOOL_RESULT, call_id=call_id)
        self._start_reply()

    def _start_reply(self) -> None:
        if self._speaker and not self._speaker.done():
            return
        self._responses += 1
        self._speaker = asyncio.create_task(self._speak(), name="fake-speak")

    async def _endpointer(self) -> None:
        """Server-VAD stand-in: reply once the caller has been quiet long enough."""
        if self.config.turn_detection.is_manual:
            return
        silence = self.config.turn_detection.silence_duration_ms or self.endpoint_ms
        while True:
            await asyncio.sleep(0.01)
            quiet_since = self._last_caller_speech
            if quiet_since is None or (self._speaker and not self._speaker.done()):
                continue
            if (self.log.clock.now() - quiet_since) * 1000.0 >= silence:
                self._last_caller_speech = None
                self._start_reply()

    async def _speak(self) -> None:
        total = int(self.output_rate * self.reply_ms / 1000.0)
        per_chunk = int(self.output_rate * self.chunk_ms / 1000.0)
        phase = np.arange(total) / self.output_rate
        tone = to_pcm(8000 * np.sin(2 * np.pi * 220.0 * phase))
        try:
            for start in range(0, total, per_chunk):
                self._on_agent_audio(tone[start * SAMPLE_WIDTH : (start + per_chunk) * SAMPLE_WIDTH])
                await asyncio.sleep(self.chunk_ms / 1000.0)
            self._on_agent_audio_done()
            self.log.emit(ev.RESPONSE_DONE, status="completed", usage={})
        except asyncio.CancelledError:
            raise
