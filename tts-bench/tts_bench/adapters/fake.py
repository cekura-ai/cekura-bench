"""A scripted provider, so the harness can be tested without a key or a network.

It behaves like a well-mannered websocket service: a fixed round-trip, a little
leading silence, audio delivered faster than realtime in 100 ms chunks, text
accepted in frames before ``finish``, and a cancel honoured within one chunk.
The audio is a deterministic tone, so two syntheses of one text are identical
byte for byte -- which is what the determinism probe should find here.
"""

from __future__ import annotations

import asyncio
import math
from typing import ClassVar

import numpy as np

from tts_bench.common.audio import to_pcm

from tts_bench.adapters.base import TTSAdapter, TTSConfig


class FakeTTSAdapter(TTSAdapter):
    name: ClassVar[str] = "fake"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 24000)
    setup_excluded: ClassVar[str] = "nothing (scripted)"

    # Tunable per test through class attributes.
    roundtrip_ms: ClassVar[float] = 120.0
    leading_silence_ms: ClassVar[float] = 60.0
    chars_per_second: ClassVar[float] = 15.0      # speaking rate
    chunk_ms: ClassVar[float] = 100.0
    speedup: ClassVar[float] = 10.0               # delivery over realtime
    cancel_lag_ms: ClassVar[float] = 30.0

    def __init__(self, config: TTSConfig, api_key: str, log, clock) -> None:
        super().__init__(config, api_key, log, clock)
        self._text: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()

    async def _connect(self) -> None:
        await asyncio.sleep(0.01)

    async def _close(self) -> None:
        for task in self._tasks.values():
            task.cancel()

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        self._text[context_id] = self._text.get(context_id, "") + text
        if first:
            self._tasks[context_id] = asyncio.create_task(self._speak(context_id))

    async def _finish(self, context_id: str) -> None:
        self._text.setdefault(context_id, "")
        self._finished(context_id)

    def _finished(self, context_id: str) -> None:
        self.contexts[context_id].meta["finished"] = True

    async def _cancel(self, context_id: str) -> None:
        self._cancelled.add(context_id)

    def _tone(self, n_samples: int, start_sample: int) -> bytes:
        rate = self.config.sample_rate
        t = (np.arange(n_samples) + start_sample) / rate
        return to_pcm(0.3 * 32767 * np.sin(2 * math.pi * 220.0 * t))

    async def _speak(self, context_id: str) -> None:
        rate = self.config.sample_rate
        await asyncio.sleep(self.roundtrip_ms / 1000.0)
        # Leading silence, then tone for as long as the text warrants.
        lead = int(rate * self.leading_silence_ms / 1000.0)
        chunk = int(rate * self.chunk_ms / 1000.0)
        produced = 0
        pending_silence = lead
        voiced_target = 0
        while True:
            if context_id in self._cancelled:
                await asyncio.sleep(self.cancel_lag_ms / 1000.0)
                self._on_done(context_id, cancelled=True)
                self._on_cancel_ack(context_id)
                return
            text = self._text.get(context_id, "")
            voiced_target = int(rate * len(text) / self.chars_per_second)
            finished = self.contexts[context_id].meta.get("finished", False)
            if produced >= lead + voiced_target:
                if finished:
                    self._on_done(context_id)
                    return
                await asyncio.sleep(0.01)      # more text may still come
                continue
            n = min(chunk, lead + voiced_target - produced)
            if pending_silence > 0:
                s = min(n, pending_silence)
                pcm = b"\x00\x00" * s + self._tone(n - s, 0)
                pending_silence -= s
            else:
                pcm = self._tone(n, produced - lead)
            self._on_audio(context_id, pcm)
            produced += n
            await asyncio.sleep(n / rate / self.speedup)
