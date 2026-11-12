"""HeyGen Voice over HTTP: one request per utterance, raw PCM streamed back.

``POST /v3/models/audio/tts/stream`` with ``output_format`` set to ``pcm`` at
the configured rate answers with a chunked ``audio/pcm`` body (without
``output_format`` it is server-sent events carrying 44.1 kHz WAV parts, which
is not what a voice agent would play). The whole text goes in one request, so
streamed input, cancel and continuation are declared exclusions, as for
OpenAI. The response headers name the rate and channel count; a body at any
other rate than the one asked for is recorded as an error rather than scored
at the wrong speed.

Before t0, each cell's session reads the voice (``GET /v3/models/audio/voices/
{voice_id}``). That warms the connection the speech request reuses, and it
turns a refused key or a voice that is missing or not ``ACTIVE`` into a void
instead of a failure of the service.

Behaviour measured live on 2026-11-05 that a reader of the numbers needs:

* ``language`` is always sent (``en`` unless option ``language`` says
  otherwise). Left out, HeyGen guesses it from the text, and a sentence that is
  mostly digits or symbols can be refused as undetectable.
* The same text, voice and model give byte-identical audio, on a new
  connection and minutes apart, while each request still takes a full
  synthesis to deliver: deterministic output, not a cache (there is no cache
  switch on this endpoint; ``force_regenerate`` is refused).
* The body is cut wherever TCP cuts it, so a chunk can end inside a sample.
  The odd byte is held until the next chunk completes it, so the arrival
  timeline counts whole samples. A body that ends on a half sample is an
  error.
* If synthesis fails after audio starts, HeyGen closes the connection without
  an error event. An unterminated chunked body raises here and is the cell's
  error; a cleanly terminated short body is only caught by the round-trip
  transcript.
* Each speech endpoint allows 30 requests a minute per workspace member. Before
  t0, a cell waits until two more requests (what ``repeat`` sends) fit under
  that limit in the last minute, so a full run is paced instead of refused.
  Option ``requests_per_minute`` sets the limit (``0`` turns pacing off).

Options: ``language`` (ISO 639-1, default ``en``), ``expressiveness_boost``
(``0.0``-``1.0``; instant voices only; HeyGen's API default ``1.0`` when
unset), ``requests_per_minute`` (default ``30``).

Reference: https://developers.heygen.com/docs/voices/heygen-voice-speech
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, ClassVar
from urllib.parse import quote

import aiohttp

from tts_bench.adapters.base import AdapterError
from tts_bench.adapters.openai_tts import OpenAITTSAdapter
from tts_bench.common.audio import SAMPLE_WIDTH

REQUESTS_PER_MINUTE = 30        # HeyGen's documented per-member limit on each speech endpoint
REQUESTS_PER_CELL = 2           # the most any full-suite probe sends on one connection (repeat)
WINDOW_S = 61.0                 # a minute, plus a second for the request to reach HeyGen's clock

# Speech requests this process has sent, oldest first, on the monotonic clock.
# Module-level because every cell builds a new adapter and the limit is the account's.
_SENT: deque[float] = deque()


def wait_for_room(sent: deque[float], needed: int, per_minute: int, now: float, window_s: float = WINDOW_S) -> float:
    """Seconds until ``needed`` more requests fit under ``per_minute`` in the last ``window_s``; 0 if they fit now.

    Drops the timestamps that have left the window, so ``sent`` stays small.
    """
    while sent and sent[0] <= now - window_s:
        sent.popleft()
    over = len(sent) + min(needed, per_minute) - per_minute
    if per_minute <= 0 or over <= 0:
        return 0.0
    return sent[over - 1] + window_s - now


class HeyGenTTSAdapter(OpenAITTSAdapter):
    name: ClassVar[str] = "heygen-speech"
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100, 48000)
    native_mulaw_8k: ClassVar[bool] = True       # output_format encoding ulaw at 8000 Hz
    setup_excluded: ClassVar[str] = ("TCP, TLS (session pre-warmed with one request that checks the voice is ACTIVE); "
                                     "any wait to stay under the account's request limit")
    base = "https://api.heygen.com"
    speech_path = "/v3/models/audio/tts/stream"

    def _prewarm_url(self) -> str:
        return f"{self.base}/v3/models/audio/voices/{quote(self.config.voice, safe='')}"

    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key}

    def _request_args(self, text: str) -> dict[str, Any]:
        """The body of one synthesis request, logged without the text."""
        payload: dict[str, Any] = {
            "model": self.config.model, "voice_id": self.config.voice, "text": text,
            "language": self.config.language or self.config.option("language", "en"),
            "output_format": {"encoding": "pcm", "sample_rate_hertz": self.config.sample_rate},
        }
        if (boost := self.config.option("expressiveness_boost")) is not None:
            payload["expressiveness_boost"] = float(boost)
        self.log.raw({**payload, "text": f"<{len(text)} chars>"}, direction="out")
        return {"json": payload}

    async def _pace(self) -> None:
        per_minute = int(self.config.option("requests_per_minute", str(REQUESTS_PER_MINUTE)))
        waited = 0.0
        while (wait := wait_for_room(_SENT, REQUESTS_PER_CELL, per_minute, time.monotonic(), WINDOW_S)) > 0:
            await asyncio.sleep(wait)
            waited += wait
        if waited:
            self.log.raw({"paced_s": round(waited, 3), "requests_per_minute": per_minute}, direction="out")

    async def _connect(self) -> None:
        await self._pace()
        self._session = aiohttp.ClientSession(headers=self._headers())
        try:
            async with self._session.get(self._prewarm_url()) as response:
                body = await response.json(content_type=None) if response.status in (200, 404) else None
                self.log.raw({"prewarm_status": response.status}, direction="in")
                if response.status in (401, 402, 403):
                    raise AdapterError(f"{self.name} rejected the key: HTTP {response.status}")
                if response.status == 404:
                    raise AdapterError(f"{self.name}: voice {self.config.voice} is not available to this key")
                status = ((body or {}).get("data") or {}).get("status") if response.status == 200 else None
                if response.status == 200 and status != "ACTIVE":
                    raise AdapterError(f"{self.name}: voice {self.config.voice} is {status}, not ACTIVE")
                # Any other answer (a busy voices endpoint) still warmed the connection; the speech request decides.
        except (aiohttp.ClientError, ValueError) as exc:      # ValueError: a voice record that is not JSON
            await self._session.close()
            raise AdapterError(f"{self.name} prewarm failed: {exc}") from exc
        except AdapterError:
            await self._session.close()
            raise

    def _format_problem(self, headers: Any) -> str | None:
        """Why a 200 response is not the audio that was asked for, if it is not."""
        content_type = headers.get("Content-Type", "")
        if not content_type.startswith("audio/pcm"):
            return f"expected audio/pcm, got {content_type or 'no content type'}"
        rate, channels = headers.get("X-Audio-Sample-Rate"), headers.get("X-Audio-Channels")
        if rate is not None and int(rate) != self.config.sample_rate:
            return f"asked for {self.config.sample_rate} Hz, got {rate} Hz"
        if channels is not None and int(channels) != 1:
            return f"expected mono, got {channels} channels"
        return None

    async def _request(self, context_id: str, text: str) -> None:
        assert self._session is not None
        args = self._request_args(text)
        _SENT.append(time.monotonic())
        try:
            async with self._session.post(f"{self.base}{self.speech_path}", **args) as response:
                self._on_response(context_id, response.status, response.headers)
                if response.status != 200:
                    body = await response.text()
                    self._on_error(context_id, f"HTTP {response.status}: {body[:300]}")
                    return
                if problem := self._format_problem(response.headers):
                    self._on_error(context_id, problem)
                    return
                carry = b""
                async for chunk in response.content.iter_any():
                    chunk = carry + chunk
                    whole = len(chunk) - len(chunk) % SAMPLE_WIDTH
                    carry = chunk[whole:]
                    self._on_audio(context_id, chunk[:whole])
            if carry:
                self._on_error(context_id, f"stream ended inside a sample ({len(carry)} trailing byte)")
                return
            self._on_done(context_id)
        except Exception as exc:  # noqa: BLE001 -- anything that stops the stream is the cell's error, never a silent timeout
            self._on_error(context_id, f"http: {exc!r}")
