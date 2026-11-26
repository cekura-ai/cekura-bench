"""HeyGen Voice over the multi-context input-streaming websocket.

``wss://api.heygen.com/v3/models/audio/tts/multi-stream-input`` speaks the same
frame shapes as ElevenLabs' multi-context socket, so this adapter is that one
with HeyGen's URL, credential, error envelope, completion rule and request
limit:

* One socket, up to five contexts, each named by ``context_id``. ``{"text": " "}``
  opens a context, text frames append to it, ``flush`` generates what is
  buffered, and audio frames carry base64 PCM and a ``contextId``.
* ``finish`` sends only the flush. HeyGen sends ``isFinal`` only when a context
  is closed, but every audio frame carries ``alignment``, the characters of the
  text as sent that the frame speaks (``with_timestamps=true`` makes the
  timings required). A context is done when its frames have spoken every
  character it was sent: on 2,495 of the 2,550 syntheses of the 2026-11-18
  campaign (both models, stop probe aside) the aligned characters equal the
  text sent, character for character. This replaces ElevenLabs' end-on-quiet,
  which would cut a recording at any pause over 1.5 s and record the missing
  words as misreadings instead of the pause.
* 54 others (on five sentences) were spoken in full, but HeyGen never aligned
  the closing full stop or question mark, so they waited out the timeout. A final
  punctuation mark normally comes in its own last frame, at most 0.46 s after
  the frame that speaks the last letter or digit (campaign, both models). So
  once every letter and digit sent has been spoken, a context may also end on
  quiet (1.5 s with no audio): every other context still ends on the frame it
  did, and a pause before the last word is still a stall, not the end.
* ``auto_mode`` (option, default ``true`` here; HeyGen's default is ``false``)
  is in the URL because it changes the number. Off, HeyGen generates once 120
  characters are buffered or on the flush, but text that arrives after a
  generation has started waits: a flush does not start it, and it is spoken
  some 7-9 s later (measured 2026-11-18 on four tries, including a flush sent
  2 s after the last frame). That is a silence mid-answer for any answer over
  120 characters sent in pieces. On, HeyGen generates at sentence ends, and
  the same answers had no gap over 0.6 s. As for ElevenLabs, the mode that was
  measured stable is the one run.
* ``close_context`` is the cancel, the only per-context stop the socket
  documents. On a flushed context HeyGen lets the audio it is generating finish
  (documented, and measured live on 2026-11-18: 15.5 s of a 17.7 s answer still
  arrived after a stop sent 300 ms in), so the cancel probe measures how much
  audio still arrives rather than assuming it stops, as for ElevenLabs'
  ``close_context``. Its ``isFinal`` is the acknowledgement.
* HeyGen closes the socket after ``close_socket``, or after an error frame. A
  close while a reply is still being spoken cuts it short (one reply in the
  campaign: close 1012, "Service restarting; reconnect", 7.6 s into a 21 s
  answer), so it is that context's error, not its end.
* The voice, model, rate and language are fixed per socket in the URL. Unknown
  query parameters and unknown frame fields are refused (HTTP 400 at the
  handshake; close 1008 with an error frame), so nothing outside the documented
  set is sent: per-context voice settings, which ElevenLabs takes, are not.

Request limit: each handshake and each new context counts toward HeyGen's 30
speech requests a minute per workspace member (measured: the 28th bare
handshake inside a minute, after three others, was refused with 429 and
``Retry-After: 60``). Before t0 a cell waits until three more requests fit, the
most a full-suite probe sends on one socket (a handshake and two contexts).
The window is the HTTP adapter's, so the two endpoints are paced together.

Options: ``language`` (ISO 639-1, default ``en``), ``expressiveness_boost``
(``0.0``-``1.0``, HeyGen's default ``1.0`` when unset), ``auto_mode``
(default ``true``), ``requests_per_minute`` (default ``30``, ``0`` turns pacing
off).

Reference: https://developers.heygen.com/docs/voices/heygen-voice-speech
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, ClassVar
from urllib.parse import urlencode

from tts_bench.adapters import heygen
from tts_bench.adapters.base import AdapterError, Synthesis
from tts_bench.adapters.elevenlabs import ElevenLabsAdapter
from tts_bench.common.audio import SAMPLE_WIDTH

REQUESTS_PER_CELL = 3       # the most a full-suite probe sends on one socket: the handshake and two contexts (repeat, continuation)


def _letters(text: str) -> int:
    """Letters and digits: what has to be spoken, whatever HeyGen does with punctuation."""
    return sum(ch.isalnum() for ch in text)


class HeyGenStreamingAdapter(ElevenLabsAdapter):
    name: ClassVar[str] = "heygen-streaming"
    ends_on_quiet: ClassVar[bool] = False          # done when every character sent has been spoken (alignment)
    native_rates: ClassVar[tuple[int, ...]] = (8000, 16000, 22050, 24000, 44100)
    native_mulaw_8k: ClassVar[bool] = False
    setup_excluded: ClassVar[str] = ("TCP, TLS, websocket upgrade, per-context open frame; "
                                     "any wait to stay under the account's request limit")
    base = "wss://api.heygen.com"
    path = "/v3/models/audio/tts/multi-stream-input"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._carry: dict[str | None, bytes] = {}
        self._aligned: dict[str | None, int] = {}
        self._letters_sent: dict[str | None, int] = {}
        self._letters_spoken: dict[str | None, int] = {}

    @property
    def auto_mode(self) -> bool:
        return (self.config.option("auto_mode", "true") or "true").lower() == "true"

    @property
    def url(self) -> str:  # type: ignore[override]
        c = self.config
        query: dict[str, str] = {
            "voice_id": c.voice, "model": c.model, "output_format": f"pcm_{c.sample_rate}",
            "language": c.language or c.option("language", "en") or "en",
            "auto_mode": str(self.auto_mode).lower(), "with_timestamps": "true", "inactivity_timeout": "60",
        }
        if (boost := c.option("expressiveness_boost")) is not None:
            query["expressiveness_boost"] = str(float(boost))
        return f"{self.base}{self.path}?{urlencode(query)}"

    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key}

    def _record_request(self) -> None:
        heygen._SENT.append(time.monotonic())

    async def _pace(self) -> None:
        per_minute = int(self.config.option("requests_per_minute", str(heygen.REQUESTS_PER_MINUTE)) or 0)
        waited = 0.0
        while (wait := heygen.wait_for_room(heygen._SENT, REQUESTS_PER_CELL, per_minute, time.monotonic(),
                                            heygen.WINDOW_S)) > 0:
            await asyncio.sleep(wait)
            waited += wait
        if waited:
            self.log.raw({"paced_s": round(waited, 3), "requests_per_minute": per_minute}, direction="out")

    async def _connect(self) -> None:
        await self._pace()
        self._record_request()
        await super()._connect()

    async def _open_context(self, context_id: str) -> None:
        if self.config.speed is not None:
            raise AdapterError(f"{self.name}: speed applies only to professional clones, not to {self.config.voice}")
        self._record_request()
        self._aligned[context_id] = self._letters_sent[context_id] = self._letters_spoken[context_id] = 0
        await self._send_json({"text": " ", "context_id": context_id})

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        self._letters_sent[context_id] = self._letters_sent.get(context_id, 0) + _letters(text)
        await super()._send_text(context_id, text, first)

    async def _finish(self, context_id: str) -> None:
        await super()._finish(context_id)
        self._done_if_spoken(context_id)          # with auto_mode the last sentence can be spoken before the flush

    def _done_if_spoken(self, context_id: str | None) -> None:
        """Done once the input is complete and the frames have spoken every character of it."""
        synthesis = self.contexts.get(context_id)
        if (synthesis is None or synthesis.t_input_done is None or synthesis.t_cancel is not None
                or self._carry.get(context_id)):
            return
        if self._aligned.get(context_id, 0) >= synthesis.chars_sent > 0:
            self._on_done(context_id, aligned_chars=self._aligned[context_id])

    def _ends_on_quiet(self, synthesis: Synthesis) -> bool:
        """Every letter and digit sent has been spoken: an unaligned closing punctuation mark need not be waited for."""
        sent = self._letters_sent.get(synthesis.context_id, 0)
        return synthesis.t_input_done is not None and sent > 0 and self._letters_spoken.get(synthesis.context_id, 0) >= sent

    def _on_closed(self, reason: str) -> None:
        for context_id, synthesis in self.contexts.items():
            if not synthesis.done.is_set() and synthesis.t_cancel is None:
                self._on_error(context_id, f"closed while speaking: {reason}")
        super()._on_closed(reason)

    def _on_message(self, message: dict[str, Any]) -> None:
        context_id = message.get("contextId")
        error = message.get("error")
        if error:
            detail = error if isinstance(error, dict) else {"message": str(error)}
            self._on_error(context_id, f"{detail.get('code', 'error')}: {detail.get('message') or message.get('message')}")
            return
        audio = message.get("audio")
        if audio:
            # A frame that ends inside a sample keeps its odd byte for the next one, so the timeline counts whole samples.
            pcm = self._carry.pop(context_id, b"") + base64.b64decode(audio)
            whole = len(pcm) - len(pcm) % SAMPLE_WIDTH
            if whole < len(pcm):
                self._carry[context_id] = pcm[whole:]
            self._on_audio(context_id, pcm[:whole])
            chars = (message.get("alignment") or {}).get("chars") or ()
            self._aligned[context_id] = self._aligned.get(context_id, 0) + len(chars)
            self._letters_spoken[context_id] = self._letters_spoken.get(context_id, 0) + _letters("".join(chars))
        if message.get("isFinal"):
            if self._carry.pop(context_id, b""):
                self._on_error(context_id, "stream ended inside a sample (1 trailing byte)")
                return
            synthesis = self.contexts.get(context_id)
            if synthesis is not None and synthesis.t_cancel is not None:
                self._on_cancel_ack(context_id)
            self._on_done(context_id, synthesized_chars=message.get("synthesizedChars"))
            return
        self._done_if_spoken(context_id)
