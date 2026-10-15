"""Deepgram Flux TTS over the ``/v2/speak`` websocket.

The same shape as Aura's ``/v1/speak`` (``Speak`` frames, ``Flush``, binary
audio, no context ids, so contexts on one socket run one after another), with
three differences that change the record:

* ``Flushed`` arrives *before* the audio of the turn, as an acknowledgement.
  The end of the turn is ``SpeechMetadata``, and it is the completion signal.
* The cancel is ``Interrupt``, answered by ``SpeechInterrupted``. ``Clear`` is
  rejected on this endpoint.
* The server opens with ``Connected`` and marks each turn with
  ``SpeechStarted``; both are logged and change no timing.
"""

from __future__ import annotations

from typing import Any, ClassVar

from tts_bench.adapters.deepgram import DeepgramSpeakAdapter


class DeepgramFluxAdapter(DeepgramSpeakAdapter):
    name: ClassVar[str] = "deepgram-flux"
    native_mulaw_8k: ClassVar[bool] = True     # encoding=mulaw
    base = "wss://api.deepgram.com/v2/speak"

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"type": "Interrupt"})

    def _on_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "SpeechMetadata":
            self._on_done(self._active, audio_duration_ms=message.get("audio_duration_ms"),
                          billable_character_count=message.get("billable_character_count"))
        elif kind == "SpeechInterrupted":
            self._on_cancel_ack(self._active, audio_played_ms=message.get("audio_played_ms"))
            self._on_done(self._active, interrupted=True)
        elif kind == "Error" or "err_code" in message or kind == "Warning":
            self._on_error(self._active, str(message))
        # Connected, SpeechStarted and the early Flushed are acknowledgements, not ends.
