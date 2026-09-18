"""xAI Grok Voice Agent over ``/v1/realtime``.

The wire protocol is OpenAI-shaped, so this adapter inherits the event handling
and overrides only what differs: the session document is flat rather than
nested, the endpoint and default model are xAI's, and the input transcript is
re-sent cumulatively while the caller is still talking. Those cumulative frames
are the trap: appended naively they would publish the caller's one sentence as
five overlapping transcripts and score the provider's ASR on our bookkeeping.

The concrete model id is pinned rather than the vendor's ``latest`` alias, since
a published row has to name what it measured after the alias has moved on.
"""

from __future__ import annotations

from typing import Any

from lane_a import events as ev
from lane_a.adapters.base import SessionConfig
from lane_a.adapters.openai_realtime import OpenAIRealtimeAdapter

URL = "wss://api.x.ai/v1/realtime"


class GrokRealtimeAdapter(OpenAIRealtimeAdapter):
    name = "grok-realtime"
    input_rate = 24000
    output_rate = 24000
    supports_manual_commit = True
    supports_text_modality = True
    emits_vad_events = True

    url = URL

    def __init__(self, *, model: str = "grok-voice-think-fast-2.0", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)

    @classmethod
    def unsupported_reason(cls, config: SessionConfig) -> str | None:
        reason = super().unsupported_reason(config)
        if reason:
            return reason
        if config.turn_detection.mode == "semantic_vad":
            return "grok-realtime has no semantic turn detection"
        return None

    def _session_payload(self) -> dict[str, Any]:
        config = self.config
        audio: dict[str, Any] = {
            "input": {"format": {"type": "audio/pcm", "rate": self.input_rate}},
            "output": {"format": {"type": "audio/pcm", "rate": self.output_rate}},
        }
        if config.transcribe_input:
            audio["input"]["transcription"] = {"model": "grok-transcribe"}
        session: dict[str, Any] = {
            "instructions": config.instructions,
            # Explicit null is what selects manual turn taking; leaving the key
            # out keeps the server default, which is its own VAD.
            "turn_detection": self._turn_detection_payload(),
            "audio": audio,
        }
        if config.voice:
            session["voice"] = config.voice
        # No text-only output mode is exposed; the text arm is text in, audio
        # out, graded from the provider's transcript of its reply.
        if config.tools:
            session["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in config.tools
            ]
        return session

    def _handle(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type", "")
        if kind == "ping":
            return  # keep-alive; not evidence of anything
        if kind == "conversation.item.input_audio_transcription.completed" and payload.get("status") == "in_progress":
            # A cumulative draft, re-sent as more speech arrives. Only the final
            # one is the provider's transcript of what we said.
            self.log.raw(payload)
            return
        if kind == "conversation.item.input_audio_transcription.updated":
            self.log.raw(payload)
            return
        super()._handle(payload)
