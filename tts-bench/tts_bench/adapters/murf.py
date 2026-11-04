"""Murf Falcon 2.1 TTS over its bidirectional WebSocket.

Each benchmark context is a Murf ``context_id``.  It is configured before
the first text frame (and therefore before t0), receives as many text frames
as the caller has, then ``end: true`` marks the utterance complete.  Murf
returns base64, headerless little-endian PCM audio and a final message for the
same context.

``clear`` is Murf's barge-in control.  It stops queued audio, but cannot take
back audio the client has already received, so the player must stop local
playout too.  A final message following clear is the only completion signal;
there is no separate clear acknowledgement.

References:
https://murf.ai/api/docs/text-to-speech/web-sockets.md
https://murf.ai/api/docs/text-to-speech/web-sockets/context-id.md
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar
from urllib.parse import urlencode

from tts_bench import events as ev
from tts_bench.adapters._ws import WebSocketAdapter
from tts_bench.adapters.base import TTSConfig

# The Falcon voice catalog (GET /v1/speech/voices?model=FALCON) names the
# conversational style per voice: "Conversational" for some, "Conversation"
# for others.  The socket accepts either name without a warning, so send the
# one the voice lists; --option style=<name> overrides it and is recorded.
VOICE_STYLES = {"en-US-judith": "Conversational"}
DEFAULT_STYLE = "Conversation"


class MurfAdapter(WebSocketAdapter):
    """Falcon 2.1, using one persistent socket and named contexts."""

    name: ClassVar[str] = "murf"
    native_rates: ClassVar[tuple[int, ...]] = (24000,)
    native_mulaw_8k: ClassVar[bool] = False
    setup_excluded: ClassVar[str] = "TCP, TLS, websocket upgrade, per-context voice_config frame"
    base = "wss://global.api.murf.ai/v1/speech/stream-input"

    @property
    def url(self) -> str:  # type: ignore[override]
        # Keep the API key in the upgrade header rather than the URL: the run
        # records the endpoint as provenance, and URLs are easy to leak in logs.
        return f"{self.base}?" + urlencode({
            "model": self.config.model,
            "sample_rate": self.config.sample_rate,
            "channel_type": "MONO",
            "format": "PCM",
            "include_usage": "true",
        })

    def _headers(self) -> dict[str, str]:
        return {"api-key": self.api_key}

    @classmethod
    def unsupported_reason(cls, config: TTSConfig, needs: tuple[str, ...] = ()) -> str | None:
        if config.model != "falcon-2.1":
            return "murf WebSocket serves Falcon 2.1 only"
        return super().unsupported_reason(config, needs)

    async def _open_context(self, context_id: str) -> None:
        synthesis = self.contexts[context_id]
        style = self.config.option("style", VOICE_STYLES.get(self.config.voice, DEFAULT_STYLE))
        synthesis.meta["request"] = {
            "model": self.config.model,
            "voice_id": self.config.voice,
            "locale": self.config.language or "en-US",
            "style": style,
            "audio_format": "pcm_s16le",
            "sample_rate": self.config.sample_rate,
        }
        await self._send_json({
            "voice_config": {
                "voiceId": self.config.voice,
                "locale": self.config.language or "en-US",
                "style": style,
                "rate": 0,
                "pitch": 0,
            },
            "context_id": context_id,
        })

    async def _send_text(self, context_id: str, text: str, first: bool) -> None:
        # Falcon uses sentence-aware predictive chunking.  The documented
        # default (strict) is intentionally left in effect unless a campaign
        # records an explicit --option predictive_chunking=<strict|lenient>.
        payload: dict[str, Any] = {"text": text, "context_id": context_id, "end": False}
        predictive_chunking = self.config.option("predictive_chunking")
        if predictive_chunking is not None:
            payload["predictive_chunking"] = predictive_chunking
        await self._send_json(payload)

    async def _finish(self, context_id: str) -> None:
        await self._send_json({"text": "", "context_id": context_id, "end": True})

    async def _cancel(self, context_id: str) -> None:
        await self._send_json({"clear": True, "context_id": context_id})

    def _record_frame_metadata(self, context_id: str | None, message: dict[str, Any]) -> None:
        if context_id not in self.contexts:
            return
        synthesis = self.contexts[context_id]
        trace_id = message.get("trace_id")
        if trace_id:
            synthesis.meta.setdefault("trace_ids", []).append(trace_id)
        if "concurrency" in message:
            synthesis.meta["concurrency"] = message["concurrency"]

    def _on_message(self, message: dict[str, Any]) -> None:
        context_id = message.get("context_id")
        self._record_frame_metadata(context_id, message)

        audio = message.get("audio")
        if audio:
            try:
                self._on_audio(context_id, base64.b64decode(audio, validate=True))
            except (TypeError, ValueError) as exc:
                self._on_error(context_id, f"murf returned invalid base64 audio: {exc}")
            return

        if message.get("timestamps") is not None and context_id in self.contexts:
            words = message["timestamps"].get("words", [])
            self.contexts[context_id].meta.setdefault("word_timestamps", []).extend(words)
            return

        if message.get("warning") is not None and context_id in self.contexts:
            warning = {"code": message.get("warning_code"), "message": message["warning"]}
            self.contexts[context_id].meta.setdefault("warnings", []).append(warning)
            return

        if message.get("error") is not None:
            error = f"{message.get('error_code', 'ERROR')}: {message['error']}"
            # Murf documents that a non-fatal error can be followed by audio;
            # retain it in the cell record and only terminate a fatal context.
            if message.get("fatal"):
                self._on_error(context_id, error)
            elif context_id in self.contexts:
                self.log.emit(ev.PROVIDER_ERROR, context_id=context_id, message=error)
                self.contexts[context_id].meta.setdefault("nonfatal_errors", []).append(error)
            else:
                self._on_error(context_id, error)
            return

        if message.get("final") and context_id in self.contexts:
            synthesis = self.contexts[context_id]
            if synthesis.t_cancel is not None:
                self._on_cancel_ack(context_id, cleared=True)
            self._on_done(context_id)

    def _on_closed(self, reason: str) -> None:
        # A bad Murf key is reported after the WebSocket upgrade as close 1008,
        # not as HTTP 401/403.  Preserve that as a refusal so the runner voids
        # the cell instead of publishing it as a service-quality failure.
        if "1008" in reason:
            self._on_error(None, f"murf connection rejected: {reason}")
            return
        for context_id, synthesis in self.contexts.items():
            if not synthesis.done.is_set():
                self._on_error(context_id, f"murf connection closed before final: {reason}")
