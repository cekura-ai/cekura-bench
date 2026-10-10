"""Azure Speech text to speech over HTTP: one SSML request per utterance, raw PCM streamed back.

Microsoft's MAI voices are served here. A model is picked by suffixing the voice
name with it in the SSML (``en-US-Harper:MAI-Voice-2.1-Flash``), so ``model``
and ``voice`` stay separate in the config and are joined only on the wire.

The key belongs to one Speech resource in one region, and the region is the
host. The default is East US, next to the campaign's cloud region; the
``region`` option overrides it and is recorded in every cell.

The whole text goes in one request, so streamed input, cancel and continuation
are declared exclusions, as for OpenAI. The connection is warmed with the voice
list on the same host before t0, so the timed part is synthesis alone. Text is
XML-escaped: a sentence holding ``&`` or ``<`` must read as written, not break
the request.
"""

from __future__ import annotations

from typing import Any, ClassVar
from xml.sax.saxutils import escape, quoteattr

from tts_bench.adapters.openai_tts import OpenAITTSAdapter

RATE = 24000
DEFAULT_REGION = "eastus"


class AzureSpeechAdapter(OpenAITTSAdapter):
    name: ClassVar[str] = "azure-speech"
    native_rates: ClassVar[tuple[int, ...]] = (RATE,)
    _PRIVATE_HEADERS = OpenAITTSAdapter._PRIVATE_HEADERS | {"ocp-apim-subscription-key"}
    speech_path = "/cognitiveservices/v1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.region = self.config.option("region", DEFAULT_REGION)
        self.base = f"https://{self.region}.tts.speech.microsoft.com"

    def _prewarm_url(self) -> str:
        return f"{self.base}/cognitiveservices/voices/list"

    def _headers(self) -> dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
            "User-Agent": "cekura-tts-bench",
        }

    def ssml(self, text: str) -> str:
        """The request body: one voice element, the model as the voice name's suffix, no style."""
        voice = quoteattr(f"{self.config.voice}:{self.config.model}")
        lang = quoteattr(self.config.language or "-".join(self.config.voice.split("-")[:2]))
        body = escape(text)
        if self.config.speed is not None:
            body = f"<prosody rate={quoteattr(f'{self.config.speed:.2f}')}>{body}</prosody>"
        return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang={lang}>'
                f"<voice name={voice}>{body}</voice></speak>")

    def _request_args(self, text: str) -> dict[str, Any]:
        self.log.raw({"voice": f"{self.config.voice}:{self.config.model}", "input": f"<{len(text)} chars>"}, direction="out")
        return {"data": self.ssml(text).encode("utf-8")}
