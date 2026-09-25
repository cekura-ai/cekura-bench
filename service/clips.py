"""The caller corpus: authored here, rendered, checksummed, versioned.

The caller content is written here because no public dataset can supply it: these
clips are replies to *our* agent, in *our* scenarios, and a corpus of unrelated
recordings cannot answer a question the agent has just asked. Nothing here is
sampled from public recordings: interaction tokens are rendered like every other
line, and noise beds are synthesized (``service/audio.py``).

Provenance travels with every clip and therefore with every published cell:
``tts:<vendor>/<voice>`` or ``human:<speaker>``. v1 may publish on TTS audio so
labelled; the human re-record of the same script is the first sensitivity release
after it. If re-recording moves a ranking, that is a finding.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from service.audio import read_wav, write_wav
from service.provenance import sha256_file, text_digest
from service.caller import Clip

MASTER_RATE = 24000  # one canonical master per clip; providers get a published resample
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
DEFAULT_TTS_MODEL = "eleven_turbo_v2_5"


@dataclass(frozen=True)
class Voice:
    label: str          # what appears in a published cell
    vendor: str
    voice_id: str
    model: str = DEFAULT_TTS_MODEL

    @property
    def provenance(self) -> str:
        return f"tts:{self.vendor}/{self.label}"


@dataclass(frozen=True)
class ClipSpec:
    """A line of authored caller speech. ``text`` is also the ASR ground truth."""

    id: str
    text: str
    note: str = ""


class Corpus:
    """Rendered clips on disk, addressed by ``clip_id`` and voice label."""

    def __init__(self, root: str | Path, version: str = "0.1.0") -> None:
        self.root = Path(root)
        self.version = version
        self.specs: dict[str, ClipSpec] = {}
        self.voices: dict[str, Voice] = {}

    # -- definition -------------------------------------------------------

    def add(self, *specs: ClipSpec) -> "Corpus":
        for spec in specs:
            self.specs[spec.id] = spec
        return self

    def add_voice(self, *voices: Voice) -> "Corpus":
        for voice in voices:
            self.voices[voice.label] = voice
        return self

    # -- rendering --------------------------------------------------------

    @staticmethod
    def text_stamp(text: str) -> str:
        """Short form of the shared digest: eight characters is plenty in a filename."""
        return text_digest(text, 8)

    def path_for(self, clip_id: str, voice: str) -> Path:
        """Render path carries a stamp of the text it was rendered from.

        Editing a line and re-running the renderer would otherwise leave the old
        audio in place -- it exists, so it is skipped -- and the published
        manifest would then assert a checksum for audio that says something else.
        Putting the text in the path makes a stale render impossible rather than
        merely unlikely.
        """
        stamp = self.text_stamp(self.specs[clip_id].text)
        return self.root / "audio" / voice / f"{clip_id}.{stamp}.wav"

    def render(self, api_key: str, voice: Voice, overwrite: bool = False) -> list[str]:
        """Render every missing clip for one voice. Returns the ids written."""
        written = []
        for spec in self.specs.values():
            target = self.path_for(spec.id, voice.label)
            if target.exists() and not overwrite:
                continue
            write_wav(target, _elevenlabs_pcm(api_key, spec.text, voice), MASTER_RATE)
            written.append(spec.id)
        return written

    # -- loading ----------------------------------------------------------

    def load(self, clip_id: str, voice: str) -> Clip:
        spec = self.specs[clip_id]
        pcm, rate = read_wav(self.path_for(clip_id, voice))
        return Clip(name=f"{clip_id}@{voice}", pcm=pcm, rate=rate, text=spec.text)

    # -- publication ------------------------------------------------------

    def manifest(self) -> dict:
        """What gets published: the script, the voices, and a checksum per file.

        Anything published becomes training data eventually, and a model that has
        seen the corpus scores well on it for the wrong reason. Releases therefore
        pair this manifest with a hidden holdout, drawn from the same script and
        scored the same way, so the published set stays auditable while the
        ranking keeps something it cannot have been fitted to.
        """
        clips = []
        for spec in sorted(self.specs.values(), key=lambda s: s.id):
            entry = {**asdict(spec), "renders": {}}
            for label, voice in self.voices.items():
                path = self.path_for(spec.id, label)
                if path.exists():
                    entry["renders"][label] = {
                        "file": path.name,
                        "provenance": voice.provenance,
                        "tts_model": voice.model,
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
            clips.append(entry)
        return {
            "corpus_version": self.version,
            "master_rate": MASTER_RATE,
            "voices": {label: asdict(voice) for label, voice in self.voices.items()},
            "clips": clips,
        }

    def write_manifest(self) -> Path:
        path = self.root / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), indent=2) + "\n", encoding="utf-8")
        return path


def _elevenlabs_pcm(api_key: str, text: str, voice: Voice) -> bytes:
    request = urllib.request.Request(
        ELEVENLABS_URL.format(voice=voice.voice_id) + f"?output_format=pcm_{MASTER_RATE}",
        data=json.dumps({"text": text, "model_id": voice.model}).encode(),
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:  # surface the vendor's reason, not a stack trace
        raise RuntimeError(f"ElevenLabs {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}") from exc
