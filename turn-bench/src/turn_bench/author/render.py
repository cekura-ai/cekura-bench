"""Renderers: ``render(text, voice, speed=1.0) -> (mono float32 samples, sample_rate)``, cached on disk.

Two implementations share the interface:

* ``MockRenderer`` — offline and deterministic. Every word becomes a harmonic
  burst (a voice-specific fundamental with a short raised-cosine attack and
  release) separated by exact digital silence; a ``<break time="1.5s"/>``
  marker becomes exactly that much silence. Because nothing is estimated, the
  renderer can also state where its speech segments are (``layout``), which is
  what the boundary-finder tests measure against.
* ``ElevenLabsRenderer`` — the service bench's TTS path (``service.clips``):
  the same request, model and 24 kHz PCM master as ``bin/render-corpus.py``,
  with the voice ids from ``service.corpus_v1``. Needs ``ELEVENLABS_API_KEY``.

``speed`` is the TTS speaking rate (1.0 = the voice's natural rate): ElevenLabs
takes it as ``voice_settings.speed`` (its documented range is 0.7-1.2), the
mock stretches every word and gap by ``1 / speed``.

``RenderCache`` keys a render by sha256 of renderer name, model, voice, text
and (when it is not 1.0) speed, and stores a 16-bit WAV plus a JSON sidecar (provenance, the file's sha256) under
a gitignored directory (``<turn-bench>/.cache/renders`` by default), so a spec
edit re-renders only the lines that changed and a dataset can record exactly
which render bytes it was composed from.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from turn_bench.assets import TURN_BENCH_ROOT
from turn_bench.author import service_bridge
from turn_bench.author.spec import split_breaks

DEFAULT_CACHE_DIR = TURN_BENCH_ROOT / ".cache" / "renders"
MASTER_RATE = 24000  # the service bench's master rate; the mock mirrors it


@dataclass(frozen=True)
class Voice:
    """A voice label shared by every renderer: the registry of the authored
    datasets' voices. The three published voices (``f-us``, ``m-us``, ``f-gb``)
    take their ElevenLabs id from ``service.corpus_v1`` (``elevenlabs_voice_id``
    None); the accent voices added in v0.4 record theirs here, chosen from the
    account's voice library by accent label. The mock only needs a fundamental."""

    label: str
    gender: str
    f0_hz: float
    accent: str = ""  # the voice library's accent label; the reports' accent axis is the caller voice
    elevenlabs_voice_id: str | None = None  # None: resolved from service.corpus_v1
    elevenlabs_name: str = ""  # the voice's name in the library, for the record

    def record(self) -> dict[str, Any]:
        return {"gender": self.gender, "accent": self.accent, "elevenlabs_voice_id": self.elevenlabs_voice_id, "elevenlabs_name": self.elevenlabs_name}


VOICES: dict[str, Voice] = {
    "f-us": Voice("f-us", "female", 210.0, "american"),
    "m-us": Voice("m-us", "male", 118.0, "american"),
    "f-gb": Voice("f-gb", "female", 196.0, "british"),
    # v0.4 accent voices (ElevenLabs voice library, 2026-10-10; both support eleven_turbo_v2_5)
    "m-in": Voice("m-in", "male", 125.0, "indian", "wJ5MX7uuKXZwFqGdWM4N", "Raj - Professional Indian English"),
    "f-au": Voice("f-au", "female", 200.0, "australian", "ys3XeJJA4ArWMhRpcX1D", "Jess - Relaxed, Conversational Australian Accent"),
}


def voice_registry(model: str = "") -> dict[str, dict[str, Any]]:
    """Every voice label with gender, accent, ElevenLabs id, name and model (for
    a dataset manifest). Published voices resolve their id from the service bench
    when it is importable; otherwise the id is left None."""
    registry = {}
    try:
        published = service_bridge.corpus_voices()
    except ImportError:
        published = {}
    for label, voice in VOICES.items():
        entry = voice.record()
        if entry["elevenlabs_voice_id"] is None and label in published:
            entry["elevenlabs_voice_id"] = published[label].voice_id
            entry["elevenlabs_name"] = entry["elevenlabs_name"] or f"service.corpus_v1 {label}"
        entry["model"] = model or (published[label].model if label in published else "")
        registry[label] = entry
    return registry


SPEED_MIN, SPEED_MAX = 0.7, 1.2  # ElevenLabs' documented range for voice_settings.speed


class Renderer(Protocol):
    name: str
    model: str

    def render(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]: ...


def check_speed(speed: float) -> float:
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        raise ValueError(f"speed {speed} outside [{SPEED_MIN}, {SPEED_MAX}]")
    return float(speed)


# ---- mock -----------------------------------------------------------------------

_WORD_CHARS = re.compile(r"[A-Za-z0-9']")


@dataclass(frozen=True)
class MockRenderer:
    """Speech-like bursts per word, exact silence for breaks; see module docstring.

    Timing constants (seconds): a word lasts ``0.09 + 0.06 * letters`` clamped
    to ``[0.15, 0.45]``; words are separated by ``word_gap_s``, plus
    ``punctuation_gap_s`` after ``.,;:?!``. All intra-segment gaps stay well
    below ``bounds.MIN_PAUSE_S`` so a segment is one detected run of speech.
    """

    sample_rate: int = MASTER_RATE
    amplitude: float = 0.3
    word_gap_s: float = 0.06
    punctuation_gap_s: float = 0.12
    lead_s: float = 0.1
    trail_s: float = 0.1
    ramp_s: float = 0.01
    name: str = "mock"
    model: str = "mock-v1"

    def _word_samples(self, word: str, speed: float = 1.0) -> int:
        letters = len(_WORD_CHARS.findall(word))
        seconds = min(0.45, max(0.15, 0.09 + 0.06 * letters)) / speed
        return int(round(seconds * self.sample_rate))

    def _gap_samples(self, word: str, speed: float = 1.0) -> int:
        extra = self.punctuation_gap_s if word and word[-1] in ".,;:?!" else 0.0
        return int(round((self.word_gap_s + extra) / speed * self.sample_rate))

    def layout(self, text: str, voice: str, speed: float = 1.0) -> list[tuple[int, int]]:
        """Exact (start, end) sample of every speech segment the render will hold."""
        speed = check_speed(speed)
        segments, breaks = split_breaks(text)
        cursor = int(round(self.lead_s * self.sample_rate))
        bounds: list[tuple[int, int]] = []
        for index, segment in enumerate(segments):
            words = segment.split()
            start = cursor
            for position, word in enumerate(words):
                cursor += self._word_samples(word, speed)
                if position < len(words) - 1:
                    cursor += self._gap_samples(word, speed)
            bounds.append((start, cursor))
            if index < len(breaks):
                cursor += int(round(breaks[index] * self.sample_rate))  # a break is a break at any speed
        return bounds

    def render(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        if voice not in VOICES:
            raise KeyError(f"unknown voice {voice!r}; known: {sorted(VOICES)}")
        speed = check_speed(speed)
        f0 = VOICES[voice].f0_hz
        rate = self.sample_rate
        seed = int.from_bytes(hashlib.sha256(f"{voice}\n{text}".encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        segments, _breaks = split_breaks(text)
        layout = self.layout(text, voice, speed)
        total = layout[-1][1] + int(round(self.trail_s * rate))
        out = np.zeros(total, dtype=np.float64)
        ramp = max(1, int(round(self.ramp_s * rate)))
        for (start, _end), segment in zip(layout, segments):
            cursor = start
            words = segment.split()
            for position, word in enumerate(words):
                n = self._word_samples(word, speed)
                t = np.arange(n) / rate
                pitch = f0 * (1.0 + 0.04 * rng.standard_normal())
                burst = np.zeros(n)
                for harmonic in range(1, 7):
                    burst += np.sin(2 * np.pi * pitch * harmonic * t + rng.uniform(0, 2 * np.pi)) / harmonic
                burst *= self.amplitude / np.max(np.abs(burst))
                envelope = np.ones(n)
                k = min(ramp, n // 2)
                if k:
                    window = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
                    envelope[:k] = window
                    envelope[-k:] = window[::-1]
                out[cursor : cursor + n] = burst * envelope
                cursor += n
                if position < len(words) - 1:
                    cursor += self._gap_samples(word, speed)
        return out.astype(np.float32), rate


# ---- ElevenLabs (via the service bench) ------------------------------------------


@dataclass(frozen=True)
class ElevenLabsRenderer:
    """Delegates to ``service.clips._elevenlabs_pcm``: one POST per line, PCM
    int16 at the service bench's master rate, voice ids from ``service.corpus_v1``.
    Break markers are passed through in the text (ElevenLabs renders SSML breaks).
    A ``speed`` other than 1.0 is the same request with ``voice_settings.speed``
    set (the service bench's request has no speed, so that one is built here)."""

    api_key: str
    name: str = "elevenlabs"
    model: str = ""  # resolved from the service bench's Voice when empty

    def __repr__(self) -> str:  # never echo the key (tracebacks, logs, provenance)
        return f"ElevenLabsRenderer(name={self.name!r}, model={self.model!r})"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "ElevenLabsRenderer":
        """The key from ``ELEVENLABS_API_KEY``, else from ``<turn-bench>/.env``
        (gitignored; ``TURN_BENCH_ENV_FILE`` overrides the path). The key stays
        in-process: it is never logged or written to any provenance record."""
        key = os.environ.get("ELEVENLABS_API_KEY") or read_env_key(env_file or default_env_file(), "ELEVENLABS_API_KEY")
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set (environment or .env)")
        return cls(api_key=key, model=service_bridge.clips().DEFAULT_TTS_MODEL)

    def service_voice(self, voice: str) -> Any:
        """The service bench ``Voice`` for a label: the published one from
        ``service.corpus_v1``, else one built from this registry's ElevenLabs id."""
        clips = service_bridge.clips()
        published = service_bridge.corpus_voices()
        if voice in published:
            return published[voice]
        entry = VOICES.get(voice)
        if entry is None or entry.elevenlabs_voice_id is None:
            raise KeyError(f"voice {voice!r} has no ElevenLabs id (service.corpus_v1 or render.VOICES); known: {sorted(VOICES)}")
        return clips.Voice(label=voice, vendor="elevenlabs", voice_id=entry.elevenlabs_voice_id, model=self.model or clips.DEFAULT_TTS_MODEL)

    def render(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        clips = service_bridge.clips()
        service_voice = self.service_voice(voice)
        speed = check_speed(speed)
        if speed == 1.0:
            pcm = clips._elevenlabs_pcm(self.api_key, text, service_voice)
        else:
            pcm = _elevenlabs_pcm_at_speed(self.api_key, text, service_voice, speed)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return samples, clips.MASTER_RATE


def _elevenlabs_pcm_at_speed(api_key: str, text: str, voice: Any, speed: float) -> bytes:
    """``service.clips._elevenlabs_pcm`` with ``voice_settings.speed``: same URL,
    output format and model; the voice's other settings stay at its defaults."""
    import urllib.error
    import urllib.request

    clips = service_bridge.clips()
    request = urllib.request.Request(
        clips.ELEVENLABS_URL.format(voice=voice.voice_id) + f"?output_format=pcm_{clips.MASTER_RATE}",
        data=json.dumps({"text": text, "model_id": voice.model, "voice_settings": {"speed": speed}}).encode(),
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:  # surface the vendor's reason, not a stack trace (never the key)
        raise RuntimeError(f"ElevenLabs {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}") from exc


def default_env_file() -> Path:
    return Path(os.environ.get("TURN_BENCH_ENV_FILE") or TURN_BENCH_ROOT / ".env")


def read_env_key(path: Path, name: str) -> str | None:
    """``NAME=value`` from a dotenv-style file (comments, blank lines and
    optional single/double quotes around the value); None when absent."""
    if not Path(path).is_file():
        return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value or None
    return None


def make_renderer(name: str, **options: Any) -> Renderer:
    if name == "mock":
        return MockRenderer(**options)
    if name == "elevenlabs":
        return ElevenLabsRenderer.from_env()
    raise KeyError(f"unknown renderer {name!r}; known: mock, elevenlabs")


# ---- cache ------------------------------------------------------------------------


@dataclass(frozen=True)
class Render:
    """One cached render and its provenance."""

    samples: np.ndarray  # float32 mono
    sample_rate: int
    sha256: str  # of the cached WAV file
    provenance: dict[str, Any]


def render_key(renderer: Renderer, text: str, voice: str, speed: float = 1.0) -> str:
    parts = [renderer.name, renderer.model, voice, text]
    if speed != 1.0:  # keys of natural-rate renders are unchanged from before speed existed
        parts.append(f"speed={speed:g}")
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


class RenderCache:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else DEFAULT_CACHE_DIR
        self.hits = 0
        self.misses = 0  # = renderer calls made (one per miss)
        self.characters = 0  # characters of text sent to the renderer on misses (the TTS cost unit)

    def paths(self, renderer: Renderer, text: str, voice: str, speed: float = 1.0) -> tuple[Path, Path]:
        key = render_key(renderer, text, voice, speed)
        base = self.directory / renderer.name / key
        return base.with_suffix(".wav"), base.with_suffix(".json")

    def get(self, renderer: Renderer, text: str, voice: str, speed: float = 1.0) -> Render:
        """The cached render, rendering (and caching) on a miss."""
        wav_path, meta_path = self.paths(renderer, text, voice, speed)
        if wav_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            samples, rate = sf.read(wav_path, dtype="float32", always_2d=False)
            self.hits += 1
            return Render(samples=samples, sample_rate=rate, sha256=meta["sha256"], provenance=meta)
        samples, rate = renderer.render(text, voice, speed)
        self.characters += len(text)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, samples, rate, subtype="PCM_16")
        # Re-read so the samples the pipeline sees are exactly the cached file's.
        samples, rate = sf.read(wav_path, dtype="float32", always_2d=False)
        digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        meta = {
            "renderer": renderer.name,
            "model": renderer.model,
            "voice": voice,
            "text": text,
            "speed": speed,
            "sample_rate": rate,
            "samples": int(len(samples)),
            "sha256": digest,
            "file": wav_path.name,
            "rendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        self.misses += 1
        return Render(samples=samples, sample_rate=rate, sha256=digest, provenance=meta)

    def stats(self) -> dict[str, Any]:
        return {"dir": str(self.directory), "hits": self.hits, "misses": self.misses, "renderer_calls": self.misses, "characters_rendered": self.characters}
