"""Optional, pinned Silero speech veto for a separate calibration scorecard.

This produces coarse evidence chunks, not exact acoustic boundaries. No model
assets are downloaded, installed or modified by this module.
"""
from __future__ import annotations

import hashlib
import math
import threading
from pathlib import Path

import numpy as np

from turn_taking_bench.adapters import smart_turn_v3

RATE = 16_000
CHUNK = 512
THRESHOLD = 0.5


class SpeechVeto:
    """Reuse the single CPU-thread Silero session with per-analysis state resets."""

    def __init__(self, model_path: Path | str):
        path = Path(model_path)
        if not path.is_file():
            raise ValueError(f"cached Silero model is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != smart_turn_v3.SILERO_VAD.sha256:
            raise ValueError("cached Silero model SHA-256 does not match the pinned asset")
        # SileroVad loads a cached CPU session with one inter/intra-op thread.
        # No asset helper is called: the exact local bytes were checked first.
        self._vad = smart_turn_v3.SileroVad(path.resolve(), reset_s=0)
        self._lock = threading.Lock()
        self.provenance = {
            "model": "silero_vad", "model_path": str(path.resolve()),
            "model_sha256": actual, "threshold": THRESHOLD,
            "threshold_rule": "probability > 0.5", "sample_rate": RATE,
            "chunk_samples": CHUNK, "chunk_ms": 1000 * CHUNK / RATE,
            "cpu_threads": 1, "reset_s": 0,
            "reset_policy": "reset recurrent/context state before every analysis; no periodic resets",
            "boundary_definition": "coarse 32 ms evidence chunks; not exact acoustic boundaries",
        }

    def analyze(self, pcm: bytes, rate: int, *, start_s: float = 0) -> dict:
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("audio must be headerless mono signed little-endian 16-bit PCM bytes")
        if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
            raise ValueError("sample rate must be a positive integer")
        if isinstance(start_s, bool) or not isinstance(start_s, (int, float)) or not math.isfinite(start_s) or start_s < 0:
            raise ValueError("start_s must be a finite nonnegative timestamp")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768.0)
        source_samples = len(samples)
        if rate != RATE and source_samples:
            # Only interpolate where both source samples exist. In particular,
            # do not extend the final sample to invent resampled tail evidence.
            count = ((source_samples - 1) * RATE) // rate + 1
            positions = np.arange(count, dtype=np.float64) * rate / RATE
            samples = np.interp(positions, np.arange(source_samples), samples).astype(np.float32)
        usable = len(samples) // CHUNK * CHUNK
        chunks = []
        with self._lock:
            self._vad.reset()
            for index in range(0, usable, CHUNK):
                end = (index + CHUNK) / RATE
                probability = float(self._vad.prob(samples[index:index + CHUNK], end))
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Silero returned an invalid speech probability")
                chunks.append({"start_s": float(start_s + index / RATE),
                               "end_s": float(start_s + end), "probability": probability})
        provenance = {**self.provenance, "input_rate": rate,
                      "input_samples": source_samples, "resampled_samples": len(samples),
                      "analyzed_samples": usable, "omitted_tail_samples_16k": len(samples) - usable,
                      "resampling": "none" if rate == RATE else "numpy linear interpolation; no edge extrapolation",
                      "normalization": "PCM int16 / 32768, float32",
                      "tail_policy": "only complete 512-sample chunks; no padding or extrapolated evidence",
                      "start_s": float(start_s)}
        return {"chunks": chunks, "provenance": provenance}


def to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    """int16 PCM -> float32 at 16 kHz, the same linear interpolation as ``analyze``."""
    samples = np.asarray(pcm, dtype=np.float32) / np.float32(32768.0)
    if rate != RATE and len(samples):
        count = ((len(samples) - 1) * RATE) // rate + 1
        positions = np.arange(count, dtype=np.float64) * rate / RATE
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    return samples


def local_probs(veto: SpeechVeto, x16: np.ndarray, width: int) -> np.ndarray:
    """Chunk k's probability from a FRESH state over chunks [k-width+1, k].

    Removes long recurrent hangover: a chunk is labelled speech only when the
    last ``width`` chunks contain speech, whatever came before them.
    """
    n = len(x16) // CHUNK
    out = np.zeros(n)
    with veto._lock:
        for k in range(n):
            veto._vad.reset()
            p = 0.0
            for j in range(max(0, k - width + 1), k + 1):
                p = float(veto._vad.prob(x16[j * CHUNK:(j + 1) * CHUNK], (j + 1) * CHUNK / RATE))
            out[k] = p
    return out


def fresh_probs(veto: SpeechVeto, x16: np.ndarray) -> np.ndarray:
    """Probabilities of every complete chunk of ``x16`` from one fresh state."""
    n = len(x16) // CHUNK
    out = np.zeros(n)
    with veto._lock:
        veto._vad.reset()
        for j in range(n):
            out[j] = float(veto._vad.prob(x16[j * CHUNK:(j + 1) * CHUNK], (j + 1) * CHUNK / RATE))
    return out
