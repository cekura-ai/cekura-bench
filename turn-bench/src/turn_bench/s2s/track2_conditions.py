"""Track 2 caller audio conditions: Track 1's seeded service transforms on the caller stream.

A Track 2 session streams pre-authored caller clips into one continuous,
real-time carrier (``service.caller.BranchingCaller``) whose gaps are digital
silence. A condition degrades what the model HEARS of the caller, both the
clips and the silence between them, with the same named, seeded
``service.transforms`` transform Track 1's condition variants were made with
(``turn_bench.author.degrade.CONDITIONS``). The model's output channel, the
speech-cessation measurement, stop detection and the no-stimulus controls are
untouched: they read only the model's audio as received.

Two stages, chosen per transform so that causality is never broken:

* ``clip`` stage (``telephone``, ``reverb``): the transform needs the whole
  input (resampler filter lookahead; RMS normalisation over the input), so it
  is applied to each whole caller clip BEFORE streaming, at the session's input
  rate. The clips are pre-authored audio, so this looks ahead only into audio
  that already exists; nothing depends on the model. The carrier's digital
  silence is left as silence (G.711 codes silence as silence; the reverb tail
  is cut at the clip end because the transform is length-preserving).
* ``stream`` stage (``noise-*``, ``dropouts``): the transform is content-
  independent per sample, so it is applied to every outbound 20 ms chunk at its
  absolute stream position, in order, clips and carrier silence alike. Noise is
  therefore present between lines as on a real call, continuous across clip
  boundaries. Output at stream sample ``p`` depends only on input at ``p``.

``noise-*``: ``service.transforms.noise`` adds ``_pink(n, rate, seed) * level /
10**(snr/20)`` with ``level`` = RMS of its input. A live stream has no whole
input, so here ``level`` is fixed per conversation, before the session, as the
RMS of that conversation's caller clips concatenated (at the input rate) --
"SNR against the caller's own audio" -- and the bed is one ``_pink`` draw of
``NOISE_PERIOD_S`` (seed 101) indexed by ``position % period`` (an inverse-FFT
draw is periodic in its length, so the loop point is seamless). With the
period equal to a stream's length and ``level`` equal to its RMS the result is
byte-identical to ``service.transforms.noise`` on that stream (tested). Track
1 took ``level`` over the whole two-speaker channel including gaps, so its
effective speech SNR is a few dB higher; Track 1 and Track 2 condition numbers
are not interchangeable (they are never blended anyway).

``dropouts``: the service transform's holes (60 ms every 500 ms from 250 ms)
at absolute stream positions; identical to the service transform on the whole
stream (tested).

Determinism: clip-stage output is a pure function of the clip (byte-identical
on every run, SHA-256 recorded); stream-stage output is a pure function of the
stream position and the transmitted samples (the noise bed is a fixed seeded
array, SHA-256 recorded), so the same clip schedule yields a byte-identical
stream. Every session also saves the transmitted stream as ``caller.wav``.
"""
from __future__ import annotations

import functools
import hashlib
from typing import Any

import numpy as np

from turn_bench.author import service_bridge
from turn_bench.author.degrade import CONDITIONS as TRACK1_CONDITIONS

service_bridge._ensure_path()
from service import transforms as T  # noqa: E402
from service.audio import to_array, to_pcm  # noqa: E402
from service.caller import BranchingCaller, Clip  # noqa: E402

SCHEMA = "turn-bench/track2-condition/1"
NOISE_PERIOD_S = 120.0
# Parameters of the service transforms (service/transforms.py TRANSFORMS; a
# changed parameter there is a new transform name, so these cannot drift
# silently -- test_track2_conditions checks them against the service output).
NOISE = {"noise-20db": {"snr_db": 20.0, "seed": 101}, "noise-10db": {"snr_db": 10.0, "seed": 101}}
DROPOUTS = {"hole_ms": 60.0, "every_ms": 500.0, "offset_ms": 250.0}
REVERB = {"rt60_s": 0.4, "direct_ratio": 0.6, "seed": 202}
TELEPHONE = {"rate_hz": 8000, "codec": "G.711 mu-law"}

STAGES = {"clean": None, "telephone": "clip", "reverb": "clip",
          "noise-20db": "stream", "noise-10db": "stream", "dropouts": "stream"}
CONDITIONS = tuple(STAGES)


def _rms(samples: np.ndarray) -> float:
    return T._rms(samples)


@functools.lru_cache(maxsize=4)
def _unit_pink(n: int, rate: int, seed: int) -> np.ndarray:
    bed = T._pink(n, rate, seed)
    bed.setflags(write=False)
    return bed


class Condition:
    """One named caller condition, bound to a session's input rate."""

    def __init__(self, name: str = "clean"):
        if name not in STAGES:
            raise KeyError(f"unknown Track 2 condition {name!r}; known: {list(STAGES)}")
        self.name = name
        self.transform = TRACK1_CONDITIONS[name]["transform"]
        self.named = T.get(self.transform)  # fail early if the service renamed it
        self.stage = STAGES[name]
        self.rate: int | None = None
        self.level: float | None = None
        self._bed: np.ndarray | None = None
        self._scaled: np.ndarray | None = None

    @property
    def params(self) -> dict[str, Any]:
        if self.name.startswith("noise-"):
            return {**NOISE[self.name], "period_s": NOISE_PERIOD_S,
                    "level": "RMS of the conversation's caller clips, concatenated, at the input rate"}
        return {"telephone": TELEPHONE, "reverb": REVERB, "dropouts": DROPOUTS}.get(self.name, {})

    @property
    def seed(self) -> int | None:
        return self.params.get("seed")

    # -- clip stage ---------------------------------------------------------

    def prepare_clips(self, clips: dict[str, Clip], rate: int) -> dict[str, Clip]:
        """Clips at the session input rate, clip-stage transform applied to each
        whole clip; fixes the noise level for stream-stage noise."""
        self.rate = rate
        out = {}
        for key, clip in clips.items():
            clip = clip.at_rate(rate)
            if self.stage == "clip":
                pcm = self.named.apply(clip.pcm, rate)
                if len(pcm) != len(clip.pcm):
                    raise RuntimeError(f"transform {self.transform} changed clip length")
                clip = Clip(clip.name, pcm, rate, clip.text)
            out[key] = clip
        if self.name.startswith("noise-"):
            joined = np.concatenate([to_array(c.pcm) for c in out.values()]).astype(np.float64) if out else np.zeros(0)
            self.set_noise(_rms(joined) or 1.0, rate)
        return out

    # -- stream stage -------------------------------------------------------

    def set_noise(self, level: float, rate: int, period: int | None = None) -> None:
        p = NOISE[self.name]
        self.rate, self.level = rate, float(level)
        self._bed = _unit_pink(period or int(round(NOISE_PERIOD_S * rate)), rate, p["seed"])
        # same expression and order as service.transforms.noise
        self._scaled = self._bed * self.level / (10 ** (p["snr_db"] / 20.0))

    def process(self, pcm: bytes, position: int) -> bytes:
        """Stream-stage transform of one outbound chunk starting at absolute
        stream sample ``position``; identity for clean and clip-stage conditions."""
        if self.stage != "stream" or not pcm:
            return pcm
        if self.name == "dropouts":
            samples = to_array(pcm).copy()
            hole = int(self.rate * DROPOUTS["hole_ms"] / 1000.0)
            step = int(self.rate * DROPOUTS["every_ms"] / 1000.0)
            start = int(self.rate * DROPOUTS["offset_ms"] / 1000.0)
            idx = position + np.arange(samples.size)
            phase = idx - start
            samples[(phase >= 0) & (phase % step < hole)] = 0
            return samples.tobytes()
        samples = to_array(pcm).astype(np.float64)
        period = self._scaled.size
        idx = (position + np.arange(samples.size)) % period
        return to_pcm(samples + self._scaled[idx])

    # -- provenance ---------------------------------------------------------

    def provenance(self) -> dict[str, Any]:
        out = {"schema": SCHEMA, "condition": self.name, "transform": self.transform,
               "transform_description": self.named.description, "track1_condition": TRACK1_CONDITIONS[self.name],
               "stage": self.stage, "seed": self.seed, "params": self.params,
               "applied_to": "caller audio only (clips and carrier silence); model output as received",
               "causality": {"clip": "whole pre-authored clips transformed before streaming (lookahead only into authored audio)",
                             "stream": "each outbound chunk transformed at its absolute stream position, in order",
                             None: "no transform"}[self.stage]}
        if self.level is not None:
            out.update(input_rate=self.rate, level_rms=self.level,
                       level_dbfs=20 * np.log10(max(self.level, 1e-9) / 32768),
                       noise_rms_dbfs=20 * np.log10(max(self.level / 10 ** (NOISE[self.name]["snr_db"] / 20), 1e-9) / 32768),
                       noise_bed_sha256=hashlib.sha256(np.ascontiguousarray(self._bed).tobytes()).hexdigest())
        return out

    def describe(self) -> dict[str, Any]:
        """Run-level record (no session-specific level)."""
        d = self.provenance()
        for k in ("input_rate", "level_rms", "level_dbfs", "noise_rms_dbfs", "noise_bed_sha256"):
            d.pop(k, None)
        return d


class ConditionedCaller(BranchingCaller):
    """BranchingCaller whose every outbound chunk passes the stream stage."""

    def __init__(self, adapter, log, condition: Condition):
        super().__init__(adapter, log)
        self.condition = condition

    def _next_chunk(self) -> bytes:
        position = self.adapter.caller_timeline.n_samples  # first sample of this chunk
        return self.condition.process(super()._next_chunk(), position)
