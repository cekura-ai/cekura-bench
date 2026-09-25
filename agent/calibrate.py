"""Measure the phone-leg detector's own error against signals with exact boundaries.

    python -m agent.calibrate

Nothing here touches a provider or a phone network. The signal is constructed, so
its true onset and offset are known to the sample, and the difference between
those and the detector's answer is the resolution floor for every latency the agent bench
could publish. A gap between two configurations smaller than that number is not
a result.

The signal is then put through the path the audio actually takes -- resampled to
8 kHz, μ-law encoded and decoded, mixed with line noise and mains hum -- because
calibrating on clean wideband audio would measure a detector that is never used.

Two cases are measured separately, because they fail differently:

* a **voiced onset**, the ordinary case;
* an **unvoiced onset**, where the turn opens on a fricative. Periodicity cannot
  see a fricative, so this is late by construction. It is reported rather than
  tuned away: the fix would be to accept loud frames regardless of periodicity,
  which is what produces false onsets on line noise.

The energy detector is run over the same signals for comparison. It is not a
candidate for this channel -- the service bench's rule already forbids it -- but a number
showing why is worth more than the rule alone.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from service import detector as energy_detector
from service.audio import pcm_to_ulaw, resample, ulaw_to_pcm
from agent import detector as phone_detector

WIDE_RATE = 16000
PHONE_RATE = 8000
TRIALS = 60
SEED = 20260919


def voiced(duration_s: float, rate: int) -> np.ndarray:
    """Voiced-speech surrogate: harmonic stack, syllabic envelope, consonant attack.

    A pure tone would flatter both detectors -- instantaneous attack, constant
    energy. Real speech starts softly and fluctuates, and the envelope is what
    makes this a fair test of the hysteresis rather than of the threshold.
    """
    t = np.arange(int(duration_s * rate)) / rate
    f0 = 120.0
    signal = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 12))
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t - np.pi / 2)
    attack = np.clip(t / 0.015, 0, 1)
    return signal * syllables * attack


def fricative(duration_s: float, rate: int, rng: np.random.Generator) -> np.ndarray:
    """An /s/-like burst: band-limited noise, no periodicity, ordinary loudness."""
    noise = rng.normal(0, 1.0, int(duration_s * rate))
    # Crude high-pass: a fricative's energy sits above the voiced band.
    return noise - np.convolve(noise, np.ones(8) / 8, mode="same")


def line_noise(size: int, rate: int, rms: float, rng: np.random.Generator) -> np.ndarray:
    """Comfort noise plus mains hum, the two things always on a phone leg."""
    t = np.arange(size) / rate
    hum = 0.35 * np.sin(2 * np.pi * 60.0 * t) + 0.15 * np.sin(2 * np.pi * 180.0 * t)
    return (rng.normal(0, 1.0, size) + hum) * rms


def through_the_phone(wide: np.ndarray, snr_db: float, rng: np.random.Generator) -> bytes:
    """The wideband signal as it arrives on the far side of the call."""
    pcm = (np.clip(wide, -1, 1) * 32767).astype(np.int16).tobytes()
    narrow = np.frombuffer(resample(pcm, WIDE_RATE, PHONE_RATE), dtype=np.int16).astype(np.float64)
    narrow /= 32768.0

    speech_rms = np.sqrt(np.mean(narrow[narrow != 0] ** 2)) if np.any(narrow) else 1e-6
    noisy = narrow + line_noise(narrow.size, PHONE_RATE, speech_rms / (10 ** (snr_db / 20.0)), rng)
    coded = (np.clip(noisy, -1, 1) * 32767).astype(np.int16).tobytes()
    return ulaw_to_pcm(pcm_to_ulaw(coded))           # the codec the leg actually uses


def build(onset_s: float, speech_s: float, tail_s: float, unvoiced: bool,
          rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    speech = voiced(speech_s, WIDE_RATE)
    speech = speech / np.abs(speech).max() * 0.35
    if unvoiced:
        burst = fricative(0.12, WIDE_RATE, rng)
        burst = burst / np.abs(burst).max() * 0.30
        speech = np.concatenate([burst, speech])
    clip = np.concatenate([np.zeros(int(onset_s * WIDE_RATE)), speech, np.zeros(int(tail_s * WIDE_RATE))])
    return clip, onset_s * 1000, onset_s * 1000 + speech.size / WIDE_RATE * 1000


def run(snr_db: float, unvoiced: bool, trials: int = TRIALS) -> dict:
    rng = np.random.default_rng(SEED)
    results: dict[str, dict[str, list[float]]] = {
        "phone": {"on": [], "off": [], "miss": []},
        "energy": {"on": [], "off": [], "miss": []},
    }
    for _ in range(trials):
        clip, true_on, true_off = build(
            rng.uniform(0.05, 0.60), rng.uniform(0.8, 2.0), 0.6, unvoiced, rng
        )
        pcm = through_the_phone(clip, snr_db, rng)
        for name, module in (("phone", phone_detector), ("energy", energy_detector)):
            on = module.detect_onset(pcm, PHONE_RATE)
            off = module.detect_offset(pcm, PHONE_RATE)
            if not on.found or not off.found:
                results[name]["miss"].append(1.0)
                continue
            results[name]["on"].append(on.time_ms - true_on)
            results[name]["off"].append(off.time_ms - true_off)

    def stats(r: dict[str, list[float]]) -> dict:
        if not r["on"]:
            return {"n": 0, "misses": len(r["miss"])}
        return {
            "n": len(r["on"]),
            "misses": len(r["miss"]),
            "onset_bias_ms": round(float(np.mean(r["on"])), 1),
            "onset_p95_ms": round(float(np.percentile(np.abs(r["on"]), 95)), 1),
            "offset_bias_ms": round(float(np.mean(r["off"])), 1),
            "offset_p95_ms": round(float(np.percentile(np.abs(r["off"]), 95)), 1),
        }

    return {
        "snr_db": snr_db,
        "onset": "unvoiced" if unvoiced else "voiced",
        "phone_detector": stats(results["phone"]),
        "energy_detector": stats(results["energy"]),
    }


def false_onsets(trials: int = TRIALS) -> dict:
    """How often each detector hears speech on a line where nobody spoke.

    This is the measurement the phone detector exists for, and the one a
    bias-and-P95 table hides completely. A late boundary is an error with a size;
    a boundary invented out of line noise is a reply latency measured from
    nothing, and it lands in the published percentiles looking like data.

    The buffers carry what a real leg carries when the caller is silent --
    comfort noise, mains hum, and the transients a line produces on its own:
    switching clicks, a burst of crosstalk, a dropout returning.
    """
    rng = np.random.default_rng(SEED)
    counts = {"phone": 0, "energy": 0}
    for _ in range(trials):
        size = int(3.0 * PHONE_RATE)
        buffer = line_noise(size, PHONE_RATE, 0.02, rng)
        for _ in range(rng.integers(1, 4)):          # transients, not speech
            at = int(rng.uniform(0.2, 2.6) * PHONE_RATE)
            width = int(rng.uniform(0.005, 0.040) * PHONE_RATE)
            buffer[at : at + width] += rng.normal(0, 0.25, min(width, size - at))
        pcm = ulaw_to_pcm(pcm_to_ulaw((np.clip(buffer, -1, 1) * 32767).astype(np.int16).tobytes()))
        for name, module in (("phone", phone_detector), ("energy", energy_detector)):
            if module.detect_onset(pcm, PHONE_RATE).found:
                counts[name] += 1
    return {
        "case": "silent line with transients",
        "trials": trials,
        "phone_detector_false_onsets": counts["phone"],
        "energy_detector_false_onsets": counts["energy"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, action="append", help="dB; repeatable")
    parser.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args()
    rows = [
        run(snr, unvoiced, args.trials)
        for snr in (args.snr or [30.0, 20.0, 10.0, 5.0, 0.0])
        for unvoiced in (False, True)
    ]
    rows.append(false_onsets(args.trials))
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
