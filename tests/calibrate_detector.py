"""Measure the detector's own error against signals whose boundaries are exact.

Nothing here touches a provider. The signal is constructed, so the true onset
and offset are known to the sample, and the difference between those and the
detector's answer is the resolution floor for every latency the benchmark
publishes. A gap between two providers smaller than this number is not a result.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lane_a.detector import detect_offset, detect_onset  # noqa: E402

RATE = 16000
TRIALS = 60


def speech_like(duration_s: float, rate: int, rng: np.random.Generator) -> np.ndarray:
    """Voiced-speech surrogate: harmonic stack, formant tilt, syllabic envelope.

    A pure tone would flatter the detector — it has an instantaneous attack and
    constant energy. Real speech starts on a consonant and fluctuates, so the
    envelope is what makes this a fair test of the hysteresis settings.
    """
    t = np.arange(int(duration_s * rate)) / rate
    f0 = 120.0
    signal = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 12))
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t - np.pi / 2)
    attack = np.clip(t / 0.015, 0, 1)          # 15 ms consonant onset
    return signal * syllables * attack


def build(true_onset_s: float, speech_s: float, tail_s: float, snr_db: float | None,
          rng: np.random.Generator) -> tuple[bytes, float, float]:
    lead = np.zeros(int(true_onset_s * RATE))
    speech = speech_like(speech_s, RATE, rng)
    speech = speech / np.abs(speech).max() * 0.35
    tail = np.zeros(int(tail_s * RATE))
    clip = np.concatenate([lead, speech, tail])

    if snr_db is not None:
        speech_rms = np.sqrt(np.mean(speech**2))
        noise = rng.normal(0, speech_rms / (10 ** (snr_db / 20.0)), clip.size)
        clip = clip + noise

    pcm = (np.clip(clip, -1, 1) * 32767).astype(np.int16).tobytes()
    return pcm, true_onset_s * 1000, (true_onset_s + speech_s) * 1000


def run(snr_db: float | None) -> dict:
    rng = np.random.default_rng(20260918)
    on_err, off_err, misses = [], [], 0
    for _ in range(TRIALS):
        onset_s = rng.uniform(0.05, 0.60)
        pcm, true_on, true_off = build(onset_s, rng.uniform(0.8, 2.0), 0.6, snr_db, rng)
        d_on, d_off = detect_onset(pcm, RATE), detect_offset(pcm, RATE)
        if not d_on.found or not d_off.found:
            misses += 1
            continue
        on_err.append(d_on.time_ms - true_on)
        off_err.append(d_off.time_ms - true_off)
    label = "clean" if snr_db is None else f"{snr_db:g} dB"
    if not on_err:
        return {"snr": label, "misses": misses, "n": 0}
    return {
        "snr": label, "misses": misses, "n": len(on_err),
        "on_bias": np.mean(on_err), "on_p95": np.percentile(np.abs(on_err), 95),
        "off_bias": np.mean(off_err), "off_p95": np.percentile(np.abs(off_err), 95),
    }


print(f"{'SNR':>8} {'n':>4} {'miss':>5} | {'onset bias':>11} {'|err| p95':>10} "
      f"| {'offset bias':>12} {'|err| p95':>10}")
print("-" * 72)
for snr in (None, 30, 20, 10, 5, 0):
    r = run(snr)
    if not r["n"]:
        print(f"{r['snr']:>8} {0:>4} {r['misses']:>5} |  no reliable detection")
        continue
    print(f"{r['snr']:>8} {r['n']:>4} {r['misses']:>5} | {r['on_bias']:>10.1f}ms "
          f"{r['on_p95']:>9.1f}ms | {r['off_bias']:>11.1f}ms {r['off_p95']:>9.1f}ms")
