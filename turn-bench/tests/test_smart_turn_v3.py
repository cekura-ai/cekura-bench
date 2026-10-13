"""Smart Turn v3 adapter: the ported pieces against their references, and the
streamed pipeline's causality. Tests that need the `smart-turn` extra or the
assets (`turn-bench setup smart_turn_v3`) skip when those are absent."""

import numpy as np
import pytest
from turnbench.sweep import commit_events

from turn_bench import assets
from turn_bench.adapters import smart_turn_v3
from turn_bench.adapters.smart_turn_v3 import OUT_HZ, EdgeCommitter, SmartTurnV3
from turn_bench.model import FRAME_S
from turn_bench.runner import stream_speaker

SAMPLE_RATE = 48000


def _extra_installed() -> bool:
    try:
        import onnxruntime, torch, torchaudio, transformers  # noqa: F401
    except ImportError:
        return False
    return True


needs_extra = pytest.mark.skipif(not _extra_installed(), reason="smart-turn extra not installed")
needs_assets = pytest.mark.skipif(
    not _extra_installed()
    or assets.verify(assets.assets_dir(smart_turn_v3.MODEL_NAME), smart_turn_v3.ASSETS),
    reason="smart-turn extra and `turn-bench setup smart_turn_v3` required",
)


def vowel_like(seconds: float, f0: float = 120.0, syllable_hz: float = 4.0) -> np.ndarray:
    """A harmonic series under three formant bumps with a syllable-rate envelope:
    enough like speech for Silero to open a segment, so the classifier runs."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for harmonic in range(1, 40):
        amplitude = sum(np.exp(-0.5 * ((harmonic * f0 - fc) / bw) ** 2) for fc, bw in ((600, 80), (1200, 100), (2500, 120)))
        signal += amplitude * np.sin(harmonic * phase)
    signal *= 0.5 * (1 + np.sin(2 * np.pi * syllable_hz * t - np.pi / 2)) ** 1.5
    return (0.3 * signal / np.max(np.abs(signal))).astype(np.float32)


def silence_vowel_silence() -> np.ndarray:
    return np.concatenate(
        [np.zeros(SAMPLE_RATE), vowel_like(2.0), np.zeros(3 * SAMPLE_RATE)]
    ).astype(np.float32)


def test_edge_committer_matches_sesame_commit_events_frame_by_frame():
    rng = np.random.default_rng(0)
    for _ in range(50):
        probs = (rng.random(400) * 0.02).astype(np.float32).tolist()
        theta = float(rng.random() * 0.02)
        committer = EdgeCommitter(theta, OUT_HZ)
        ours = [(k + 1) / OUT_HZ for k, p in enumerate(probs) if committer.step(k, p)]
        assert ours == commit_events(probs, OUT_HZ, theta)


@needs_extra
def test_streaming_resampler_is_bit_identical_to_torchaudio_batch_resample():
    import torch
    import torchaudio

    rng = np.random.default_rng(1)
    seconds = 2.0
    x = (0.5 * rng.standard_normal(int(seconds * SAMPLE_RATE)) + 0.2 * np.sin(np.arange(int(seconds * SAMPLE_RATE)) * 0.05)).astype(np.float32)
    frame = int(FRAME_S * SAMPLE_RATE)
    resampler = smart_turn_v3.StreamingResampler(SAMPLE_RATE, smart_turn_v3.SR)
    pieces = [resampler.push(x[i : i + frame]) for i in range(0, len(x), frame)]
    pieces.append(resampler.flush())
    streamed = np.concatenate(pieces)
    batch = torchaudio.functional.resample(torch.from_numpy(x), SAMPLE_RATE, smart_turn_v3.SR).numpy()
    assert streamed.shape == batch.shape
    assert np.array_equal(streamed, batch)


@needs_assets
def test_streamed_pipeline_commits_causally_in_both_timing_modes():
    audio = silence_vowel_silence()
    reference = stream_speaker(SmartTurnV3(commit_time="reference"), audio, SAMPLE_RATE)
    causal_model = SmartTurnV3(commit_time="causal")
    causal = stream_speaker(causal_model, audio, SAMPLE_RATE)  # runner raises CausalityError otherwise

    # The reference pipeline always fires INT on its first frame (1 - 0 > theta_int),
    # and the vowel burst must have opened a segment that the classifier scored.
    assert reference.events[0].kind == "interruption" and reference.events[0].commit_time_s == 1 / OUT_HZ
    assert any(event.kind == "eot" for event in reference.events)
    diagnostics = causal_model.diagnostics()
    assert diagnostics["predict_calls"] > 0
    assert len(diagnostics["scores"]) == causal.frames // 4  # one score per whole 80 ms frame

    # Same decisions in both modes; the causal stamp is the 20 ms frame end in
    # which the decision became available: 20-40 ms after the reference's frame end.
    assert [e.kind for e in causal.events] == [e.kind for e in reference.events]
    for ours, theirs in zip(causal.events, reference.events):
        assert theirs.commit_time_s <= ours.commit_time_s <= theirs.commit_time_s + 2 * FRAME_S + 1e-9
        assert ours.commit_time_s <= causal.audio_time_s

