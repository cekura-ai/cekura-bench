"""Pinned offline VAD evidence; model execution is mocked only at its boundary."""
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from turn_taking_bench.adapters import smart_turn_v3
from turn_taking_bench.s2s.authored_vad import SpeechVeto


def test_missing_or_wrong_model_fails_before_constructing_inference(tmp_path, monkeypatch):
    constructed = []
    monkeypatch.setattr(smart_turn_v3, "SileroVad", lambda *a, **k: constructed.append(True))
    with pytest.raises(ValueError, match="missing"):
        SpeechVeto(tmp_path / "missing.onnx")
    model = tmp_path / "incorrect.onnx"
    model.write_bytes(b"not the pinned model")
    with pytest.raises(ValueError, match="SHA-256"):
        SpeechVeto(model)
    assert not constructed


def fake_veto(tmp_path, monkeypatch):
    model = tmp_path / "offline-fixture.onnx"
    model.write_bytes(b"offline model boundary fixture")
    monkeypatch.setattr(smart_turn_v3, "SILERO_VAD", replace(smart_turn_v3.SILERO_VAD,
        sha256=hashlib.sha256(model.read_bytes()).hexdigest()))

    class FakeVad:
        def __init__(self, path, reset_s):
            assert path == model.resolve() and reset_s == 0
            self.resets = 0
            self.inputs = []

        def reset(self):
            self.resets += 1

        def prob(self, samples, end_s):
            assert samples.shape == (512,) and samples.dtype == np.float32
            self.inputs.append((samples.copy(), end_s))
            return 0.75

    monkeypatch.setattr(smart_turn_v3, "SileroVad", FakeVad)
    return SpeechVeto(model)


def test_full_chunks_normalization_timestamps_and_state_reset(tmp_path, monkeypatch):
    veto = fake_veto(tmp_path, monkeypatch)
    signal = np.tile(np.array([-32768, 0, 16384, 32767], dtype="<i2"), 260)
    evidence = veto.analyze(signal.tobytes(), 16000, start_s=3)
    assert evidence["chunks"] == [
        {"start_s": 3.0, "end_s": 3.032, "probability": 0.75},
        {"start_s": 3.032, "end_s": 3.064, "probability": 0.75}]
    np.testing.assert_array_equal(veto._vad.inputs[0][0][:4], [-1, 0, 0.5, 32767 / 32768])
    assert evidence["provenance"]["omitted_tail_samples_16k"] == 16
    assert evidence["provenance"]["model_sha256"] == smart_turn_v3.SILERO_VAD.sha256
    assert evidence["provenance"]["threshold"] == 0.5
    assert evidence["provenance"]["resampling"] == "none"
    assert veto.analyze(b"\0\0" * 511, 16000)["chunks"] == []
    assert veto._vad.resets == 2


def test_linear_resampling_uses_observed_support_and_omits_partial_chunks(tmp_path, monkeypatch):
    veto = fake_veto(tmp_path, monkeypatch)
    signal = np.arange(769, dtype="<i2")
    result = veto.analyze(signal.tobytes(), 24000)
    expected = np.interp(np.arange(513) * 1.5, np.arange(769), signal / 32768).astype(np.float32)
    np.testing.assert_array_equal(veto._vad.inputs[0][0], expected[:512])
    assert len(result["chunks"]) == 1
    assert result["provenance"]["resampled_samples"] == 513
    assert result["provenance"]["omitted_tail_samples_16k"] == 1
    assert "linear interpolation" in result["provenance"]["resampling"]
    # Upsampling must not extrapolate a final value to fabricate a full chunk.
    assert veto.analyze(b"\0\0" * 256, 8000)["chunks"] == []


@pytest.mark.parametrize("pcm,rate,start", [(b"odd", 16000, 0), (bytearray(1024), 16000, 0),
    (b"", 0, 0), (b"", True, 0), (b"", 16000.0, 0), (b"", 16000, float("nan")),
    (b"", 16000, -1)])
def test_invalid_pcm_contract_rejected(tmp_path, monkeypatch, pcm, rate, start):
    veto = fake_veto(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        veto.analyze(pcm, rate, start_s=start)
