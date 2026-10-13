import numpy as np

from turn_bench.adapters.rms_vad import RmsVad
from turn_bench.model import Event
from turn_bench.runner import stream_speaker

SAMPLE_RATE = 16000


def silence_tone_silence(seconds=(1.0, 1.0, 1.0), amplitude=0.5):
    """1 s silence, 1 s of a 440 Hz tone, 1 s silence, at 16 kHz float32."""
    before, during, after = (int(s * SAMPLE_RATE) for s in seconds)
    t = np.arange(during) / SAMPLE_RATE
    tone = amplitude * np.sin(2 * np.pi * 440.0 * t)
    return np.concatenate([np.zeros(before), tone, np.zeros(after)]).astype(np.float32)


def test_onset_is_interruption_and_offset_is_eot_at_window_ends():
    run = stream_speaker(RmsVad(), silence_tone_silence(), SAMPLE_RATE)
    # Window 50 (1.00-1.02 s) is the first loud one: the flip is known at its end.
    # Window 100 (2.00-2.02 s) is the first quiet one after the tone.
    assert run.events == [Event("interruption", 51 * 0.02), Event("eot", 101 * 0.02)]
    assert run.frames == 150


def test_first_window_never_flips_and_quiet_signal_emits_nothing():
    # Audio that starts loud: the reference loop starts at window 1, so the
    # loud first window seeds the state and emits no event.
    loud = (0.5 * np.ones(SAMPLE_RATE)).astype(np.float32)
    assert stream_speaker(RmsVad(), loud, SAMPLE_RATE).events == []
    quiet = (0.005 * np.ones(SAMPLE_RATE)).astype(np.float32)  # RMS below 0.01
    assert stream_speaker(RmsVad(), quiet, SAMPLE_RATE).events == []
