"""A streaming form of the service bench's published resampler.

``service.audio.resample`` is the one filter every provider's audio goes
through in the service bench (Kaiser-windowed polyphase FIR, ``np.convolve``
in ``same`` mode over the whole clip). The caller channel reaches an S2S model
one 20 ms frame at a time, and resampling each frame on its own would put a
filter edge — a click — every 20 ms. ``StreamingResampler`` keeps the filter
and the arithmetic of the published function and only changes *when* the
output is produced: a sample is emitted as soon as every input it depends on
has arrived, so the stream is the published function's output delayed by half
the filter (``delay_s``: about 1.3 ms for 48 kHz → 24 kHz, 2 ms for 48 kHz →
16 kHz). ``tests/test_s2s_resample.py`` checks the two agree sample for sample.

The filter design below is copied from ``service.audio.resample`` rather than
imported because that function computes the taps inside its body; the test
pins the two together.
"""

from __future__ import annotations

from math import gcd

import numpy as np


def design(src_rate: int, dst_rate: int) -> tuple[np.ndarray, int, int]:
    """``(taps, up, down)`` exactly as ``service.audio.resample`` builds them."""
    divisor = gcd(src_rate, dst_rate)
    up, down = dst_rate // divisor, src_rate // divisor
    taps = 2 * 32 * max(up, down) + 1
    n = np.arange(taps) - (taps - 1) / 2.0
    cutoff = 1.0 / max(up, down)
    h = np.sinc(cutoff * n) * np.kaiser(taps, 8.6)
    h *= up / h.sum()
    return h, up, down


class StreamingResampler:
    """Rational resampler fed in pieces; ``process`` returns float64 samples at
    ``dst_rate`` and ``flush`` the tail the published filter would pad with zeros."""

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.identity = src_rate == dst_rate
        if self.identity:
            self.h, self.up, self.down, self.half = np.ones(1), 1, 1, 0
        else:
            self.h, self.up, self.down = design(src_rate, dst_rate)
            self.half = (len(self.h) - 1) // 2
        self._buffer = np.zeros(0, dtype=np.float64)  # upsampled ("stuffed") input not yet consumed
        self._buffer_start = 0  # absolute stuffed index of _buffer[0]
        self._stuffed_total = 0  # stuffed samples received so far
        self._next_output = 0  # next output index m (output m = filtered[m * down])

    @property
    def delay_s(self) -> float:
        """How far the stream lags the published whole-clip function."""
        return self.half / (self.src_rate * self.up)

    def process(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float64)
        if self.identity:
            return samples
        if samples.size:
            stuffed = np.zeros(samples.size * self.up, dtype=np.float64)
            stuffed[:: self.up] = samples
            self._buffer = np.concatenate([self._buffer, stuffed])
            self._stuffed_total += stuffed.size
        return self._emit()

    def flush(self) -> np.ndarray:
        """The outputs that still need future input, computed as the published
        function would: with zeros past the end. Call once, at the end."""
        if self.identity:
            return np.zeros(0)
        # Outputs exist for every stuffed index < total (``same`` mode keeps the
        # input length); feed the zeros that complete their windows.
        target = self._stuffed_total
        pad = max(0, target - 1 + self.half - (self._buffer_start + self._buffer.size) + 1)
        if pad:
            self._buffer = np.concatenate([self._buffer, np.zeros(pad)])
        out = self._emit(limit_total=target)
        return out

    def _emit(self, limit_total: int | None = None) -> np.ndarray:
        available = self._buffer_start + self._buffer.size  # stuffed samples present (incl. flush zeros)
        last_index = available - 1 - self.half  # last filtered index whose window is complete
        bound = self._stuffed_total if limit_total is None else limit_total
        last_index = min(last_index, bound - 1)
        first_index = self._next_output * self.down
        if last_index < first_index:
            return np.zeros(0)
        last_output = last_index // self.down
        last_index = last_output * self.down
        # filtered[i] = sum_j h[j] * stuffed[i + half - j]: window stuffed[i - half .. i + half]
        seg_start = first_index - self.half
        seg_end = last_index + self.half  # inclusive
        lead = max(0, -seg_start)
        seg = self._buffer[max(0, seg_start) - self._buffer_start : seg_end - self._buffer_start + 1]
        if lead:
            seg = np.concatenate([np.zeros(lead), seg])
        filtered = np.convolve(seg, self.h, mode="valid")  # filtered[k] = y[first_index + k]
        out = filtered[:: self.down]
        self._next_output = last_output + 1
        keep_from = self._next_output * self.down - self.half
        drop = max(0, keep_from - self._buffer_start)
        if drop:
            self._buffer = self._buffer[drop:]
            self._buffer_start += drop
        return out
