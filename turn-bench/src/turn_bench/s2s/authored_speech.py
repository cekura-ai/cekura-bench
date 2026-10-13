"""Track 2 v0.7: did the model STOP SPEAKING? (speech cessation, not output quiet)

Computed on the reference client's heard audio (playout clock, cuts applied),
5 ms frames, pinned Silero VAD (stateful pass over a 1 s context + 1.6 s
tail; fresh-state passes for short sounds). Constants are in ``policy()``.

* Speaking at injection: the stateful VAD region (p>0.5 entry, p<0.35 exit,
  widened 250 ms before onsets) covers t, with an energetic speech frame in
  [t-30, t+20] ms (or VAD-positive audible energy when noise masks it).
* Energy threshold convention: max(-50 dBFS lowered for quieter speech,
  model floor + 6 dB). The floor is taken from VAD non-speech frames OUTSIDE
  the speech region (its 250 ms pre-onset pad excluded: Silero lags onsets).
  Isolated bursts < 30 ms are transients (clicks).
* Sounds, not frame runs: runs of energetic frames separated by < 50 ms
  (closures, soft fricatives) are one sound, judged together (v0.7b).
* Stop = start of >= 200 ms without speech after the speech at injection:
  - silent path: 200 ms below the silent level (-60 dBFS, level-relative);
  - VAD path: the stateful VAD RELEASES (p<0.35); that decides THAT speech
    ended. The residual level after the release (25th pct, the model's own
    floor) + 6 dB places WHERE (last energetic frame before the release, plus
    the sound continuous with it across < 50 ms dips). After that, EVERY
    sound counts, inside or outside the stateful VAD region (a playout cut can
    end a word's onset before Silero re-enters). Isolated short sounds
    (<=300 ms, >=50 ms after the previous sound) and sounds the stateful VAD
    never entered are judged alone by a fresh VAD pass and spectral flatness:
    clearly non-speech (p<=0.15, flat) is ignored (breaths, clicks, noise);
    uncertain -> exclusion.
* Exclusions (never guesses): residual noise within 3 dB of the speech level;
  a residual never observed absent before the injection (no pre-injection
  output 3 dB quieter than it: it may have been under the speech all along);
  a raised threshold over a NON-stationary residual (p90-p10 > 3 dB: softer
  sound can hide under it); masked pause lengths; unobservable release; a
  "kept speaking" verdict without a fresh-VAD speech confirmation in the last
  400 ms of the window.
* Latency is retained only when the threshold was not raised by residual
  sound (residual <= -50 dBFS and threshold <= convention + 1 dB).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from turn_bench.s2s.authored_vad import CHUNK, RATE as VAD_RATE, fresh_probs, local_probs, to_16k

SCHEMA = "turn-bench/track2-speech-cessation/4"
FRAME_S = 0.005
OBSERVATION_S = 1.2
WINDOW_S = 1.0
QUIET_S = 0.2
CONTEXT_S = 1.0
ABS_THRESHOLD_DB = -50.0
SILENT_DB = -60.0
MARGIN_DB = 6.0
LEVEL_REF_DB = -20.0  # speech at/above -20 dBFS uses the absolute thresholds as declared
FLOOR_MIN_FRAMES = 20
TIMING_MAX_FLOOR_DB = -50.0
VAD_ON, VAD_OFF = 0.5, 0.35
LOCAL_CHUNKS = 6
STATIONARY_DB = 3.0
SNR_MIN_DB = 3.0
BACKGROUND_DB = 3.0
FLOOR_PCT = 25
BURST_MAX_P = 0.15
BURST_MIN_FLATNESS = 0.3
MASKED_PAUSE_MARGIN_S = 0.15
GLOBAL_FLOOR_GAP_DB = 15.0
DECAY_S = 0.256
RELEASE_P = 0.85
ANALYSIS_TAIL_S = 1.6
CHUNK_S = 0.032
VAD_PAD_BEFORE_S, VAD_PAD_AFTER_S = 0.25, 0.032
TRANSIENT_MAX_S, TRANSIENT_ISOLATION_S = 0.03, 0.05
OVERLAP_PRE_S, OVERLAP_POST_S = 0.03, 0.02
DISAGREE_MIN_S, DISAGREE_LEVEL_DB, DISAGREE_MOD_DB = 0.1, 15.0, 3.0
RESIDUAL_SPREAD_DB = 3.0


def policy() -> dict[str, Any]:
    return {"schema": SCHEMA, "metric": "speech cessation (model stopped speaking)",
            "frame_ms": FRAME_S * 1000, "observation_ms": OBSERVATION_S * 1000,
            "stop_window_ms": WINDOW_S * 1000, "speech_free_ms": QUIET_S * 1000,
            "vad_context_ms": CONTEXT_S * 1000, "abs_threshold_dbfs": ABS_THRESHOLD_DB,
            "silent_dbfs": SILENT_DB, "floor_margin_db": MARGIN_DB,
            "level_rule": "abs/silent thresholds lowered by (VAD-positive median level - (-20 dBFS)) when the speech is quieter than -20 dBFS",
            "floor": "median energy of frames in stateful-VAD p<0.35 chunks outside the speech region (pre-onset pad excluded); <100 ms of them -> no floor",
            "sound_grouping_ms": TRANSIENT_ISOLATION_S * 1000,
            "residual_spread_max_db": RESIDUAL_SPREAD_DB,
            "background_rule": f"masked residual excluded unless pre-injection p5 <= residual - {BACKGROUND_DB} dB",
            "timing_max_floor_dbfs": TIMING_MAX_FLOOR_DB,
            "vad": {"speech": f"p > {VAD_ON}", "nonspeech_floor": f"p < {VAD_OFF}",
                    "labels": f"fresh-state Silero over the last {LOCAL_CHUNKS} x 32 ms chunks (no long hangover)",
                    "confirm": "fresh-state Silero over the candidate 200 ms has no chunk > 0.5",
                    "pad_before_ms": VAD_PAD_BEFORE_S * 1000, "pad_after_ms": VAD_PAD_AFTER_S * 1000},
            "transient": {"max_ms": TRANSIENT_MAX_S * 1000, "isolation_ms": TRANSIENT_ISOLATION_S * 1000},
            "overlap_ms": [-OVERLAP_PRE_S * 1000, OVERLAP_POST_S * 1000],
            "disagreement": {"min_ms": DISAGREE_MIN_S * 1000, "within_speech_level_db": DISAGREE_LEVEL_DB,
                             "min_modulation_db": DISAGREE_MOD_DB},
            "boundary": "first transmitted caller PCM sample (injection); not acoustic onset"}


def heard(adapter: Any, begin: float, end: float) -> np.ndarray:
    """Heard agent PCM on the playout clock over [begin, end); unheard = 0."""
    timeline = adapter.agent_timeline
    rate = timeline.rate
    values = np.zeros(max(0, int(round((end - begin) * rate))), dtype=np.int16)
    pcm = adapter.agent_pcm
    for chunk in timeline.chunks:
        left0, right0 = timeline.playout_span(chunk.seq)
        left, right = max(left0, begin), min(right0, end)
        if right <= left:
            continue
        dst = max(0, int(round((left - begin) * rate)))
        src = chunk.first_sample + max(0, int(round((left - left0) * rate)))
        length = min(int(round((right - left) * rate)), len(values) - dst, chunk.last_sample - src)
        if length > 0:
            # copy only this slice: never hold a view of a live, growing buffer
            part = np.frombuffer(bytes(pcm[src * 2:(src + length) * 2]), dtype="<i2")
            values[dst:dst + len(part)] = part
    return values


def _energy(x: np.ndarray, rate: int) -> np.ndarray:
    n = int(round(rate * FRAME_S))
    usable = len(x) // n * n
    if not usable:
        return np.zeros(0)
    frames = x[:usable].astype(np.float64).reshape(-1, n)
    rms = np.sqrt((frames * frames).mean(axis=1)) / 32768.0
    return np.maximum(20 * np.log10(np.maximum(rms, 1e-9)), -120.0)


def _runs(mask: np.ndarray):
    """(start, end) of True runs."""
    if not len(mask):
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def _groups(mask: np.ndarray, gap: int):
    """Sounds: True runs joined across gaps shorter than ``gap`` frames.

    A word is not a run of frames: stop closures and soft fricatives leave
    5-40 ms dips inside it. Sounds separated by less than the transient
    isolation are one sound (judged together), never separate bursts.
    """
    out: list[list[int]] = []
    for a, b in _runs(mask):
        if out and a - out[-1][1] < gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _chain_end(act: np.ndarray, start: int, gap: int) -> int:
    """One past the end of the sound sounding at ``start`` (gaps < ``gap`` joined).

    ``start`` itself if no sound is active at it or ended less than ``gap``
    frames before it.
    """
    n = len(act)
    if start >= n:
        return start
    back = np.flatnonzero(act[max(0, start - gap):start + 1])
    if not len(back):
        return start
    end = max(0, start - gap) + int(back[-1]) + 1
    while end < n:
        if act[end]:
            end += 1
            continue
        ahead = np.flatnonzero(act[end:end + gap])
        if not len(ahead):
            break
        end += int(ahead[0])
    return max(end, start)


class Analysis:
    """Speech evidence for heard audio x starting at absolute time t0.

    Two Silero views of the same audio: a STATEFUL pass (reliable inside
    speech, but its recurrent state lingers for up to ~0.3 s over residual
    noise after speech ends) and FRESH-state windows of ``LOCAL_CHUNKS`` chunks
    (a window is negative only if it holds no speech: no lingering state).
    """

    def __init__(self, x: np.ndarray, rate: int, t0: float, vad: Any, level_until: float | None = None):
        if rate % 200:
            raise ValueError("agent rate must be a multiple of 200 Hz for 5 ms frames")
        self.t0, self.rate, self.vad = t0, rate, vad
        self.energy = _energy(x, rate)
        n = len(self.energy)
        x16 = to_16k(x, rate) if len(x) else np.zeros(0, np.float32)
        self.x16 = x16
        self.nonspeech_bursts: list[tuple[int, int]] = []
        self.uncertain_bursts: list[tuple[int, int]] = []
        self.state = fresh_probs(vad, x16) if len(x16) else np.zeros(0)
        per = CHUNK_S / FRAME_S
        self._fr = lambda k: (int(round(k * per)), int(round((k + 1) * per)))
        neg, pos, region = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        inside, begin = False, 0
        pad = int(round(VAD_PAD_BEFORE_S / FRAME_S)); pad_after = int(round(VAD_PAD_AFTER_S / FRAME_S))
        for k, p in enumerate(self.state):
            a, b = self._fr(k)
            if p < VAD_OFF:
                neg[a:b] = True
            if p > VAD_ON:
                pos[a:b] = True
            if not inside and p > VAD_ON:
                inside, begin = True, a
            elif inside and p < VAD_OFF:
                region[max(0, begin - pad):a + pad_after] = True
                inside = False
        if inside:
            region[max(0, begin - pad):] = True
        # Floor frames: VAD non-speech OUTSIDE the speech region. The region's
        # 250 ms pre-onset pad is excluded because Silero lags speech onsets
        # (a clip's leading ramp is speech the VAD has not caught yet).
        floor_mask = neg & ~region
        self.neg_frames = int(floor_mask.sum())
        # Speech level from VAD-positive frames up to the injection (the model's
        # level while it is speaking), not the lingering state over later noise.
        lvl = pos.copy()
        if level_until is not None:
            lvl[max(0, int(round((level_until - t0) / FRAME_S))):] = False
        if lvl.sum() < FLOOR_MIN_FRAMES:
            lvl = pos
        self.vad_level = float(np.median(self.energy[lvl])) if lvl.sum() >= FLOOR_MIN_FRAMES else None
        # the quiet end of the model's own speech: residual noise above it can
        # hide soft speech, so cessation into such noise is not identifiable
        pre = self.energy[:max(0, int(round((level_until - t0) / FRAME_S)))] if level_until is not None else self.energy[:0]
        # the quietest pre-injection output: residual sound louder than it may
        # have been under the speech all along (it was never seen absent)
        self.pre_p5 = float(np.percentile(pre, 5)) if len(pre) >= 40 else None
        rel = 0.0 if self.vad_level is None else min(0.0, self.vad_level - LEVEL_REF_DB)
        self.abs_threshold = ABS_THRESHOLD_DB + rel
        self.silent_db = SILENT_DB + rel
        self.floor = float(np.median(self.energy[floor_mask])) if self.neg_frames >= FLOOR_MIN_FRAMES else None
        if self.floor is not None and self.vad_level is not None and self.floor > self.vad_level - GLOBAL_FLOOR_GAP_DB:
            self.floor = None  # "non-speech" frames at speech level: not a usable floor
        self.threshold = (self.abs_threshold if self.floor is None
                          else max(self.abs_threshold, self.floor + MARGIN_DB))
        self.region, self.pos = region, pos
        self.active = _active(self, self.threshold)
        self.speech = self.active & region
        self.speech_level = float(np.median(self.energy[self.speech])) if self.speech.any() else None

    def frame(self, t: float) -> int:
        return int(round((t - self.t0) / FRAME_S))

    def time(self, i: int) -> float:
        return self.t0 + i * FRAME_S

    def speaking_at(self, t: float, causal: bool = False) -> bool:
        a = max(0, self.frame(t - OVERLAP_PRE_S))
        b = min(len(self.speech), self.frame(t + (0 if causal else OVERLAP_POST_S)))
        at = min(len(self.region) - 1, self.frame(t))
        if a >= b or at < 0 or not self.region[at]:
            return False  # the VAD must place the injection inside speech
        if self.speech[a:b].any():
            return True
        # Speech masked by residual output noise: VAD-positive with audible energy.
        return bool(self.pos[a:b].any() and (self.energy[a:b] >= self.abs_threshold).any())

    def negative_windows(self, i: int):
        """(start_frame, end_frame) of fresh windows with no speech, starting at/after frame i."""
        for k, p in enumerate(self.local):
            if p < VAD_ON:
                a = self._fr(k - LOCAL_CHUNKS + 1)[0]
                b = self._fr(k)[1]
                if a >= i:
                    yield a, b

    def state_decaying(self, q: int) -> bool:
        """Stateful VAD confidence has clearly fallen 128-256 ms after frame q."""
        a, b = q + int(round(0.128 / FRAME_S)), q + int(round(DECAY_S / FRAME_S))
        vals = [p for k, p in enumerate(self.state) if self._fr(k)[0] >= a and self._fr(k)[1] <= b + 1]
        return bool(vals) and float(np.mean(vals)) < RELEASE_P

    def state_released(self, i: int, j: int) -> bool | None:
        """Stateful VAD shows release (p < RELEASE_P) within frames [i, j]; None if unobserved."""
        for k, p in enumerate(self.state):
            a, b = self._fr(k)
            if a >= i and b <= j and p < RELEASE_P:
                return True
        return False if self._fr(len(self.state) - 1)[1] >= j else None


def _active(an: "Analysis", threshold: float) -> np.ndarray:
    active = an.energy >= threshold
    iso = int(round(TRANSIENT_ISOLATION_S / FRAME_S))
    n = len(active)
    for a, b in _runs(active):
        if (b - a) * FRAME_S < TRANSIENT_MAX_S - 1e-9:
            if (a == 0 or not active[max(0, a - iso):a].any()) and (b == n or not active[b:b + iso].any()):
                active[a:b] = False
    return active


def _burst_kind(an: "Analysis", prev_end: int, a: int, b: int) -> str:
    """Fresh-state VAD over an isolated short sound alone: speech / uncertain / nonspeech."""
    pad = int(round(0.032 / FRAME_S))
    lo, hi = max(prev_end + 2, a - pad), min(len(an.energy), b + pad)
    x16 = an.x16[int(round(lo * FRAME_S * 16000)):int(round(hi * FRAME_S * 16000))]
    peak = float(fresh_probs(an.vad, x16).max(initial=0.0))
    if peak > VAD_ON:
        return "speech"
    if peak > BURST_MAX_P or _flatness(x16) < BURST_MIN_FLATNESS:
        return "uncertain"  # not clearly non-speech: a tonal (voiced) sound, or the VAD hesitates
    return "nonspeech"


def _flatness(x16: np.ndarray) -> float:
    """Median spectral flatness of 32 ms frames: ~1 for noise/clicks, small for voiced speech."""
    if len(x16) < CHUNK:
        return 1.0
    frames = x16[:len(x16) // CHUNK * CHUNK].reshape(-1, CHUNK).astype(np.float64)
    power = np.abs(np.fft.rfft(frames * np.hanning(CHUNK), axis=1)) ** 2 + 1e-20
    flat = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    return float(np.median(flat))


def _last_speech_end(an: "Analysis", act: np.ndarray, i: int, r: int) -> int:
    """One past the last speech frame in [i, r). An isolated trailing burst
    (>= 50 ms after the previous sound, <= 300 ms long) inside the VAD's
    lingering state counts only if a fresh-state VAD pass over it finds speech:
    breaths, clicks and noise bursts after the last word are not speech."""
    iso = int(round(TRANSIENT_ISOLATION_S / FRAME_S))
    runs = [(a + i, b + i) for a, b in _groups(act[i:r], iso)]
    longest = int(round(0.3 / FRAME_S))
    while len(runs) >= 2:
        a, b = runs[-1]
        if a - runs[-2][1] < iso or b - a > longest:
            break
        kind = _burst_kind(an, runs[-2][1], a, b)
        if kind == "speech":
            break
        if kind == "uncertain":
            an.uncertain_bursts.append((a, b))
            break
        an.nonspeech_bursts.append(runs.pop())
    return runs[-1][1] if runs else i


def stop_after(an: Analysis, t: float, limit: float) -> dict[str, Any]:
    """First speech cessation at/after injection t (stop <= limit).

    Speech has ended when the stateful VAD RELEASES (p < 0.35) -- or, with no
    VAD needed, when the output is below the silent level for 200 ms. The
    release comes late over residual noise (recurrent hangover), so it only
    decides THAT speech ended; the boundary is placed by energy: the residual
    level after the release (the model's own floor) + 6 dB is the threshold,
    and the stop is one frame past the last above-threshold (non-transient)
    frame before the release. It counts if no speech frame follows for 200 ms.
    """
    quiet = int(round(QUIET_S / FRAME_S))
    n = len(an.speech)
    lo, hi = max(0, an.frame(t - OVERLAP_PRE_S)), min(n, an.frame(t + OVERLAP_POST_S))
    pre = an.speech[lo:hi]
    i0 = lo + int(np.flatnonzero(pre)[-1]) + 1 if pre.any() else an.frame(t)
    last_q = an.frame(limit)
    # 1) clean cessation: 200 ms below the silent level after a speech frame
    silent_stop, q = None, i0
    while q <= last_q and q + quiet <= n:
        nxt = np.flatnonzero(an.speech[q:q + quiet])
        if len(nxt):
            q += int(nxt[-1]) + 1
            continue
        if bool((an.energy[q:q + quiet] < an.silent_db).all()):
            silent_stop = q
            break
        q += 1
    # 2) cessation into residual sound: VAD release, energy-placed boundary
    found, i, bounded = None, i0, False
    # releases are the stateful VAD's speech->non-speech TRANSITIONS: a low
    # chunk inside a pause (e.g. while Silero lags the next onset) is not the
    # end of speech, and the 'residual' after it would be the next words
    releases = [an._fr(k)[0] for k, p in enumerate(an.state)
                if p < VAD_OFF and (k == 0 or an.state[k - 1] >= VAD_OFF)]
    for r in releases:
        if r <= i:
            continue
        if silent_stop is not None and i >= silent_stop:
            break
        if r + 10 > n:
            if i <= last_q:
                found = {"stop_s": None, "unconfirmed_candidate_s": an.time(i)}
            break
        # The release says THAT speech ended; energy places WHERE. Energy that
        # is still continuous with the speech at the release belongs to it.
        # first estimate of the residual level: frames after the release
        local = float(np.percentile(an.energy[r:min(n, r + quiet)], FLOOR_PCT))
        thr = max(an.abs_threshold, an.threshold, local + MARGIN_DB)
        iso = int(round(TRANSIENT_ISOLATION_S / FRAME_S))
        for _ in range(2):
            act_e = _active(an, thr)
            # the sound still going on at the release, across closure dips
            end = _chain_end(act_e, r, iso)
            q = _last_speech_end(an, act_e, i, end)
            post = an.energy[q:min(n, q + quiet)]
            if len(post) < 10:
                break
            local = float(np.percentile(post, FLOOR_PCT))
            thr = max(an.abs_threshold, an.threshold, local + MARGIN_DB)
        if q > last_q:
            break
        act = _active(an, thr)
        for a, b in an.nonspeech_bursts:
            act[a:b] = False  # VAD-rejected bursts (breath, clicks, noise) are not speech
        # The next speech after q. Every sound counts, inside or outside the
        # stateful VAD region (Silero lags onsets, and a playout cut can end a
        # word before the VAD re-enters). Short isolated sounds, and long ones
        # the VAD never entered, are judged on their own (fresh VAD + flatness).
        nxt, prev, late_residual = n, q, False
        for a, b in _groups(act[q:], iso):
            a, b = a + q, b + q
            if a - q >= quiet:
                nxt = a
                break
            short = b - a <= int(round(0.3 / FRAME_S))
            if (a - prev >= iso and short) or not an.region[a:b].any():
                kind = _burst_kind(an, prev, a, b)
                if kind == "nonspeech":
                    an.nonspeech_bursts.append((a, b))
                    # a lasting non-speech sound that starts inside the quiet
                    # window is residual sound the floor estimate did not see
                    late_residual = late_residual or not short
                    prev = b
                    continue
                if kind == "uncertain":
                    an.uncertain_bursts.append((a, b))
            nxt = a
            break
        if nxt - q >= quiet:
            # Latency is only as precise as the threshold: residual sound that
            # raised it above the clean-output convention can hide speech tails.
            masked = late_residual or (local is not None and (local > TIMING_MAX_FLOOR_DB or
                                                              thr > an.abs_threshold + 1.0))
            if an.uncertain_bursts:
                found = {"stop_s": None, "uncertain_burst_s": an.time(an.uncertain_bursts[-1][0])}
            elif masked and _residual_spread(an, q, local) > RESIDUAL_SPREAD_DB:
                # the threshold was raised by residual sound that is not
                # stationary: softer sound (speech) can hide under it
                found = {"stop_s": None, "nonstationary_residual_s": an.time(q), "local_floor_dbfs": local}
            elif masked and nxt < n and nxt - q < quiet + int(round(MASKED_PAUSE_MARGIN_S / FRAME_S)):
                # speech resumes after a pause whose length residual noise may hide
                found = {"stop_s": None, "masked_pause_s": an.time(q)}
            elif local is not None and masked and (an.pre_p5 is None or an.pre_p5 > local - BACKGROUND_DB):
                # the residual sound was never observed ABSENT before the probe
                # (no pre-injection output 3 dB quieter than it): it may have been
                # under the speech all along, and speech softer than it was never
                # observable, so neither is its end
                found = {"stop_s": None, "background_candidate_s": an.time(q), "local_floor_dbfs": local}
            elif local is not None and an.vad_level is not None and local > an.vad_level - SNR_MIN_DB:
                found = {"stop_s": None, "low_snr_candidate_s": an.time(q), "local_floor_dbfs": local}
            else:
                found = {"stop_s": an.time(q), "placement": "vad_release_bounded" if bounded else "vad_release+energy",
                         "local_floor_dbfs": local,
                         "threshold_dbfs": thr, "masked": masked,
                         "release_s": an.time(r)}
            break
        i = nxt
    if silent_stop is not None:
        keys = ("stop_s", "low_snr_candidate_s", "unconfirmed_candidate_s", "uncertain_burst_s", "masked_pause_s",
                "background_candidate_s", "nonstationary_residual_s")
        when = next((found[k] for k in keys if found and found.get(k) is not None), None)
        if when is None or an.frame(when) > silent_stop:
            return {"stop_s": an.time(silent_stop), "placement": "silent", "masked": False}
    return found or {"stop_s": None}


def _residual_spread(an: Analysis, q: int, local: float) -> float:
    """p90 - p10 (dB) of the 5 ms energies in the 200 ms after the stop,
    after the decay to the residual (<= 50 ms of frames above local + 3 dB
    are the speech tail). Stationary noise: ~1.5 dB; soft speech or other
    modulated sound under the raised threshold: more."""
    quiet = int(round(QUIET_S / FRAME_S))
    seg = an.energy[q:q + quiet]
    skip = 0
    while skip < min(10, len(seg)) and seg[skip] > local + 3.0:
        skip += 1
    seg = seg[skip:]
    if len(seg) < 10:
        return 0.0
    return float(np.percentile(seg, 90) - np.percentile(seg, 10))


def _disagreement(an: Analysis, a: int, b: int) -> bool:
    """Sustained energetic, modulated audio the VAD calls non-speech."""
    if an.speech_level is None:
        return False
    mask = (an.energy >= an.threshold) & ~an.region
    mask[:max(0, a)] = False
    mask[b:] = False
    for s, e in _runs(mask):
        if (e - s) * FRAME_S >= DISAGREE_MIN_S - 1e-9:
            seg = an.energy[s:e]
            if np.median(seg) >= an.speech_level - DISAGREE_LEVEL_DB and np.std(seg) >= DISAGREE_MOD_DB:
                return True
    return False


# every void reason score_event itself can produce (an offline re-score
# re-evaluates these; any other void was decided live and is kept)
SCORING_VOIDS = frozenset({
    "missing_or_invalid_actual_timing", "insufficient_stop_observation", "next_caller_contaminates_observation",
    "invalid_agent_pcm_timeline", "no_agent_audio", "agent_not_speaking_at_injection",
    "speech_continuation_unconfirmed", "trailing_sound_speech_uncertain", "pause_length_masked_by_residual_noise",
    "nonstationary_residual_under_raised_threshold", "persistent_background_masks_speech_end",
    "residual_output_noise_within_6db_of_speech", "speech_cessation_unconfirmed_in_observation",
    "energy_vad_disagreement"})
SCORING_OUTPUT_KEYS = ("verdict", "hit", "fp", "stopped", "stop_ms", "latency_retained", "measurement")


def _void(record, reason, evidence=None):
    return {**record, "void": reason, "verdict": None, "hit": None, "fp": None, "stopped": None,
            "stop_ms": None, "latency_retained": None, "measurement": evidence}


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def score_event(record: dict[str, Any], adapter: Any, vad: Any) -> dict[str, Any]:
    """Score a stimulus (barge_in/backchannel) or no-stimulus control point.

    ``caller_start_s`` is the injection (or control) instant on the playout
    clock; ``observation_end_s`` and ``next_caller_start_s`` bound validity.
    """
    if record.get("void"):
        return _void(record, str(record["void"]))
    kind = record.get("kind")
    if kind not in {"barge_in", "backchannel", "control"}:
        raise ValueError(f"unknown Track 2 event kind: {kind!r}")
    t, observed = record.get("caller_start_s"), record.get("observation_end_s")
    if not (_finite(t) and _finite(observed)):
        return _void(record, "missing_or_invalid_actual_timing")
    if observed < t + OBSERVATION_S - 1e-8:
        return _void(record, "insufficient_stop_observation")
    nxt = record.get("next_caller_start_s")
    if nxt is not None and (not _finite(nxt) or t <= nxt < t + OBSERVATION_S - 1e-8):
        return _void(record, "next_caller_contaminates_observation")
    timeline = adapter.agent_timeline
    if timeline.rate <= 0 or len(bytes(adapter.agent_pcm)) != timeline.n_samples * 2:
        return _void(record, "invalid_agent_pcm_timeline")
    if not timeline.chunks:
        return _void(record, "no_agent_audio")
    t0 = max(0.0, t - CONTEXT_S)
    tail = min(observed, t + ANALYSIS_TAIL_S)  # beyond 1.2 s only to confirm VAD release
    an = Analysis(heard(adapter, t0, tail), timeline.rate, t0, vad, level_until=t)
    evidence = {"policy_schema": SCHEMA, "clock": "reference_client_playout",
                "floor_dbfs": an.floor, "threshold_dbfs": an.threshold, "vad_level_dbfs": an.vad_level,
                "speech_level_dbfs": an.speech_level, "stop_window_tail_s": None,
                "causal_yield_verified": False}
    if not an.speaking_at(t):
        return _void(record, "agent_not_speaking_at_injection", evidence)
    found = stop_after(an, t, t + WINDOW_S)
    evidence["stop_search"] = {k: v for k, v in found.items()}
    stop = found.get("stop_s")
    if stop is None and not any(found.get(k) is not None for k in (
            "low_snr_candidate_s", "unconfirmed_candidate_s", "uncertain_burst_s", "masked_pause_s",
            "background_candidate_s", "nonstationary_residual_s")):
        # "Kept speaking" must be positive evidence: a fresh-state VAD pass over
        # the last 400 ms of the window (any non-stop leaves speech in it) has to
        # find speech; residual noise the lingering state mistakes for speech fails.
        a, b = an.frame(t + WINDOW_S - 0.2), an.frame(t + OBSERVATION_S)
        seg = an.x16[int(round(a * FRAME_S * 16000)):int(round(b * FRAME_S * 16000))]
        if not (fresh_probs(vad, seg) > VAD_ON).any():
            return _void(record, "speech_continuation_unconfirmed", evidence)
    if stop is None and found.get("uncertain_burst_s") is not None:
        return _void(record, "trailing_sound_speech_uncertain", evidence)
    if stop is None and found.get("masked_pause_s") is not None:
        return _void(record, "pause_length_masked_by_residual_noise", evidence)
    if stop is None and found.get("nonstationary_residual_s") is not None:
        return _void(record, "nonstationary_residual_under_raised_threshold", evidence)
    if stop is None and found.get("background_candidate_s") is not None:
        return _void(record, "persistent_background_masks_speech_end", evidence)
    if stop is None and found.get("low_snr_candidate_s") is not None:
        return _void(record, "residual_output_noise_within_6db_of_speech", evidence)
    if stop is None and found.get("unconfirmed_candidate_s") is not None:
        return _void(record, "speech_cessation_unconfirmed_in_observation", evidence)
    within = stop is not None and stop - t <= WINDOW_S + 1e-9
    decision_end = an.frame(stop if within else t + OBSERVATION_S)
    if _disagreement(an, an.frame(t), decision_end):
        return _void(record, "energy_vad_disagreement", evidence)
    retained = bool(within and not found.get("masked") and (an.floor is None or an.floor <= TIMING_MAX_FLOOR_DB))
    out = {**record, "void": None, "stopped": bool(within),
           "stop_ms": round(max(0.0, stop - t) * 1000, 3) if within else None,
           "latency_retained": retained if within else None, "measurement": evidence}
    if kind == "barge_in":
        out.update(verdict="pass" if within else "fail", hit=bool(within), fp=None)
    elif kind == "backchannel":
        out.update(verdict="fail" if within else "pass", hit=None, fp=bool(within))
    else:
        out.update(verdict=None, hit=None, fp=None)
    cut = any(t <= c <= (stop if within else t + WINDOW_S) + FRAME_S for c in timeline.cuts)
    out["measurement"]["playback_cut_before_stop"] = cut
    return out


SPEAK_CONTEXT_S = 0.6


def speaking_in(x: np.ndarray, rate: int, now: float, vad: Any) -> bool:
    """Causal 'is the model speaking now' on heard audio ending at ``now``."""
    if len(x) < rate * 0.1:
        return False
    t0 = now - len(x) / rate
    return Analysis(x, rate, t0, vad).speaking_at(now, causal=True)


def speaking_now(adapter: Any, now: float, vad: Any) -> bool:
    """Same rule, from a recording: live injection and matched controls agree."""
    x = heard(adapter, max(0.0, now - SPEAK_CONTEXT_S), now)
    return speaking_in(x, adapter.agent_timeline.rate, now, vad)
