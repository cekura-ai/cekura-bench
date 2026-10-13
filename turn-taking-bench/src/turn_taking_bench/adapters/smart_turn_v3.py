"""Pipecat Smart Turn v3, TurnBench's baseline pipeline, as a streaming adapter.
Ported from TurnBench (MIT licence); see THIRD_PARTY_NOTICES.md.

A port of TurnBench's ``baselines/smart_turn_v3/predict.py`` (Sathvik Udupa,
2026) and the two pieces of pipecat-ai/smart-turn it runs on
(``record_and_predict.SileroVAD`` and ``inference.predict_endpoint``) into the
``TurnModel`` interface. The logic is the reference's, statement for statement;
only its shape changed from "whole channel in, score array out" to "one 20 ms
frame in, events out":

1. The 48 kHz channel is resampled to 16 kHz with torchaudio's sinc kernel.
   The reference resamples the whole channel at once; here each output sample
   is computed as soon as the 21 input samples past it (0.44 ms of lookahead,
   the kernel's right half) have arrived, with the same oneDNN convolution, so
   the samples are bit-identical to the batch result.
2. Silero VAD on 512-sample chunks (32 ms) drives the reference's
   accumulate / settle state machine: a segment opens on speech (with 200 ms
   of pre-roll), Smart Turn scores it after 1 s of trailing silence, re-scores
   every 8 chunks while settling, and closes after 2 s.
3. The score (P(turn complete)) is sampled onto a 12.5 Hz grid, last write per
   frame wins, exactly as the reference fills its ``scores`` array, including
   its ``int(i / SR * OUT_HZ)`` frame index (a float expression that assigns
   some boundary chunks to the previous frame; ported verbatim).
4. Events are committed with the rule of TurnBench's ``turnbench.sweep.commit_events``
   (rising edge above theta, 2 s refractory, time = frame end), applied
   incrementally as each 12.5 Hz frame becomes final. EOT uses the score,
   INT uses ``1 - score`` in float32, at the operating point TurnBench swept
   (theta_eot = 0.0082, theta_int = 0.0155).

Two places where the reference is not causal, and what this port does:

- Frame timing (``commit_time`` config). The reference stamps an event at the
  end of its 12.5 Hz frame, but a frame's score is written by the last 32 ms
  chunk *starting* inside it, which can end up to 32 ms after the frame does;
  the resampler adds 0.44 ms. ``commit_time="reference"`` (default) keeps the
  reference's timestamps so predictions can be compared with TurnBench's file;
  those stamps precede the audio they depend on by up to ~32 ms. The runner
  accepts them because the adapter only emits an event after that audio was
  pushed. ``commit_time="causal"`` stamps the end of the 20 ms frame in which
  the decision was actually available (20 ms later, 40 ms at the boundary
  chunks the float frame index assigns backwards), which satisfies TurnBench's
  causality rule to the letter.
- VAD state reset (``vad_reset_s``). The reference's Silero wrapper zeroes its
  recurrent state every 5 s of *wall-clock* time, so its decisions depend on
  how fast the machine runs. This port resets every 5 s of *audio* time, which
  is what that wall clock means when the pipeline runs live (and what the
  harness's ``--realtime`` mode reproduces). ``vad_reset_s=0`` disables it.

Everything heavy (torch, torchaudio, onnxruntime, transformers) is imported on
first use so the base harness imports without the ``smart-turn`` extra.
"""

from __future__ import annotations

import math
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from turn_taking_bench import assets
from turn_taking_bench.assets import GitCheckout, HfFile, UrlFile
from turn_taking_bench.model import FRAME_S, Event

MODEL_NAME = "smart_turn_v3"

# ---- pinned assets -------------------------------------------------------------

SMART_TURN_REPO = "https://github.com/pipecat-ai/smart-turn"
SMART_TURN_COMMIT = "4786657e242dfe77dd138699ac564ee074a2a543"
SMART_TURN_HF_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_HF_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
# The reference runs the "-cpu" file without CUDA and the "-gpu" file with it;
# both are v3.1. The -cpu graph is the quantised 8.7 MB export, the -gpu graph
# the 32 MB float32 one. TurnBench's published dev results come from the -gpu
# graph, and the two graphs disagree enough to move EOT recall by ~4 points
# (see README), so "gpu" is the default: it runs on CPU too, about 25 % slower.
ONNX_FILES = {
    "cpu": HfFile(
        name="smart-turn-v3.1-cpu.onnx",
        repo_id=SMART_TURN_HF_REPO,
        filename="smart-turn-v3.1-cpu.onnx",
        revision=SMART_TURN_HF_REVISION,
        sha256="fb68d55c2d542ce79e44b12013bfd571e90df8594ab096d757198e851b0c6594",
    ),
    "gpu": HfFile(
        name="smart-turn-v3.1-gpu.onnx",
        repo_id=SMART_TURN_HF_REPO,
        filename="smart-turn-v3.1-gpu.onnx",
        revision=SMART_TURN_HF_REVISION,
        sha256="a32f7445d5076029472b6c9f7a71005df576ea19d5f929021200f535b962af84",
    ),
}
# record_and_predict.ensure_model fetches this from silero-vad's master branch;
# pinned here to the commit that last touched the file (sha256 identical).
SILERO_VAD = UrlFile(
    name="silero_vad.onnx",
    url=(
        "https://raw.githubusercontent.com/snakers4/silero-vad/"
        "bfdc0193023f121ea5b3cc7b176dbed570a68a59/src/silero_vad/data/silero_vad.onnx"
    ),
    sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
)
SMART_TURN_SOURCE = GitCheckout(name="smart-turn", url=SMART_TURN_REPO, commit=SMART_TURN_COMMIT)

ASSETS: list[assets.Asset] = [SMART_TURN_SOURCE, ONNX_FILES["cpu"], ONNX_FILES["gpu"], SILERO_VAD]

# ---- reference constants (baselines/smart_turn_v3/predict.py), verbatim -------

SR = 16_000
CHUNK = 512
OUT_HZ = 12.5
VAD_THRESH = 0.5

_CHUNK_MS = CHUNK / SR * 1000.0
_PRE_CHUNKS = math.ceil(200.0 / _CHUNK_MS)
_STOP_CHUNKS = math.ceil(1000.0 / _CHUNK_MS)
_SETTLE_CHUNKS = math.ceil(2000.0 / _CHUNK_MS)
_MAX_CHUNKS = math.ceil(8000.0 / _CHUNK_MS)
_SETTLE_STRIDE = 8

# Operating point from TurnBench's dev sweep (baselines/smart_turn_v3/README.md).
THETA_EOT = 0.0082
THETA_INT = 0.0155
# turnbench.sweep.REFRACTORY_S
REFRACTORY_S = 2.0

# record_and_predict.SileroVAD
VAD_CONTEXT = 64
VAD_RESET_S = 5.0
# inference.py / audio_utils.py
MAX_SEGMENT_S = 8

DEFAULT_CONFIG: dict[str, Any] = {
    "theta_eot": THETA_EOT,
    "theta_int": THETA_INT,
    "onnx_variant": "gpu",  # "gpu" (full-precision graph, TurnBench's published results) | "cpu" (quantised)
    "commit_time": "reference",  # "reference" | "causal"
    "vad_reset_s": VAD_RESET_S,  # audio seconds between Silero state resets; 0 disables
    "onnx_threads": 0,  # intra-op threads for the Smart Turn session; 0 = onnxruntime default
    "assets_dir": None,  # default: <turn-taking-bench>/.assets/smart_turn_v3 or $TURN_TAKING_BENCH_ASSETS_DIR
}


# ---- shared sessions (one per process) -----------------------------------------


@lru_cache(maxsize=None)
def _smart_turn_session(path: str, intra_op_threads: int):
    """inference.build_session, plus an optional intra-op thread count."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    if intra_op_threads:
        so.intra_op_num_threads = intra_op_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=None)
def _feature_extractor():
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor(chunk_length=MAX_SEGMENT_S)


@lru_cache(maxsize=None)
def _silero_session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)


# ---- ported pieces -------------------------------------------------------------


def truncate_audio_to_last_n_seconds(audio_array: np.ndarray, n_seconds: int = MAX_SEGMENT_S, sample_rate: int = SR):
    """audio_utils.truncate_audio_to_last_n_seconds, verbatim."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)
    return audio_array


class SmartTurnClassifier:
    """inference.predict_endpoint: Whisper log-mel features -> P(turn complete)."""

    def __init__(self, onnx_path: Path, intra_op_threads: int = 0) -> None:
        self.session = _smart_turn_session(str(onnx_path), int(intra_op_threads))
        self.feature_extractor = _feature_extractor()
        self.calls = 0

    def probability(self, audio_array: np.ndarray) -> float:
        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=MAX_SEGMENT_S)
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=SR,
            return_tensors="np",
            padding="max_length",
            max_length=MAX_SEGMENT_S * SR,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)
        outputs = self.session.run(None, {"input_features": input_features})
        self.calls += 1
        return outputs[0][0].item()


class SileroVad:
    """record_and_predict.SileroVAD with the 5 s state reset on audio time."""

    def __init__(self, onnx_path: Path, reset_s: float = VAD_RESET_S) -> None:
        self.session = _silero_session(str(onnx_path))
        self.reset_s = float(reset_s)
        self.reset()

    def reset(self) -> None:
        self._init_states()
        self._last_reset_t = 0.0

    def _init_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)

    def prob(self, chunk_f32: np.ndarray, t_end_s: float) -> float:
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK:
            raise ValueError(f"Expected {CHUNK} samples, got {x.shape[1]}")
        x = np.concatenate((self._context, x), axis=1)
        ort_inputs = {"input": x.astype(np.float32), "state": self._state, "sr": np.array(SR, dtype=np.int64)}
        out, self._state = self.session.run(None, ort_inputs)
        self._context = x[:, -VAD_CONTEXT:]
        # Reference: maybe_reset() on wall-clock time; here on the chunk's audio time.
        if self.reset_s and (t_end_s - self._last_reset_t) >= self.reset_s:
            self._init_states()
            self._last_reset_t = t_end_s
        return float(out[0][0])


class StreamingResampler:
    """torchaudio.functional.resample, sample for sample, one frame at a time.

    The batch version zero-pads the signal by ``width`` on the left and
    ``width + orig`` on the right and runs one strided convolution. Output
    ``q`` reads padded input ``[orig*q, orig*q + len(kernel))``, so it is
    computable once ``orig*q + len(kernel)`` input samples (with the left pad)
    have arrived. Running the same oneDNN convolution over whatever prefix is
    complete gives the batch result bit for bit (``torch.mkldnn_convolution``,
    the kernel torch picks for the whole channel; plain ``conv1d`` on short
    inputs takes a different code path and differs in the last ulp).
    """

    def __init__(self, orig_freq: int, new_freq: int) -> None:
        import torch
        import torchaudio.functional.functional as F

        self.torch = torch
        gcd = math.gcd(int(orig_freq), int(new_freq))
        self.orig = int(orig_freq) // gcd
        self.new = int(new_freq) // gcd
        kernel, self.width = F._get_sinc_resample_kernel(orig_freq, new_freq, gcd, dtype=torch.float32)
        self.kernel = kernel  # (new, 1, kernel_len)
        self.kernel_len = kernel.shape[-1]
        self.weight = kernel[:, :, None, :].contiguous()  # (new, 1, 1, kernel_len) for the 2-D op
        self.reset()

    def reset(self) -> None:
        self._buffer = np.zeros(self.width, dtype=np.float32)  # left pad, then unconsumed input
        self._next_q = 0  # next convolution output position
        self._consumed = 0  # input samples (incl. left pad) dropped from the buffer
        self._input_len = 0

    def _convolve(self, data: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(data))[None, None, None, :]
        if hasattr(torch, "mkldnn_convolution"):
            out = torch.mkldnn_convolution(x, self.weight, None, [0, 0], [1, self.orig], [1, 1], 1)
        else:  # not bit-exact with the batch path, but the same filter
            out = torch.nn.functional.conv2d(x, self.weight, stride=[1, self.orig])
        return out[0, :, 0, :].transpose(0, 1).reshape(-1).numpy()

    def _drain(self) -> np.ndarray:
        """Compute every output position whose input window is complete."""
        available = self._consumed + len(self._buffer)  # padded-input samples seen
        q_end = (available - self.kernel_len) // self.orig + 1  # exclusive
        if q_end <= self._next_q:
            return np.zeros(0, dtype=np.float32)
        start = self.orig * self._next_q - self._consumed
        stop = self.orig * (q_end - 1) + self.kernel_len - self._consumed
        out = self._convolve(self._buffer[start:stop])
        self._next_q = q_end
        keep_from = self.orig * q_end - self._consumed
        self._buffer = self._buffer[keep_from:]
        self._consumed += keep_from
        return out

    def push(self, samples: np.ndarray) -> np.ndarray:
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        self._input_len += len(samples)
        return self._drain()

    def flush(self) -> np.ndarray:
        """Append the batch version's right pad and emit the remaining samples,
        truncated to its ``ceil(new * length / orig)`` output length."""
        produced_before = self._next_q * self.new
        self._buffer = np.concatenate([self._buffer, np.zeros(self.width + self.orig, dtype=np.float32)])
        out = self._drain()
        target_length = math.ceil(self.new * self._input_len / self.orig)
        return out[: max(0, target_length - produced_before)]


class EdgeCommitter:
    """turnbench.sweep.commit_events, one frame at a time: an event at every
    rising edge of ``prob > theta``, deduplicated by a refractory that only a
    committed event extends."""

    def __init__(self, theta: float, frame_rate_hz: float, refractory_s: float = REFRACTORY_S) -> None:
        self.theta = float(theta)
        self.refractory_frames = max(1, round(refractory_s * frame_rate_hz))
        self.above = False
        self.last: int | None = None

    def step(self, frame: int, prob: float) -> bool:
        above = prob > self.theta
        rising = above and not self.above
        self.above = above
        if rising and (self.last is None or frame - self.last >= self.refractory_frames):
            self.last = frame
            return True
        return False


# ---- the adapter ---------------------------------------------------------------


class SmartTurnV3:
    """Smart Turn v3 pipeline on one speaker channel, frame by frame."""

    def __init__(
        self,
        *,
        theta_eot: float = THETA_EOT,
        theta_int: float = THETA_INT,
        onnx_variant: str = "gpu",
        commit_time: str = "reference",
        vad_reset_s: float = VAD_RESET_S,
        onnx_threads: int = 0,
        assets_dir: str | None = None,
    ) -> None:
        if onnx_variant not in ONNX_FILES:
            raise ValueError(f"onnx_variant must be one of {sorted(ONNX_FILES)}, got {onnx_variant!r}")
        if commit_time not in ("reference", "causal"):
            raise ValueError(f"commit_time must be 'reference' or 'causal', got {commit_time!r}")
        self.theta_eot, self.theta_int = float(theta_eot), float(theta_int)
        self.commit_time = commit_time
        self.directory = assets.assets_dir(MODEL_NAME, assets_dir)
        onnx = ONNX_FILES[onnx_variant]
        assets.require(self.directory, [onnx, SILERO_VAD], MODEL_NAME)
        self.classifier = SmartTurnClassifier(self.directory / onnx.name, onnx_threads)
        self.vad = SileroVad(self.directory / SILERO_VAD.name, vad_reset_s)
        self._resampler: StreamingResampler | None = None
        self._resamplers: dict[int, StreamingResampler] = {}
        self._sample_rate: int | None = None

    # -- stream lifecycle

    def reset(self, sample_rate: int) -> None:
        self._sample_rate = int(sample_rate)
        if self._sample_rate != SR:
            resampler = self._resamplers.get(self._sample_rate)
            if resampler is None:
                resampler = self._resamplers[self._sample_rate] = StreamingResampler(self._sample_rate, SR)
            resampler.reset()
            self._resampler = resampler
        else:
            self._resampler = None
        self.vad.reset()
        self._pending = np.zeros(0, dtype=np.float32)  # 16 kHz samples not yet chunked
        self._chunk_start = 0  # 16 kHz index of the next chunk, the reference's `i`
        self._pre: deque[np.ndarray] = deque(maxlen=_PRE_CHUNKS)
        self._seg: list[np.ndarray] = []
        self._active = False
        self._sil = 0
        self._extra = 0
        self._cur = 0.0
        self._pending_idx: int | None = None  # 12.5 Hz frame still open to later writes
        self._pending_score = np.float32(0.0)
        self._finalized = 0  # 12.5 Hz frames committed so far
        self._committers = {
            "eot": EdgeCommitter(self.theta_eot, OUT_HZ),
            "interruption": EdgeCommitter(self.theta_int, OUT_HZ),
        }
        self._scores: list[float] = []
        self._frames_pushed = 0
        self._vad_chunks = 0
        self._predict_calls_at_reset = self.classifier.calls

    def push(self, frame: np.ndarray, t_end_s: float) -> list[Event]:
        if self._sample_rate is None:
            raise RuntimeError("reset(sample_rate) must be called before push()")
        self._frames_pushed += 1
        audio16 = self._resampler.push(frame) if self._resampler is not None else frame.astype(np.float32, copy=False)
        return self._consume(audio16, t_end_s)

    def flush(self, t_end_s: float) -> list[Event]:
        events: list[Event] = []
        if self._resampler is not None:
            events.extend(self._consume(self._resampler.flush(), t_end_s))
        # The reference keeps floor(duration * 12.5) frames; the audio heard is
        # whole 20 ms frames, so that is the number of whole 80 ms frames in it.
        frames_used = self._frames_pushed // round(1.0 / (OUT_HZ * FRAME_S))
        while self._finalized < frames_used:
            score = self._pending_score if self._finalized == self._pending_idx else np.float32(0.0)
            events.extend(self._commit_frame(self._finalized, score, t_end_s))
        return events

    def diagnostics(self) -> dict[str, Any]:
        return {
            "frame_rate_hz": OUT_HZ,
            "scores": self._scores,  # P(turn complete) per 12.5 Hz frame, float32 values
            "vad_chunks": self._vad_chunks,
            "predict_calls": self.classifier.calls - self._predict_calls_at_reset,
        }

    # -- the reference's chunk loop, fed incrementally

    def _consume(self, audio16: np.ndarray, t_end_s: float) -> list[Event]:
        events: list[Event] = []
        if len(audio16):
            self._pending = np.concatenate([self._pending, audio16]) if len(self._pending) else audio16
        while len(self._pending) >= CHUNK:
            c = self._pending[:CHUNK].astype(np.float32)
            self._pending = self._pending[CHUNK:]
            i = self._chunk_start
            self._chunk_start += CHUNK
            events.extend(self._process_chunk(c, i, t_end_s))
        return events

    def _process_chunk(self, c: np.ndarray, i: int, t_end_s: float) -> list[Event]:
        chunk_end_s = (i + CHUNK) / SR
        sp = self.vad.prob(c, chunk_end_s) > VAD_THRESH
        self._vad_chunks += 1
        out_idx = int(i / SR * OUT_HZ)  # reference expression, float quirks included

        if not self._active:
            self._pre.append(c)
            if sp:
                self._cur = 0.0; self._seg = list(self._pre) + [c]; self._active = True; self._sil = 0; self._extra = 0
        elif self._extra == 0:
            self._seg.append(c)
            if len(self._seg) > _MAX_CHUNKS:
                self._seg.pop(0)
            self._sil = 0 if sp else self._sil + 1
            if self._sil >= _STOP_CHUNKS:
                self._extra = 1
                self._cur = self.classifier.probability(np.concatenate(self._seg))
        else:
            if sp:
                self._cur = 0.0; self._seg = [c]; self._active = True; self._sil = 0; self._extra = 0
            else:
                self._seg.append(c); self._extra += 1
                if self._extra % _SETTLE_STRIDE == 1:
                    self._cur = self.classifier.probability(np.concatenate(self._seg))
                if self._extra >= _SETTLE_CHUNKS:
                    self._active = False; self._sil = 0; self._extra = 0; self._pre.clear()

        # scores[out_idx] = cur, last write wins. Whether this chunk is the last
        # one written to its frame depends only on the next chunk's index, so the
        # frame is final right here, without waiting for the next chunk's audio.
        events: list[Event] = []
        score = np.float32(self._cur)
        if int((i + CHUNK) / SR * OUT_HZ) != out_idx:
            while self._finalized < out_idx:  # frames nothing wrote keep the reference's 0.0
                events.extend(self._commit_frame(self._finalized, np.float32(0.0), t_end_s))
            events.extend(self._commit_frame(out_idx, score, t_end_s))
            self._pending_idx = None
        else:
            self._pending_idx, self._pending_score = out_idx, score
        return events

    def _commit_frame(self, frame: int, score: np.float32, t_end_s: float) -> list[Event]:
        """One 12.5 Hz frame is final: apply commit_events' rule to both tasks.
        EOT reads the score, INT reads ``1.0 - score`` in float32, as the
        reference builds its two probability arrays."""
        self._finalized = frame + 1
        self._scores.append(float(score))
        commit_s = (frame + 1) / OUT_HZ if self.commit_time == "reference" else t_end_s
        events: list[Event] = []
        if self._committers["eot"].step(frame, float(score)):
            events.append(Event("eot", commit_s))
        if self._committers["interruption"].step(frame, float(np.float32(1.0) - score)):
            events.append(Event("interruption", commit_s))
        return events


def build(config: dict[str, Any]) -> SmartTurnV3:
    known = set(DEFAULT_CONFIG)
    unknown = set(config) - known
    if unknown:
        raise KeyError(f"unknown smart_turn_v3 config keys: {sorted(unknown)}; known: {sorted(known)}")
    return SmartTurnV3(**{key: config[key] for key in config})


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    """What `run.json` records about the model: the pinned assets as installed
    (commit, sha256) and the library versions that compute the scores."""
    directory = assets.assets_dir(MODEL_NAME, config.get("assets_dir"))
    record = assets.load_record(directory).get("assets", {})
    onnx = ONNX_FILES[config.get("onnx_variant", DEFAULT_CONFIG["onnx_variant"])]
    info: dict[str, Any] = {
        "assets_dir": str(directory),
        "smart_turn_source": record.get(SMART_TURN_SOURCE.name) or {"url": SMART_TURN_REPO, "commit": SMART_TURN_COMMIT, "installed": False},
        "onnx": record.get(onnx.name) or {"repo_id": onnx.repo_id, "filename": onnx.filename, "revision": onnx.revision, "sha256": onnx.sha256, "installed": False},
        "silero_vad": record.get(SILERO_VAD.name) or {"url": SILERO_VAD.url, "sha256": SILERO_VAD.sha256, "installed": False},
        "libraries": {},
    }
    from importlib import metadata

    for name in ("torch", "torchaudio", "onnxruntime", "transformers"):
        try:
            info["libraries"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info["libraries"][name] = None
    return info
