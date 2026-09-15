"""Provider-independent audio pacing and append-only timing evidence."""
import asyncio
import json
import time
import queue
import threading
from pathlib import Path

import numpy as np

from .data import FRAME_BYTES, FRAME_SECONDS
from .timing import deadline_timer, wait_until


class EventLog:
    def __init__(self, path: Path):
        self.handle = path.open("x", encoding="utf-8")
        self.origin = time.perf_counter()
        self.pending = queue.SimpleQueue()
        self.error = None
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()

    def _write(self):
        try:
            while (event := self.pending.get()) is not None:
                self.handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                self.handle.flush()
        except Exception as exc:
            self.error = exc

    def now(self) -> float:
        return time.perf_counter() - self.origin

    def emit(self, kind: str, *, at: float | None = None, **fields):
        event = {"kind": kind, "time_seconds": self.now() if at is None else at, **fields}
        if self.error:
            raise RuntimeError("Event log writer failed") from self.error
        self.pending.put(event)
        return event

    def close(self):
        import os
        self.pending.put(None)
        self.worker.join()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if self.error:
            raise RuntimeError("Event log writer failed") from self.error


def transmitted_silence_frames(config):
    """Absent means the historical 50-frame contract, including saved replays."""
    count = config.get('transmitted_silence_frames', 50)
    if type(count) is not int or count not in (0, 50):
        raise ValueError('Expected zero or 50 transmitted silence frames')
    if count != 50 and config.get('provider') != 'inworld':
        raise ValueError('Only Inworld supports omitting the prepared silence tail')
    return count


async def stream_audio(pcm: bytes, speech_frames: int, send_audio, finalize, log: EventLog, *, sample_rate=16000,
                       transmitted_silence_frames=50):
    if sample_rate not in (16000, 24000):
        raise ValueError("Unsupported sample rate")
    frame_bytes = sample_rate // 50 * 2
    if len(pcm) % frame_bytes or len(pcm) // frame_bytes != speech_frames + 50:
        raise ValueError("Audio must have whole 20 ms frames and exactly 50 silence frames")
    if type(transmitted_silence_frames) is not int or transmitted_silence_frames not in (0, 50):
        raise ValueError('Expected zero or 50 transmitted silence frames')
    if transmitted_silence_frames == 0:
        # The frozen file stays unchanged. Never omit actual audio: only the
        # verified, artificial zero tail may be excluded from this transport.
        if any(pcm[speech_frames * frame_bytes:]):
            raise ValueError('Cannot omit a nonzero audio tail')
        pcm = pcm[:speech_frames * frame_bytes]
    async with deadline_timer() as timer:
        start = log.now()
        due = start + FRAME_SECONDS
        t0 = None
        for index in range(len(pcm) // frame_bytes):
            ideal = start + (index + 1) * FRAME_SECONDS
            # Recover toward the original timeline without bursts: consecutive starts
            # are scheduled at least 19 ms apart, inside the frozen 18 ms minimum.
            await wait_until(log.origin + due, time.perf_counter, timer=timer)
            sent_at = log.now()
            if index == 0:
                log.emit('audio_start', at=sent_at)
            await send_audio(pcm[index * frame_bytes:(index + 1) * frame_bytes])
            completed_at = log.now()
            log.emit("audio_sent", at=sent_at, index=index, bytes=frame_bytes, sample_rate=sample_rate,
                     phase="speech" if index < speech_frames else "silence",
                     ideal_seconds=ideal, scheduled_seconds=due,
                     wakeup_delay_ms=max(0, sent_at - due) * 1000,
                     send_duration_ms=(completed_at - sent_at) * 1000,
                     send_completed_seconds=completed_at)
            due = max(ideal + FRAME_SECONDS, sent_at + .019, completed_at + .001)
            if index == speech_frames - 1:
                # t0 is measured locally when sending the last speech frame completes.
                t0 = completed_at
                log.emit("speech_end", at=t0, index=index)
                await finalize(t0)
        log.emit("audio_complete", t0_seconds=t0)


def pacing_metrics(events: list[dict], *, transmitted_silence_frames=50) -> dict:
    if type(transmitted_silence_frames) is not int or transmitted_silence_frames not in (0, 50):
        raise ValueError('Expected zero or 50 transmitted silence frames')
    if any(e["kind"] == "transport_packetization" for e in events):
        from .assemblyai_pacing import pacing_metrics as packet_metrics
        return packet_metrics(events)
    frames = [e for e in events if e["kind"] == "audio_sent"]
    if len(frames) < 2:
        return {"valid": False, "frames": len(frames), "reason": "too few frames"}
    reasons = []
    silence = sum(e.get("phase") == "silence" for e in frames)
    rates = {e.get("sample_rate", 16000) for e in frames}
    rate = next(iter(rates))
    if len(rates) != 1 or rate not in (16000, 24000):
        reasons.append("invalid_sample_rate")
        rate = 16000
    if silence != transmitted_silence_frames or any(e.get("bytes") != rate // 50 * 2 for e in frames):
        reasons.append("invalid_frame_shape")
    if any(type(e.get("index")) is not int or e["index"] != index
           for index, e in enumerate(frames)):
        reasons.append("invalid_frame_indexes")
    speech_frames = len(frames) - transmitted_silence_frames
    if speech_frames < 1 or any(e.get("phase") != ("speech" if index < speech_frames else "silence")
                               for index, e in enumerate(frames)):
        reasons.append("invalid_frame_phase_order")

    completion_count = sum("send_completed_seconds" in e for e in frames)
    timing_evidence = ("complete" if completion_count == len(frames) else
                       "legacy_unavailable" if completion_count == 0 else "incomplete")
    if timing_evidence == "incomplete":
        reasons.append("incomplete_send_timing")
    required = ("time_seconds", "ideal_seconds")
    optional = ("scheduled_seconds", "send_completed_seconds", "send_duration_ms", "wakeup_delay_ms")
    def finite_number(value):
        return type(value) in (int, float) and np.isfinite(value)
    if any(not finite_number(e.get(key)) for e in frames for key in required) or any(
            not finite_number(e[key]) for e in frames for key in optional if key in e):
        # Do not let NaN comparisons pass a gate or leak nonfinite JSON metrics.
        return dict(valid=False, frames=len(frames), silence_frames=silence,
                    gate_reasons=reasons + ["invalid_timing_fields"],
                    send_timing_evidence=timing_evidence)

    sent = np.array([e["time_seconds"] for e in frames], dtype=float)
    ideal = np.array([e["ideal_seconds"] for e in frames], dtype=float)
    gaps = np.diff(sent) * 1000
    lag = (sent - ideal) * 1000
    elapsed = float(sent[-1] - sent[0])
    expected = (len(frames) - 1) * FRAME_SECONDS
    if min(sent) < 0 or min(ideal) < 0 or min(np.diff(ideal)) <= 0 or min(gaps) <= 0:
        reasons.append("invalid_timestamp_order")
    if min(gaps) < 18:
        reasons.append("send_gap_below_18ms")
    if max(gaps) > 40:
        reasons.append("send_gap_above_40ms")
    if not .98 <= elapsed / expected <= 1.02:
        reasons.append("span_drift_above_2pct")
    durations = [(e['send_completed_seconds'] - e['time_seconds']) * 1000
                 for e in frames if 'send_completed_seconds' in e]
    wakes = [e['wakeup_delay_ms'] for e in frames if 'wakeup_delay_ms' in e]
    scheduled = [e['scheduled_seconds'] for e in frames if 'scheduled_seconds' in e]
    if (any(value < 0 for value in durations + wakes + scheduled)
            or any(e.get('send_duration_ms', 0) < 0 for e in frames)
            or any(e.get('scheduled_seconds', e['time_seconds']) > e['time_seconds'] for e in frames)
            or any(right <= left for left, right in zip(scheduled, scheduled[1:]))
            or any(e.get('send_completed_seconds', e['time_seconds']) > frames[index + 1]['time_seconds']
                   for index, e in enumerate(frames[:-1]))):
        if "invalid_timestamp_order" not in reasons:
            reasons.append("invalid_timestamp_order")
    if any(value > 40 for value in durations) or any(e.get('send_duration_ms', 0) > 40 for e in frames):
        reasons.append("send_duration_above_40ms")
    if any(abs(e['send_duration_ms'] - (e['send_completed_seconds'] - e['time_seconds']) * 1000) > .000001
           for e in frames if 'send_duration_ms' in e and 'send_completed_seconds' in e):
        reasons.append("inconsistent_send_duration")
    return dict(frames=len(frames), silence_frames=silence, expected_span_seconds=expected,
                actual_span_seconds=elapsed, actual_over_ideal=elapsed / expected,
                interval_ms_p50=float(np.percentile(gaps, 50)), interval_ms_p90=float(np.percentile(gaps, 90)),
                interval_ms_min=float(min(gaps)), interval_ms_max=float(max(gaps)),
                max_schedule_lag_ms=float(max(lag)),
                gate_reasons=reasons,
                send_timing_evidence=timing_evidence,
                send_duration_ms_max=max(durations) if durations else None,
                wakeup_delay_ms_max=max(wakes) if wakes else None,
                diagnosis='Send duration includes transport backpressure and local scheduling; neither proves provider cause.',
                valid=not reasons)


def read_events(path: Path, allow_truncated_final=False) -> list[dict]:
    lines = path.read_text().splitlines(keepends=True)
    events = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_truncated_final and index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise ValueError(f"Corrupted event log: {path.name}, line {index + 1}") from None
    return events
