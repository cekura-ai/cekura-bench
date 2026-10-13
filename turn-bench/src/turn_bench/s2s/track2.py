"""Track 2 v0.7: authored interruption/backchannel sessions, speech-cessation metric.

Separate from Track 1 and from the v0.6 ``authored-run`` (left unchanged):

* Sessions use ``SYSTEM_PROMPT_TRACK2`` (Track 1's prompt minus the brevity
  sentence), identical for every model; its text and SHA-256 are recorded.
* Probe offsets are placed relative to the model's ACTUAL reply: the anchor is
  the first heard speech onset of the new reply (whole reply scanned, no prefix
  limit), and offsets come from a fixed, model-independent schedule
  (``schedule``). Injection happens at the first poll at/after the target at
  which the model is causally heard to be speaking (<= 0.5 s grace).
* No-stimulus control points: the same offsets and the same speaking rule are
  evaluated offline in every reply at moments with no caller audio near and
  no earlier stimulus into that reply; their stop rate is the natural-stop base
  rate used for excess yield.
* Scoring: ``authored_speech`` (VAD-primary speech cessation).
* Caller conditions (``--condition``, default clean): Track 1's seeded service
  transforms applied to the caller stream only (``track2_conditions``); every
  row, ``run.json``, ``result.json`` and summary records the condition.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from turn_bench.assets import assets_dir
from turn_bench.author import service_bridge
from turn_bench.runner import harness_git_state, turnbench_provenance, utc_now
from turn_bench.s2s import SYSTEM_PROMPT_TRACK2, default_config, prompt_sha256
from turn_bench.s2s import authored_speech as speech
from turn_bench.s2s.authored_plan import build_plan, load_clips
from turn_bench.s2s.authored_runner import SafeLog, _alive, _json, _play, _until
from turn_bench.s2s.authored_vad import SpeechVeto
from turn_bench.s2s.model import adapter_class, api_key, provider_entry, resolve_model, session_config
from turn_bench.s2s.onsets import response_starts
from turn_bench.s2s.track2_conditions import Condition, ConditionedCaller

service_bridge._ensure_path()
from service.caller import BranchingCaller, VOID_SLIP_MS  # noqa: E402
from service.events import Clock  # noqa: E402

SCHEMA = "turn-bench/track2-live-run/2"
SUMMARY_SCHEMA = "turn-bench/track2-summary/2"
VAD_MODEL = assets_dir("smart_turn_v3") / "silero_vad.onnx"
BARGE_OFFSETS_S = (1.0, 2.0, 3.0)
BACKCHANNEL_FIRST_S, BACKCHANNEL_STEP_S, BACKCHANNEL_MAX = 2.5, 2.5, 4
CONTROL_OFFSETS_S = (1.0, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0)
SPEAK_GRACE_S, POLL_S = 0.5, 0.02
CONTROL_CALLER_GUARD_S = 1.0
# Host pacing (v0.7d): a slip over VOID_SLIP_MS voids only the events and controls whose
# window [stimulus - PACING_GUARD_S, stimulus + observation] overlaps it; a single stall
# far from a stimulus says nothing about that stimulus. A stall this long voids the session.
PACING_GUARD_S = 1.0
SESSION_VOID_SLIP_MS = 250.0
ONSET_DBFS, ONSET_FRAMES = -50.0, 3
QUIET_DBFS, QUIET_HOLD_S = -50.0, 0.8
FINAL_REPLY_MAX_S = 12.5
REANCHOR_TIMEOUT_S = 6.0
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 2000, 7


def prompt_info() -> dict[str, str]:
    return {"text": SYSTEM_PROMPT_TRACK2, "sha256": prompt_sha256(SYSTEM_PROMPT_TRACK2),
            "track1_prompt_sha256": prompt_sha256()}


# ---- schedule ---------------------------------------------------------------

def snap_barge(offset: float | None) -> float:
    if offset is None:  # NB-24 offset_from_end_s: the end is unknowable causally
        return BARGE_OFFSETS_S[-1]
    return BARGE_OFFSETS_S[0] if offset < 1.5 else BARGE_OFFSETS_S[1] if offset < 2.5 else BARGE_OFFSETS_S[2]


def schedule(conversation: dict) -> dict[str, dict]:
    """Model-independent nominal offsets (s after the reply's first speech)."""
    out, per_reply = {}, Counter()
    for item in conversation["items"]:
        if item["speaker"] != "caller" or item["kind"] not in {"barge_in", "backchannel"}:
            continue
        k = per_reply[item["into"]]
        per_reply[item["into"]] += 1
        if item["kind"] == "barge_in":
            end_relative = item.get("offset_from_end_s") is not None
            nominal = snap_barge(None if end_relative else item.get("offset_s"))
            rule = ("authored offset_from_end_s is noncausal; fixed late onset offset"
                    if end_relative else "authored offset snapped to {1,2,3} s")
        else:
            nominal = BACKCHANNEL_FIRST_S + BACKCHANNEL_STEP_S * k if k < BACKCHANNEL_MAX else None
            rule = f"backchannel #{k + 1} into this reply: {BACKCHANNEL_FIRST_S} s + {BACKCHANNEL_STEP_S} s steps"
        out[item["id"]] = {"nominal_offset_s": nominal, "authored_offset_s": item.get("offset_s"),
                           "authored_offset_from_end_s": item.get("offset_from_end_s"), "rule": rule}
    return out


def build_t2_plan(specs_dir: Path, cache_dir: Path, *, ids=None, caller_voice="f-us", split="test") -> dict:
    """v0.6 cache-only preflight, plus the causal schedule; NB-24 is re-admitted."""
    plan = build_plan(specs_dir, cache_dir, ids=ids, caller_voice=caller_voice, split=split)
    counts = Counter()
    for conv in plan["conversations"]:
        if conv["void"] and conv["void"].startswith("noncausal offset_from_end_s"):
            conv["readmitted_from"] = conv["void"]
            conv["void"] = None
        conv["schedule"] = schedule(conv)
        for item in conv["items"]:
            if item["speaker"] == "caller" and item["kind"] in {"barge_in", "backchannel"}:
                counts[item["kind"]] += 1
    plan["schema"] = "turn-bench.track2-plan.v2"
    plan["t2_counts"] = dict(counts)
    plan["exclusions"] = [c for c in plan["exclusions"] if not c["reason"].startswith("noncausal")]
    return plan


# ---- live helpers -------------------------------------------------------------

class Onsets:
    """Incremental first-speech-onset search per response (whole reply, no prefix cap)."""

    def __init__(self):
        self.scanned: dict[int, int] = {}

    def first(self, adapter, start: int, end: int) -> int | None:
        rate = adapter.output_rate
        frame = rate // 100
        pos = self.scanned.get(start, start)
        pcm = np.frombuffer(bytes(adapter.agent_pcm[pos * 2:end * 2]), dtype="<i2")
        usable = len(pcm) // frame * frame
        if usable < frame * ONSET_FRAMES:
            return None
        rms = np.sqrt((pcm[:usable].astype(np.float64).reshape(-1, frame) ** 2).mean(axis=1)) / 32768
        loud = 20 * np.log10(np.maximum(rms, 1e-9)) >= ONSET_DBFS
        for i in range(len(loud) - ONSET_FRAMES + 1):
            if loud[i:i + ONSET_FRAMES].all():
                return pos + i * frame
        # keep the last frames for a run that straddles the next poll
        self.scanned[start] = pos + max(0, len(loud) - ONSET_FRAMES + 1) * frame
        return None


async def _anchor(caller, adapter, mark, earliest, timeout_s, onsets):
    deadline = caller.log.clock.now() + timeout_s
    while caller.log.clock.now() < deadline:
        _alive(caller, adapter)
        starts = response_starts(adapter.log.events)
        for response, start in enumerate(starts):
            if start < mark:
                continue
            end = starts[response + 1] if response + 1 < len(starts) else adapter.agent_timeline.n_samples
            sample = onsets.first(adapter, start, end)
            if sample is None:
                continue
            at = adapter.agent_timeline.playout_time_of_sample(sample)
            chunk = adapter.agent_timeline.chunk_of_sample(sample)
            if at is None or chunk is None:
                continue
            begin, stop = adapter.agent_timeline.playout_span(chunk.seq)
            if not (stop > begin and begin <= at < stop):
                continue  # never heard (cut before playout)
            if at < earliest:
                continue
            await _until(caller, adapter, at)
            return {"t": at, "sample": sample, "response": response, "response_sample": start,
                    "observed_at_s": caller.log.clock.now(),
                    "detector": {"frame_ms": 10, "onset_dbfs": ONSET_DBFS, "onset_frames": ONSET_FRAMES,
                                 "search": "whole response, incremental"}}
        await asyncio.sleep(POLL_S)
    return None


def _silent(adapter, begin, end) -> bool:
    x = speech.heard(adapter, begin, end)
    if not len(x):
        return True
    e = speech._energy(x, adapter.agent_timeline.rate)
    return bool((e < QUIET_DBFS).all())


def _unplayed_silent(adapter, now) -> bool:
    timeline = adapter.agent_timeline
    raw = np.frombuffer(bytes(adapter.agent_pcm), dtype="<i2")
    for chunk in reversed(timeline.chunks):
        begin, end = timeline.playout_span(chunk.seq)
        if end <= now:
            break
        first = chunk.first_sample + max(0, int((now - begin) * timeline.rate))
        part = raw[first:chunk.last_sample].astype(np.float64)
        if len(part) and 20 * math.log10(max(np.sqrt((part ** 2).max()) / 32768, 1e-9)) >= QUIET_DBFS + 10:
            return False
    return True


async def _quiet(caller, adapter, timeout_s):
    """Buffer drained + 0.2 s, or (continuous-stream providers) 0.8 s heard
    silence with nothing audible still queued."""
    deadline = caller.log.clock.now() + timeout_s
    while caller.log.clock.now() < deadline:
        _alive(caller, adapter)
        now = caller.log.clock.now()
        end = adapter.agent_timeline.playout_end()
        if end is not None and now >= end + 0.2:
            return end
        if end is not None and _silent(adapter, now - QUIET_HOLD_S, now) and _unplayed_silent(adapter, now):
            return now - QUIET_HOLD_S
        await asyncio.sleep(POLL_S)
    raise RuntimeError("agent quiet timeout; caller turn cannot be placed")


def _new_reply_before(adapter, anchor, target):
    for sample in response_starts(adapter.log.events):
        if sample <= anchor["response_sample"]:
            continue
        at = adapter.agent_timeline.playout_time_of_sample(sample)
        if at is not None and at <= target:
            return True
    return False


def _event(conversation, item, repeat, plan_row):
    return {"conversation_id": conversation["conversation_id"], "repeat": repeat,
            "spec_id": conversation["spec_id"], "caller_voice": conversation.get("caller_voice"),
            "event_id": item["id"], "kind": item["kind"], "into": item["into"],
            **plan_row, "anchor_s": None, "target_s": None, "caller_start_s": None,
            "caller_end_s": None, "void": None, "stage": "planned"}


async def _drive(conversation, clips, caller, adapter, repeat, records, anchors, vad,
                 onset_timeout_s, quiet_timeout_s, checkpoint):
    spoken, live, per_reply, last_stim = {}, {}, {}, None
    onsets = Onsets()
    sched = conversation["schedule"]
    for index, item in enumerate(conversation["items"]):
        _alive(caller, adapter)
        if item["speaker"] == "agent":
            trigger = spoken.get(item.get("trigger_caller_id"))
            mark, earliest = (trigger["agent_mark_before_caller"], trigger["caller_end_s"]) if trigger else (0, 0.0)
            current = await _anchor(caller, adapter, mark, earliest, onset_timeout_s, onsets)
            if current is not None and any(a is not None and a["response_sample"] == current["response_sample"]
                                           for a in anchors.values()):
                current = None  # a script marker cannot reuse an old reply
            anchors[item["id"]] = current
            continue
        if item["kind"] == "turn":
            after = item.get("after")
            if after is None and index and conversation["items"][index - 1]["speaker"] == "agent":
                after = conversation["items"][index - 1]["id"]
            # Any ordinary turn waits for the latest anchored live reply to go
            # quiet (also after backchannels into it), so it never becomes an
            # accidental extra barge-in.
            last_agent = next((p["id"] for p in reversed(conversation["items"][:index])
                               if p["speaker"] == "agent"), None)
            if (after in anchors and anchors[after] is not None) or (
                    after is None and anchors.get(last_agent) is not None):
                end = await _quiet(caller, adapter, quiet_timeout_s)
                target = end + (item.get("hold_s") or item.get("gap_s") or 0)
            elif after in spoken:
                target = spoken[after]["caller_end_s"] + (item.get("hold_s") or item.get("gap_s") or 0)
            elif not spoken:
                target = caller.log.clock.now() + conversation["timing"].get("start_s", 0)
            else:
                target = caller.log.clock.now() + (item.get("hold_s") or item.get("gap_s") or 0)
            if last_stim is not None:  # never cut a probe's 1.2 s observation short
                target = max(target, last_stim["caller_start_s"] + speech.OBSERVATION_S + 0.05)
            await _until(caller, adapter, max(target, caller.log.clock.now()))
            spoken[item["id"]] = await _play(caller, adapter, clips[item["id"]])
            continue
        record = _event(conversation, item, repeat, sched[item["id"]])
        records.append(record)
        anchor = live.get(item["into"], anchors.get(item["into"]))
        if anchor is None:
            record["void"] = "no_live_agent_reply_anchor"; checkpoint(); continue
        for attempt in range(2):
            if (item["kind"] == "backchannel" and last_stim is not None
                    and _new_reply_before(adapter, anchor, caller.log.clock.now())):
                # The model answered an earlier backchannel with a new reply: later
                # backchannels go into the reply it is speaking now, scheduled from
                # that reply's own first speech (same fixed offsets).
                fresh = await _anchor(caller, adapter, last_stim["agent_mark_before_caller"],
                                      last_stim["caller_start_s"], REANCHOR_TIMEOUT_S, onsets)
                if fresh is not None and all(fresh["response_sample"] != a["response_sample"]
                                             for a in anchors.values() if a is not None):
                    record["reanchored_from_s"] = anchor["t"]
                    anchor = live[item["into"]] = fresh
                    anchors[f"{item['into']}#{item['id']}"] = fresh  # hosts controls too
            if item["kind"] == "backchannel":
                k = per_reply.get(anchor["response_sample"], 0)
                record.setdefault("plan_nominal_offset_s", record["nominal_offset_s"])
                record["nominal_offset_s"] = (BACKCHANNEL_FIRST_S + BACKCHANNEL_STEP_S * k
                                              if k < BACKCHANNEL_MAX else None)
            if record["nominal_offset_s"] is None:
                record["void"] = "beyond_schedule"; break
            record["host"] = next(k for k, a in anchors.items() if a is anchor)
            target = anchor["t"] + record["nominal_offset_s"]
            record.update(anchor_s=anchor["t"], anchor_sample=anchor["sample"], target_s=target, stage="anchored")
            if caller.log.clock.now() > target + SPEAK_GRACE_S:
                record["void"] = "target_passed_before_scheduling"; break
            await _until(caller, adapter, max(target, caller.log.clock.now()))
            decided = audio_until = None
            while caller.log.clock.now() <= target + SPEAK_GRACE_S + 1e-9:
                now = caller.log.clock.now()
                if _new_reply_before(adapter, anchor, now):
                    record["void"] = "anchor_reply_replaced_before_injection"; break
                snap = speech.heard(adapter, max(0.0, now - speech.SPEAK_CONTEXT_S), now)
                if await asyncio.to_thread(speech.speaking_in, snap, adapter.agent_timeline.rate, now, vad):
                    decided = caller.log.clock.now(); audio_until = now; break
                await _until(caller, adapter, now + POLL_S)
            if (record["void"] == "anchor_reply_replaced_before_injection" and attempt == 0
                    and item["kind"] == "backchannel" and last_stim is not None):
                record["void"] = None  # the model answered the previous backchannel: re-anchor once
                record["reanchor_at_target"] = True
                continue
            break
        if record["void"]:
            checkpoint(); continue
        if decided is None:
            record["void"] = "agent_not_speaking_at_target"; checkpoint(); continue
        actual = await _play(caller, adapter, clips[item["id"]])
        spoken[item["id"]] = actual
        last_stim = actual
        per_reply[anchor["response_sample"]] = per_reply.get(anchor["response_sample"], 0) + 1
        record.update(actual, decided_at_s=decided, decision_audio_until_s=audio_until,
                      decision_compute_ms=(decided - audio_until) * 1000, stage="injected",
                      injection_lateness_ms=(actual["caller_start_s"] - decided) * 1000,
                      actual_offset_s=actual["caller_start_s"] - anchor["t"])
        if abs(record["injection_lateness_ms"]) > VOID_SLIP_MS:
            record["void"] = "injection_timing_outside_25ms"
        checkpoint()
    last_needed = max((r["caller_start_s"] + speech.OBSERVATION_S for r in records
                       if r.get("caller_start_s") is not None), default=caller.log.clock.now())
    # Let the final reply play out (bounded) so it can host no-stimulus controls.
    final = [a for a in anchors.values() if a is not None]
    if final:
        horizon = final[-1]["t"] + FINAL_REPLY_MAX_S
        try:
            await asyncio.wait_for(_quiet(caller, adapter, max(0.1, horizon - caller.log.clock.now())),
                                   timeout=max(0.2, horizon - caller.log.clock.now() + 0.5))
        except (RuntimeError, asyncio.TimeoutError):
            pass
    await _until(caller, adapter, max(last_needed, caller.log.clock.now() + 0.2))
    return spoken


# ---- offline controls -----------------------------------------------------------

def controls(conversation, anchors, records, adapter, caller_spans, observed, vad, repeat):
    """No-stimulus control points at the schedule's offsets in every reply."""
    out = []
    starts = response_starts(getattr(adapter, "events", []))
    stim_into = {}
    for r in records:
        if r.get("caller_start_s") is not None:
            stim_into.setdefault(r.get("host") or r["into"], []).append(r["caller_start_s"])
    agents = [i["id"] for i in conversation["items"] if i["speaker"] == "agent"]
    closing = agents[-1] if agents else None
    for marker, anchor in anchors.items():
        # The closing reply (to the final caller turn) never hosts a stimulus and
        # ends naturally soon: it is not a matched control context.
        if anchor is None or marker == closing:
            continue
        later = [adapter.agent_timeline.playout_time_of_sample(s) for s in starts if s > anchor["response_sample"]]
        replaced = min([t for t in later if t is not None], default=math.inf)
        first_stim = min(stim_into.get(marker, []), default=math.inf)
        for x in CONTROL_OFFSETS_S:
            row = {"conversation_id": conversation["conversation_id"], "repeat": repeat,
                   "spec_id": conversation["spec_id"], "caller_voice": conversation.get("caller_voice"),
                   "event_id": f"control:{marker}@{x}", "kind": "control", "into": marker,
                   "nominal_offset_s": x, "anchor_s": anchor["t"], "target_s": anchor["t"] + x,
                   "void": None, "observation_end_s": observed}
            tau, chosen = anchor["t"] + x, None
            while tau <= anchor["t"] + x + SPEAK_GRACE_S + 1e-9:
                if tau >= replaced:
                    row["void"] = "anchor_reply_replaced_before_injection"; break
                if tau >= first_stim:
                    row["void"] = "reply_already_stimulated"; break
                if speech.speaking_now(adapter, tau, vad):
                    chosen = tau; break
                tau += POLL_S
            if row["void"] is None and chosen is None:
                row["void"] = "agent_not_speaking_at_target"
            if row["void"] is None:
                if chosen + speech.OBSERVATION_S > first_stim:
                    row["void"] = "stimulus_inside_control_window"
                elif any(s < chosen + speech.OBSERVATION_S and e > chosen - CONTROL_CALLER_GUARD_S
                         for s, e in caller_spans):
                    row["void"] = "caller_audio_near_control"
            if row["void"] is None:
                row.update(caller_start_s=chosen, actual_offset_s=chosen - anchor["t"], stage="injected")
            out.append(row)
    return out


def slip_windows(events) -> list[tuple[float, float]]:
    """Intervals of caller audio sent more than VOID_SLIP_MS late, merged.

    Each slip log line is emitted when the late chunk goes out; the stream was
    behind for ``slip_ms`` before that.
    """
    rows = [e.as_json() if hasattr(e, "as_json") else e for e in events]  # live Events or saved events.jsonl rows
    spans = sorted((e["t"] - e["slip_ms"] / 1000.0, e["t"]) for e in rows
                   if e.get("kind") == "caller.pacing.slip" and e.get("slip_ms", 0) > VOID_SLIP_MS)
    out: list[list[float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _stalled(row, stalls) -> bool:
    """True when a stall overlaps this row's stimulus (or control) window."""
    start = row.get("caller_start_s")
    if start is None or not stalls:
        return False
    end = max(start + speech.OBSERVATION_S, row.get("caller_end_s") or start)
    return any(a < end and b > start - PACING_GUARD_S for a, b in stalls)


# ---- one conversation ---------------------------------------------------------------

async def run_conversation(conversation, cache_dir, config, out, *, repeat=1, vad=None,
                           adapter_factory=None, adapter_kwargs=None, clips=None,
                           onset_timeout_s=20.0, quiet_timeout_s=60.0, timeout_s=900.0, condition="clean"):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    cond = condition if isinstance(condition, Condition) else Condition(condition)
    rows, ctl_rows = await _run_conversation(conversation, cache_dir, config, out, repeat=repeat, vad=vad,
                                             adapter_factory=adapter_factory, adapter_kwargs=adapter_kwargs,
                                             clips=clips, onset_timeout_s=onset_timeout_s,
                                             quiet_timeout_s=quiet_timeout_s, timeout_s=timeout_s, cond=cond)
    for row in [*rows, *ctl_rows]:
        row["condition"] = cond.name
    path = out / "result.json"
    result = json.loads(path.read_text())
    for key in ("events", "controls"):
        for row in result.get(key) or []:
            row["condition"] = cond.name
    result["condition"] = cond.provenance()
    _json(path, result)
    return rows, ctl_rows


async def _run_conversation(conversation, cache_dir, config, out, *, repeat, vad, adapter_factory, adapter_kwargs,
                            clips, onset_timeout_s, quiet_timeout_s, timeout_s, cond):
    vad = vad or SpeechVeto(VAD_MODEL)
    planned = [i for i in conversation["items"] if i["speaker"] == "caller" and i["kind"] in {"barge_in", "backchannel"}]
    sched = conversation["schedule"]
    if conversation.get("void"):
        rows = [{**_event(conversation, i, repeat, sched[i["id"]]), "void": conversation["void"]} for i in planned]
        rows = [speech.score_event(r, None, vad) for r in rows]
        _json(out / "result.json", {"events": rows, "controls": [], "void": conversation["void"]})
        return rows, []
    clips = clips if clips is not None else load_clips(conversation, Path(cache_dir))
    entry = provider_entry(config["provider"])
    config = {**config, "instructions": SYSTEM_PROMPT_TRACK2}
    session = session_config(config, entry)
    if session.instructions != SYSTEM_PROMPT_TRACK2 or session.turn_detection.is_manual:
        raise ValueError("Track 2 requires the fixed Track 2 prompt and native turn detection")
    records, anchors, session_error = [], {}, None
    try:
        key = api_key(config)
        clock = Clock(); log = SafeLog(clock, out / "events.jsonl", out / "raw.jsonl", secret=key)
        cls = adapter_factory or adapter_class(config)
        adapter = cls(model=resolve_model(config), api_key=key, config=session, log=log, **(adapter_kwargs or {}))
    except Exception as exc:
        reason = f"{type(exc).__name__}: session setup failed before streaming"
        rows = [speech.score_event({**_event(conversation, i, repeat, sched[i["id"]]), "void": reason}, None, vad)
                for i in planned]
        _json(out / "result.json", {"events": rows, "controls": [], "error": reason})
        if "log" in locals():
            log.close()
        return rows, []
    if cond.stage is None:
        caller = BranchingCaller(adapter, log)
    else:  # degrade the caller stream only; the model's output is measured as received
        clips = cond.prepare_clips(clips, adapter.input_rate)
        caller = ConditionedCaller(adapter, log, cond)
    checkpoint = lambda: _json(out / "stimuli.json", {"events": log.clean(records)})
    observed = 0.0
    try:
        reason = cls.unsupported_reason(session)
        if reason:
            raise RuntimeError(f"unsupported session: {reason}")
        await asyncio.wait_for(adapter.connect(), timeout=40.0)
        async with caller:
            await asyncio.wait_for(_drive(conversation, clips, caller, adapter, repeat, records, anchors, vad,
                                          onset_timeout_s, quiet_timeout_s, checkpoint), timeout=timeout_s)
            observed = clock.now()
    except Exception as exc:
        session_error = f"{type(exc).__name__}: session did not complete"
        observed = clock.now()
    finally:
        await caller.stop()
        try:
            await asyncio.wait_for(adapter.close(), timeout=30.0)
        except Exception as exc:
            session_error = session_error or f"{type(exc).__name__}: session close failed"
    seen = {r["event_id"] for r in records}
    for item in planned:
        if item["id"] not in seen:
            records.append({**_event(conversation, item, repeat, sched[item["id"]]),
                            "void": session_error or "event_not_reached"})
    stalls = slip_windows(adapter.log.events)
    pacing = {**caller.pacing(), "pacing_void": caller.max_slip_ms > SESSION_VOID_SLIP_MS,
              "max_slip_ms_unrounded": caller.max_slip_ms, "policy": "event-local",
              "guard_s": PACING_GUARD_S, "session_void_slip_ms": SESSION_VOID_SLIP_MS,
              "stalls": [[round(a, 3), round(b, 3)] for a, b in stalls]}
    timeline_data = {"caller": adapter.caller_timeline.as_json(), "agent": adapter.agent_timeline.as_json(),
                     "observed_until_s": observed}
    _json(out / "timelines.json", timeline_data)
    _json(out / "anchors.json", anchors)
    audio = service_bridge.audio()
    caller_tl = audio.AudioTimeline.from_json(timeline_data["caller"])
    measurement = SimpleNamespace(agent_pcm=bytes(adapter.agent_pcm),
                                  agent_timeline=audio.AudioTimeline.from_json(timeline_data["agent"]),
                                  events=list(adapter.log.events))
    spans = []
    for u in caller.utterances:
        s = caller_tl.time_of_sample(u.first_sample); e = caller_tl.time_of_sample(u.last_sample - 1)
        if s is not None and e is not None:
            spans.append((s, e + 1 / adapter.input_rate))
    scored = []
    for record in records:
        record["observation_end_s"] = observed
        record["pacing"] = pacing
        if record.get("caller_first_sample") is not None:
            if cond.stage is not None:  # what was actually transmitted for this clip, condition applied
                record["sent_stream_sha256"] = hashlib.sha256(bytes(
                    adapter.caller_pcm[record["caller_first_sample"] * 2:record["caller_last_sample"] * 2])).hexdigest()
            record["caller_start_s"] = caller_tl.time_of_sample(record["caller_first_sample"])
            record["caller_end_s"] = caller_tl.time_of_sample(record["caller_last_sample"] - 1) + 1 / adapter.input_rate
            record["next_caller_start_s"] = min((s for s, _ in spans if s > record["caller_start_s"] + 1e-6), default=None)
        if session_error:
            record["void"] = record.get("void") or session_error
        if pacing["pacing_void"] or _stalled(record, stalls):
            record["void"] = "host_pacing_outside_25ms"
        scored.append(speech.score_event(record, measurement, vad))
    ctl = [] if session_error or pacing["pacing_void"] else controls(
        conversation, anchors, records, measurement, spans, observed, vad, repeat)
    ctl_scored = []
    for row in ctl:
        if row.get("caller_start_s") is not None:
            row["next_caller_start_s"] = min((s for s, _ in spans if s > row["caller_start_s"]), default=None)
        if _stalled(row, stalls):
            row["void"] = "host_pacing_outside_25ms"
        ctl_scored.append(speech.score_event(row, measurement, vad))
    if adapter.caller_pcm:
        audio.write_wav(out / "caller.wav", bytes(adapter.caller_pcm), adapter.input_rate)
    if adapter.agent_pcm:
        audio.write_wav(out / "agent.wav", bytes(adapter.agent_pcm), adapter.output_rate)
    _json(out / "result.json", {"events": log.clean(scored), "controls": ctl_scored, "error": session_error,
                                "provider": config["provider"], "model": adapter.model,
                                "prompt": prompt_info(), "pacing": pacing, "observed_until_s": observed,
                                "speech_policy": speech.policy(), "vad": vad.provenance,
                                "session_acknowledged": log.clean(adapter.session_ack),
                                "usage": service_bridge.metrics().usage(adapter)})
    log.close()
    return scored, ctl_scored


# ---- summary ----------------------------------------------------------------------------

def _pct(v, q):
    return float(np.percentile(v, q)) if v else None


def _stats(events, ctls, kind):
    ev = [r for r in events if r["kind"] == kind]
    sc = [r for r in ev if not r.get("void")]
    stops = sum(bool(r["stopped"]) for r in sc)
    rate = stops / len(sc) if sc else None
    by_x = {}
    for r in ctls:
        if not r.get("void"):
            by_x.setdefault(r["nominal_offset_s"], []).append(bool(r["stopped"]))
    matched = [r for r in sc if r["nominal_offset_s"] in by_x]
    base = (sum(np.mean(by_x[r["nominal_offset_s"]]) for r in matched) / len(matched)) if matched else None
    m_rate = (sum(bool(r["stopped"]) for r in matched) / len(matched)) if matched else None
    return {"stop_rate": rate, "n": len(sc), "stops": stops, "matched_n": len(matched),
            "matched_stop_rate": m_rate, "control_base_stop_rate": base,
            "excess_stop_rate": None if base is None else m_rate - base,
            "latencies": [r["stop_ms"] for r in sc if r["stopped"] and r.get("latency_retained")]}


def summarize(events, ctls, cluster="conversation_id"):
    """cluster: conversation_id (one voice) or spec_id (voice variants of a template together)."""
    clusters = sorted({r[cluster] for r in events} | {r[cluster] for r in ctls})
    ev_by = {c: [r for r in events if r[cluster] == c] for c in clusters}
    ct_by = {c: [r for r in ctls if r[cluster] == c] for c in clusters}
    conditions = sorted({r.get("condition") or "clean" for r in [*events, *ctls]}) or ["clean"]
    result = {"schema": SUMMARY_SCHEMA, "speech_policy": speech.policy(), "prompt": prompt_info(),
              "condition": conditions[0] if len(conditions) == 1 else "mixed", "conditions": conditions,
              "caller_voices": sorted({r.get("caller_voice") for r in [*events, *ctls] if r.get("caller_voice")}),
              "bootstrap": {"method": "percentile, whole-conversation clusters (events and controls together)", "cluster": cluster,
                            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "level": 0.95}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [[clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
             for _ in range(BOOTSTRAP_DRAWS)] if len(clusters) >= 2 else []
    for kind, name in (("barge_in", "interruption"), ("backchannel", "backchannel")):
        point = _stats(events, ctls, kind)
        boot = {"stop_rate": [], "control_base_stop_rate": [], "excess_stop_rate": [], "p50": [], "p90": []}
        for d in draws:
            s = _stats([r for c in d for r in ev_by[c]], [r for c in d for r in ct_by[c]], kind)
            for k in ("stop_rate", "control_base_stop_rate", "excess_stop_rate"):
                if s[k] is not None:
                    boot[k].append(s[k])
            if s["latencies"]:
                boot["p50"].append(_pct(s["latencies"], 50)); boot["p90"].append(_pct(s["latencies"], 90))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
        lat = point.pop("latencies")
        sel = [r for r in events if r["kind"] == kind]
        injected = [r for r in sel if r.get("stage") == "injected"]
        row = {**point, "cells": len(sel), "injected": len(injected),
               "scored_of_injected": (point["n"] / len(injected)) if injected else None,
               "scored_of_planned": (point["n"] / len(sel)) if sel else None,
               "void_reasons": dict(Counter(r["void"] for r in sel if r.get("void"))),
               "excluded_after_injection": dict(Counter(r["void"] for r in injected if r.get("void"))),
               "conversations": len({r["conversation_id"] for r in sel if not r.get("void")}),
               "ci95": {k: ci(boot[k]) for k in ("stop_rate", "control_base_stop_rate", "excess_stop_rate")},
               "stop_ms": {"p50": _pct(lat, 50), "p90": _pct(lat, 90), "n": len(lat),
                           "p50_ci95": ci(boot["p50"]), "p90_ci95": ci(boot["p90"])},
               "by_offset": {str(x): {"n": sum(1 for r in sel if r["nominal_offset_s"] == x and not r.get("void")),
                                      "stops": sum(1 for r in sel if r["nominal_offset_s"] == x and not r.get("void") and r["stopped"]),
                                      "planned": sum(1 for r in sel if r["nominal_offset_s"] == x)}
                             for x in sorted({r["nominal_offset_s"] for r in sel if r["nominal_offset_s"] is not None})}}
        if kind == "barge_in":
            row["headline"] = "yield rate = stop_rate; excess yield = matched stop rate - matched no-stimulus base rate"
        else:
            row["continuation_rate"] = None if point["stop_rate"] is None else 1 - point["stop_rate"]
            row["headline"] = "continuation rate = 1 - stop_rate; excess stop = matched stop rate - base rate (lower is better)"
        result[name] = row
    sc = [r for r in ctls if not r.get("void")]
    result["controls"] = {"cells": len(ctls), "scored": len(sc),
                          "void_reasons": dict(Counter(r["void"] for r in ctls if r.get("void"))),
                          "by_offset": {str(x): {"n": sum(1 for r in sc if r["nominal_offset_s"] == x),
                                                 "stops": sum(1 for r in sc if r["nominal_offset_s"] == x and r["stopped"])}
                                        for x in CONTROL_OFFSETS_S}}
    result["notes"] = [
        "Experimental Track 2; never blended with Track 1 or into one rank.",
        "Stop = model stopped SPEAKING: >=200 ms without VAD-confirmed speech, within 1.0 s of the first transmitted caller sample; full 1.2 s observation required.",
        "Excess = stop rate at stimulus points minus the no-stimulus stop rate at the same nominal offsets (same speaking rule), conversation-cluster bootstrap.",
        "Backchannel handling = continuing to speak through it; semantic content after it is not measured.",
        "Latency only where the residual output floor <= -50 dBFS and the stop was energy-placed.",
    ]
    return result


def render(summary, meta) -> str:
    def f(v, p=3):
        return "n/a" if v is None else f"{v:.{p}f}"
    def ci(v):
        return "" if not v else f" [{v[0]:.3f}, {v[1]:.3f}]"
    cond = summary.get("condition") or "clean"
    lines = [f"# Track 2 (experimental) — {meta.get('provider')} / {meta.get('model')}"
             + ("" if cond == "clean" else f" — caller condition: {cond}"), ""]
    i, b = summary["interruption"], summary["backchannel"]
    lines += ["| | n scored / injected / planned | rate | no-stimulus base | excess |", "|---|---|---|---|---|",
              f"| Interruption yield | {i['n']} / {i['injected']} / {i['cells']} | {f(i['stop_rate'])}{ci(i['ci95']['stop_rate'])} | "
              f"{f(i['control_base_stop_rate'])}{ci(i['ci95']['control_base_stop_rate'])} | {f(i['excess_stop_rate'])}{ci(i['ci95']['excess_stop_rate'])} |",
              f"| Backchannel stop (lower better) | {b['n']} / {b['injected']} / {b['cells']} | {f(b['stop_rate'])}{ci(b['ci95']['stop_rate'])} | "
              f"{f(b['control_base_stop_rate'])}{ci(b['ci95']['control_base_stop_rate'])} | {f(b['excess_stop_rate'])}{ci(b['ci95']['excess_stop_rate'])} |", ""]
    s = i["stop_ms"]
    lines += [f"Yield latency p50 / p90: {f(s['p50'],0)} / {f(s['p90'],0)} ms (n={s['n']}).", "",
              f"Controls: {summary['controls']['scored']} scored of {summary['controls']['cells']}; by offset {json.dumps(summary['controls']['by_offset'])}", ""]
    for name in ("interruption", "backchannel"):
        lines.append(f"- {name} voids: {json.dumps(summary[name]['void_reasons'])}")
    lines += [""] + [f"- {n}" for n in summary["notes"]] + [""]
    return "\n".join(lines)


def run_t2(plan: dict, out: Path, *, provider: str, model: str | None = None, env_file: str | None = None,
           condition: str = "clean", adapter_factory=None, adapter_kwargs=None,
           turn_detection: str = "server_vad") -> dict:
    cond = Condition(condition)
    for conversation in plan["conversations"]:
        if not conversation.get("void"):
            load_clips(conversation, Path(plan["cache_dir"]))
    config = default_config(provider)
    config["env_file"] = env_file
    if model:
        config["model"] = model
    config["turn_detection"] = turn_detection
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    vad = SpeechVeto(VAD_MODEL)
    _json(out / "plan.json", plan)
    info = {"schema": SCHEMA, "status": "running", "started_at": utc_now(), "provider": provider,
            "model": resolve_model(config), "prompt": prompt_info(), "speech_policy": speech.policy(),
            "vad": vad.provenance, "harness": harness_git_state(), "turnbench": turnbench_provenance(),
            "label": "EXPERIMENTAL Track 2; separate from Track 1; no combined rank",
            "condition": cond.describe(), "turn_detection": turn_detection, "conversations": []}
    _json(out / "run.json", info)
    events, ctls = [], []

    async def sessions():
        for conversation in plan["conversations"]:
            directory = out / f"{conversation['conversation_id']}-r1"
            ev, ct = await run_conversation(conversation, plan["cache_dir"], config, directory, vad=vad,
                                            condition=cond.name, adapter_factory=adapter_factory,
                                            adapter_kwargs=adapter_kwargs)
            events.extend(ev); ctls.extend(ct)
            info["conversations"].append({"conversation_id": conversation["conversation_id"], "repeat": 1,
                                          "events": len(ev), "voids": sum(bool(r.get("void")) for r in ev),
                                          "controls": len(ct)})
            _json(out / "run.json", info)
            _json(out / "t2-events.json", {"events": events, "controls": ctls})
    asyncio.run(sessions())
    summary = summarize(events, ctls)
    summary.update(provider=provider, model=info["model"], run_id=out.name, ranking_permitted=False,
                   condition_provenance=cond.describe())
    _json(out / "t2-summary.json", summary)
    (out / "t2-summary.md").write_text(render(summary, summary))
    info.update(status="complete", finished_at=utc_now())
    _json(out / "run.json", info)
    return summary


def run_condition(info: dict) -> str:
    """Caller condition of a saved run (runs from before conditions existed are clean)."""
    return (info.get("condition") or {}).get("condition") or "clean"


def combine(run_dirs: list[Path], out: Path) -> dict:
    """Pool runs of one model (e.g. two caller voices); clusters = templates."""
    events, ctls, meta, conds, cond_info = [], [], None, set(), None
    for d in run_dirs:
        data = json.loads((Path(d) / "t2-events.json").read_text())
        events += data["events"]; ctls += data["controls"]
        info = json.loads((Path(d) / "run.json").read_text())
        key = (info["provider"], info["model"])
        if meta is not None and meta != key:
            raise ValueError("combine only runs of the same provider/model")
        condition = run_condition(info)
        if conds and condition not in conds:
            raise ValueError(f"combine only runs of one caller condition (got {sorted(conds | {condition})}); "
                             "pool each condition separately")
        conds.add(condition); cond_info = cond_info or info.get("condition")
        meta = key
    summary = summarize(events, ctls, cluster="spec_id")
    summary.update(provider=meta[0], model=meta[1], sources=[str(Path(d).resolve()) for d in run_dirs],
                   ranking_permitted=False, condition_provenance=cond_info)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    _json(out / "t2-summary.json", summary)
    (out / "t2-summary.md").write_text(render(summary, summary))
    return summary


# ---- offline re-score ------------------------------------------------------------------

RESCORE_SCHEMA = "turn-bench/track2-rescore/1"


def _saved_measurement(conv_dir: Path):
    """Heard agent audio of one saved session (agent.wav + timelines.json), or None."""
    tl_path = conv_dir / "timelines.json"
    if not tl_path.is_file():
        return None, None
    data = json.loads(tl_path.read_text())
    timeline = service_bridge.audio().AudioTimeline.from_json(data["agent"])
    pcm = b""
    wav_path = conv_dir / "agent.wav"
    if wav_path.is_file():
        import wave

        with wave.open(str(wav_path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != timeline.rate:
                raise ValueError(f"{wav_path}: expected mono 16-bit PCM at {timeline.rate} Hz")
            pcm = w.readframes(w.getnframes())
    return SimpleNamespace(agent_pcm=pcm, agent_timeline=timeline), data.get("observed_until_s")


def rescore_row(row: dict, measurement, vad) -> dict:
    """Re-run the current speech-cessation scoring on one saved event/control row.

    The injection (or control) instant, observation bounds and every void the
    live run decided (no anchor, timing, pacing, session errors, control
    selection) are kept; only the scoring step and its own voids are re-run.
    """
    base = {k: v for k, v in row.items() if k not in speech.SCORING_OUTPUT_KEYS}
    if base.get("void") in speech.SCORING_VOIDS:
        base["void"] = None
    if base.get("void") is None and measurement is None:
        base["void"] = "rescore_saved_audio_missing"
    return speech.score_event(base, measurement, vad)


def rescore_run(run_dir: Path, out: Path, *, vad_model: Path = VAD_MODEL, allow_incomplete: bool = False) -> dict:
    """Re-score a saved ``t2-run`` directory offline (no provider contact).

    Reads ``run.json`` and every ``<conversation>-r<k>/result.json`` with its
    saved ``agent.wav`` + ``timelines.json``; writes a new run directory under
    ``out`` (``run.json``, ``t2-events.json``, ``t2-summary.{json,md}``,
    ``rescore-diff.json``) that ``t2-combine`` accepts. ``run_dir`` is only read.
    """
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    if out == run_dir or run_dir in out.parents:
        raise ValueError("refusing to write a re-score inside the source run directory")
    info = json.loads((run_dir / "run.json").read_text())
    if info.get("schema") != SCHEMA:
        raise ValueError(f"not a Track 2 run ({info.get('schema')!r})")
    if info.get("status") != "complete" and not allow_incomplete:
        raise ValueError(f"run status is {info.get('status')!r}, not complete (still running?)")
    order = [f"{c['conversation_id']}-r{c.get('repeat', 1)}" for c in info.get("conversations", [])]
    found = sorted(p.parent.name for p in run_dir.glob("*-r*/result.json"))
    names = [n for n in order if n in found] + [n for n in found if n not in order]
    vad = SpeechVeto(vad_model)
    events, ctls, diff = [], [], {"rows": 0, "outcome_changed": 0, "void_changed": 0, "stop_ms_changed": 0,
                                  "by_transition": Counter()}
    for name in names:
        result = json.loads((run_dir / name / "result.json").read_text())
        measurement, _ = _saved_measurement(run_dir / name)
        for src_rows, dst in ((result.get("events", []), events), (result.get("controls", []), ctls)):
            for old in src_rows:
                new = rescore_row(old, measurement, vad)
                dst.append(new)
                diff["rows"] += 1
                before = old.get("void") or ("stopped" if old.get("stopped") else "kept_speaking")
                after = new.get("void") or ("stopped" if new.get("stopped") else "kept_speaking")
                if before != after:
                    diff["by_transition"][f"{before} -> {after}"] += 1
                    if old.get("void") or new.get("void"):
                        diff["void_changed"] += 1
                    else:
                        diff["outcome_changed"] += 1
                elif not new.get("void") and new.get("stop_ms") != old.get("stop_ms"):
                    diff["stop_ms_changed"] += 1
    diff["by_transition"] = dict(diff["by_transition"].most_common())
    out.mkdir(parents=True, exist_ok=False)
    summary = summarize(events, ctls)
    summary.update(provider=info["provider"], model=info["model"], run_id=run_dir.name, ranking_permitted=False,
                   condition_provenance=info.get("condition"))
    if summary["condition"] != run_condition(info) and events:
        raise ValueError(f"rows say condition {summary['condition']!r}, run.json says {run_condition(info)!r}")
    rescore = {"schema": RESCORE_SCHEMA, "at": utc_now(), "source_run": str(run_dir),
               "source_speech_policy_schema": (info.get("speech_policy") or {}).get("schema"),
               "speech_policy": speech.policy(), "vad": vad.provenance, "harness": harness_git_state(),
               "provider_contacted": False,
               "kept_from_live_run": "injection and control instants, observation bounds, live/selection voids "
                                     "(decided by the source run's own speaking rule)",
               "rescored": "speech-cessation scoring and its own exclusions"}
    summary["rescore"] = rescore
    _json(out / "run.json", {**info, "rescore": rescore})
    _json(out / "t2-events.json", {"events": events, "controls": ctls})
    _json(out / "t2-summary.json", summary)
    _json(out / "rescore-diff.json", diff)
    (out / "t2-summary.md").write_text(render(summary, summary))
    return {"summary": summary, "diff": diff}
