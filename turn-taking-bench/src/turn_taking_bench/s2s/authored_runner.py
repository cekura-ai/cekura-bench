"""Opt-in, experimental live caller for the authored interaction scenarios.

The v0.5 runner, official scorer, and service bench are unchanged. Caller
stimulus boundaries are transmitted PCM sample indices, not inferred acoustic
onsets. Agent scripts are control markers only; they are never spoken or put
in the model prompt. Preflight rejects unavailable inputs before any session.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from turn_taking_bench.author import service_bridge
from turn_taking_bench.runner import harness_git_state, turnbench_provenance, utc_now
from turn_taking_bench.s2s import SYSTEM_PROMPT, default_config, prompt_sha256
from turn_taking_bench.s2s.authored_metrics import render_markdown, score_event, summarize
from turn_taking_bench.s2s.authored_plan import load_clips
from turn_taking_bench.s2s.model import adapter_class, api_key, provider_entry, resolve_model, session_config
from turn_taking_bench.s2s.onsets import response_starts, speech_segments

service_bridge._ensure_path()
from service.caller import BranchingCaller, VOID_SLIP_MS
from service.events import Clock, EventLog

SCHEMA = "turn-taking-bench/authored-live-run/1"
BOUNDARY = "first transmitted PCM sample; exact stimulus injection, not acoustic onset"
OBSERVATION_S = 1.2


def _json(path: Path, value: Any) -> None:
    # Replace one whole checkpoint so a reviewer never reads a half JSON file.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


class SafeLog(EventLog):
    """Reuse normalized/raw logs while omitting credentials from payloads."""

    def __init__(self, *args, secret: str = "", **kwargs):
        self._secret = secret
        super().__init__(*args, **kwargs)

    def clean(self, value):
        if isinstance(value, dict):
            sensitive = {"api_key", "apikey", "authorization", "password", "secret",
                         "access_key", "secret_access_key", "session_token", "token"}
            return {k: "[redacted]" if k.lower().replace("-", "_") in sensitive else self.clean(v)
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        if isinstance(value, str) and self._secret:
            return value.replace(self._secret, "[redacted]")
        return value

    def emit(self, kind, at=None, **data):
        return super().emit(kind, at=at, **self.clean(data))

    def raw(self, payload, direction="in"):
        return super().raw(self.clean(payload), direction)


def _alive(caller, adapter):
    if caller.failed or adapter.closed.is_set() or any(e.kind == 'session.error' for e in adapter.log.events):
        raise RuntimeError("provider session or caller carrier closed")


async def _until(caller, adapter, target):
    while caller.log.clock.now() < target:
        _alive(caller, adapter)
        await asyncio.sleep(min(0.01, target - caller.log.clock.now()))
    _alive(caller, adapter)


async def _play(caller, adapter, clip):
    """BranchingCaller accounting, with completion after the final send."""
    mark = adapter.agent_timeline.n_samples
    utterance = await caller.play(clip)
    # BranchingCaller signals completion while preparing the final chunk, just
    # before send_audio records it. Do not look up its sample before that send.
    while adapter.caller_timeline.n_samples < utterance.last_sample:
        _alive(caller, adapter)
        await asyncio.sleep(0.001)
    start = adapter.caller_timeline.time_of_sample(utterance.first_sample)
    last = adapter.caller_timeline.time_of_sample(utterance.last_sample - 1)
    if start is None or last is None:
        raise RuntimeError("transmitted caller samples are not on the timeline")
    return {"caller_start_s": start, "caller_end_s": last + 1 / adapter.input_rate,
            "caller_first_sample": utterance.first_sample, "caller_last_sample": utterance.last_sample,
            "detected_speech_start_sample": utterance.speech_start_sample,
            "detected_speech_end_sample": utterance.speech_end_sample,
            "detected_speech_bounds_are_exact": False, "boundary_definition": BOUNDARY,
            "sent_pcm_sha256": utterance.clip.sha256, "sent_rate": utterance.clip.rate,
            "agent_mark_before_caller": mark}


async def _anchor(caller, adapter, mark: int, earliest: float, timeout_s: float):
    """Observe speech, not a provider's audio-start flag or leading padding."""
    deadline = caller.log.clock.now() + timeout_s
    while caller.log.clock.now() < deadline:
        _alive(caller, adapter)
        starts = response_starts(adapter.log.events)
        rate = adapter.output_rate
        for response, start in enumerate(starts):
            if start < mark:
                continue
            end = starts[response + 1] if response + 1 < len(starts) else adapter.agent_timeline.n_samples
            # Only the new reply's first eight seconds are needed to find its
            # first speech onset. Scanning prior conversations during each live
            # poll would itself create pacing slips.
            end = min(end, start + 8 * rate)
            segments = speech_segments(bytes(adapter.agent_pcm[start * 2:end * 2]), rate)
            if not segments:
                continue
            sample = start + segments[0].start_frame * int(rate * 0.01)
            # Use the actual buffered client clock, including playback cuts.
            chunk = adapter.agent_timeline.chunk_of_sample(sample)
            if chunk is None:
                continue
            begin, end = adapter.agent_timeline.playout_span(chunk.seq)
            at = adapter.agent_timeline.playout_time_of_sample(sample)
            if at is not None and earliest <= at < end and end > begin:
                await _until(caller, adapter, at)
                return {"t": at, "sample": sample, "response": response,
                        "response_sample": start, "observed_at_s": caller.log.clock.now(),
                        "detector": {"frame_ms": 10.0, "segment_start_frame": segments[0].start_frame,
                                     "search_prefix_max_s": 8.0}}
        await asyncio.sleep(0.02)
    return None


async def _quiet(caller, adapter, timeout_s):
    deadline = caller.log.clock.now() + timeout_s
    while caller.log.clock.now() < deadline:
        _alive(caller, adapter)
        end = adapter.agent_timeline.playout_end()
        if end is not None and caller.log.clock.now() >= end + 0.2:
            return end
        await asyncio.sleep(0.02)
    raise RuntimeError("agent quiet timeout; caller turn cannot be placed")


def _new_reply_before(adapter, anchor, target):
    for sample in response_starts(adapter.log.events):
        if sample <= anchor["response_sample"]:
            continue
        at = adapter.agent_timeline.playout_time_of_sample(sample)
        if at is not None and at <= target:
            return True
    return False


def _event(conversation, item, repeat):
    return {"conversation_id": conversation['conversation_id'], "repeat": repeat,
            "source_conversation_id": conversation["conversation_id"],
            "spec_id": conversation["spec_id"], "event_id": item["id"],
            "kind": item["kind"], "offset_s": item["offset_s"], "into": item["into"],
            "boundary_definition": BOUNDARY, "anchor_s": None, "target_s": None,
            "caller_start_s": None, "caller_end_s": None, "void": None}


async def _drive(conversation, clips, caller, adapter, repeat, records,
                 onset_timeout_s, quiet_timeout_s, checkpoint):
    anchors, spoken = {}, {}
    for index, item in enumerate(conversation["items"]):
        _alive(caller, adapter)
        if item["speaker"] == "agent":
            trigger = spoken.get(item.get("trigger_caller_id"))
            mark, earliest = (trigger["agent_mark_before_caller"], trigger["caller_end_s"]) if trigger else (0, 0.0)
            anchors[item["id"]] = await _anchor(caller, adapter, mark, earliest, onset_timeout_s)
            current = anchors[item["id"]]
            if current is not None and any(a is not None and a['response_sample'] == current['response_sample']
                                           for name, a in anchors.items() if name != item['id']):
                anchors[item['id']] = None  # a script marker cannot reuse an old reply
            continue
        if item["kind"] == "turn":
            after = item.get("after")
            if after is None and index and conversation['items'][index - 1]['speaker'] == 'agent':
                after = conversation['items'][index - 1]['id']
            if after in anchors and anchors[after] is not None:
                end = await _quiet(caller, adapter, quiet_timeout_s)
                target = end + (item.get("hold_s") or item.get("gap_s") or 0)
            elif after in spoken:
                target = spoken[after]["caller_end_s"] + (item.get("hold_s") or item.get("gap_s") or 0)
            elif not spoken:
                target = caller.log.clock.now() + conversation["timing"].get("start_s", 0)
            else:
                target = caller.log.clock.now() + (item.get("hold_s") or item.get("gap_s") or 0)
            await _until(caller, adapter, target)
            spoken[item["id"]] = await _play(caller, adapter, clips[item["id"]])
            continue
        record = _event(conversation, item, repeat)
        records.append(record)
        anchor = anchors.get(item["into"])
        if anchor is None:
            record["void"] = "no_live_agent_speech_anchor"
            checkpoint()
            continue
        target = anchor["t"] + item["offset_s"]
        record.update(anchor_s=anchor["t"], anchor_sample=anchor["sample"],
                      anchor_observed_at_s=anchor['observed_at_s'], target_s=target)
        await _until(caller, adapter, target)
        if _new_reply_before(adapter, anchor, target):
            record["void"] = "anchor_reply_replaced_before_injection"
        elif not adapter.agent_speaking:
            record["void"] = "agent_finished_before_injection_point"
        elif caller.log.clock.now() - target > VOID_SLIP_MS / 1000:
            record["void"] = "injection_target_already_missed"
        else:
            actual = await _play(caller, adapter, clips[item["id"]])
            spoken[item["id"]] = actual
            record.update(actual)
            record["injection_lateness_ms"] = (actual["caller_start_s"] - target) * 1000
            if abs(record["injection_lateness_ms"]) > VOID_SLIP_MS:
                record["void"] = "injection_timing_outside_25ms"
        checkpoint()
    last_needed = max((r["caller_start_s"] + OBSERVATION_S for r in records
                       if r.get("caller_start_s") is not None), default=caller.log.clock.now())
    await _until(caller, adapter, max(last_needed, caller.log.clock.now() + 0.2))
    return spoken


async def run_conversation(conversation, cache_dir, config, out, *, repeat=1,
                           adapter_factory=None, adapter_kwargs=None,
                           onset_timeout_s=20.0, quiet_timeout_s=45.0, timeout_s=600.0):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    planned = [i for i in conversation["items"] if i["speaker"] == "caller" and i["kind"] in {"barge_in", "backchannel"}]
    records = []; session_error = None
    if conversation.get("void"):
        records = [{**_event(conversation, i, repeat), "void": conversation["void"]} for i in planned]
        _json(out / "result.json", {"events": records, "void": conversation["void"]})
        return records
    clips = load_clips(conversation, Path(cache_dir))  # before reading a key or connecting
    entry = provider_entry(config["provider"])
    session = session_config(config, entry)
    if session.instructions != SYSTEM_PROMPT or session.turn_detection.is_manual:
        raise ValueError("authored run requires the fixed prompt and native turn detection")
    try:
        key = api_key(config)
        clock = Clock(); log = SafeLog(clock, out / "events.jsonl", out / "raw.jsonl", secret=key)
        cls = adapter_factory or adapter_class(config)
        adapter = cls(model=resolve_model(config), api_key=key, config=session, log=log, **(adapter_kwargs or {}))
    except Exception as exc:
        reason = f'{type(exc).__name__}: session setup failed before streaming'
        records = [score_event({**_event(conversation, i, repeat), 'void': reason}, None) for i in planned]
        _json(out / 'result.json', {'events': records, 'error': reason})
        if 'log' in locals():
            log.close()
        return records
    caller = BranchingCaller(adapter, log)
    spoken = {}
    checkpoint = lambda: _json(out / "stimuli.json", {"events": log.clean(records)})
    observed = 0.0
    try:
        reason = cls.unsupported_reason(session)
        if reason:
            raise RuntimeError(f"unsupported session: {reason}")
        await asyncio.wait_for(adapter.connect(), timeout=40.0)
        async with caller:
            spoken = await asyncio.wait_for(
                _drive(conversation, clips, caller, adapter, repeat, records,
                       onset_timeout_s, quiet_timeout_s, checkpoint), timeout=timeout_s)
            observed = clock.now()
    except Exception as exc:
        # Provider exceptions can echo arbitrary request contents. Their class
        # is enough for a safe top-level exclusion; normalized logs hold evidence.
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
            records.append({**_event(conversation, item, repeat), "void": session_error or "event_not_reached"})
    pacing = {**caller.pacing(), 'pacing_void': caller.max_slip_ms > VOID_SLIP_MS,
              'max_slip_ms_unrounded': caller.max_slip_ms}
    timeline_data = {"caller": adapter.caller_timeline.as_json(),
                     "agent": adapter.agent_timeline.as_json(), "observed_until_s": observed}
    _json(out / "timelines.json", timeline_data)
    audio = service_bridge.audio()
    canonical_caller = audio.AudioTimeline.from_json(timeline_data['caller'])
    measurement = SimpleNamespace(agent_pcm=bytes(adapter.agent_pcm),
                                  agent_timeline=audio.AudioTimeline.from_json(timeline_data['agent']))
    # Every caller start, including ordinary turns, can contaminate a window.
    starts = [(u.first_sample, canonical_caller.time_of_sample(u.first_sample)) for u in caller.utterances]
    scored = []
    for record in records:
        record["observation_end_s"] = observed
        record["pacing"] = pacing
        if record["caller_start_s"] is not None:
            # Score the persisted clock representation, so a re-read of the
            # artifact files reproduces each cell exactly (timestamps at 1 us).
            record['caller_start_s'] = canonical_caller.time_of_sample(record['caller_first_sample'])
            record['caller_end_s'] = canonical_caller.time_of_sample(record['caller_last_sample'] - 1) + 1 / adapter.input_rate
            record["next_caller_start_s"] = min((at for sample, at in starts
                                                if at is not None and sample > record['caller_first_sample']), default=None)
        if session_error:
            record["void"] = session_error
        if pacing['pacing_void']:
            record["void"] = "host_pacing_outside_25ms"
        scored.append(score_event(record, measurement))
    if adapter.caller_pcm:
        audio.write_wav(out / "caller.wav", bytes(adapter.caller_pcm), adapter.input_rate)
    if adapter.agent_pcm:
        audio.write_wav(out / "agent.wav", bytes(adapter.agent_pcm), adapter.output_rate)
    _json(out / "result.json", {"events": log.clean(scored), "error": session_error,
                                "provider": config["provider"], "model": adapter.model,
                                "prompt_sha256": prompt_sha256(), "boundary_definition": BOUNDARY,
                                "pacing": pacing, "observed_until_s": observed,
                                "session_acknowledged": log.clean(adapter.session_ack),
                                "usage": service_bridge.metrics().usage(adapter)})
    log.close()
    return scored


def run_authored(plan: dict, out: Path, *, provider: str, model: str | None = None,
                 repeats: int = 1, env_file: str | None = None,
                 track1: Path | None = None) -> dict:
    """One session at a time per provider; independent providers can be capped externally."""
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    # Validate every selected clip before spending on the first provider session.
    for conversation in plan["conversations"]:
        load_clips(conversation, Path(plan["cache_dir"]))
    config = default_config(provider)
    config["env_file"] = env_file
    if model:
        config["model"] = model
    track1_data = None
    if track1 is not None:
        report = json.loads((Path(track1) / 'report.json').read_text())
        run = json.loads((Path(track1) / 'run.json').read_text())
        if (run['model']['name'] != f's2s:{provider}'
                or run['model']['provenance']['model'] != resolve_model(config)
                or run['model']['provenance']['prompt_sha256'] != prompt_sha256()
                or run['turnbench']['commit'] != turnbench_provenance()['commit']):
            raise ValueError('Track 1 provider/model/prompt/scorer does not match the authored run')
        # Validate every field needed by the optional report before a session.
        # A missing/undefined score must not fail formatting after provider spend.
        try:
            count = report['conversations_scored']
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError('invalid conversation count')
            cells = []
            for name in ('pause_handling', 'turn_taking'):
                value = report['blocks']['caller']['scorecard'][name]['value']
                if value is not None and (isinstance(value, bool) or
                        not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError('invalid score value')
                cells.append('n/a' if value is None else f'{value:.3f}')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Track 1 report lacks valid caller scorecard fields') from exc
        track1_data = (report, run)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    _json(out / "plan.json", plan)
    info = {"schema": SCHEMA, "status": "running", "started_at": utc_now(),
            "provider": provider, "model": resolve_model(config), "repeats": repeats,
            "prompt_sha256": prompt_sha256(), "boundary_definition": BOUNDARY,
            "harness": harness_git_state(), "turnbench": turnbench_provenance(),
            "label": "EXPERIMENTAL; offline validation does not authorize ranking or publication",
            "conversations": []}
    _json(out / "run.json", info)
    records = []

    async def sessions():
        for repeat in range(1, repeats + 1):
            for conversation in plan["conversations"]:
                directory = out / f"{conversation['conversation_id']}-r{repeat}"
                result = await run_conversation(conversation, plan["cache_dir"], config, directory, repeat=repeat)
                records.extend(result)
                info["conversations"].append({"conversation_id": conversation["conversation_id"],
                                              "repeat": repeat, "events": len(result),
                                              "voids": sum(bool(r.get('void')) for r in result)})
                _json(out / "run.json", info)
                _json(out / "authored-events.json", records)
    asyncio.run(sessions())
    summary = summarize(records)
    summary.update(provider=provider, model=info["model"], run_id=out.name,
                   plan_counts=plan["counts"], prompt_sha256=prompt_sha256(),
                   ranking_permitted=False, track1_unchanged=True)
    _json(out / "authored-interaction.json", summary)
    (out / "authored-interaction.md").write_text(render_markdown(summary), encoding="utf-8")
    if track1 is not None:
        # Attach a read-only copy of the existing caller rows, never rescore or
        # overwrite v0.5. The tracks use different stimuli; no overall ranking.
        report, run = track1_data
        combined = {"label": "EXPERIMENTAL authored Track 2 + unchanged Track 1",
                    "provider": provider, "model": info['model'], "ranking_permitted": False,
                    "overall": None, "overall_reason": "different stimuli and unresolved Track 2 validity; no combined rank",
                    "track1": {"source": str(Path(track1).resolve()),
                               "conversations": report['conversations_scored'],
                               "caller": report['blocks']['caller']['scorecard']},
                    "track2": summary}
        _json(out / "authored-scorecard.json", combined)
        heading = ("# Experimental combined scorecard\n\nTrack 1 caller rows are copied unchanged from "
                   + str(Path(track1).resolve()) + ". Track 2 uses separate authored interaction stimuli. No combined overall or ranking.\n\n"
                   + "| Track 1 pause handling | Track 1 turn taking |\n|---|---|\n"
                   + f"| {cells[0]} | {cells[1]} |\n\n")
        (out / 'authored-scorecard.md').write_text(heading + render_markdown(summary))
    info.update(status="complete", finished_at=utc_now(), events=len(records), scored=summary["scored"])
    _json(out / "run.json", info)
    return summary


def replay_authored(run_dir: Path, out: Path) -> dict:
    """Replay acoustic scores from PCM/timelines, preserving saved exclusions."""
    run_dir, out = Path(run_dir), Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite replay output: {out}')
    info = json.loads((run_dir / 'run.json').read_text())
    audio = service_bridge.audio()
    rows = []
    for conv in info['conversations']:
        directory = run_dir / f"{conv['conversation_id']}-r{conv['repeat']}"
        result = json.loads((directory / 'result.json').read_text())
        if not any(not r.get('void') for r in result['events']):
            rows.extend(result['events'])
            continue
        timelines = json.loads((directory / 'timelines.json').read_text())
        pcm, rate = audio.read_wav(directory / 'agent.wav')
        adapter = SimpleNamespace(agent_pcm=pcm, agent_timeline=audio.AudioTimeline.from_json(timelines['agent']))
        if adapter.agent_timeline.rate != rate:
            raise ValueError('agent WAV rate does not match recorded timeline')
        for record in result['events']:
            replayed = score_event(record, adapter)
            if replayed != record:
                raise ValueError(f"saved cell does not reproduce: {record['conversation_id']} / {record['event_id']}")
            rows.append(replayed)
    summary = summarize(rows)
    summary.update(provider=info['provider'], model=info['model'], run_id=run_dir.name,
                   ranking_permitted=False, track1_unchanged=True, replayed_from=str(run_dir.resolve()))
    out.mkdir(parents=True)
    _json(out / 'authored-interaction.json', summary)
    (out / 'authored-interaction.md').write_text(render_markdown(summary))
    return summary
