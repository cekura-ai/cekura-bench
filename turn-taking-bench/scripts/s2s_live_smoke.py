"""Live smoke for the Azure Realtime and GPT-Live adapters (one session each).

Streams the first ``--seconds`` of a clean caller channel (channel L of an
el-v0.3 listening file, resampled 48 -> 24 kHz with the service bench's
published filter) at real-time pace in 20 ms chunks, then ``--tail`` seconds
of silence, and reports: whether the session opened, what it acknowledged, the
resolved model id, the first reply's onset relative to the end of the caller's
first turn (from the authored cues), the reply transcripts and usage.

Keys are read in-process (environment, then turn-taking-bench/.env, then the repo
root's .env, or ``--env-file``); nothing prints them. Artifacts (agent/caller
wav, events.jsonl, raw.jsonl, state.json) go to ``--out``.

    uv run --with websockets==17.2 python scripts/s2s_live_smoke.py --provider gpt-live --out /tmp/smoke
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from turn_taking_bench.author import service_bridge

service_bridge._ensure_path()

from service import audio as sa  # noqa: E402
from service import events as ev  # noqa: E402
from service.adapters.base import SessionConfig  # noqa: E402

PROMPT = (
    "You are the phone assistant for a medical practice's appointments line. "
    "Callers ring to book, change or cancel appointments and to ask simple questions. "
    "Reply briefly, in one or two short sentences, then wait for the caller to speak. "
    "Speak naturally, as on a phone call. Never mention that you are an AI."
)


def read_env(name: str, env_file: str | None) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    root = service_bridge.REPO_ROOT
    for path in [Path(env_file)] if env_file else [root / "turn-taking-bench" / ".env", root / ".env"]:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = raw.strip().removeprefix("export ").partition("=")
            if sep and key.strip() == name and value.strip():
                return value.strip().strip("\"'")
    return None


def first_caller_end(cues: Path) -> float:
    with cues.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["speaker"] == "caller" and row["event"] == "eot_positive":
                return float(row["time_s"])
    raise ValueError("no caller turn end in cues")


def make_adapter(provider: str, log, env_file):
    config = SessionConfig(instructions=PROMPT)
    if provider == "azure-realtime":
        from turn_taking_bench.s2s.azure_realtime import AzureRealtimeAdapter

        return AzureRealtimeAdapter(
            api_key=read_env("AZURE_VOICE_API_KEY", env_file) or "",
            endpoint=read_env("AZURE_VOICE_ENDPOINT", env_file),
            log=log, config=config,
        )
    if provider == "gpt-live":
        from turn_taking_bench.s2s.gpt_live import GPTLiveAdapter

        key = read_env("OPENAI_API_KEY", env_file) or read_env("OpenAI", env_file) or ""
        return GPTLiveAdapter(api_key=key, log=log, config=config)
    raise SystemExit(f"unknown provider {provider}")


async def run(args) -> dict:
    wav = Path(args.wav)
    data, rate = sf.read(wav, dtype="int16", always_2d=True)
    caller = data[: int(args.seconds * rate), 0].astype("<i2").tobytes()
    pcm = sa.resample(caller, rate, 24000)
    turn_end_s = first_caller_end(wav.with_name(wav.stem + ".cues.csv"))

    out = Path(args.out) / args.provider
    out.mkdir(parents=True, exist_ok=True)
    clock = ev.Clock()
    log = ev.EventLog(clock, out / "events.jsonl", out / "raw.jsonl")
    adapter = make_adapter(args.provider, log, args.env_file)
    result: dict = {"provider": args.provider, "wav": wav.name, "first_caller_turn_end_s": turn_end_s}
    try:
        await adapter.connect()
    except Exception as exc:  # noqa: BLE001
        result["connected"] = False
        result["error"] = str(exc)[:500]
        return result
    result["connected"] = True
    rate_out = adapter.input_rate
    if rate_out != 24000:
        pcm = sa.resample(pcm, 24000, rate_out)
    pcm += sa.silence(rate_out, args.tail * 1000)
    chunks = list(sa.iter_chunks(pcm, rate_out, 20.0))
    start = clock.now()
    for i, chunk in enumerate(chunks):
        target = start + i * 0.020
        delay = target - clock.now()
        if delay > 0:
            await asyncio.sleep(delay)
        if adapter.closed.is_set():
            break
        await adapter.send_audio(chunk, t_send=max(target, clock.now()))
    await adapter.close()
    log.close()

    turn_end_wall = start + turn_end_s
    onsets = []
    for event in log.of_kind(ev.AGENT_AUDIO_START):
        at = adapter.agent_timeline.time_of_sample(int(event.data["sample"]))
        onsets.append(round(at - start, 3) if at is not None else None)
    after = [t for t in onsets if t is not None and t >= turn_end_s]
    state = adapter.state()
    result.update(
        replied=bool(onsets),
        reply_onsets_s=onsets,
        first_reply_latency_ms=round(1000 * (after[0] - turn_end_s)) if after else None,
        onsets_before_turn_end=[t for t in onsets if t is not None and t < turn_end_s],
        resolved_model=getattr(adapter, "resolved_model", None),
        session_id=adapter.session_id,
        agent_text=adapter.agent_text,
        caller_text=adapter.caller_text,
        vad_events=[(e.kind, round(e.t - start, 3)) for e in log.of_kind(ev.VAD_SPEECH_START, ev.VAD_SPEECH_END)],
        usage=[e.data.get("usage") for e in log.of_kind(ev.RESPONSE_DONE) if e.data.get("usage")],
        live_usage=state.get("usage"),
        errors=[e.data for e in log.of_kind(ev.SESSION_ERROR)],
    )
    sa.write_wav(out / "agent.wav", bytes(adapter.agent_pcm), adapter.output_rate)
    sa.write_wav(out / "caller.wav", bytes(adapter.caller_pcm), adapter.input_rate)
    (out / "state.json").write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["azure-realtime", "gpt-live"])
    parser.add_argument("--wav", required=True, help="caller WAV to stream (e.g. a rendered call from your local build)")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--tail", type=float, default=4.0)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("connected") else 1)


if __name__ == "__main__":
    main()
