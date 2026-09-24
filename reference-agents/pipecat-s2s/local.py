"""Run the reference agent on this machine the way the platform runs it in the cloud.

The platform never dials a number for an agent bench row. It asks Pipecat Cloud
to start a session with a body -- provider, model, agent definition, a run id --
and gets back a Daily room; the simulated caller then joins that room and the
call happens there. This file does the same three things on one machine, with
nothing changed in ``bot.py``: the same body, a room of its own, the same
``bot()`` entry point through the same ``DailyRunnerArguments``, the same
tracing SDK, and the same payload posted at the end -- except that the payload
lands on a receiver here instead of on the platform, where it can be read.

That is the point of this file. When a run scores strangely, the question is
whether the harness or the model produced it, and the fastest way to answer is
to place one call on a laptop and read exactly what the platform would have
been sent: the transcript, the tool rows, the log, the record. Three commands::

    python local.py receive                 # the stand-in for the platform: keeps every payload
    python local.py call --provider gemini-live --agent-dir medicare
                                            # mint a room, answer in it, post the payload
    python local.py inspect data/local-runs/<file>.json
                                            # read the payload the way the score will read it

``call`` prints the room and a second token before it answers. Join the room
from a browser to be the caller yourself, or hand the room and that token to a
simulated caller. Everything the agent needs comes
from a ``.env`` at the repository root (see the README's credential table); the
Daily key may also be spelled ``daily_api_key`` there, which this file maps.

Nothing here is used by a deployed run and nothing here changes what one does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = ROOT / "data" / "local-runs"
STAMP = re.compile(r"^\[\d\d:\d\d\] ")
CONTROL_TOKEN = re.compile(r"<ctrl\d+>")


# ── call: the agent, in a room, the way a session starts it ──────────────────

# Settings that make a run local rather than deployed. Re-applied after the
# agent is imported, because importing it reloads the repository's .env.
_RIG_SETTINGS = ("CEKURA_HOST", "CEKURA_OTEL_TRACES", "BENCH_LOG_LEVEL")


def _environment(env_file: Path, receiver: str) -> dict[str, str]:
    """The deployment's environment, assembled from a local file.

    Credentials stay in the file and reach this process only; nothing is
    printed. The tracing SDK is pointed at the local receiver and its span
    exporter is switched off, so a call finalises promptly instead of retrying
    against a collector it cannot reach. The SDK needs *a* key and agent id to
    run at all, so placeholders stand in when the file has none -- the receiver
    does not check them.
    """
    from dotenv import dotenv_values

    values = {k: v for k, v in dotenv_values(env_file).items() if v} if env_file.exists() else {}
    env = {**os.environ, **values}
    if not env.get("DAILY_API_KEY") and env.get("daily_api_key"):
        env["DAILY_API_KEY"] = env["daily_api_key"]
    env.setdefault("CEKURA_API_KEY", "local")
    env.setdefault("CEKURA_AGENT_ID", "0")
    env["CEKURA_HOST"] = receiver
    env.setdefault("CEKURA_OTEL_TRACES", "0")
    env.setdefault("BENCH_LOG_LEVEL", "DEBUG")
    return env


def _room(api_key: str, minutes: int) -> tuple[str, str, str]:
    """A fresh room and two tokens: one for the agent, one for whoever calls it.

    The room is created here rather than by Pipecat's development runner, which
    always names the room it creates, and some Daily domains refuse a named
    room. Everything else is what it does: an expiring room, an owner token
    each side, and the same ``DailyRunnerArguments`` handed to the same
    ``bot()`` the deployment runs.
    """

    def post(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"https://api.daily.co/v1/{path}",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise SystemExit(f"Daily refused {path}: {error.read().decode('utf-8', 'replace')}") from None

    expiry = int(time.time()) + minutes * 60
    room = post("rooms", {"privacy": "private", "properties": {"exp": expiry, "eject_at_room_exp": True}})
    tokens = [
        post("meeting-tokens", {"properties": {"room_name": room["name"], "is_owner": True, "exp": expiry}})["token"]
        for _ in range(2)
    ]
    return room["url"], tokens[0], tokens[1]


def call(args: argparse.Namespace) -> int:
    """Run one call: mint a room, then answer in it with the row's configuration.

    The body is the row's configuration plus a run id -- the same keys the agent
    reads from a session in the cloud, sent the same way.
    """
    env = _environment(Path(args.env), args.receiver)
    if not env.get("DAILY_API_KEY"):
        print("DAILY_API_KEY (or daily_api_key) is not set; no room can be created", file=sys.stderr)
        return 2

    body = {
        "s2s_provider": args.provider,
        "agent_dir": args.agent_dir,
        "cekura_run_id": args.run_id or int(time.time()),
    }
    if args.model:
        body["s2s_model"] = args.model
    if args.voice:
        body["s2s_voice"] = args.voice
    for extra in args.set or []:
        key, _, value = extra.partition("=")
        body[key] = value

    room_url, agent_token, caller_token = _room(env["DAILY_API_KEY"], args.minutes)
    session_id = f"local-{args.provider}-{int(time.time())}"
    # Printed and flushed before the agent starts, because whoever is going to
    # call has to be able to join before the agent gives up waiting.
    print(json.dumps({
        "sessionId": session_id, "dailyRoom": room_url,
        "dailyToken": agent_token, "callerToken": caller_token, "body": body,
    }), flush=True)
    print(f"\njoin as the caller: {room_url}", file=sys.stderr, flush=True)

    os.environ.update(env)
    sys.path.insert(0, str(HERE))
    sys.argv = [sys.argv[0]]  # the agent's own runner must not see our arguments
    import asyncio

    import bot  # noqa: E402 -- after the environment is in place, as a deployment imports it

    # Importing the agent loads the repository's own .env over the environment,
    # exactly as it does in the cloud. The few settings that point the run at
    # this machine rather than at the platform are therefore re-applied after it.
    os.environ.update({key: env[key] for key in _RIG_SETTINGS if key in env})
    from pipecat.runner.types import DailyRunnerArguments  # noqa: E402

    arguments = DailyRunnerArguments(
        room_url=room_url, token=agent_token, body=body, session_id=session_id
    )
    arguments.handle_sigint = False
    try:
        asyncio.run(asyncio.wait_for(bot.bot(arguments), timeout=args.timeout))
    except asyncio.TimeoutError:
        print(f"the agent was still in the call after {args.timeout}s", file=sys.stderr)
        return 1
    return 0


# ── receive: the stand-in for the platform ───────────────────────────────────

class _Receiver(BaseHTTPRequestHandler):
    out: Path = RUNS

    def do_POST(self) -> None:  # noqa: N802 -- http.server's name
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {"raw": raw.decode("utf-8", "replace")}
        session = str(payload.get("session_id") or "no-session")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / f"{stamp}-{session}{'' if self.path.count('/') < 3 else '-' + self.path.strip('/').split('/')[-2]}.json"
        path.write_text(json.dumps(payload, indent=1))
        print(f"\n{self.path} -> {path}")
        if "transcript" in payload:
            print(report(payload, path))
        body = b'{"success": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"receiver up\n")

    def log_message(self, *_args) -> None:  # quiet; the payload summary is the output
        return


def receive(args: argparse.Namespace) -> int:
    _Receiver.out = Path(args.out)
    server = ThreadingHTTPServer((args.host, args.port), _Receiver)
    print(f"receiving on http://{args.host}:{args.port}; payloads kept under {args.out}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


# ── inspect: read a payload the way the score reads it ───────────────────────

def checks_for(payload: dict) -> tuple[list[tuple[str, bool, str]], dict, list, dict]:
    """Every check a run has to pass, with what each one saw."""
    transcript = payload.get("transcript") or []
    meta = ((payload.get("provider_data") or {}).get("custom_metadata")) or {}
    logs = payload.get("logs") or []

    users = [row for row in transcript if row.get("role") == "user"]
    # A caller row that opens by repeating its own first words is the mark of a
    # service that restates the whole turn on every final.
    restated = 0
    for row in users:
        words = str(row.get("content") or "").split()
        if len(words) >= 6 and " ".join(words[:3]) in " ".join(words[3:]):
            restated += 1
    agent_text = [row for row in transcript if row.get("role") == "assistant" and row.get("content")]
    requests = [row for row in transcript if row.get("role") == "assistant" and row.get("tool_calls")]
    requested = [call.get("function", {}).get("name") for row in requests for call in row["tool_calls"]]
    control = sum(1 for row in transcript if CONTROL_TOKEN.search(json.dumps(row.get("content") or "")))
    recorded = meta.get("tool_calls") or []
    usage = meta.get("usage") or {}
    stamped = sum(1 for line in logs if STAMP.match(line.get("message", "")))
    # What a native row needs before its cost can sit beside another's: either
    # the tokens split out by audio, or the seconds a per-second model billed.
    priceable = ", ".join(
        key for key in ("input_audio_tokens", "output_audio_tokens", "live_audio_seconds") if key in usage
    )
    debug = sum(1 for line in logs if line.get("level") == "DEBUG")
    reply = (meta.get("timing") or {}).get("reply") or {}

    checks = [
        ("caller turns in the transcript", len(users) > 0, f"{len(users)}"),
        ("agent turns with words", len(agent_text) > 0, f"{len(agent_text)}"),
        ("no control tokens in what the agent said", control == 0, f"{control} row(s)"),
        ("tool rows in the transcript match the record",
         len(requested) == len(recorded), f"{len(requested)} in transcript, {len(recorded)} on the record"),
        ("record was finished (tools + usage present)", "tool_call_count" in meta and "usage" in meta,
         ", ".join(sorted(k for k in ("tool_call_count", "usage") if k in meta)) or "neither"),
        ("no caller row restates itself", restated == 0, f"{restated} of {len(users)} row(s)"),
        ("usage was reported by the provider", usage.get("usage_reports", 0) > 0, f"{usage.get('usage_reports', 0)} report(s)"),
        # Audio tokens cost a multiple of text tokens, so a native row reporting
        # one undifferentiated total cannot be priced on the same basis as one
        # that splits them -- and a cost column that mixes the two is not a
        # comparison. A cascade is billed by seconds and characters instead.
        ("usage can be priced beside the other rows",
         meta.get("stack") != "native" or bool(priceable), priceable or "totals only: the speech half is not separable"),
        ("reply latency was measured", bool(reply),
         f"p50 {reply['p50_ms']} ms, p90 {reply['p90_ms']} ms over {reply['count']} reply(s)"
         if reply else "no reply interval recorded"),
        ("log lines carry the call clock", bool(logs) and stamped == len(logs), f"{stamped}/{len(logs)}"),
        ("log includes the framework's DEBUG lines", debug > 0, f"{debug} DEBUG of {len(logs)}"),
    ]
    return checks, meta, recorded if recorded else requested, usage


def report(payload: dict, path: Path | None = None) -> str:
    """One screen on a run: what reached the record, and what did not."""
    checks, meta, tools, usage = checks_for(payload)
    recorded = (((payload.get("provider_data") or {}).get("custom_metadata")) or {}).get("tool_calls") or []
    requested = [] if recorded else tools
    misses = sum(
        1 for row in (payload.get("transcript") or [])
        if row.get("role") == "tool" and "no_match" in json.dumps(row.get("content", ""))
    )
    width = max(len(name) for name, _, _ in checks)
    lines = []
    if path is not None:
        lines.append(f"run: {path.name}")
    lines.append(
        "config: " + ", ".join(
            f"{k}={meta.get(k)}" for k in ("s2s_provider", "s2s_model", "agent_definition", "turn_source", "agent_commit", "worker_call")
            if k in meta
        )
    )
    for name, ok, detail in checks:
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    if recorded:
        lines.append("tools: " + ", ".join(
            f"{call['name']}:{call.get('resolution', 'exact' if call.get('matched') else 'none')}"
            for call in recorded
        ))
    elif requested:
        lines.append("tools (transcript only): " + ", ".join(str(n) for n in requested))
    if recorded or requested:
        if misses:
            lines.append(f"  {misses} tool answer(s) were no_match")
    if usage:
        lines.append("usage: " + ", ".join(f"{k}={v}" for k, v in usage.items()))
    return "\n".join(lines)


def inspect(args: argparse.Namespace) -> int:
    failed = 0
    for name in args.payload:
        path = Path(name)
        payload = json.loads(path.read_text())
        print(report(payload, path), end="\n\n")
        failed += sum(1 for _, ok, _ in checks_for(payload)[0] if not ok)
        if args.logs:
            for line in payload.get("logs") or []:
                print(f"{line.get('level', ''):8} {line.get('message', '')}")
    return 1 if failed else 0


# ── entry ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("call", help="mint a room and answer in it with one row's configuration")
    p.add_argument("--provider", required=True)
    p.add_argument("--agent-dir", required=True, choices=sorted(d.name for d in (ROOT / "agent-definitions").iterdir() if d.is_dir()))
    p.add_argument("--model")
    p.add_argument("--voice")
    p.add_argument("--run-id", type=int, help="stands in for the platform's run id")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="any other session key")
    p.add_argument("--env", default=str(ROOT / ".env"), help="dotenv file holding the credentials")
    p.add_argument("--receiver", default="http://127.0.0.1:8765", help="where the tracing SDK posts the payload")
    p.add_argument("--minutes", type=int, default=20, help="how long the room lives")
    p.add_argument("--timeout", type=int, default=600, help="give up if the call has not ended by then")
    p.set_defaults(run=call)

    p = sub.add_parser("receive", help="stand in for the platform webhook and keep what the SDK posts")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--out", default=str(RUNS), help="where payloads are written")
    p.set_defaults(run=receive)

    p = sub.add_parser("inspect", help="read one or more payloads the way the score reads them")
    p.add_argument("payload", nargs="+")
    p.add_argument("--logs", action="store_true", help="print the captured log after the summary")
    p.set_defaults(run=inspect)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
