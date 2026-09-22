#!/usr/bin/env python3
"""Exercise the production Gemini Live setup without Twilio.

This is deliberately a small diagnostic, not a substitute for the PSTN tests.
It proves that each exact canonical configuration is accepted by the API and
records the time to the first model event after the configured greeting is
requested.  It never prints API keys, audio bytes, or prompt contents.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from google import genai
from google.genai import types

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR / "definitions"
MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live-extended-thinking")
LIVE_API_VERSION = "v1alpha"


def workflow_for(number: str) -> dict[str, object]:
    key = {"19715717785": "appointments", "19713912400": "insurance"}[number]
    root = ROOT / key
    return {
        "key": key,
        "prompt": (root / "system-prompt.txt").read_text(),
        "greeting": (root / "first-message.txt").read_text().strip(),
        "tools": json.loads((root / "tool-definitions.json").read_text()),
    }


def live_config(workflow: dict[str, object]) -> types.LiveConnectConfig:
    tools = workflow["tools"]
    assert isinstance(tools, list)
    declarations = [
        types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters=tool["parameters"],
            behavior="NON_BLOCKING",
        )
        for tool in tools
    ]
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        thinking_config=types.ThinkingConfig(
            thinking_level=os.environ.get("GEMINI_THINKING_LEVEL", "HIGH"),
            include_thoughts=False,
        ),
        system_instruction=types.Content(parts=[types.Part(text=str(workflow["prompt"]))]),
        tools=[types.Tool(function_declarations=declarations)],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


async def probe(number: str, timeout: float, continuous_silence: bool) -> dict[str, object]:
    workflow = workflow_for(number)
    started = time.monotonic()
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options={"api_version": LIVE_API_VERSION},
    )
    async with client.aio.live.connect(model=MODEL, config=live_config(workflow)) as session:
        connected_ms = round((time.monotonic() - started) * 1000)
        await session.send_client_content(
            turns=[types.Content(
                role="user",
                parts=[types.Part(text=f"Begin the phone call. Say exactly: {workflow['greeting']}")],
            )],
            turn_complete=True,
        )

        async def send_silence() -> None:
            while True:
                await session.send_realtime_input(
                    audio=types.Blob(data=b"\0" * 640, mime_type="audio/pcm;rate=16000"),
                )
                await asyncio.sleep(0.02)

        async def receive_first_model_event() -> dict[str, object]:
            event_count = 0
            async for message in session.receive():
                event_count += 1
                content = message.server_content
                has_audio = bool(
                    content
                    and content.model_turn
                    and any(part.inline_data for part in content.model_turn.parts)
                )
                text = bool(content and content.output_transcription and content.output_transcription.text)
                if has_audio or text or message.tool_call:
                    return {
                        "event_count": event_count,
                        "has_audio": has_audio,
                        "has_output_transcription": text,
                        "has_tool_call": bool(message.tool_call),
                    }
            raise RuntimeError("Gemini receive stream ended before a model event")

        silence_task = asyncio.create_task(send_silence()) if continuous_silence else None
        try:
            first = await asyncio.wait_for(receive_first_model_event(), timeout=timeout)
        finally:
            if silence_task:
                silence_task.cancel()
                await asyncio.gather(silence_task, return_exceptions=True)
        return {
            "workflow": workflow["key"],
            "model": MODEL,
            "api_version": LIVE_API_VERSION,
            "continuous_silence": continuous_silence,
            "connected_ms": connected_ms,
            "first_model_event_ms": round((time.monotonic() - started) * 1000),
            **first,
        }


async def tool_cycle_probe(timeout: float, response_key: str = "output", full_appointment_cycle: bool = False) -> dict[str, object]:
    """Run natural canonical tool turns without Twilio for comparison evidence."""
    workflow = workflow_for("19715717785")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": LIVE_API_VERSION})
    started = time.monotonic()
    events: list[dict[str, object]] = []

    def record(stage: str, **extra: object) -> None:
        event = {"stage": stage, "ms": round((time.monotonic() - started) * 1000), **extra}
        events.append(event)
        print(json.dumps(event), flush=True)

    async with client.aio.live.connect(model=MODEL, config=live_config(workflow)) as session:
        record("connected", response_key=response_key)

        async def send_turn(text: str) -> None:
            await session.send_client_content(
                turns=[types.Content(role="user", parts=[types.Part(text=text)])], turn_complete=True,
            )
            record("client_turn_sent")

        async def respond(fc: object) -> None:
            if fc.name not in {tool["name"] for tool in workflow["tools"]}:
                raise RuntimeError(f"Undeclared tool: {fc.name}")
            record("mock_request_started", name=fc.name, tool_id=fc.id, args=fc.args)
            async with httpx.AsyncClient(timeout=20) as http:
                response = await http.post(
                    f"https://new-prod.cekura.ai/test_framework/v1/aiagents/20909/tool/{fc.name}/",
                    json=fc.args or {},
                )
                response.raise_for_status()
                result = response.json()
            record("mock_response_received", name=fc.name, result=result)
            await session.send_tool_response(function_responses=[
                types.FunctionResponse(id=fc.id, name=fc.name, response={response_key: result}),
            ])
            record("tool_response_sent", name=fc.name, tool_id=fc.id)

        turns = [
            f"Begin the phone call. Say exactly: {workflow['greeting']}",
            "I need to reschedule my upcoming appointment.",
            "My phone number is 617-555-9210.",
        ]
        if full_appointment_cycle:
            # The first availability request intentionally remains in the
            # sequence because it reproduced the real PSTN failure.
            turns += [
                "I would like to move it to July 9th in the morning.",
                "Let's try once more, please.",
                "Let's do 10:15 with Dr. Patel.",
            ]
        for text in turns:
            await send_turn(text)
            await asyncio.wait_for(drain_interaction(session, respond, record), timeout)
        if not any(e["stage"] == "tool_response_sent" and e["name"] == "lookup_patient" for e in events):
            raise RuntimeError("Interaction ended without lookup_patient; probe did not pass")
        validate_lookup_continuation(events)
        if full_appointment_cycle:
            actual = [e["name"] for e in events if e["stage"] == "tool_response_sent"]
            expected = ["lookup_patient", "check_availability", "book_appointment", "cancel_appointment"]
            if actual != expected:
                raise RuntimeError(f"Full cycle tool order mismatch: {actual}; expected {expected}")
    return {"workflow": workflow["key"], "model": MODEL, "api_version": LIVE_API_VERSION, "events": events}


def validate_lookup_continuation(events):
    """A filler or fabricated system error is not a successful tool cycle.

    Deliberately conservative diagnostic check against this fixture's returned
    provider, not a replacement for Cekura's workflow scoring.
    """
    results = [e for e in events if e["stage"] == "mock_response_received" and e["name"] == "lookup_patient"]
    if not results:
        raise RuntimeError("No lookup result to validate")
    result_event = results[-1]
    appointments = result_event["result"].get("upcoming_appointments", [])
    providers = [a["provider"].lower().replace("dr.", "").strip() for a in appointments if a.get("provider")]
    text = "".join(e["text"] for e in events if e["stage"] == "output_transcription" and e["ms"] > result_event["ms"]).lower()
    if not providers or not any(provider in text for provider in providers):
        raise RuntimeError("Tool result was returned, but no grounded appointment readback was observed; probe did not pass")


async def drain_interaction(session, respond, record):
    """Keep receiving across utterances and dispatch tools before awaiting IDLE.

    The reader and tool workers share a TaskGroup: worker errors surface and all
    children are awaited before the caller closes the SDK session.
    """
    seen = set()
    async with asyncio.TaskGroup() as workers:
        while True:
            async for message in session.receive():
                if message.go_away:
                    raise RuntimeError("Gemini sent goAway")
                if message.tool_call_cancellation:
                    raise RuntimeError("Tool cancelled during uninterrupted diagnostic")
                if message.tool_call:
                    for fc in message.tool_call.function_calls or []:
                        if not fc.id or fc.id in seen:
                            raise RuntimeError(f"Missing or duplicate tool ID: {fc.id}")
                        seen.add(fc.id)
                        record("tool_call", name=fc.name, tool_id=fc.id)
                        workers.create_task(respond(fc))
                sc = message.server_content
                if not sc:
                    continue
                if sc.output_transcription and sc.output_transcription.text:
                    record("output_transcription", text=sc.output_transcription.text)
                if sc.interrupted:
                    record("interrupted")
                status = getattr(sc.interaction_status, "value", sc.interaction_status)
                if status:
                    record("interaction_status", status=status)
                if status == "IDLE":
                    return


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--workflow", choices=("appointments", "insurance", "both"), default="both")
    parser.add_argument("--continuous-silence", action="store_true")
    parser.add_argument("--tool-cycle", action="store_true")
    parser.add_argument("--tool-response-key", choices=("output", "result"), default="output",
                        help="Diagnostic wrapper comparison only; does not modify the deployed bridge")
    parser.add_argument("--full-appointment-cycle", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required")
    if args.tool_cycle:
        print(json.dumps(await tool_cycle_probe(args.timeout, args.tool_response_key, args.full_appointment_cycle), sort_keys=True), flush=True)
        return 0
    numbers = {
        "appointments": "19715717785",
        "insurance": "19713912400",
    }
    selected = numbers if args.workflow == "both" else {args.workflow: numbers[args.workflow]}
    for number in selected.values():
        print(json.dumps(await probe(number, args.timeout, args.continuous_silence), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
