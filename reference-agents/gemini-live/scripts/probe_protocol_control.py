"""Diagnostic controls only; never imported by the production bridge.

Compare canonical setup with one changed setting, and capture server messages
before the SDK converts them. No model switch, scheduling override or fake tool
result. Text turns isolate the protocol; PSTN remains the deployment gate.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from google import genai
from google.genai import types
from app import MODEL, LIVE_API_VERSION, call_tool, live_config, workflow_for


async def probe(variant, sample=1):
    start = time.monotonic()
    def record(stage, **data):
        print(json.dumps({"variant": variant, "sample": sample, "ms": round((time.monotonic()-start)*1000),
                          "stage": stage, **data}), flush=True)
    workflow = workflow_for("19715717785")
    config = live_config(workflow)
    # Freeze the deployed 00021-rab baseline even if app.live_config changes.
    config.thinking_config.include_thoughts = False
    if variant == "omit-thoughts":
        config.thinking_config.include_thoughts = None
    elif variant == "minimal":
        config.system_instruction = "You are a clinic scheduling assistant. Use the available tools to answer appointment questions accurately."
    elif variant in ("low-thinking", "medium-thinking"):
        config.thinking_config.thinking_level = types.ThinkingLevel(variant.split("-")[0].upper())
    calls = []
    outputs = []
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": LIVE_API_VERSION})
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            original_recv = session._ws.recv
            async def traced_recv(*args, **kwargs):
                raw = await original_recv(*args, **kwargs)
                payload = json.loads(raw)
                sc = payload.get("serverContent", {})
                parts = sc.get("modelTurn", {}).get("parts", [])
                non_audio = [{k:v for k,v in p.items() if k not in ("inlineData", "thoughtSignature")} for p in parts]
                non_audio = [p for p in non_audio if p]
                other = {k:v for k,v in payload.items() if k not in ("serverContent", "usageMetadata")}
                if non_audio or other:
                    record("raw_server", parts=non_audio, other=other)
                return raw
            session._ws.recv = traced_recv
            record("connected", model=MODEL, include_thoughts=config.thinking_config.include_thoughts,
                   thinking_level=str(config.thinking_config.thinking_level))
            async def respond(fc):
                calls.append(fc.name)
                record("tool_call", name=fc.name, id=fc.id, args=fc.args)
                result = await call_tool(workflow, fc.name, fc.args or {})
                body = result if variant == "flat-response" else {"result" if variant == "result-response" else "output": result}
                response = types.FunctionResponse(id=fc.id, name=fc.name, response=body)
                if variant == "camel-response":
                    await session._ws.send(json.dumps({"toolResponse": {"functionResponses": [
                        response.model_dump(mode="json", by_alias=True, exclude_none=True)]}}))
                else:
                    await session.send_tool_response(function_responses=[response])
                record("tool_response_sent", name=fc.name, result=result)

            async def drain():
                async with asyncio.TaskGroup() as group:
                    while True:
                        async for message in session.receive():
                            if message.tool_call:
                                for fc in message.tool_call.function_calls or []:
                                    group.create_task(respond(fc))
                            sc = message.server_content
                            if sc and sc.output_transcription and sc.output_transcription.text:
                                text = sc.output_transcription.text
                                outputs.append(text)
                                record("output", text=text)
                            if sc and sc.interaction_status:
                                status = getattr(sc.interaction_status, "value", sc.interaction_status)
                                record("status", value=status)
                                if status == "IDLE":
                                    return

            for utterance in [
                f"Begin the phone call. Say exactly: {workflow['greeting']}",
                "I need to reschedule my upcoming appointment.",
                "My phone number is 617-555-9210.",
            ]:
                await session.send_client_content(turns=[types.Content(role="user", parts=[types.Part(text=utterance)])], turn_complete=True)
                record("user", text=utterance)
                await asyncio.wait_for(drain(), 65)
        record("complete", calls=calls, output="".join(outputs))
    except Exception as exc:
        record("failed", error=repr(exc), calls=calls, output="".join(outputs))
    finally:
        await client.aio.aclose()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="production,omit-thoughts")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    variants = args.variants.split(",")
    if any(v not in ("production", "omit-thoughts", "minimal", "camel-response", "result-response", "flat-response", "low-thinking", "medium-thinking") for v in variants):
        parser.error("Unknown diagnostic variant")
    await asyncio.gather(*(probe(v, sample) for v in variants for sample in range(1, args.repeat + 1)))


if __name__ == "__main__":
    asyncio.run(main())
