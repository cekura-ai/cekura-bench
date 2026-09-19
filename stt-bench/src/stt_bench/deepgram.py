"""Deepgram Nova streaming protocol. No pre-recorded transcription calls."""
import asyncio
import contextlib
import json
from urllib.parse import urlencode

from websockets.asyncio.client import connect

from .streaming import EventLog, stream_audio


def query_url(config: dict) -> str:
    if config["endpoint"] != "wss://api.deepgram.com/v1/listen":
        raise ValueError("Expected the Deepgram Nova streaming endpoint")
    if config.get("version") in (None, "", "latest", "beta"):
        raise ValueError("An exact model version is required")
    query = {**config["query"], "model": config["model"], "version": config["version"],
             "language": config["language"]}
    required = {"encoding": "linear16", "sample_rate": 16000, "channels": 1,
                "endpointing": False, "interim_results": True}
    if any(query.get(k) != v for k, v in required.items()):
        raise ValueError("Provider config conflicts with the streaming protocol")
    return config["endpoint"] + "?" + urlencode({
        k: str(v).lower() if isinstance(v, bool) else v for k, v in query.items()})


async def exchange(ws, pcm: bytes, speech_frames: int, config: dict, log: EventLog):
    finalized = asyncio.Event()
    speech_end = None

    async def receive():
        async for raw in ws:
            at = log.now()  # Local receipt time, before parsing or writing to disk.
            message = json.loads(raw)
            log.emit("provider_message", at=at, message=message)
            if message.get("type") == "Error":
                raise RuntimeError("Provider returned an error; see saved provider_message")
            if message.get("type") == "Results" and message.get("from_finalize") and message.get("is_final"):
                finalized.set()

    async def finalize(t0):
        nonlocal speech_end
        speech_end = t0
        log.emit("finalize_requested", t0_seconds=t0)
        await ws.send(json.dumps({"type": "Finalize"}))
        log.emit("finalize_sent", t0_seconds=t0)

    receiver = asyncio.create_task(receive())
    try:
        # Stop on receive failure as well as send failure; never let a broken receiver
        # silently collect a successful-looking timing run.
        sender = asyncio.create_task(stream_audio(pcm, speech_frames, ws.send, finalize, log))
        try:
            done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                await receiver
                raise RuntimeError("Provider closed before audio streaming completed")
            await sender
        finally:
            if not sender.done():
                sender.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender
        # Acknowledge all 50 silence frames even when Finalize responds early.
        try:
            remaining = max(0, config["finalize_timeout_seconds"] - (log.now() - speech_end))
            if not finalized.is_set():
                await asyncio.wait_for(finalized.wait(), remaining)
        except TimeoutError:
            log.emit("finalize_ack_missing")
        # Keep the observation window open even if acknowledgment arrives early.
        await asyncio.sleep(max(0, speech_end + 1.0 - log.now()))
        # CloseStream also flushes buffers. Results caused by this cleanup may be
        # re-scored for accuracy, but must never become headline finalize latency.
        log.emit("close_stream_requested")
        await ws.send(json.dumps({"type": "CloseStream"}))
        try:
            await asyncio.wait_for(receiver, config["close_timeout_seconds"])
        except TimeoutError:
            log.emit("close_stream_timeout")
    finally:
        if not receiver.done():
            receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver


async def transcribe(pcm: bytes, speech_frames: int, config: dict, key: str, log: EventLog):
    url = query_url(config)
    log.emit("connection_requested", url=url)
    async with connect(url, additional_headers={"Authorization": f"Token {key}"},
                       open_timeout=15, close_timeout=5, max_size=8 * 1024 * 1024) as ws:
        log.emit("connection_open")
        await exchange(ws, pcm, speech_frames, config, log)


def reduce_events(events: list[dict], config: dict) -> dict:
    """Derive transcripts and timings offline from untouched provider messages."""
    t0 = next((e["time_seconds"] for e in events if e["kind"] == "speech_end"), None)
    close_at = next((e["time_seconds"] for e in events if e["kind"] == "close_stream_requested"), float("inf"))
    requested = next((e["time_seconds"] for e in events if e["kind"] == "finalize_requested"), float("inf"))
    messages = [e for e in events if e["kind"] == "provider_message"]
    results = [e for e in messages if e["message"].get("type") == "Results"]
    boundary = next((e for e in results if e["message"].get("from_finalize")
                     and e["message"].get("is_final") and requested <= e["time_seconds"] < close_at), None)
    cutoff = boundary["time_seconds"] if boundary else float("inf")
    segments, seen = [], set()
    first_partial = None
    model_versions, model_uuids = set(), set()
    last_transcript_at = None
    for event in results:
        message, at = event["message"], event["time_seconds"]
        metadata = message.get("metadata", {})
        info = metadata.get("model_info", {})
        if info.get("version"):
            model_versions.add(info["version"])
        if metadata.get("model_uuid"):
            model_uuids.add(metadata["model_uuid"])
        alternatives = message.get("channel", {}).get("alternatives", [])
        transcript = alternatives[0].get("transcript", "") if alternatives else ""
        if t0 is not None and t0 <= at < min(cutoff, close_at) and not message.get("is_final") and transcript and first_partial is None:
            first_partial = {"text": transcript, "received_seconds": at, "latency_ms": (at - t0) * 1000}
        if not message.get("is_final") or at > cutoff:
            continue
        # Deduplicate exact repeated segment events. Equal words in different time ranges remain.
        key = (tuple(message.get("channel_index", [])), message.get("start"), message.get("duration"), transcript)
        if key not in seen:
            seen.add(key)
            if transcript:
                segments.append((message.get("start", 0), transcript))
                last_transcript_at = at
    # CloseStream's terminal Metadata includes model information even with empty transcripts.
    for event in messages:
        message = event["message"]
        if message.get("type") == "Metadata":
            for uuid, info in message.get("model_info", {}).items():
                model_uuids.add(uuid)
                if info.get("version"):
                    model_versions.add(info["version"])
    model_verified = (model_versions == {config["version"]}
                      and model_uuids == {config["expected_model_uuid"]})
    errored = any(e["kind"] == "error" for e in events)
    stream_complete = any(e["kind"] == "audio_complete" for e in events)
    closed_cleanly = any(e["message"].get("type") == "Metadata" and e["time_seconds"] >= close_at for e in messages)
    transcript_complete = not errored and stream_complete and closed_cleanly
    completion_at = next((e['time_seconds'] for e in messages
                          if e['message'].get('type') == 'Metadata' and e['time_seconds'] >= close_at), None)
    latency = (boundary["time_seconds"] - t0) * 1000 if boundary and t0 is not None else None
    return dict(transcript=" ".join(text for _, text in sorted(segments, key=lambda s: s[0])),
                transcript_complete=transcript_complete, model_verified=model_verified,
                model_versions=sorted(model_versions), model_uuids=sorted(model_uuids),
                t0_seconds=t0, first_partial_after_t0=first_partial,
                final_transcript_received_seconds=last_transcript_at,
                finalize_ack_received_seconds=boundary["time_seconds"] if boundary else None,
                finalize_latency_ms=latency,
                finalize_latency_status="observed" if latency is not None else "missing_finalize_ack",
                completion_received_seconds=completion_at,
                completion_latency_ms=(completion_at - t0) * 1000 if transcript_complete and t0 is not None else None,
                completion_timed_out=any(e['kind'] == 'close_stream_timeout' for e in events),
                transport_failed=errored,
                transcript_completion_basis="close_stream_metadata" if closed_cleanly else "unconfirmed")


def transcript_at(events: list[dict], cutoff: float) -> dict:
    """Deepgram mono segment finals plus the latest replaceable partial at receipt time.

    When a final covers only part of a previous partial, retain timestamped suffix
    words. Without word timing that state is explicitly unsupported until replaced.
    """
    finals, seen = [], set()
    epsilon = 1e-5  # Provider decimal audio timestamps can differ by float rounding.
    partial = None
    ambiguous = False
    conflict = False
    for event in sorted(events, key=lambda e: e['time_seconds']):
        if event['time_seconds'] > cutoff:
            break
        message = event.get('message', {})
        if event['kind'] != 'provider_message' or message.get('type') != 'Results':
            continue
        alt = (message.get('channel', {}).get('alternatives') or [{}])[0]
        start = message.get('start', 0)
        end = start + message.get('duration', 0)
        text = alt.get('transcript', '')
        if not message.get('is_final'):
            partial = {'start': start, 'end': end, 'text': text, 'words': alt.get('words')}
            ambiguous = False
            # Stale messages entirely covered by final segments cannot replace them.
            if any(start >= a - epsilon and end <= b + epsilon for a, b, _ in finals):
                partial = None
            continue
        key = (start, end, text)
        if key in seen:
            continue
        seen.add(key)
        if text:
            if any(start < b - epsilon and end > a + epsilon for a, b, _ in finals):
                conflict = True
            finals.append((start, end, text))
        if partial and partial['start'] < end - epsilon and partial['end'] > start + epsilon:
            if partial['end'] > end + 1e-6:
                if not partial['text']:
                    partial = None
                elif not partial['words'] or any('start' not in w for w in partial['words']):
                    ambiguous = True
                    partial = None
                else:
                    suffix = [w for w in partial['words'] if w.get('start', -1) >= end - 1e-6]
                    partial = {'start': end, 'end': partial['end'], 'words': suffix,
                               'text': ' '.join(w.get('punctuated_word', w.get('word', '')) for w in suffix)}
            else:
                partial = None
    final_text = ' '.join(t for _, _, t in sorted(finals))
    pending = partial['text'] if partial else ''
    # Overlapping latest partials are not safely concatenable without another event.
    if partial and any(partial['start'] < b - epsilon and partial['end'] > a + epsilon for a, b, _ in finals):
        ambiguous = True
    return {'text': ' '.join(t for t in (final_text, pending) if t),
            'final_text': final_text, 'partial_text': pending,
            'provisional': bool(pending),
            'reconstruction_status': 'unsupported_overlap' if ambiguous or conflict else 'supported'}
