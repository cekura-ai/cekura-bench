"""One synthetic phrase only; no dataset, benchmark runner, retries, or metrics."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import wave
from stt_bench.catalog import model_config
from stt_bench.credentials import credential
from stt_bench.providers import transcribe, reduce_events
from stt_bench.streaming import EventLog, read_events


async def smoke(audio, out):
    with wave.open(str(audio), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        frames = wav.getnframes()
        assert 0 < frames <= 16000 * 3, 'Smoke phrase is capped at three seconds'
        pcm = wav.readframes(frames)
    speech_frames = (frames + 319) // 320
    pcm += bytes(speech_frames * 640 - len(pcm) + 32000)
    assert len(pcm) <= 16000 * 2 * 4, 'Hard cap: four seconds including silence'
    key, _ = credential('reson8', env_file=None)
    assert key, 'RESON_API_KEY must be injected for this one command'
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental repeat billing on re-entry.
    with (out / 'attempt.json').open('x') as f:
        json.dump(dict(provider_requests_max=1, retries=0, planned_audio_seconds=len(pcm)/32000,
                       benchmark_started=False, started_at=datetime.now(timezone.utc).isoformat()), f)
    log = EventLog(out / 'events.jsonl')
    c = json.loads(model_config('reson8-realtime').read_text())
    error = None
    try:
        async with asyncio.timeout(30):
            await transcribe(pcm, speech_frames, c, key, log)
    except Exception as exc:
        error = type(exc).__name__
        log.emit('error', error_type=error)
    finally:
        log.close()
    events = read_events(out / 'events.jsonl')
    result = reduce_events(events, c)
    passed = not error and result['transcript_complete'] and result['model_verified'] and bool(result['transcript'])
    summary = dict(passed=passed, kind='single_synthetic_phrase_protocol_smoke',
                   provider_requests=1, retries=0, benchmark_started=False,
                   planned_audio_seconds=len(pcm)/32000,
                   submitted_audio_seconds=sum(e.get('bytes',0) for e in events if e['kind']=='audio_sent')/32000,
                   error_type=error,
                   transcript=result['transcript'], transcript_complete=result['transcript_complete'],
                   flush_confirmations=[e['message'] for e in events if e['kind']=='provider_message' and e['message'].get('type')=='flush_confirmation'],
                   completed_at=datetime.now(timezone.utc).isoformat())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(smoke(args.audio, args.out)))
