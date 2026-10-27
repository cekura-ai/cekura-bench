"""IBM Granite Speech: whole-file transcription over OpenAI chat completions.

Like the NVIDIA batch row, the model hears a complete recording in one request,
so only eventual accuracy and the request's round trip are measured; results
are labelled ``interface: batch`` and are not comparable with the streaming
rows' deadline or latency columns.

The model is trained on segments of up to about two minutes and its card says to
split longer audio before decoding; IBM advised pieces of 30 s to a minute.
Public clips and turns are sent whole. A private recording runs about fifteen
minutes, so it is cut at pauses into pieces of at most ``max_chunk_seconds``:
each cut is the middle of a gap of at least ``min_cut_gap_seconds`` between two
reference words, or, where a window has no such pause, its widest gap between
words, so no word is split. The pieces are transcribed in order and joined with
a space, and the joined text is scored against the recording's whole reference,
exactly as a single transcript would be. Cut points come from the reference's
word timings, which is a choice a deployment would make with a voice-activity
detector instead.
"""
import argparse
import asyncio
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import sha256, write_json
from .score import aggregate_wer, word_errors

MODEL = 'ibm-granite-speech-5-0-2b'
PUBLIC = Path('datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/full')
PRIVATE = Path('reports/assemblyai-private-20260914/dataset')
PRIVATE_MANIFEST = Path('workspaces/private-longform-recovery-v2/dataset/manifest.json')
TURNS = Path('datasets/private-turns-v1/automatic-v1')
USER_AGENT = 'cekura-stt-bench/1'


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=Path(f'config/batch/{MODEL}.json')):
    config = json.loads(Path(path).read_text())
    if config.get('interface') != 'batch' or config.get('provider') != 'ibm':
        raise ValueError('Expected the IBM Granite batch configuration')
    return config


def credentials(config, environ=None):
    environ = environ or os.environ
    url = environ.get(config['endpoint_env'], '').strip()
    token = environ.get(config['token_env'], '').strip()
    if not url or not token:
        raise ValueError(f"Missing {config['endpoint_env']} or {config['token_env']}")
    return url, token


def cut_points(words, seconds, max_chunk, min_gap, min_chunk=0.0):
    """Chunk boundaries in seconds, always in a gap between two words.

    Each cut is the latest pause of at least ``min_gap`` that keeps the piece within ``max_chunk``. A window with
    no such pause is cut in its widest gap between words, looked for first among cuts that leave the piece at
    least ``min_chunk`` long.
    """
    gaps = [(a['end'] + (b['start'] - a['end']) / 2, b['start'] - a['end']) for a, b in zip(words, words[1:])
            if b['start'] > a['end']]
    cuts, start = [], 0.0
    while seconds - start > max_chunk:
        window = [gap for gap in gaps if start < gap[0] <= start + max_chunk]
        pauses = [mid for mid, width in window if width >= min_gap]
        if pauses:
            start = pauses[-1]
        else:
            late = [gap for gap in window if gap[0] - start >= min_chunk] or window
            if not late:
                raise ValueError(f'no gap between words within {max_chunk}s after {start:.1f}s')
            start = max(late, key=lambda gap: (gap[1], gap[0]))[0]
        cuts.append(round(start, 3))
    return cuts


def items(scope, config, limit=None):
    """Frozen public clips, private recordings and/or private turns, each bound to its audio hash."""
    out = []
    if scope in ('public', 'all', 'smoke'):
        manifest = PUBLIC.parent / 'smoke' if scope == 'smoke' else PUBLIC
        for clip in json.loads((manifest / 'manifest.json').read_text())['clips']:
            out.append(dict(clip_id=clip['clip_id'], cohort='public', reference=clip['reference'],
                            audio=str(manifest / clip['audio']), audio_sha256=clip['audio_sha256'],
                            seconds=clip['submitted_seconds'], cuts=[]))
    if scope in ('private', 'all'):
        for clip in json.loads(PRIVATE_MANIFEST.read_text())['clips']:
            audio = clip['audio']['16000']
            out.append(dict(clip_id=clip['clip_id'], cohort='private', reference=clip['reference'],
                            audio=str(PRIVATE / audio['path']), audio_sha256=audio['sha256'],
                            seconds=clip['submitted_seconds'],
                            cuts=cut_points(clip['words'], clip['submitted_seconds'],
                                            config['max_chunk_seconds'], config['min_cut_gap_seconds'],
                                            config.get('min_chunk_seconds', 0.0))))
    if scope in ('turns', 'all'):
        for clip in json.loads((TURNS / 'manifest.json').read_text())['clips']:
            out.append(dict(clip_id=clip['clip_id'], cohort='turns', reference=clip['reference'],
                            audio=str(TURNS / clip['audio']), audio_sha256=clip['audio_sha256'],
                            seconds=clip['submitted_seconds'], cuts=[]))
    return out[:limit] if limit else out


def pieces(path, cuts):
    """The recording as WAV bytes, whole or cut at ``cuts`` (seconds)."""
    with wave.open(path) as source:
        rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
        pcm = source.readframes(source.getnframes())
    if not cuts:
        return [Path(path).read_bytes()]
    frame = width * channels
    bounds = [0] + [int(c * rate) * frame for c in cuts] + [len(pcm)]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as sink:
            sink.setnchannels(channels); sink.setsampwidth(width); sink.setframerate(rate)
            sink.writeframes(pcm[a:b])
        out.append(buffer.getvalue())
    return out


def request_body(config, wav_bytes, seconds):
    audio = base64.b64encode(wav_bytes).decode()
    return {'model': config['model'], 'temperature': config['temperature'],
            'max_tokens': max(config['min_max_tokens'], int(seconds * config['max_tokens_per_audio_second'])),
            'messages': [{'role': 'user', 'content': [
                {'type': 'audio_url', 'audio_url': {'url': f'data:audio/wav;base64,{audio}'}},
                {'type': 'text', 'text': config['prompt']}]}]}


def failure_class(status):
    if status in (401, 403):
        return 'authentication'
    if status == 429:
        return 'rate_limit'
    if status and status >= 500:
        return 'server'
    return 'request' if status else 'transport'


def post(url, token, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', 'User-Agent': USER_AGENT,
                                              'Authorization': f'Bearer {token}'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors='replace')


async def transcribe_one(url, token, config, item, attempt):
    if sha256(Path(item['audio'])) != item['audio_sha256']:
        raise ValueError(f"Frozen audio changed: {item['clip_id']}")
    bounds = [0.0] + item['cuts'] + [item['seconds']]
    record = dict(clip_id=item['clip_id'], cohort=item['cohort'], attempt=attempt, started_at=now(),
                  cuts=item['cuts'], pieces=[])
    t0 = time.perf_counter()
    texts = []
    for index, (wav, (start, end)) in enumerate(zip(pieces(item['audio'], item['cuts']), zip(bounds, bounds[1:]))):
        body = request_body(config, wav, end - start)
        p0 = time.perf_counter()
        try:
            status, text = await asyncio.to_thread(post, url, token, body, config['request_timeout_seconds'])
        except Exception as exc:  # noqa: BLE001 -- the reason is part of the record
            return {**record, 'round_trip_ms': (time.perf_counter() - t0) * 1000, 'valid': False,
                    'failure_class': 'transport', 'error': repr(exc)[:500], 'failed_piece': index}
        piece = dict(index=index, start=start, end=end, http_status=status, max_tokens=body['max_tokens'],
                     round_trip_ms=(time.perf_counter() - p0) * 1000)
        if status != 200:
            record['pieces'].append(piece)
            return {**record, 'round_trip_ms': (time.perf_counter() - t0) * 1000, 'valid': False,
                    'failure_class': failure_class(status), 'error': text[:500], 'failed_piece': index}
        payload = json.loads(text)
        choice = payload['choices'][0]
        piece.update(served_model=payload.get('model'), finish_reason=choice.get('finish_reason'),
                     usage=payload.get('usage'), hypothesis=choice['message'].get('content') or '')
        record['pieces'].append(piece)
        # A transcript cut off by the token budget is incomplete, not a model answer, unless the config keeps such
        # pieces: then the text the model returned up to the cap is scored as it is, and the piece is flagged.
        finished = choice.get('finish_reason') == 'stop'
        if payload.get('model') != config['model'] or not (finished or config.get('keep_capped_pieces')):
            return {**record, 'round_trip_ms': (time.perf_counter() - t0) * 1000, 'valid': False,
                    'failure_class': 'incomplete' if not finished else 'model_identity', 'failed_piece': index}
        if not finished:
            record.setdefault('capped_pieces', []).append(index)
        texts.append(piece['hypothesis'].strip())
    hypothesis = ' '.join(t for t in texts if t)
    return {**record, 'round_trip_ms': (time.perf_counter() - t0) * 1000, 'valid': True, 'hypothesis': hypothesis,
            'served_model': config['model'], 'word_errors': word_errors(item['reference'], hypothesis)}


async def run(scope, out, config, url, token, workers, max_attempts=3, limit=None):
    out.mkdir(parents=True, exist_ok=True)
    journal = out / 'attempts.jsonl'
    done = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            done.setdefault(row['clip_id'], []).append(row)
    plan = items(scope, config, limit)
    write_json(out / 'plan.json', dict(model=MODEL, scope=scope, created_at=now(), config=config,
                                      items=[{k: v for k, v in i.items() if k != 'reference'} for i in plan]))
    queue = asyncio.Queue()
    for item in plan:
        rows = done.get(item['clip_id'], [])
        if not any(r['valid'] for r in rows) and len(rows) < max_attempts:
            queue.put_nowait(item)

    async def worker():
        while not queue.empty():
            item = queue.get_nowait()
            rows = done.setdefault(item['clip_id'], [])
            while len(rows) < max_attempts and not any(r['valid'] for r in rows):
                row = await transcribe_one(url, token, config, item, len(rows) + 1)
                rows.append(row)
                with journal.open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f"{item['clip_id']} attempt {row['attempt']} {'ok' if row['valid'] else row['failure_class']} "
                      f"{row['round_trip_ms']:.0f} ms", flush=True)
                if row.get('failure_class') == 'authentication':
                    raise RuntimeError('Endpoint refused the request; stopping')
                if not row['valid'] and row.get('failure_class') in ('server', 'transport', 'rate_limit'):
                    await asyncio.sleep(config['retry_wait_seconds'])
    await asyncio.gather(*(worker() for _ in range(workers)))
    return summarize(out, plan, done)


def summarize(out, plan, done):
    """Eventual WER per cohort from the first valid attempt; every attempt stays in the journal."""
    cohorts = {}
    for cohort in sorted({i['cohort'] for i in plan}):
        planned = [i for i in plan if i['cohort'] == cohort]
        valid = [next(r for r in done.get(i['clip_id'], []) if r['valid'])
                 for i in planned if any(r['valid'] for r in done.get(i['clip_id'], []))]
        failures = Counter(r['failure_class'] for i in planned for r in done.get(i['clip_id'], []) if not r['valid'])
        trips = sorted(r['round_trip_ms'] for r in valid)
        cohorts[cohort] = dict(planned=len(planned), usable=len(valid),
                               first_attempt_valid=sum(1 for i in planned if done.get(i['clip_id']) and done[i['clip_id']][0]['valid']),
                               failures=dict(failures), errors=aggregate_wer([r['word_errors'] for r in valid]),
                               round_trip_ms_p50=trips[len(trips) // 2] if trips else None)
    summary = dict(model=MODEL, interface='batch', generated_at=now(), cohorts=cohorts,
                   note='Whole-file transcription: eventual accuracy only; no deadline or streaming latency')
    write_json(out / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='IBM Granite Speech whole-file transcription on frozen inputs')
    parser.add_argument('--scope', choices=('smoke', 'public', 'private', 'turns', 'all'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    config = load_config()
    url, token = credentials(config)
    summary = asyncio.run(run(args.scope, args.out, config, url, token, args.workers or config['workers'], limit=args.limit))
    print(json.dumps(summary['cohorts'], indent=2))


if __name__ == '__main__':
    main()
