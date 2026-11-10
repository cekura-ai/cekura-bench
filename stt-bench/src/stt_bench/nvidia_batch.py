"""NVIDIA Nemotron Transcribe: whole-file transcription over OpenAI chat completions.

Unlike every streaming adapter here, the model hears the complete recording in
one request, so there is no transcript at speech end and no streaming timing:
only eventual accuracy and the request's round trip are measured. Results are
labelled ``interface: batch`` and are not comparable with the streaming rows'
deadline or latency columns.

The frozen 16 kHz input (the speech plus its one-second tail) is sent as the
file. The model is a reasoning model; thinking is disabled because with it on,
the model answers the caller's request instead of transcribing it.
"""
import argparse
import asyncio
import base64
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import sha256, write_json
from .score import aggregate_wer, word_errors

MODEL = 'nvidia-nemotron-3-5-transcribe'
PUBLIC = Path('datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/full')
PRIVATE = Path('reports/assemblyai-private-20260914/dataset')
PRIVATE_MANIFEST = Path('workspaces/private-longform-recovery-v2/dataset/manifest.json')
TURNS = Path('datasets/private-turns-v1/automatic-v1')
# Cloudflare in front of the endpoint refuses urllib's default user agent.
USER_AGENT = 'cekura-stt-bench/1'


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path=Path(f'config/batch/{MODEL}.json')):
    config = json.loads(Path(path).read_text())
    if config.get('interface') != 'batch' or config.get('provider') != 'nvidia':
        raise ValueError('Expected the NVIDIA batch configuration')
    return config


def endpoint(config, environ=None):
    url = (environ or os.environ).get(config['endpoint_env'], '').strip()
    if not url:
        raise ValueError(f"Missing {config['endpoint_env']}")
    return url


def items(scope, limit=None):
    """Frozen public clips, private recordings and/or private turns, each bound to its audio hash."""
    out = []
    if scope in ('public', 'all', 'smoke'):
        manifest = PUBLIC.parent / 'smoke' if scope == 'smoke' else PUBLIC
        for clip in json.loads((manifest / 'manifest.json').read_text())['clips']:
            out.append(dict(clip_id=clip['clip_id'], cohort='public', reference=clip['reference'],
                            audio=str(manifest / clip['audio']), audio_sha256=clip['audio_sha256'],
                            seconds=clip['submitted_seconds']))
    if scope in ('private', 'all'):
        for clip in json.loads(PRIVATE_MANIFEST.read_text())['clips']:
            audio = clip['audio']['16000']
            out.append(dict(clip_id=clip['clip_id'], cohort='private', reference=clip['reference'],
                            audio=str(PRIVATE / audio['path']), audio_sha256=audio['sha256'],
                            seconds=clip['submitted_seconds']))
    if scope in ('turns', 'all'):
        # Each turn is its own request, as each is its own session for the streaming models.
        for clip in json.loads((TURNS / 'manifest.json').read_text())['clips']:
            out.append(dict(clip_id=clip['clip_id'], cohort='turns', reference=clip['reference'],
                            audio=str(TURNS / clip['audio']), audio_sha256=clip['audio_sha256'],
                            seconds=clip['submitted_seconds']))
    return out[:limit] if limit else out


def request_body(config, wav_bytes, seconds):
    audio = base64.b64encode(wav_bytes).decode()
    return {'model': config['model'], 'temperature': config['temperature'],
            'max_tokens': max(config['min_max_tokens'], int(seconds * config['max_tokens_per_audio_second'])),
            'chat_template_kwargs': config['chat_template_kwargs'],
            'messages': [{'role': 'user', 'content': [
                {'type': 'input_audio', 'input_audio': {'data': audio, 'format': config['audio']['format']}},
                {'type': 'text', 'text': config['prompt']}]}]}


def failure_class(status, body):
    if status in (401, 403):
        return 'authentication'
    if status == 429:
        return 'rate_limit'
    if status and status >= 500:
        return 'server'
    return 'request' if status else 'transport'


def post(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', 'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors='replace')


async def transcribe_one(url, config, item, attempt):
    wav = Path(item['audio']).read_bytes()
    if sha256(Path(item['audio'])) != item['audio_sha256']:
        raise ValueError(f"Frozen audio changed: {item['clip_id']}")
    body = request_body(config, wav, item['seconds'])
    record = dict(clip_id=item['clip_id'], cohort=item['cohort'], attempt=attempt, started_at=now(),
                  request_bytes=len(json.dumps(body)), max_tokens=body['max_tokens'])
    t0 = time.perf_counter()
    try:
        status, text = await asyncio.to_thread(post, url, body, config['request_timeout_seconds'])
        record['round_trip_ms'] = (time.perf_counter() - t0) * 1000
        record['http_status'] = status
        if status != 200:
            return {**record, 'valid': False, 'failure_class': failure_class(status, text), 'error': text[:500]}
        payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001 -- the reason is part of the record
        return {**record, 'round_trip_ms': (time.perf_counter() - t0) * 1000, 'valid': False,
                'failure_class': 'transport', 'error': repr(exc)[:500]}
    choice = payload['choices'][0]
    hypothesis = choice['message'].get('content') or ''
    record.update(served_model=payload.get('model'), finish_reason=choice.get('finish_reason'),
                  usage=payload.get('usage'), system_fingerprint=payload.get('system_fingerprint'),
                  hypothesis=hypothesis)
    # A transcript cut off by the token budget is incomplete, not a model answer.
    if choice.get('finish_reason') != 'stop' or payload.get('model') != config['model']:
        return {**record, 'valid': False, 'failure_class': 'incomplete' if choice.get('finish_reason') != 'stop' else 'model_identity'}
    return {**record, 'valid': True, 'word_errors': word_errors(item['reference'], hypothesis)}


async def run(scope, out, config, url, workers, max_attempts=2, limit=None):
    out.mkdir(parents=True, exist_ok=True)
    journal = out / 'attempts.jsonl'
    done = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            done.setdefault(row['clip_id'], []).append(row)
    plan = items(scope, limit)
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
                row = await transcribe_one(url, config, item, len(rows) + 1)
                rows.append(row)
                with journal.open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f"{item['clip_id']} attempt {row['attempt']} {'ok' if row['valid'] else row['failure_class']} "
                      f"{row['round_trip_ms']:.0f} ms", flush=True)
                if row.get('failure_class') == 'authentication':
                    raise RuntimeError('Endpoint refused the request; stopping')
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
    parser = argparse.ArgumentParser(description='NVIDIA Nemotron whole-file transcription on frozen inputs')
    parser.add_argument('--scope', choices=('smoke', 'public', 'private', 'turns', 'all'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    config = load_config()
    summary = asyncio.run(run(args.scope, args.out, config, endpoint(config),
                              args.workers or config['workers'], limit=args.limit))
    print(json.dumps(summary['cohorts'], indent=2))


if __name__ == '__main__':
    main()
