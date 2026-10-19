"""Accuracy-only runs of transcription endpoints that take a whole recording per request.

Such an endpoint answers only after the upload ends, so none of the streaming
latencies is comparable with the streaming providers. This path records
transcripts only and scores them with the unchanged WER and entity scorers.
Each clip is the frozen audio exactly as streamed to providers, sent as WAV.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import base64
import fcntl
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .catalog import dataset_definition
from .data import load_manifest, sha256, write_json
from .huggingface_data import verify_prepared
from .score import NORMALIZATION, aggregate_wer, entity_errors, word_errors

BATCH_MODELS = {name: Path('config/batch-models') / (name + '.json') for name in (
    'nemotron-3.5-transcribe-30b-a4b',
)}
CI = dict(iterations=10000, seed=42, method='percentile bootstrap of summed errors over summed reference words',
          resampled='clips')
RETRYABLE = {408, 429, 500, 502, 503, 504}
ATTEMPTS = 4


class NoRedirect(urllib.request.HTTPRedirectHandler):
    # A redirect from the endpoint is a sign-in page, never a transcript.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def batch_config(name):
    if name not in BATCH_MODELS:
        raise ValueError(f'Unknown batch model {name!r}; available: {", ".join(BATCH_MODELS)}')
    config = json.loads(BATCH_MODELS[name].read_text())
    if config['model_id'] != name or config['api'] != 'openai-chat-completions-audio':
        raise ValueError('Batch model config does not match selector')
    return config


def request_payload(config, wav):
    return {
        'model': config['model'],
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': config['prompt']},
            {'type': 'audio_url', 'audio_url': {'url': 'data:audio/wav;base64,' + base64.b64encode(wav).decode('ascii')}},
        ]}],
        'temperature': 0,
        'max_tokens': config['max_tokens'],
        'stream': False,
        **config.get('extra_body', {}),
    }


def read_transcript(response):
    """The transcript of a complete answer; an incomplete or failed one raises."""
    if response.get('error'):
        raise RuntimeError(f"Endpoint error: {response['error']}")
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise RuntimeError(f"Incomplete transcript: finish_reason={choice.get('finish_reason')!r}")
    return choice['message']['content'].strip()


def transcribe_wav(config, wav, sleep=time.sleep):
    request = urllib.request.Request(config['endpoint'], data=json.dumps(request_payload(config, wav)).encode(),
                                     headers={'Content-Type': 'application/json', 'Accept': 'application/json',
                                              'User-Agent': config['user_agent']}, method='POST')
    opener = urllib.request.build_opener(NoRedirect())
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with opener.open(request, timeout=config['timeout_s']) as response:
                return read_transcript(json.load(response)), attempt
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE or attempt == ATTEMPTS:
                raise RuntimeError(f'HTTP {error.code} from the endpoint') from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == ATTEMPTS:
                raise
        sleep(2 ** attempt)


def bootstrap_ci(rows, iterations=CI['iterations'], seed=CI['seed']):
    errors = np.array([r['substitutions'] + r['insertions'] + r['deletions'] for r in rows], dtype=float)
    words = np.array([r['reference_words'] for r in rows], dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(rows), size=(iterations, len(rows)))
    rates = errors[picks].sum(axis=1) / words[picks].sum(axis=1) * 100
    return [round(float(np.percentile(rates, 2.5)), 3), round(float(np.percentile(rates, 97.5)), 3)]


def transcribe_subset(manifest_path, out, transcribe, concurrency):
    """Append one JSON line per clip; a rerun sends only the clips not yet saved."""
    manifest = load_manifest(manifest_path)
    path = out / 'transcripts.jsonl'
    done = {json.loads(line)['clip_id'] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [clip for clip in manifest['clips'] if clip['clip_id'] not in done]
    for clip in todo:
        if sha256(manifest_path.parent / clip['audio']) != clip['audio_sha256']:
            raise ValueError(f"Frozen audio changed: {clip['clip_id']}")
    with path.open('a') as handle, ThreadPoolExecutor(concurrency) as pool:
        futures = {pool.submit(transcribe, (manifest_path.parent / clip['audio']).read_bytes()): clip for clip in todo}
        for index, future in enumerate(as_completed(futures), len(done) + 1):
            clip = futures[future]
            try:
                text, attempts = future.result()
                row = dict(clip_id=clip['clip_id'], transcript=text, attempts=attempts)
            except Exception as error:   # recorded as unusable, so one bad answer cannot stop the run
                row = dict(clip_id=clip['clip_id'], transcript=None, error=str(error)[:300])
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            print(f"[{index}/{len(manifest['clips'])}] {clip['clip_id']}", flush=True)
    return manifest


def score_subset(manifest, out):
    saved = {r['clip_id']: r for r in map(json.loads, (out / 'transcripts.jsonl').read_text().splitlines())}
    clips = []
    for clip in manifest['clips']:
        text = (saved.get(clip['clip_id']) or {}).get('transcript')
        usable = bool(text and text.strip())
        clips.append(dict(clip_id=clip['clip_id'], condition=clip['condition'], transcript=text, usable=usable,
                          error=(saved.get(clip['clip_id']) or {}).get('error'),
                          word_errors=word_errors(clip['reference'], text) if usable else None,
                          entities=entity_errors(clip['reference'], text or '', clip.get('entities'))))
    usable = [c['word_errors'] for c in clips if c['usable']]
    annotated = [c['entities'] for c in clips if c['entities']['status'] == 'annotated']
    entities = None
    if annotated:
        by_type = {}
        for item in annotated:
            for kind, counts in item['by_type'].items():
                total = by_type.setdefault(kind, {'errors': 0, 'reference_entities': 0})
                total['errors'] += counts['errors']
                total['reference_entities'] += counts['reference_entities']
        entities = dict(errors=sum(e['errors'] for e in annotated),
                        reference_entities=sum(e['reference_entities'] for e in annotated), by_type=by_type)
    summary = dict(planned=len(clips), transcribed=len(saved), usable=len(usable), wer=aggregate_wer(usable),
                   wer_ci95_percent=bootstrap_ci(usable) if usable else None, ci_method=CI, entities=entities,
                   normalization=NORMALIZATION)
    write_json(out / 'results.json', dict(summary=summary, clips=clips))
    return summary


def batch_benchmark(dataset, model, run_id=None, resume=False):
    definition = dataset_definition(dataset)
    config = batch_config(model)
    root = verify_prepared(definition)
    if resume and not run_id:
        raise ValueError('--resume requires --run-id')
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('run-id may contain only letters, digits, underscores and hyphens')
    runs = Path('runs') / 'batch' / dataset / model / run_id
    runs.mkdir(parents=True, exist_ok=resume)
    with (runs / '.workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('This batch workflow is already running') from None
        identity = dict(dataset=definition, model_id=model, config=config,
                        manifests={s: sha256(root / s / 'manifest.json') for s in ('smoke', 'full')},
                        source_sha256=sha256(Path(__file__)))
        status_path = runs / 'benchmark.json'
        if resume:
            if json.loads(status_path.read_text())['identity'] != identity:
                raise ValueError('Resume requires the same dataset, model configuration and source')
        else:
            write_json(status_path, dict(run_id=run_id, identity=identity, status='starting', summaries={}))
        state = json.loads(status_path.read_text())
        for subset in ('smoke', 'full'):
            out = runs / subset
            out.mkdir(exist_ok=True)
            manifest = transcribe_subset(root / subset / 'manifest.json', out, lambda wav: transcribe_wav(config, wav),
                                         config['concurrency'])
            summary = score_subset(manifest, out)
            state['summaries'][subset] = summary
            state['status'] = f'{subset}_complete'
            write_json(status_path, state)
            if subset == 'smoke' and summary['usable'] != summary['planned']:
                raise ValueError(f'Smoke clips produced no usable transcript; inspect {out}')
            print(f"{subset}: WER {summary['wer']['wer']:.4f} on {summary['usable']}/{summary['planned']} clips", flush=True)
        state['status'] = 'complete'
        write_json(status_path, state)
        return state
