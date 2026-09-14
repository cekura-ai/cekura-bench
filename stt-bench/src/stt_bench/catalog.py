"""Explicit dataset and model names, independent of artifact directory names."""
import json
from pathlib import Path

DATASETS = {'pipecat-stt-benchmark': Path('config/datasets/pipecat-stt-benchmark.json')}
MODELS = {name: Path('config/models') / (name + '.json') for name in (
    'assemblyai-universal-3-5-pro',
    'reson8-realtime',
    'gradium-default',
    'deepgram-nova-3',
    'deepgram-nova-2',
    'deepgram-flux-en',
    'deepgram-flux-multilingual',
    'openai-gpt-realtime-whisper',
    'openai-gpt-4o-transcribe',
    'openai-gpt-4o-mini-transcribe',
    'gemini-3.5-transcribe-live',
    'elevenlabs-scribe-v2-realtime',
    'speechmatics-standard',
    'speechmatics-enhanced',
    'cartesia-ink-2',
    'google-chirp-2',
    'google-chirp-3',
    'soniox-stt-rt-v5',
    'smallest-pulse',
    'sarvam-saaras-v3-realtime',
    'inworld-stt-1',
)}
BLOCKED_MODELS = {}


def dataset_definition(name):
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset {name!r}; available: {", ".join(DATASETS)}')
    definition = json.loads(DATASETS[name].read_text())
    if definition['id'] != name:
        raise ValueError('Dataset definition ID does not match selector')
    return definition


def model_config(name):
    if name not in MODELS:
        raise ValueError(f'Unknown model {name!r}; available: {", ".join(MODELS)}')
    return MODELS[name]


def dataset_root(definition):
    return Path('datasets') / definition['id'] / definition['revision']


def dataset_identity(manifest):
    return {key: manifest.get(key) for key in
            ('dataset_id', 'dataset', 'source_revision', 'split', 'subset', 'language')}
