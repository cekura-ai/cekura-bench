"""Provider registry used by capture, replay and readiness checks."""
from . import deepgram, provider_protocol, chirp, trial_providers, gradium, reson8, assemblyai, assemblyai_min_latency, speechmatics_agent
from .credentials import credential
from .streaming import transmitted_silence_frames
from . import gemini_live

MODELS = {
    'assemblyai': {'universal-3-5-pro', 'universal-3-6-pro'},
    'reson8': {'realtime'},
    'gradium': {'default'},
    **trial_providers.MODELS,
    'deepgram': {'nova-2', 'nova-3', 'flux-general-en', 'flux-general-multi'},
    'openai': {'gpt-realtime-whisper', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe'},
    'gemini': {'gemini-3.5-transcribe-live', *gemini_live.MODELS},
    'elevenlabs': {'scribe_v2_realtime'},
    'speechmatics': {'standard', 'enhanced', 'linden-1'},
    'cartesia': {'ink-2'},
    'google': {'chirp_2', 'chirp_3'},
}
ENDPOINTS = {
    'assemblyai': {'wss://streaming.assemblyai.com/v3/ws'},
    'reson8': {'wss://api.reson8.dev/v1/speech-to-text/realtime'},
    'gradium': {'wss://api.gradium.ai/api/speech/asr'},
    **trial_providers.ENDPOINTS,
    'deepgram': {'wss://api.deepgram.com/v1/listen', 'wss://api.deepgram.com/v2/listen'},
    'openai': {'wss://api.openai.com/v1/realtime'},
    'gemini': {'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'},
    'elevenlabs': {'wss://api.elevenlabs.io/v1/speech-to-text/realtime'},
    'speechmatics': {'wss://us.rt.speechmatics.com/v2', 'wss://global.rt.speechmatics.com/v2/agent'},
    'cartesia': {'wss://api.cartesia.ai/stt/websocket'},
    'google': {'https://us-speech.googleapis.com', 'https://us-central1-speech.googleapis.com'},
}


def is_nova(config):
    return config['provider'] == 'deepgram' and config['model'] in ('nova-2', 'nova-3')


def validate(config):
    transmitted_silence_frames(config)
    p = config.get('provider')
    if p not in MODELS or config.get('model') not in MODELS[p]:
        raise ValueError('Unknown provider/model combination')
    if config.get('endpoint') not in ENDPOINTS[p]:
        raise ValueError('Unexpected provider endpoint')
    if config['model'] in gemini_live.MODELS:
        gemini_live.validate(config)
    if p == 'speechmatics':
        if config['model'] == 'linden-1':
            speechmatics_agent.validate(config)
        elif config['endpoint'] != 'wss://us.rt.speechmatics.com/v2':
            raise ValueError('Legacy Speechmatics requires its legacy endpoint')
    if p == 'google':
        chirp.validate(config)
    if p == 'gradium':
        gradium.validate(config)
    if p == 'reson8':
        reson8.validate(config)
    if p in trial_providers.MODELS:
        trial_providers.validate(config)
    if is_nova(config):
        deepgram.query_url(config)
    else:
        if config.get('sample_rate') != (24000 if p in ('openai', 'gradium') else 16000):
            raise ValueError('Provider sample rate mismatch')
        if p == 'deepgram' and not config['endpoint'].endswith('/v2/listen'):
            raise ValueError('Flux requires the v2 endpoint')
        if not config.get('completion_basis') or not config.get('finalization'):
            raise ValueError('Explicit completion contract required')
    for name in ('finalize_timeout_seconds', 'close_timeout_seconds'):
        if not isinstance(config.get(name), (int, float)) or not 0 < config[name] <= 30:
            raise ValueError('Provider timeout must be between 0 and 30 seconds')
    return config


def sample_rate(config):
    return config.get('sample_rate', 16000)


async def transcribe(pcm, speech_frames, config, key, log):
    validate(config)
    adapter = (deepgram if is_nova(config) else chirp if config['provider'] == 'google'
               else gemini_live if config['model'] in gemini_live.MODELS
               else speechmatics_agent if config['model'] == 'linden-1'
               else assembly_adapter(config) if config['provider'] == 'assemblyai'
               else gradium if config['provider'] == 'gradium'
               else reson8 if config['provider'] == 'reson8'
               else trial_providers if config['provider'] in trial_providers.MODELS else provider_protocol)
    await adapter.transcribe(pcm, speech_frames, config, key, log)


def assembly_adapter(config):
    return assemblyai_min_latency if config.get('mode') == 'min_latency' else assemblyai


def reduce_events(events, config):
    if config['model'] in gemini_live.MODELS:
        return gemini_live.replay(events, config)[1]
    if config['model'] == 'linden-1':
        return speechmatics_agent.replay(events, config)[1]
    if config['provider'] == 'assemblyai':
        return assembly_adapter(config).replay(events, config)[1]
    if config['provider'] == 'reson8':
        return reson8.replay(events, config)[1]
    if config['provider'] == 'gradium':
        return gradium.replay(events, config)[1]
    if config['provider'] in trial_providers.MODELS:
        return trial_providers.replay(events, config)[1]
    if config['provider'] == 'google':
        return chirp.replay(events, config)[1]
    return deepgram.reduce_events(events, config) if is_nova(config) else provider_protocol.replay(events, config)[1]


def transcript_at(events, cutoff, config=None):
    if config and config['model'] in gemini_live.MODELS:
        return gemini_live.replay(events, config, cutoff)[0]
    if config and config['model'] == 'linden-1':
        return speechmatics_agent.replay(events, config, cutoff)[0]
    if config and config['provider'] == 'assemblyai':
        return assembly_adapter(config).replay(events, config, cutoff)[0]
    # Old report callers and v3 files retain the exact Nova reconstruction.
    if config is None or is_nova(config):
        return deepgram.transcript_at(events, cutoff)
    if config['provider'] == 'reson8':
        return reson8.replay(events, config, cutoff)[0]
    if config['provider'] == 'gradium':
        return gradium.replay(events, config, cutoff)[0]
    if config['provider'] in trial_providers.MODELS:
        return trial_providers.replay(events, config, cutoff)[0]
    if config['provider'] == 'google':
        return chirp.replay(events, config, cutoff)[0]
    return provider_protocol.replay(events, config, cutoff)[0]


def verify_model(config):
    validate(config)
    if is_nova(config):
        import json
        import urllib.request
        from datetime import datetime, timezone
        with urllib.request.urlopen('https://api.deepgram.com/v1/models', timeout=30) as response:
            catalog = json.load(response)
        matches = [m for m in catalog.get('stt', []) if m.get('uuid') == config['expected_model_uuid']
                   and m.get('version') == config['version'] and m.get('streaming')]
        if len(matches) != 1:
            raise ValueError('Pinned Deepgram model version/UUID is absent from the public catalog')
        return dict(checked_at=datetime.now(timezone.utc).isoformat(), model=matches[0],
                    source='https://api.deepgram.com/v1/models', verification='catalog_pin')
    return {'model': {'name': config['model'], 'version': config['version']},
            'verification': 'configured_alias_only', 'live_access': 'pending_smoke',
            'source': config['model_source']}


def require_credential(config):
    key, _ = credential(config['provider'])
    if not key:
        raise ValueError(f'Missing credential for {config["provider"]}')
    if config['provider'] == 'google':
        chirp.credentials_from_json(key)
    return key


def cost_total(values):
    values = list(values)
    return sum(values) if all(v is not None for v in values) else None
