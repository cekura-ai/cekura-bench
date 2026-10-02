"""Resolve existing credential names without changing files or logging values."""
import os
from pathlib import Path
from dotenv import dotenv_values

NAMES = {
    'assemblyai': ('ASSEMBLYAI_API_KEY', 'ASSEMBLY_API_KEY'),
    'reson8': ('RESON_API_KEY',),
    'gradium': ('GRADIUM_API_KEY',),
    'soniox': ('SONIOX_API_KEY',),
    'smallest': ('SMALLEST_API_KEY',),
    'sarvam': ('SARVAM_API_KEY',),
    'inworld': ('INWORLD_API_KEY',),
    'deepgram': ('DEEPGRAM_API_KEY',),
    'openai': ('OPENAI_API_KEY', 'OpenAI'),
    'gemini': ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'Google'),
    'google': ('GOOGLE_SERVICE_ACCOUNT_JSON', 'VERTEX_CREDS', 'GOOGLE_APPLICATION_CREDENTIALS', 'GCP'),
    'elevenlabs': ('ELEVENLABS_API_KEY', 'ElevenLabs'),
    'speechmatics': ('SPEECHMATICS_API_KEY', 'Speechmatics'),
    'cartesia': ('CARTESIA_API_KEY', 'Cartesia'),
}


def credential(provider, environ=None, env_file='.env'):
    """Process environment wins over the dotenv file; canonical name wins in each."""
    env = os.environ if environ is None else environ
    file = dotenv_values(env_file, interpolate=False) if env_file else {}
    for source in (env, file):
        for name in NAMES[provider]:
            if source.get(name) and source[name].strip():
                value = source[name].strip()
                if provider == 'google' and not value.startswith('{') and len(value) < 1024:
                    path = Path(value).expanduser()
                    if path.is_file():
                        value = path.read_text().strip()
                return value, name
    return '', None


def command_environment(provider, **kwargs):
    key, _ = credential(provider, **kwargs)
    if not key:
        raise ValueError(f'Missing {NAMES[provider][0]} (configured aliases are accepted)')
    if provider == 'google':
        from .chirp import credentials_from_json
        credentials_from_json(key)
    return {NAMES[provider][0]: key}


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, '[REDACTED]') if secret else value
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, dict):
        return {k: '[REDACTED]' if k.lower() in {'key', 'api_key', 'authorization', 'token', 'audio_base_64'}
                else redact(v, secret) for k, v in value.items()}
    return value
