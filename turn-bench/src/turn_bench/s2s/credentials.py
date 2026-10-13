"""Provider credentials, read in-process and never written anywhere.

The service registry names each provider's credential variable
(``ProviderEntry.credential_env``). The value is taken from the environment,
else from a dotenv file: ``turn-bench/.env``, then the repository root's
``.env``. Aliases follow the stt-bench credentials loader (the Google key may be
called ``GEMINI_AUTHORIZATION``, ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``).

Nothing here logs, prints or returns anything but the one value asked for, and
``describe`` reports only *which* variable was found, never its content.
"""

from __future__ import annotations

import os
from pathlib import Path

from turn_bench.assets import TURN_BENCH_ROOT

ALIASES: dict[str, tuple[str, ...]] = {
    "GEMINI_AUTHORIZATION": ("GEMINI_AUTHORIZATION", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "GEMINI_API_KEY": ("GEMINI_API_KEY", "GEMINI_AUTHORIZATION", "GOOGLE_API_KEY"),
}

DEFAULT_ENV_FILES: tuple[Path, ...] = (TURN_BENCH_ROOT / ".env", TURN_BENCH_ROOT.parent / ".env")


class MissingCredential(RuntimeError):
    pass


def dotenv_values(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines of a dotenv file; quotes stripped, comments and blank
    lines ignored, ``export`` prefixes accepted. Values are not expanded."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _sep, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def credential(name: str, env_file: str | os.PathLike[str] | None = None) -> str:
    """The credential ``name`` (or one of its aliases): environment first, then
    ``env_file`` if given, then the default dotenv files. Raises
    ``MissingCredential`` naming the variable, never echoing any value."""
    found = describe(name, env_file)
    if found is None:
        raise MissingCredential(f"{name} is not set (aliases: {', '.join(ALIASES.get(name, (name,)))})")
    source, key = found
    if source == "environment":
        return os.environ[key]
    return dotenv_values(Path(source))[key]


def describe(name: str, env_file: str | os.PathLike[str] | None = None) -> tuple[str, str] | None:
    """Where ``name`` would be read from: ``("environment", KEY)`` or
    ``(path, KEY)``; None when it is nowhere. For the record, not the value."""
    names = ALIASES.get(name, (name,))
    for key in names:
        if os.environ.get(key):
            return ("environment", key)
    files = [Path(env_file)] if env_file else []
    files += list(DEFAULT_ENV_FILES)
    for path in files:
        values = dotenv_values(path)
        for key in names:
            if values.get(key):
                return (str(path), key)
    return None
