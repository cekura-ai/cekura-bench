"""Account settings an S2S adapter needs beyond its one API key.

The service bench constructs an adapter with a single ``api_key``. Deepslate
also needs account identifiers and the ElevenLabs key that voices its replies,
so its adapter reads those itself when they are not passed as keyword
arguments: environment first, then the dotenv file named by
``TURN_BENCH_ENV_FILE`` (if set), then ``turn-bench/.env``, then the repository
root's ``.env``. Values are returned to the caller only; nothing here logs,
prints or records them.
"""

from __future__ import annotations

import os
from pathlib import Path

from turn_bench.assets import TURN_BENCH_ROOT

ENV_FILES: tuple[Path, ...] = (TURN_BENCH_ROOT / ".env", TURN_BENCH_ROOT.parent / ".env")


def _dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _sep, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def setting(name: str) -> str | None:
    """``name`` from the environment or the default dotenv files; None if unset."""
    if os.environ.get(name):
        return os.environ[name]
    extra = os.environ.get("TURN_BENCH_ENV_FILE")
    for path in ((Path(extra),) if extra else ()) + ENV_FILES:
        value = _dotenv(path).get(name)
        if value:
            return value
    return None
