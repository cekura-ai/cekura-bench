"""Import the service bench's modules (``service/`` at the cekura-bench root).

turn-bench is its own uv project and ``service`` is not a distributable package
(no pyproject; the service bench's own entry points put the repository root on
``sys.path``, see ``bin/render-corpus.py``). So this module does the same thing,
once, and re-exports what the authoring pipeline uses:

* ``service.transforms`` — the named, seeded degradation transforms
* ``service.detector``  — frame energy / noise floor / onset constants
* ``service.audio``     — int16 PCM helpers and the published Kaiser resampler
* ``service.clips`` + ``service.corpus_v1`` — the ElevenLabs request, the 24 kHz
  master rate and the three published voices (loaded lazily: only the
  ElevenLabs renderer needs them)

``service.transforms``/``detector``/``audio`` depend on numpy only, which
turn-bench already pins. The repository root can be overridden with
``CEKURA_BENCH_ROOT`` when turn-bench is checked out elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

from turn_bench.assets import TURN_BENCH_ROOT

REPO_ROOT = Path(os.environ.get("CEKURA_BENCH_ROOT") or TURN_BENCH_ROOT.parent).resolve()


def _ensure_path() -> None:
    if not (REPO_ROOT / "service" / "__init__.py").is_file():
        raise ImportError(
            f"service bench not found at {REPO_ROOT / 'service'}; set CEKURA_BENCH_ROOT to the "
            "cekura-bench checkout that holds service/"
        )
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def transforms() -> ModuleType:
    _ensure_path()
    from service import transforms as module

    return module


def detector() -> ModuleType:
    _ensure_path()
    from service import detector as module

    return module


def audio() -> ModuleType:
    _ensure_path()
    from service import audio as module

    return module


def clips() -> ModuleType:
    """``service.clips``: ``_elevenlabs_pcm``, ``Voice``, ``MASTER_RATE``."""
    _ensure_path()
    from service import clips as module

    return module


def corpus_voices() -> dict[str, object]:
    """The service bench's published voices by label (``f-us``, ``m-us``, ``f-gb``)."""
    _ensure_path()
    from service.corpus_v1 import VOICES

    return {voice.label: voice for voice in VOICES}


def transform_names() -> list[str]:
    return sorted(transforms().TRANSFORMS)


# ---- the service bench itself, for the speech-to-speech models -------------------
#
# ``turn_bench.s2s`` opens provider sessions with the service bench's adapters
# and runs its interaction probes unchanged. Everything below is imported, never
# copied: the adapters, the event vocabulary, the caller, the probes and the
# runner are the service bench's own, so a number produced here is produced by
# the same code that produces the service bench's. These need ``websockets``
# (``uv sync --extra s2s``); the modules above do not.


def service_module(name: str) -> ModuleType:
    """``service.<name>`` with the repository root on ``sys.path``."""
    _ensure_path()
    import importlib

    return importlib.import_module(f"service.{name}")


def events() -> ModuleType:
    """``service.events``: the normalized event kinds, ``Clock`` and ``EventLog``."""
    return service_module("events")


def adapters_base() -> ModuleType:
    """``service.adapters.base``: ``RealtimeAdapter``, ``SessionConfig``, ``TurnDetection``, errors."""
    return service_module("adapters.base")


def registry() -> ModuleType:
    """``service.registry``: ``PROVIDERS`` (adapter class, default model, credential name, voice)."""
    return service_module("registry")


def metrics() -> ModuleType:
    """``service.metrics``: ``usage`` and the service bench's own readouts."""
    return service_module("metrics")


def caller() -> ModuleType:
    """``service.caller``: pacing constants (``CHUNK_MS``, ``VOID_SLIP_MS``)."""
    return service_module("caller")


def probes() -> ModuleType:
    """``service.probes``: ``BargeIn``, ``BackchannelTolerance`` and the rest."""
    return service_module("probes")


def runner() -> ModuleType:
    """``service.runner``: ``RunSpec`` and ``Runner`` (one session per cell)."""
    return service_module("runner")


def report() -> ModuleType:
    """``service.report``: ``load_run``, ``summarize_run``, ``render_markdown`` and the rate/latency summaries."""
    return service_module("report")


def corpus_v1() -> ModuleType:
    """``service.corpus_v1``: ``build()`` the published caller corpus, ``VOICES``."""
    return service_module("corpus_v1")
