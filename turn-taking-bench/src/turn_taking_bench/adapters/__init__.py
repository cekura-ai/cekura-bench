"""Model registry: name -> (factory, default config), plus optional hooks.

An adapter module may also define ``ASSETS`` (pinned downloads that
``turn-taking-bench setup <name>`` installs, see ``turn_taking_bench.assets``) and
``provenance(config)`` (what ``run.json`` records about the model).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from turn_taking_bench import assets, s2s
from turn_taking_bench.adapters import cascade_flux, rms_vad, smart_turn_v3
from turn_taking_bench.model import ModelFactory, TurnModel

MODELS: dict[str, tuple[ModelFactory, dict[str, Any]]] = {
    "rms_vad": (rms_vad.build, rms_vad.DEFAULT_CONFIG),
    "smart_turn_v3": (smart_turn_v3.build, smart_turn_v3.DEFAULT_CONFIG),
    "cascade_flux": (cascade_flux.build, cascade_flux.DEFAULT_CONFIG),
    # Speech-to-speech providers behind the service bench's adapters
    # (turn_taking_bench/s2s): one live session per conversation, streamed in real time.
    **{f"{s2s.MODEL_PREFIX}{provider}": (s2s.build, s2s.default_config(provider)) for provider in s2s.S2S_PROVIDERS},
}

ASSETS: dict[str, list[assets.Asset]] = {
    "smart_turn_v3": smart_turn_v3.ASSETS,
}

PROVENANCE: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "smart_turn_v3": smart_turn_v3.provenance,
    "cascade_flux": cascade_flux.provenance,
    **{f"{s2s.MODEL_PREFIX}{provider}": s2s.provenance for provider in s2s.S2S_PROVIDERS},
}


def is_live(name: str) -> bool:
    """Whether a model opens a provider session per conversation (one at a time)."""
    return name.startswith(s2s.MODEL_PREFIX)


def model_names() -> list[str]:
    return sorted(MODELS)


def resolve_config(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The model's default config with `overrides` applied; unknown names fail."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(model_names())}")
    config = dict(MODELS[name][1])
    config.update(overrides or {})
    return config


def build_model(name: str, config: dict[str, Any]) -> TurnModel:
    factory, _defaults = MODELS[name]
    return factory(config)


def model_provenance(name: str, config: dict[str, Any]) -> dict[str, Any] | None:
    hook = PROVENANCE.get(name)
    return hook(config) if hook else None
