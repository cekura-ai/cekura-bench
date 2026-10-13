"""Read-only preflight for live callers driven by our authored conversations.

Agent scripts are markers, never clips or model instructions. Every caller
clip must already exist in the ElevenLabs cache; this module cannot render.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from turn_taking_bench.author import service_bridge
from turn_taking_bench.author.build import render_item
from turn_taking_bench.author.render import Render, RenderCache, VOICES
from turn_taking_bench.author.spec import ConversationSpec, Item, load_specs

SCHEMA = "turn-taking-bench.authored-live-plan.v1"


class PreflightError(ValueError):
    """Missing or invalid input: never repaired with a paid render or a guess."""


@dataclass(frozen=True)
class CacheOnlyRenderer:
    name: str = "elevenlabs"
    model: str = "eleven_turbo_v2_5"

    def render(self, text: str, voice: str, speed: float = 1.0):
        raise PreflightError("live preflight is cache-only; rendering is forbidden")


class VerifiedCache(RenderCache):
    """RenderCache's key contract with strict validation and no write path."""

    def get(self, renderer, text: str, voice: str, speed: float = 1.0) -> Render:
        wav, sidecar = self.paths(renderer, text, voice, speed)
        if not wav.is_file() or not sidecar.is_file():
            raise PreflightError(f"missing cached WAV/sidecar: {wav}")
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            digest = hashlib.sha256(wav.read_bytes()).hexdigest()
            expected = {"renderer": renderer.name, "model": renderer.model,
                        "voice": voice, "text": text, "speed": speed,
                        "sample_rate": 24000, "file": wav.name, "sha256": digest}
            legacy_natural_speed = "speed" not in meta and speed == 1.0
            # render_key explicitly retains pre-speed natural-rate cache keys.
            # The canonical key/path and all remaining metadata must validate;
            # missing speed for a non-natural key remains an error.
            for field, value in expected.items():
                if field == "speed" and legacy_natural_speed:
                    continue
                if meta.get(field) != value:
                    raise PreflightError(f"cached {field} mismatch: {sidecar}")
            info = sf.info(wav)
            if (info.channels != 1 or info.samplerate != 24000
                    or info.subtype != "PCM_16" or info.format != "WAV"
                    or info.frames <= 0 or meta.get("samples") != info.frames):
                raise PreflightError(f"cached WAV format/sample count mismatch: {wav}")
            samples, rate = sf.read(wav, dtype="float32", always_2d=False)
            if not np.isfinite(samples).all():
                raise PreflightError(f"non-finite cached samples: {wav}")
        except PreflightError:
            raise
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            raise PreflightError(f"invalid cached WAV/sidecar: {wav}") from exc
        self.hits += 1
        if legacy_natural_speed:
            meta = {**meta, "legacy_natural_speed": True,
                    "speed_contract": "natural-rate legacy key preserved by render_key; sidecar speed absent"}
        return Render(samples, rate, digest, meta)


def _pcm(render: Render) -> bytes:
    # PCM_16 decoding is exact in float32. Multiplication by 32768 round-trips
    # the original integers, including -32768, rather than attenuating by 32767.
    return np.clip(np.rint(render.samples * 32768.0), -32768, 32767).astype("<i2").tobytes()


def _clip_record(cache: VerifiedCache, renderer: CacheOnlyRenderer,
                 voice: str, item: Item, render: Render) -> dict[str, Any]:
    texts = item.segments if item.splice_breaks else [item.text]
    parts = []
    for text in texts:
        wav, sidecar = cache.paths(renderer, text, voice, item.speed)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        parts.append({"wav_path": str(wav.resolve()), "meta_path": str(sidecar.resolve()),
                      "sha256": meta["sha256"], "text": text,
                      "legacy_natural_speed": "speed" not in meta and item.speed == 1.0})
    return {"sha256": render.sha256, "pcm_sha256": hashlib.sha256(_pcm(render)).hexdigest(),
            "sample_rate": render.sample_rate, "samples": len(render.samples),
            "provenance": render.provenance, "cache_parts": parts,
            "legacy_natural_speed": any(part["legacy_natural_speed"] for part in parts),
            "boundary_definition": "first transmitted PCM sample; not detected acoustic onset",
            "composition": "authored splice_breaks with existing speech detector segment trim" if item.splice_breaks else "cached WAV PCM unchanged"}


def _select(specs: list[ConversationSpec], ids: list[str] | None,
            caller_voice: str) -> list[tuple[ConversationSpec, str]]:
    if caller_voice not in VOICES:
        raise PreflightError(f"unknown caller voice {caller_voice!r}")
    by_template: dict[str, list[ConversationSpec]] = {}
    by_conversation = {s.conversation_id: s for s in specs}
    for spec in specs:
        by_template.setdefault(spec.id, []).append(spec)
    requested = ids if ids is not None else [
        sid for sid, rows in by_template.items() if rows[0].family in {"barge_in", "backchannels"}]
    if not requested:
        raise PreflightError("no authored interaction conversations selected")
    if len(set(requested)) != len(requested):
        raise PreflightError("duplicate requested IDs")
    chosen = []
    templates_seen: set[str] = set()
    for requested_id in requested:
        if requested_id in by_template:
            matches = [s for s in by_template[requested_id] if s.voices["caller"] == caller_voice]
            if not matches:
                raise PreflightError(f"spec {requested_id}: no pairing for caller voice {caller_voice}")
            spec = matches[0]  # source pairing order is stable, agent audio is unused
            rule = "first declared pairing for caller voice; scripted agent audio is not streamed"
            if len(matches) > 1:
                rule += f"; selected {spec.pairing} from {len(matches)} matching pairings"
        elif requested_id in by_conversation:
            spec = by_conversation[requested_id]
            if spec.voices["caller"] != caller_voice:
                raise PreflightError(f"conversation {requested_id}: caller voice differs from {caller_voice}")
            rule = "explicit conversation ID"
        else:
            raise PreflightError(f"unknown spec/conversation ID {requested_id!r}")
        if spec.id in templates_seen:
            raise PreflightError(f"ambiguous selection: template {spec.id} selected more than once")
        templates_seen.add(spec.id)
        chosen.append((spec, rule))
    return chosen


def _splits(specs_dir: Path) -> tuple[dict[str, str], dict | None]:
    path = specs_dir.parent / "splits.json"
    if not path.is_file():
        return {}, None
    payload = path.read_bytes()
    raw = json.loads(payload)
    split_of = {}
    for name in ("dev", "test", "holdout"):
        for sid in raw.get(name, []):
            if sid in split_of:
                raise PreflightError(f"duplicate split assignment for {sid}")
            split_of[sid] = name
    return split_of, {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}


def build_plan(specs_dir: Path, cache_dir: Path, *, ids: list[str] | None = None,
               caller_voice: str = "f-us", split: str = "test") -> dict:
    if split not in {"dev", "test"}:
        raise PreflightError("live authored plans support only dev or test splits")
    specs_dir, cache_dir = Path(specs_dir), Path(cache_dir)
    specs = load_specs(specs_dir)
    split_of, split_source = _splits(specs_dir)
    if split_source is None and ids is None:
        raise PreflightError("without splits.json, explicit IDs are required for local fixtures")
    selection_ids = ids
    if ids is None:
        # Public dev/test membership is an allowlist. Unlisted templates must
        # never be inferred to be holdout or enter a paid run by default.
        selection_ids = list(dict.fromkeys(s.id for s in specs
            if split_of.get(s.id) == split and s.family in {"barge_in", "backchannels"}))
    selected = _select(specs, selection_ids, caller_voice)
    if split_source is not None:
        for spec, _ in selected:
            if split_of.get(spec.id) != split:
                raise PreflightError(f"spec {spec.id}: not explicitly assigned to requested {split} split")
    cache, renderer = VerifiedCache(cache_dir), CacheOnlyRenderer()
    conversations, exclusions = [], []
    counts = {"conversations": len(selected), "eligible_conversations": 0,
              "excluded_conversations": 0, "caller_clips": 0,
              "barge_in": 0, "backchannel": 0, "excluded_events": 0}
    for spec, selection in selected:
        items, clips = [], {}
        trigger = None
        end_relative = [i.id for i in spec.items if i.offset_from_end_s is not None]
        reason = ("noncausal offset_from_end_s requires the future live agent speech end: "
                  + ", ".join(end_relative)) if end_relative else None
        for item in spec.items:
            raw = asdict(item)
            if item.kind == "turn":
                raw["gap_s"] = spec.gap_for(item)
            if item.speaker == "agent":
                raw["trigger_caller_id"] = item.after or trigger
            else:
                render = render_item(cache, renderer, caller_voice, item)
                clips[item.id] = _clip_record(cache, renderer, caller_voice, item, render)
                counts["caller_clips"] += 1
                if item.kind in {"barge_in", "backchannel"}:
                    raw["runtime_requirement"] = "agent speech must still be active at the exact onset-relative offset; otherwise void"
                    if reason:
                        counts["excluded_events"] += 1
                    else:
                        counts[item.kind] += 1
                if item.kind != "backchannel":
                    trigger = item.id
            items.append(raw)
        conversation = {"conversation_id": spec.conversation_id, "spec_id": spec.id,
                        "source_sha256": spec.source_sha256, "source_path": spec.source_path,
                        "pairing": spec.pairing, "caller_voice": caller_voice,
                        "selection_rule": selection, "family": spec.family,
                        "subtype": spec.subtype, "split": split_of.get(spec.id),
                        "timing": asdict(spec.timing), "items": items, "clips": clips, "void": reason}
        conversations.append(conversation)
        if reason:
            exclusions.append({"conversation_id": spec.conversation_id, "spec_id": spec.id,
                               "reason": reason, "event_ids": end_relative})
            counts["excluded_conversations"] += 1
        else:
            counts["eligible_conversations"] += 1
    return {"schema": SCHEMA, "cache_dir": str(cache_dir.resolve()), "caller_voice": caller_voice,
            "splits_source": split_source, "split": split, "conversations": conversations,
            "exclusions": exclusions, "counts": counts}


def load_clips(plan_conversation: dict, cache_dir: Path) -> dict:
    """Reload and revalidate every caller clip before opening a paid session."""
    cache, renderer = VerifiedCache(Path(cache_dir)), CacheOnlyRenderer()
    caller = service_bridge.caller()
    result = {}
    for raw in plan_conversation["items"]:
        if raw["speaker"] != "caller":
            continue
        item = Item(**{k: v for k, v in raw.items() if k in Item.__dataclass_fields__})
        render = render_item(cache, renderer, plan_conversation["caller_voice"], item)
        record = _clip_record(cache, renderer, plan_conversation["caller_voice"], item, render)
        if record != plan_conversation["clips"][item.id]:
            raise PreflightError(f"clip changed since preflight: {item.id}")
        result[item.id] = caller.Clip(item.id, _pcm(render), render.sample_rate, item.text)
    return result
