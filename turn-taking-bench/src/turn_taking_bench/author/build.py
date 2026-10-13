"""Specs -> rendered lines -> bounds -> timeline -> labels -> dataset directory.

``build_dataset`` is what ``turn-taking-bench author build`` runs.

Pauses: a ``<break time="1.5s"/>`` marker is only what the script *asked* for;
what the renderer produced is what a model hears, so the labels follow the
*detected* boundaries (``bounds.scripted_segments``). ``pause_report`` records,
per break, the scripted length next to the detected one so the renderer's
fidelity is visible (``ground_truth.json`` ``pauses``, and the build log). A
break that did not come out as a detectable pause (or a sentence pause that
did) is a ``BuildError`` naming the spec, the item and the line; the build goes
on to the other conversations and reports every failure together. A line with
``splice_breaks`` is rendered one segment per TTS call and joined with exactly
the scripted silence (``render_item``), for the case where the TTS stretches a
filler across the break instead of pausing. A degradation
transform (``service.transforms.TRANSFORMS``: ``clean``, ``noise-20db``,
``noise-10db``, ``telephone``, ``reverb``, ``clipping``, ``dropouts``) is
applied to each finished channel, after composition, so the labels describe
the clean timeline and the transform only changes what a model hears. The
transforms are seeded and length-preserving; the name is recorded in the
dataset metadata and manifest.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from turn_taking_bench.author import service_bridge
from turn_taking_bench.author.bounds import MIN_PAUSE_S, Bounds, BoundsError, scripted_segments, speech_segments
from turn_taking_bench.author.compose import OUTPUT_RATE, ComposeError, compose
from turn_taking_bench.author.labels import annotation_tracks, planted_events
from turn_taking_bench.author.package import BuiltConversation, encode_audio, write_dataset
from turn_taking_bench.author.render import Render, RenderCache, Renderer
from turn_taking_bench.author.spec import ConversationSpec, Item, load_specs


class BuildError(ValueError):
    """A rendered line does not carry the pauses its script asked for."""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def line_bounds(spec: ConversationSpec, item: Item, samples: np.ndarray, sample_rate: int) -> Bounds:
    """The render's segments, one per scripted segment; a ``BuildError`` naming
    the spec, item and line when the scripted pauses are not in the audio."""
    min_pause_s = MIN_PAUSE_S if item.min_pause_s is None else item.min_pause_s
    try:
        return scripted_segments(samples, sample_rate, len(item.segments), min_pause_s=min_pause_s)
    except BoundsError as exc:
        raise BuildError(
            f"spec {spec.id}, item {item.id!r} ({item.speaker}, breaks {item.breaks}): {exc}; line: {item.text!r}"
        ) from exc


def render_item(cache: RenderCache, renderer: Renderer, voice: str, item: Item) -> Render:
    """The item's render: one TTS call for the line, or with ``splice_breaks`` one
    per segment joined with exactly the scripted silence (provenance lists the
    segment renders; the sha256 covers them in order)."""
    if not item.splice_breaks:
        return cache.get(renderer, item.text, voice, item.speed)
    parts = [cache.get(renderer, segment, voice, item.speed) for segment in item.segments]
    rate = parts[0].sample_rate
    if any(part.sample_rate != rate for part in parts):
        raise BuildError(f"item {item.id!r}: segment renders differ in sample rate")
    pieces: list[np.ndarray] = []
    for index, part in enumerate(parts):
        # Trim each segment to its detected speech so the join holds exactly the
        # scripted silence (a render's own lead-in/tail would otherwise add to it).
        runs = speech_segments(part.samples, rate, min_pause_s=0.0).segments
        if not runs:
            raise BuildError(f"item {item.id!r}: no speech detected in the render of {item.segments[index]!r}")
        pieces.append(part.samples[int(round(runs[0].start_s * rate)) : int(round(runs[-1].end_s * rate))])
        if index < len(item.breaks):
            pieces.append(np.zeros(int(round(item.breaks[index] * rate)), dtype=np.float32))
    digest = hashlib.sha256("\n".join(part.sha256 for part in parts).encode()).hexdigest()
    return Render(
        samples=np.concatenate(pieces).astype(np.float32),
        sample_rate=rate,
        sha256=digest,
        provenance={"spliced": True, "breaks_s": item.breaks, "segments": [part.provenance for part in parts]},
    )


def pause_report(spec: ConversationSpec, item: Item, bounds: Bounds) -> list[dict[str, Any]]:
    """Scripted vs detected length of every ``<break/>`` in ``item``'s render."""
    segments, breaks = item.segments, item.breaks
    found = bounds.segments
    if len(found) != len(segments):  # line_bounds guarantees this; keep the invariant loud
        raise BuildError(f"spec {spec.id}, item {item.id!r}: {len(segments)} scripted segment(s), {len(found)} detected")
    report = [
        {
            "kind": "break",
            "index": index,
            "scripted_s": scripted,
            "detected_s": round(found[index + 1].start_s - found[index].end_s, 3),
            "spliced": item.splice_breaks,
            "before": segments[index],
            "after": segments[index + 1],
        }
        for index, scripted in enumerate(breaks)
    ]
    report += [
        {"kind": "unscripted", "start_s": round(gap.start_s, 3), "end_s": round(gap.end_s, 3), "detected_s": round(gap.end_s - gap.start_s, 3)}
        for gap in bounds.unscripted_pauses
    ]
    return report


def apply_transform(name: str, samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """One named service-bench transform on a float32 channel (int16 in and out)."""
    transforms = service_bridge.transforms()
    audio = service_bridge.audio()
    transform = transforms.get(name)
    pcm = audio.to_pcm(np.asarray(samples, dtype=np.float64) * 32767.0)
    degraded = audio.to_array(transform.apply(pcm, sample_rate))
    if len(degraded) != len(samples):
        raise RuntimeError(f"transform {name!r} changed the length ({len(samples)} -> {len(degraded)})")
    return (degraded.astype(np.float32) / 32768.0).astype(np.float32)


def build_conversation(
    spec: ConversationSpec,
    renderer: Renderer,
    cache: RenderCache,
    *,
    transform: str = "clean",
    output_rate: int = OUTPUT_RATE,
) -> BuiltConversation:
    renders: dict[str, tuple[Render, Any]] = {}
    pauses: dict[str, list[dict[str, Any]]] = {}
    for item in spec.items:
        render = render_item(cache, renderer, spec.voices[item.speaker], item)
        bounds = line_bounds(spec, item, render.samples, render.sample_rate)
        renders[item.id] = (render, bounds)
        report = pause_report(spec, item, bounds)
        if report:
            pauses[item.id] = report
    timeline = compose(spec, renders, output_rate=output_rate)
    channels = {channel: apply_transform(transform, samples, output_rate) for channel, samples in timeline.channels.items()}
    return BuiltConversation(
        timeline=timeline,
        tracks=annotation_tracks(timeline),
        planted=planted_events(timeline),
        channels=channels,
        transform=transform,
        renderer={"name": renderer.name, "model": renderer.model},
        pauses=pauses,
    )


def pause_table(built: list[BuiltConversation]) -> str:
    """Scripted vs detected length of every break in the build, one line each,
    then any unscripted pause (a natural gap at least as long as a break minimum)."""
    lines = [f"{'spec':>6} {'item':>5} {'speaker':>7} {'scripted':>9} {'detected':>9} {'delta':>7}"]
    for conversation in built:
        for item_id, reports in conversation.pauses.items():
            speaker = conversation.timeline.item(item_id).item.speaker
            for report in reports:
                if report["kind"] == "break":
                    delta = report["detected_s"] - report["scripted_s"]
                    lines.append(
                        f"{conversation.timeline.spec.id:>6} {item_id:>5} {speaker:>7} "
                        f"{report['scripted_s']:>8.2f}s {report['detected_s']:>8.2f}s {delta:>+6.2f}s"
                        + (" spliced" if report.get("spliced") else "")
                    )
                else:
                    lines.append(
                        f"{conversation.timeline.spec.id:>6} {item_id:>5} {speaker:>7} {'(none)':>9} {report['detected_s']:>8.2f}s "
                        f"{'':>7} unscripted pause at {report['start_s']:.2f}s in the render, inside one turn"
                    )
    return "\n".join(lines) if len(lines) > 1 else ""


def estimate_characters(specs: list[ConversationSpec], renderer: Renderer, cache: RenderCache) -> dict[str, Any]:
    """What a build of ``specs`` would send to the renderer: the characters of
    every (text, voice, speed) not already in the cache, counted once each (a
    line repeated across specs or pairings is one render), and the number of
    renderer calls. Run before an ElevenLabs build so the cost is known first."""
    pending: dict[tuple[str, str, float], int] = {}
    cached = 0
    for spec in specs:
        for item in spec.items:
            voice = spec.voices[item.speaker]
            texts = item.segments if item.splice_breaks else [item.text]
            for text in texts:
                wav_path, meta_path = cache.paths(renderer, text, voice, item.speed)
                if wav_path.is_file() and meta_path.is_file():
                    cached += 1
                else:
                    pending[(text, voice, item.speed)] = len(text)
    return {
        "conversations": len(specs),
        "renderer_calls": len(pending),
        "characters": sum(pending.values()),
        "cached_lines": cached,
    }


def build_dataset(
    specs_dir: Path,
    out: Path,
    renderer: Renderer,
    *,
    transform: str = "clean",
    cache: RenderCache | None = None,
    name: str | None = None,
    max_characters: int | None = None,
) -> dict[str, Any]:
    """Build every spec under `specs_dir` into the dataset directory `out`; returns the manifest.

    ``max_characters`` stops the build before any rendering when the estimate of
    characters still to render exceeds it (a cost guard for paid renderers)."""
    service_bridge.transforms().get(transform)  # fail early on an unknown name
    cache = cache or RenderCache()
    specs = load_specs(Path(specs_dir))
    estimate = estimate_characters(specs, renderer, cache)
    log(
        f"build: {estimate['conversations']} conversations; {estimate['renderer_calls']} lines to render "
        f"({estimate['characters']} characters), {estimate['cached_lines']} cached"
    )
    if max_characters is not None and estimate["characters"] > max_characters:
        raise BuildError(
            f"the build would render {estimate['characters']} characters, over the limit of {max_characters}; nothing rendered"
        )
    built: list[BuiltConversation] = []
    errors: list[str] = []
    for spec in specs:
        try:
            conversation = build_conversation(spec, renderer, cache, transform=transform)
        except (BuildError, ComposeError) as exc:  # keep going: report every failing line at once
            log(f"{spec.id} ({spec.conversation_id}): FAILED: {exc}")
            errors.append(str(exc))
            continue
        counts = conversation.planted.counts()
        log(
            f"{spec.id} ({spec.conversation_id}, {spec.pairing or '/'.join(spec.voices.values())}): "
            f"{len(spec.items)} items, {conversation.timeline.duration_s:.1f}s, "
            f"planted eot+={counts['eot_pos']} eot-={counts['eot_neg']} int+={counts['int_pos']} int-={counts['int_neg']}"
        )
        encode_audio(conversation)  # keep only FLAC bytes per finished conversation
        built.append(conversation)
    table = pause_table(built)
    if table:
        log("pauses: scripted (spec) vs detected (render)\n" + table)
    log(
        f"renders: {cache.hits} cache hits, {cache.misses} renderer calls, "
        f"{cache.characters} characters rendered ({renderer.name}:{renderer.model})"
    )
    if errors:
        raise BuildError(f"{len(errors)} conversation(s) failed to build; nothing written:\n" + "\n".join(errors))
    manifest = write_dataset(
        Path(out),
        built,
        name=name,
        generated_by={
            "specs_dir": str(Path(specs_dir)),
            "renderer": {"name": renderer.name, "model": renderer.model},
            "transform": transform,
            "render_cache": cache.stats(),
        },
    )
    log(f"wrote {out}: {manifest['conversations']['count']} conversations, {manifest['total_bytes'] / 1e6:.1f} MB")
    return manifest
