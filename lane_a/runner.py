"""Runs probes and writes the artifacts a result is recomputable from.

A run produces a directory, not a number. Each cell carries provider, model,
configuration, corpus version, voice, harness commit and methodology version.
Published results drift from the code that produced them as soon as either can
change without the other, and a number whose method cannot be pinned down is not
a measurement. Stamping every cell is what keeps the two attached.

One session per cell, deliberately. Sharing a session across probes would let one
probe's conversation history change the next probe's behaviour, and the resulting
row would describe an ordering as much as a provider.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from lane_a import events as ev
from lane_a.adapters.base import AdapterError, SessionConfig, ToolSpec, TurnDetection
from lane_a.audio import write_wav
from lane_a.caller import BranchingCaller
from lane_a.clips import Corpus
from lane_a.probes import Probe, ProbeContext, ProbeResult
from lane_a.registry import PROVIDERS
from lane_a.tools.server import MockToolServer

METHODOLOGY_VERSION = "lane-a/0.1"


@dataclass
class RunSpec:
    """One measurement campaign."""

    provider: str
    probes: Sequence[Probe]
    configs: Sequence[TurnDetection]
    voices: Sequence[str]
    repeats: int = 5
    model: str | None = None
    voice_name: str | None = None          # the provider's own output voice
    instructions: str = ""
    tools: tuple[ToolSpec, ...] = ()
    modality: str = "audio"
    suite: str | None = None               # agent-definitions/<suite>, enables the tool server
    corpus_root: str = "corpus/lane-a"
    out_root: str = "data/lane-a"
    label: str = ""


@dataclass
class Cell:
    """A published row, before aggregation."""

    provider: str
    model: str
    config: str
    voice: str
    repeat: int
    probe: str
    variant: str
    verdict: str | None
    void: str | None
    values: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {**asdict(self), **{}}


def harness_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- an unversioned checkout is a caveat, not a crash
        return "unknown"


class Runner:
    def __init__(self, spec: RunSpec, corpus: Corpus, api_key: str) -> None:
        self.spec = spec
        self.corpus = corpus
        self.api_key = api_key
        self.entry = PROVIDERS[spec.provider]
        self.tools = MockToolServer(spec.suite) if spec.suite else None
        self.model = spec.model or self.entry.default_model
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.root = Path(spec.out_root) / f"{started}-{spec.provider}{'-' + spec.label if spec.label else ''}"
        self.cells: list[Cell] = []

    # -- one cell ---------------------------------------------------------

    async def run_cell(self, probe: Probe, config: TurnDetection, voice: str, repeat: int) -> Cell:
        slug = f"{probe.slug}/{config.label}/{voice}/r{repeat}"
        directory = self.root / probe.slug / config.label / voice / f"r{repeat}"
        clock = ev.Clock()
        log = ev.EventLog(clock, directory / "events.jsonl", directory / "raw.jsonl")

        tools = MockToolServer(self.spec.suite) if self.spec.suite else None
        session = SessionConfig(
            instructions=self.spec.instructions or (tools.system_prompt if tools else ""),
            voice=self.spec.voice_name or self.entry.default_voice,
            tools=self.spec.tools or (tools.tool_specs() if tools else ()),
            turn_detection=config,
            modality=self.spec.modality,
            transcribe_input=self.spec.modality == "audio",
        )
        adapter = self.entry.adapter(model=self.model, api_key=self.api_key, log=log, config=session)
        result = ProbeResult(probe.name)
        try:
            async with adapter:
                caller = BranchingCaller(adapter, log)
                # The text arm sends no audio at all, so the carrier stays parked:
                # streaming silence into a text-modality session would measure
                # nothing and could be refused outright.
                if self.spec.modality == "audio":
                    caller.start()
                try:
                    result = await probe.run(
                        ProbeContext(
                            adapter=adapter, caller=caller, corpus=self.corpus,
                            voice=voice, log=log, tools=tools,
                        )
                    )
                finally:
                    await caller.stop()
                if caller.max_slip_ms > 25.0:
                    # Our own host, not the provider. Voiding is the honest call.
                    result.void = result.void or f"caller pacing slipped {caller.max_slip_ms:.0f} ms"
        except AdapterError as exc:
            result = ProbeResult(probe.name, void=f"provider refused the session: {exc}")
        except Exception as exc:  # noqa: BLE001
            result = ProbeResult(probe.name, void=f"harness error: {exc!r}")
        finally:
            if adapter.agent_pcm:
                write_wav(directory / "agent.wav", bytes(adapter.agent_pcm), adapter.output_rate)
            if adapter.caller_pcm:
                write_wav(directory / "caller.wav", bytes(adapter.caller_pcm), adapter.input_rate)
            # The timestamps that make the wavs measurable. Without them the
            # artifacts show what was said but not when it became audible, and a
            # published latency could not be recomputed by anyone but us.
            (directory / "timelines.json").write_text(
                json.dumps(
                    {
                        "caller": adapter.caller_timeline.as_json(),
                        "agent": adapter.agent_timeline.as_json(),
                        "wall_origin": clock.wall_origin,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            log.close()

        cell = Cell(
            provider=self.spec.provider,
            model=self.model,
            config=config.label,
            voice=voice,
            repeat=repeat,
            probe=probe.name,
            variant=probe.slug,
            verdict=result.verdict,
            void=result.void,
            values=result.values,
            artifacts={"dir": str(directory), "slug": slug},
        )
        self.cells.append(cell)
        return cell

    # -- the campaign -----------------------------------------------------

    async def run(self, on_cell=None) -> Path:
        for probe in self.spec.probes:
            for config in self.spec.configs:
                for voice in self.spec.voices:
                    for repeat in range(1, self.spec.repeats + 1):
                        cell = await self.run_cell(probe, config, voice, repeat)
                        if on_cell:
                            on_cell(cell)
                        await asyncio.sleep(0.5)  # be a polite client, not a load test
        return self.write_results()

    def write_results(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        provenance = {
            "methodology_version": METHODOLOGY_VERSION,
            "harness_commit": harness_commit(),
            "corpus_version": self.corpus.version,
            "provider": self.spec.provider,
            "model": self.model,
            "output_voice": self.spec.voice_name or self.entry.default_voice,
            "configs": [c.label for c in self.spec.configs],
            "caller_voices": list(self.spec.voices),
            "repeats": self.spec.repeats,
            "modality": self.spec.modality,
            "tool_contract": self.spec.suite,
            "discloses": list(self.entry.discloses),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        (self.root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        with open(self.root / "cells.jsonl", "w", encoding="utf-8") as handle:
            for cell in self.cells:
                handle.write(json.dumps(cell.as_json()) + "\n")
        (self.root / "manifest.json").write_text(
            json.dumps(self.corpus.manifest(), indent=2) + "\n", encoding="utf-8"
        )
        return self.root
