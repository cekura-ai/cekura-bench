"""Runs TTS probes and writes what a published number can be recomputed from.

A run is a directory, the plan lands before the first
connection, each cell's record lands as it finishes, and an interrupted run is a
partial result rather than a lost one. One connection per cell, warmed before
t0, so no cell's number depends on what the previous cell did to the socket.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from tts_bench.common import provenance as prov
from tts_bench.common.audio import write_wav
from tts_bench.common.events import Clock, EventLog

from tts_bench import METHODOLOGY_VERSION
from tts_bench.adapters.base import AdapterError, TTSAdapter, TTSConfig
from tts_bench.corpus import CORPUS_VERSION, ITEMS, SENTINEL_ITEM, Item
from tts_bench.probes import OneShot, Probe, ProbeContext, ProbeResult
from tts_bench.registry import PROVIDERS

# The sentinel: one fixed cell -- one-shot, one prose item -- at the head of
# every run whatever the run is about. Its spread across runs is the day's
# noise of instrument plus provider, and a ranking gap smaller than that spread
# is not a ranking. Published beside the results, never folded into them.
SENTINEL_PROBE = OneShot()
SENTINEL_REPEATS = 3


@dataclass
class RunSpec:
    provider: str
    probes: Sequence[Probe]
    items: Sequence[Item] = ITEMS
    repeats: int = 3
    model: str | None = None
    voice: str | None = None
    sample_rate: int = 24000
    options: tuple[tuple[str, str], ...] = ()
    sentinel: bool = True
    corpus_version: str = CORPUS_VERSION
    out_root: str = "data/tts"
    label: str = "run"
    wait_between_cells_s: float = 0.0


@dataclass(frozen=True)
class PlannedCell:
    probe: Probe
    item: Item
    repeat: int
    sentinel: bool = False

    @property
    def cell_id(self) -> str:
        head = "sentinel" if self.sentinel else self.probe.slug
        return f"{head}/{self.item.id}/r{self.repeat}"

    def as_json(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id, "probe": self.probe.name, "variant": self.probe.slug,
            "params": self.probe.params(), "item": self.item.id, "cohort": self.item.cohort,
            "repeat": self.repeat, "sentinel": self.sentinel,
        }


@dataclass
class Cell:
    provider: str
    model: str
    voice: str
    sample_rate: int
    probe: str
    variant: str
    item: str
    cohort: str
    repeat: int
    sentinel: bool
    verdict: str | None
    void: str | None
    values: dict[str, Any]
    artifacts: dict[str, str]
    started_utc: str
    duration_s: float
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


class Runner:
    def __init__(self, spec: RunSpec, api_key: str) -> None:
        self.spec = spec
        self.entry = PROVIDERS[spec.provider]
        self.api_key = api_key
        self.config = TTSConfig(
            model=spec.model or self.entry.default_model,
            voice=spec.voice or self.entry.default_voice,
            sample_rate=spec.sample_rate,
            options=tuple(spec.options),
        )
        self.planned = self.plan()
        self.cells: list[Cell] = []
        self.started = datetime.now(timezone.utc)
        self.root = Path(spec.out_root) / f"{self.started.strftime('%Y%m%dT%H%M%SZ')}-{spec.provider}-{spec.label}"
        self.harness = prov.harness_state()
        self.environment = prov.environment()
        self._cells_file = None

    # -- planning -----------------------------------------------------------

    def plan(self) -> list[PlannedCell]:
        by_id = {item.id: item for item in ITEMS}
        cells: list[PlannedCell] = []
        if self.spec.sentinel:
            cells += [PlannedCell(SENTINEL_PROBE, by_id[SENTINEL_ITEM], r, sentinel=True) for r in range(1, SENTINEL_REPEATS + 1)]
        for probe in self.spec.probes:
            # A probe with a fixed item runs on that item alone; content is not what it measures.
            items = [by_id.get(probe.fixed_item) or self._find(probe.fixed_item)] if probe.fixed_item else list(self.spec.items)
            for item in items:
                for repeat in range(1, self.spec.repeats + 1):
                    cells.append(PlannedCell(probe, item, repeat))
        return cells

    def _find(self, item_id: str | None) -> Item:
        for item in self.spec.items:
            if item.id == item_id:
                return item
        raise KeyError(f"probe needs corpus item {item_id!r}, which this corpus lacks")

    # -- the run ------------------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        return {
            "run_id": self.root.name,
            "methodology_version": METHODOLOGY_VERSION,
            "provider": self.spec.provider,
            "model": self.config.model,
            "voice": self.config.voice,
            "sample_rate": self.config.sample_rate,
            "config": self.config.as_json(),
            "adapter": self.entry.adapter.name,
            "capabilities": {
                "transport": self.entry.adapter.transport,
                "streamed_input": self.entry.adapter.supports_streamed_input,
                "cancel": self.entry.adapter.supports_cancel,
                "continuation": self.entry.adapter.supports_continuation,
                "native_rates": list(self.entry.adapter.native_rates),
                "native_mulaw_8k": self.entry.adapter.native_mulaw_8k,
                "setup_excluded_from_t0": self.entry.adapter.setup_excluded,
            },
            "corpus_version": self.spec.corpus_version,
            "items": len(self.spec.items),
            "repeats": self.spec.repeats,
            "sentinel": self.spec.sentinel,
            "probes": [{"name": p.name, "variant": p.slug, "params": p.params(), "needs": list(p.needs)} for p in self.spec.probes],
            "started_utc": self.started.isoformat(),
            "harness": self.harness,
            "environment": self.environment,
        }

    def open_run(self) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        prov.write_json(self.root / "provenance.json", self._provenance())
        prov.write_json(self.root / "plan.json", {"run_id": self.root.name, "cells": [c.as_json() for c in self.planned]})
        if self.harness.get("dirty"):
            patch = prov.harness_patch()
            if patch:
                (self.root / "harness.patch").write_text(patch)
        self._cells_file = open(self.root / "cells.jsonl", "a", encoding="utf-8")

    async def run(self, on_cell: Callable[[Cell], None] | None = None) -> Path:
        self.open_run()
        try:
            for planned in self.planned:
                cell = await self.run_cell(planned)
                if on_cell:
                    on_cell(cell)
                if self.spec.wait_between_cells_s:
                    await asyncio.sleep(self.spec.wait_between_cells_s)
        finally:
            self._write_summary(finished=True)
            if self._cells_file:
                self._cells_file.close()
        return self.root

    def _make_adapter(self, log: EventLog, clock: Clock) -> TTSAdapter:
        return self.entry.adapter(self.config, self.api_key, log, clock)

    async def run_cell(self, planned: PlannedCell) -> Cell:
        probe, item = planned.probe, planned.item
        directory = self.root / planned.cell_id
        directory.mkdir(parents=True, exist_ok=True)
        clock = Clock()
        log = EventLog(clock, directory / "events.jsonl", directory / "raw.jsonl")
        started_utc = datetime.now(timezone.utc).isoformat()
        adapter = self._make_adapter(log, clock)
        result = ProbeResult(probe.name)
        failure: dict[str, Any] | None = None
        excluded = self.entry.adapter.unsupported_reason(self.config, probe.needs)
        try:
            if excluded:
                result = ProbeResult(probe.name, void=f"configuration not supported: {excluded}")
            else:
                async with adapter:
                    async def new_adapter() -> TTSAdapter:
                        return self._make_adapter(log, clock)
                    result = await probe.run(ProbeContext(adapter=adapter, item=item, clock=clock, log=log, new_adapter=new_adapter))
        except AdapterError as exc:
            failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            result = ProbeResult(probe.name, void=f"provider refused: {exc}")
        except Exception as exc:  # noqa: BLE001 -- a harness bug must leave its trace in the record
            failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            result = ProbeResult(probe.name, void=f"harness error: {exc!r}")
        finally:
            syntheses = result.syntheses or list(adapter.contexts.values())
            for synthesis in syntheses:
                if synthesis.pcm:
                    write_wav(directory / f"audio-{synthesis.context_id}.wav", bytes(synthesis.pcm), synthesis.rate)
            prov.write_json(
                directory / "syntheses.json",
                {"wall_origin": clock.wall_origin, "syntheses": [s.as_json() for s in syntheses]},
                indent=None,
            )
            log.close()

        cell = Cell(
            provider=self.spec.provider, model=self.config.model, voice=self.config.voice,
            sample_rate=self.config.sample_rate, probe=probe.name, variant=probe.slug,
            item=item.id, cohort=item.cohort, repeat=planned.repeat, sentinel=planned.sentinel,
            verdict=result.verdict, void=result.void, values=result.values,
            artifacts={"dir": str(directory), "slug": planned.cell_id},
            started_utc=started_utc, duration_s=round(clock.now(), 3),
            error=None if failure is None else f"{failure['type']}: {failure['message']}",
        )
        prov.write_json(directory / "cell.json", {
            "schema": "tts-bench/cell/1",
            "methodology_version": METHODOLOGY_VERSION,
            "run_id": self.root.name,
            "cell_id": planned.cell_id,
            "identity": {
                "provider": self.spec.provider, "model": self.config.model, "voice": self.config.voice,
                "sample_rate": self.config.sample_rate, "config": self.config.as_json(), "adapter": self.entry.adapter.name,
                "probe": probe.name, "variant": probe.slug, "params": probe.params(),
                "item": item.id, "cohort": item.cohort, "repeat": planned.repeat, "sentinel": planned.sentinel,
                "corpus_version": self.spec.corpus_version,
            },
            "text": {"sent": item.text, "spoken_reference": item.spoken_reference, "chars": len(item.text), "words": item.words},
            "capabilities": adapter.capabilities(),
            "result": {"verdict": result.verdict, "void": result.void, "values": result.values},
            "error": failure,
            "timing": {"started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
                       "duration_s": round(clock.now(), 3), "wall_origin": clock.wall_origin},
            "harness": self.harness,
            "environment": self.environment,
            "artifacts": prov.file_inventory(directory),
        })
        self.cells.append(cell)
        if self._cells_file:
            self._cells_file.write(json.dumps(cell.as_json()) + "\n")
            self._cells_file.flush()
        self._write_summary(finished=False)
        return cell

    def _write_summary(self, finished: bool) -> None:
        prov.write_json(self.root / "summary.json", {
            "run_id": self.root.name, "finished": finished,
            "planned": len(self.planned), "completed": len(self.cells),
            "voids": sum(1 for c in self.cells if c.void), "errors": sum(1 for c in self.cells if c.error),
            "passed": sum(1 for c in self.cells if c.verdict == "pass"),
            "failed": sum(1 for c in self.cells if c.verdict == "fail"),
        })
