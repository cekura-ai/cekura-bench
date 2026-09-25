"""Runs TTS probes and writes what a published number can be recomputed from.

A run is a directory in the store (``tts_bench.store``). The plan, the corpus and
the provenance land before the first connection; each cell lands whole as it
finishes; an interrupted run is a partial result that ``resume`` completes
rather than a lost one. One connection per cell, opened before t0, so no cell's
number depends on what the previous cell did to the socket.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from tts_bench.common import provenance as prov
from tts_bench.common.audio import write_wav
from tts_bench.common.events import Clock, EventLog

from tts_bench import METHODOLOGY_VERSION
from tts_bench import corpus as corpus_mod
from tts_bench import store
from tts_bench.adapters.base import AdapterError, TTSAdapter, TTSConfig
from tts_bench.corpus import CORPUS_VERSION, ITEMS, SENTINEL_ITEM, Item
from tts_bench.probes import OneShot, Probe, ProbeContext, ProbeResult, probe_from_json
from tts_bench.registry import PROVIDERS

# The sentinel: one fixed cell -- one-shot, one prose item -- at the head of
# every run whatever the run is about. Its spread across runs is the day's
# noise of instrument plus provider, and a ranking gap smaller than that spread
# is not a ranking. Published beside the results, never folded into them.
SENTINEL_PROBE = OneShot()
SENTINEL_REPEATS = 3
HEARTBEAT_S = 60.0


class ResumeRefused(RuntimeError):
    """The run on disk and the one asked for are not the same measurement."""


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
    store: str | None = None          # default: tts_bench.store.default_store()
    label: str = "run"
    wait_between_cells_s: float = 0.0

    @classmethod
    def from_run(cls, run_dir: Path) -> "RunSpec":
        """The spec a stored run was started with, so resuming needs nothing but the directory."""
        provenance = json.loads((run_dir / "provenance.json").read_text())
        version, items = corpus_mod.load(run_dir / "corpus.json")
        config = provenance["config"]
        return cls(
            provider=provenance["provider"], probes=[probe_from_json(p) for p in provenance["probes"]],
            items=items, repeats=provenance["repeats"], model=config["model"], voice=config["voice"],
            sample_rate=config["sample_rate"], options=tuple(config["options"].items()),
            sentinel=provenance["sentinel"], corpus_version=version, store=str(run_dir.parent),
            label=provenance.get("label", "run"),
        )


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
    artifacts: dict[str, str]         # "slug": the cell's directory, relative to the run directory
    started_utc: str
    duration_s: float
    error: str | None = None
    attempt: int = 1
    session: int = 1

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _headline(values: dict[str, Any]) -> str:
    """The one number a person watching the log wants for this kind of cell."""
    for name in ("ttfa_ms", "cancel_to_last_chunk_ms", "duration_delta_ms", "ttfa_max_ms"):
        if values.get(name) is not None:
            return f"{name}={values[name]}"
    return ""


def _is_exclusion(void: str | None) -> bool:
    return bool(void) and void.startswith("configuration not supported")


class Runner:
    def __init__(self, spec: RunSpec, api_key: str, resume: str | Path | None = None,
                 allow_harness_change: bool = False, echo: bool = False) -> None:
        self.spec = spec
        self.entry = PROVIDERS[spec.provider]
        self.api_key = api_key
        self.echo = echo
        self.allow_harness_change = allow_harness_change
        model = spec.model or self.entry.default_model
        self.config = TTSConfig(
            model=model,
            voice=spec.voice or self.entry.voice_for(model),
            sample_rate=spec.sample_rate,
            options=tuple(spec.options),
        )
        self.planned = self.plan()
        self.cells: list[Cell] = []                    # the cells run in this session
        self.harness = prov.harness_state()
        self.environment = prov.environment()
        self.resuming = resume is not None
        if resume is not None:
            self.root = Path(resume)
            self.started = datetime.fromisoformat(json.loads((self.root / "provenance.json").read_text())["started_utc"])
        else:
            self.started = datetime.now(timezone.utc)
            base = Path(spec.store) if spec.store else store.default_store()
            # A hosted model's id can carry an owner prefix ("owner/model"); a run id is one path segment.
            model = re.sub(r"[^A-Za-z0-9._-]+", "_", self.config.model)
            self.root = base / f"{self.started.strftime('%Y%m%dT%H%M%SZ')}-{spec.provider}-{model}-{spec.label}"
        self.session = 1
        self._latest: dict[str, dict[str, Any]] = {}
        self._attempts: dict[str, int] = {}
        self._todo: list[PlannedCell] = []
        self._cells_file = None
        self.log: store.RunLog | None = None
        self._current: str | None = None

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

    def fingerprint(self) -> str:
        """Everything that decides what a cell measures. A resume must match it exactly."""
        payload = {
            "methodology": METHODOLOGY_VERSION, "provider": self.spec.provider, "adapter": self.entry.adapter.name,
            "config": self.config.as_json(), "corpus_version": self.spec.corpus_version,
            "cells": [[c.cell_id, c.probe.params(), c.item.text, c.item.spoken_reference] for c in self.planned],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def pending(self) -> list[PlannedCell]:
        """Cells with no usable record: never run, or void for a reason other than a declared exclusion.

        A fail verdict is a measurement and is kept. A provider refusal or a
        harness error measured nothing, so it runs again; the earlier attempt
        stays on disk under ``superseded/``.
        """
        out = []
        for planned in self.planned:
            row = self._latest.get(planned.cell_id)
            if row is None or (row.get("void") and not _is_exclusion(row["void"])):
                out.append(planned)
        return out

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
            "label": self.spec.label,
            "fingerprint": self.fingerprint(),
            "probes": [{"name": p.name, "variant": p.slug, "params": p.params(), "needs": list(p.needs)} for p in self.spec.probes],
            "started_utc": self.started.isoformat(),
            "harness": self.harness,
            "environment": self.environment,
        }

    def _patch_sha256(self) -> str | None:
        patch = prov.harness_patch() if self.harness.get("dirty") else ""
        return hashlib.sha256(patch.encode()).hexdigest() if patch else None

    def open_run(self) -> None:
        if self.resuming:
            self._check_resume()
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            prov.write_json(self.root / "provenance.json", self._provenance())
            prov.write_json(self.root / "plan.json", {"run_id": self.root.name, "fingerprint": self.fingerprint(),
                                                      "cells": [c.as_json() for c in self.planned]})
            prov.write_json(self.root / "corpus.json", corpus_mod.as_json(self.spec.items, self.spec.corpus_version))
            if self.harness.get("dirty"):
                patch = prov.harness_patch()
                if patch:
                    (self.root / "harness.patch").write_text(patch)
        self.session = len(store.read_jsonl(self.root / "sessions.jsonl")) + 1
        for row in store.read_jsonl(self.root / "cells.jsonl"):
            slug = row["artifacts"]["slug"]
            self._latest[slug] = row
            self._attempts[slug] = self._attempts.get(slug, 0) + 1
        self._todo = self.pending()
        kept = len(self.planned) - len(self._todo)
        with open(self.root / "sessions.jsonl", "a", encoding="utf-8") as handle:
            store.append_line(handle, {
                "session": self.session, "started_utc": datetime.now(timezone.utc).isoformat(),
                "resumed": self.resuming, "harness": self.harness, "patch_sha256": self._patch_sha256(),
                "harness_change_allowed": self.allow_harness_change,
                "planned": len(self.planned), "kept": kept, "to_run": len(self._todo), "pid": os.getpid(),
            })
        shutil.rmtree(self.root / store.PARTIAL, ignore_errors=True)     # a cell killed mid-write measured nothing
        self._cells_file = open(self.root / "cells.jsonl", "a", encoding="utf-8")
        self.log = store.RunLog(self.root, planned=len(self.planned), echo=self.echo)
        self.log.done = kept
        commit = self.harness.get("commit", "")[:9] + (" dirty" if self.harness.get("dirty") else "")
        self.log.event("resume" if self.resuming else "run_start",
                       f"{self.root.name} session {self.session}: {len(self._todo)} to run, {kept} kept of "
                       f"{len(self.planned)} · harness {commit}",
                       session=self.session, to_run=len(self._todo), kept=kept, planned=len(self.planned), harness=self.harness)

    def _check_resume(self) -> None:
        plan = json.loads((self.root / "plan.json").read_text())
        if plan.get("fingerprint") != self.fingerprint():
            raise ResumeRefused(f"{self.root.name}: the plan on disk does not match this configuration")
        first = store.read_jsonl(self.root / "sessions.jsonl")[:1]
        started = first[0] if first else {"harness": json.loads((self.root / "provenance.json").read_text())["harness"]}
        same = (started["harness"].get("commit") == self.harness.get("commit")
                and started.get("patch_sha256") == self._patch_sha256())
        if not same and not self.allow_harness_change:
            was, now = started["harness"].get("commit", ""), self.harness.get("commit", "")
            change = (f"harness {was[:9]}, now {now[:9]}" if was != now
                      else f"harness {now[:9]} with different uncommitted changes")
            raise ResumeRefused(f"{self.root.name}: started at {change}; a run measured by two versions of the code "
                                "is two runs (allow_harness_change overrides, and is recorded)")

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            if self.log:
                self.log.heartbeat(self._current)

    async def run(self, on_cell: Callable[[Cell], None] | None = None) -> Path:
        self.open_run()
        beat = asyncio.create_task(self._heartbeat())
        completed = False
        try:
            for planned in self._todo:
                cell = await self.run_cell(planned)
                if on_cell:
                    on_cell(cell)
                if self.spec.wait_between_cells_s:
                    await asyncio.sleep(self.spec.wait_between_cells_s)
            completed = True
        except BaseException as exc:
            if self.log:
                self.log.event("interrupted", f"stopped in {self._current}: {type(exc).__name__}",
                               cell_id=self._current, error_class=type(exc).__name__)
            raise
        finally:
            beat.cancel()
            self._write_summary()
            if self._cells_file:
                self._cells_file.close()
            if self.log:
                if completed:
                    c = self.log.counts()
                    self.log.event("run_end", f"{self.root.name} session {self.session} done: {c['done']}/{c['planned']} "
                                   f"cells, voids {c['voids']}, errors {c['errors']}", **c)
                self.log.close()
            store.write_manifest(self.root)          # last, so it covers the logs' final lines
        return self.root

    def _make_adapter(self, log: EventLog, clock: Clock) -> TTSAdapter:
        return self.entry.adapter(self.config, self.api_key, log, clock)

    async def run_cell(self, planned: PlannedCell) -> Cell:
        probe, item = planned.probe, planned.item
        slug = planned.cell_id
        self._current = slug
        attempt = self._attempts.get(slug, 0) + 1
        directory = self.root / store.PARTIAL / slug
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        if self.log:
            self.log.event("cell_start", f"{slug} attempt {attempt}", cell_id=slug, attempt=attempt)
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
            artifacts={"slug": slug},
            started_utc=started_utc, duration_s=round(clock.now(), 3),
            error=None if failure is None else f"{failure['type']}: {failure['message']}",
            attempt=attempt, session=self.session,
        )
        prov.write_json(directory / "cell.json", {
            "schema": "tts-bench/cell/1",
            "methodology_version": METHODOLOGY_VERSION,
            "run_id": self.root.name,
            "cell_id": slug,
            "attempt": attempt,
            "session": self.session,
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
        final = self._land(directory, slug, attempt)
        self.cells.append(cell)
        self._latest[slug] = cell.as_json()
        self._attempts[slug] = attempt
        if self._cells_file:
            store.append_line(self._cells_file, cell.as_json())
        if self.log:
            excluded_cell = _is_exclusion(cell.void)
            status = "excluded" if excluded_cell else ("void" if cell.void else (cell.verdict or "measured"))
            self.log.cell_finished(slug, status, cell.duration_s, None if excluded_cell else cell.void,
                                   failure, store.raw_tail(final / "raw.jsonl") if failure else None, _headline(cell.values))
        self._write_summary()
        self._current = None
        return cell

    def _land(self, partial: Path, slug: str, attempt: int) -> Path:
        """Move a finished cell into place; an earlier attempt moves aside and is never overwritten."""
        final = self.root / slug
        if final.exists():
            aside = self.root / store.SUPERSEDED / slug / f"attempt-{attempt - 1}"
            aside.parent.mkdir(parents=True, exist_ok=True)
            os.replace(final, aside)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, final)
        shutil.rmtree(self.root / store.PARTIAL, ignore_errors=True)    # one cell in flight at a time
        return final

    def _write_summary(self) -> None:
        cells = list(self._latest.values())
        prov.write_json(self.root / "summary.json", {
            "run_id": self.root.name, "finished": len(cells) >= len(self.planned),
            "planned": len(self.planned), "completed": len(cells), "sessions": self.session,
            "voids": sum(1 for c in cells if c.get("void")), "errors": sum(1 for c in cells if c.get("error")),
            "passed": sum(1 for c in cells if c.get("verdict") == "pass"),
            "failed": sum(1 for c in cells if c.get("verdict") == "fail"),
        })
