"""Runs probes and writes the artifacts a result is recomputable from.

A run produces a directory, not a number. Each cell carries provider, model,
configuration, corpus version, voice, harness commit and methodology version.
Published results drift from the code that produced them as soon as either can
change without the other, and a number whose method cannot be pinned down is not
a measurement. Stamping every cell is what keeps the two attached.

Everything is written as it happens rather than at the end. A campaign against a
live provider costs money, takes real time and cannot be reproduced later -- the
model behind the endpoint changes -- so a crash in the last cell must not take
the first twenty with it. The plan lands before the first connection, each cell's
record lands as that cell finishes, and the run summary is rewritten as it goes.
An interrupted run is therefore a partial result, not a lost one.

One session per cell, deliberately. Sharing a session across probes would let one
probe's conversation history change the next probe's behaviour, and the resulting
row would describe an ordering as much as a provider.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from service import events as ev
from service import provenance as prov
from service.adapters.base import AdapterError, SessionClosed, SessionConfig, ToolSpec, TurnDetection
from service.audio import write_wav
from service.caller import VOID_SLIP_MS, BranchingCaller
from service.clips import Corpus
from service.metrics import usage
from service.probes import Probe, ProbeContext, ProbeResult, ResponseLatency
from service.registry import PROVIDERS
from service.transforms import TRANSFORMS
from mock_tools.server import MockToolServer

METHODOLOGY_VERSION = "service/0.2"

# The sentinel: one fixed cell, same probe, same configuration, same voice, run
# at the head of every campaign regardless of what the campaign is about. Its
# spread across campaigns is the run-to-run noise of the instrument plus the
# provider on that day, and a ranking gap smaller than that spread is not a
# ranking. Published beside the results, never folded into them.
SENTINEL_PROBE = ResponseLatency(clip_id="open.book", name="sentinel_latency")
SENTINEL_CONFIG = TurnDetection("server_vad", silence_duration_ms=500)
SENTINEL_REPEATS = 3

# The prompt every open-loop probe runs under, and the one the sentinel always
# runs under: the sentinel is pinned to audio, this prompt and no tools whatever
# the campaign around it is measuring, or its spread would not be comparable
# across campaigns.
DEFAULT_INSTRUCTIONS = (
    "You are Riley, the receptionist at Cedar Valley Family Practice. "
    "Answer in one or two short sentences. Never mention that you are an AI."
)


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
    transforms: Sequence[str] = ("clean",)  # degradations applied to the caller audio, by name
    sentinel: bool = True                  # run the fixed sentinel cells at the head of the campaign
    corpus_root: str = "corpus/service"
    out_root: str = "data/service"
    label: str = ""


@dataclass
class Cell:
    """A published row, before aggregation."""

    provider: str
    model: str
    config: str
    voice: str
    transform: str
    repeat: int
    probe: str
    variant: str
    verdict: str | None
    void: str | None
    values: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    started_utc: str = ""
    duration_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedCell:
    """One cell the campaign intends to produce, named before it runs."""

    probe: Probe
    config: TurnDetection
    voice: str
    transform: str
    repeat: int
    sentinel: bool = False

    @property
    def cell_id(self) -> str:
        """One spelling of a cell's identity, used as its id *and* as its path.

        The plan, the directory layout and the audit all compare these strings.
        Three hand-written copies of the convention would agree only by
        coincidence, and a single divergence would report an intact run as
        entirely missing.
        """
        return f"{self.probe.slug}/{self.config.label}/{self.voice}/{self.transform}/r{self.repeat}"

    def as_json(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "probe": self.probe.name,
            "variant": self.probe.slug,
            "probe_params": prov.describe(self.probe),
            "config": self.config.label,
            "voice": self.voice,
            "transform": self.transform,
            "repeat": self.repeat,
            "sentinel": self.sentinel,
        }


def cell_id(probe: Probe, config: TurnDetection, voice: str, repeat: int, transform: str = "clean") -> str:
    return PlannedCell(probe, config, voice, transform, repeat).cell_id


class Runner:
    def __init__(self, spec: RunSpec, corpus: Corpus, api_key: str) -> None:
        self.spec = spec
        self.corpus = corpus
        self.api_key = api_key
        self.entry = PROVIDERS[spec.provider]
        self.tools = MockToolServer(spec.suite) if spec.suite else None
        self.model = spec.model or self.entry.default_model
        started = datetime.now(timezone.utc)
        self.started_utc = started.isoformat()
        self.root = Path(spec.out_root) / (
            f"{started.strftime('%Y%m%dT%H%M%SZ')}-{spec.provider}{'-' + spec.label if spec.label else ''}"
        )
        self.harness = prov.harness_state()
        self.environment = prov.environment()
        self.cells: list[Cell] = []
        self.planned = self.plan()
        self._cells_file = None

    # -- the record, before anything is measured --------------------------

    def plan(self) -> list[PlannedCell]:
        """Every cell this run intends to produce, named before it runs.

        Written up front so an interrupted campaign shows what is missing rather
        than only what survived. Without it, a directory of twelve cells and a
        directory of twelve cells from a run that meant to do twenty are
        indistinguishable. Sentinel cells come first, so even a campaign cut
        short has the day's noise floor on record.
        """
        for name in self.spec.transforms:
            if name not in TRANSFORMS:
                raise KeyError(f"unknown transform {name!r}")
        planned: list[PlannedCell] = []
        if self.spec.sentinel:
            voice = "f-us" if "f-us" in self.corpus.voices else next(iter(self.spec.voices))
            planned += [
                PlannedCell(SENTINEL_PROBE, SENTINEL_CONFIG, voice, "clean", repeat, sentinel=True)
                for repeat in range(1, SENTINEL_REPEATS + 1)
            ]
        planned += [
            PlannedCell(probe, config, voice, transform, repeat)
            for probe in self.spec.probes
            for config in self.spec.configs
            for voice in self.spec.voices
            for transform in self.spec.transforms
            for repeat in range(1, self.spec.repeats + 1)
        ]
        return planned

    def open_run(self) -> None:
        """Lay down provenance, plan and corpus manifest before the first call."""
        self.root.mkdir(parents=True, exist_ok=True)
        prov.write_json(self.root / "provenance.json", self._provenance())
        prov.write_json(self.root / "plan.json", {"run_id": self.root.name, "cells": [c.as_json() for c in self.planned]})
        prov.write_json(self.root / "manifest.json", self.corpus.manifest())
        # A commit identifies the code only when the tree is clean. When it is
        # not, the difference travels with the run instead of being lost.
        if self.harness["dirty"]:
            patch = prov.harness_patch()
            if patch:
                (self.root / "harness.patch").write_text(patch + "\n", encoding="utf-8")
        if self._cells_file is None:
            self._cells_file = open(self.root / "cells.jsonl", "a", encoding="utf-8")

    def _provenance(self) -> dict[str, Any]:
        return {
            "schema": prov.RUN_SCHEMA,
            "run_id": self.root.name,
            "methodology_version": METHODOLOGY_VERSION,
            "harness": self.harness,
            "environment": self.environment,
            "corpus_version": self.corpus.version,
            "provider": self.spec.provider,
            "adapter": self.entry.adapter.name,
            "model": self.model,
            "output_voice": self.spec.voice_name or self.entry.default_voice,
            "configs": [c.label for c in self.spec.configs],
            "caller_voices": list(self.spec.voices),
            "transforms": {name: TRANSFORMS[name].description for name in self.spec.transforms},
            "sentinel": None if not self.spec.sentinel else {
                "probe": SENTINEL_PROBE.slug, "config": SENTINEL_CONFIG.label, "repeats": SENTINEL_REPEATS,
            },
            "repeats": self.spec.repeats,
            "modality": self.spec.modality,
            "probes": [{"name": p.name, "variant": p.slug, "params": prov.describe(p)} for p in self.spec.probes],
            "tool_contract": self.spec.suite,
            "suite_label": self.spec.label,
            "discloses": list(self.entry.discloses),
            "planned_cells": len(self.planned),
            "started_utc": self.started_utc,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }

    # -- one cell ---------------------------------------------------------

    async def run_cell(self, planned: PlannedCell) -> Cell:
        probe, config, voice, repeat = planned.probe, planned.config, planned.voice, planned.repeat
        slug = planned.cell_id
        directory = self.root / slug
        directory.mkdir(parents=True, exist_ok=True)
        clock = ev.Clock()
        log = ev.EventLog(clock, directory / "events.jsonl", directory / "raw.jsonl")
        started_utc = datetime.now(timezone.utc).isoformat()

        if planned.sentinel:
            tools = None
            modality = "audio"
            session = SessionConfig(
                instructions=DEFAULT_INSTRUCTIONS, first_message=None,
                voice=self.spec.voice_name or self.entry.default_voice, tools=(),
                turn_detection=config, modality=modality, transcribe_input=True,
            )
        else:
            tools = MockToolServer(self.spec.suite) if self.spec.suite else None
            modality = self.spec.modality
            session = SessionConfig(
                instructions=self.spec.instructions or (tools.system_prompt if tools else ""),
                first_message=tools.first_message if tools else None,
                voice=self.spec.voice_name or self.entry.default_voice,
                tools=self.spec.tools or (tools.tool_specs() if tools else ()),
                turn_detection=config,
                modality=modality,
                transcribe_input=modality == "audio",
            )
        adapter = self.entry.adapter(model=self.model, api_key=self.api_key, log=log, config=session)
        result = ProbeResult(probe.name)
        caller: BranchingCaller | None = None
        failure: dict[str, Any] | None = None
        excluded = self.entry.adapter.unsupported_reason(session)
        try:
            if excluded:
                # A declared exclusion, recorded as a cell so the gap is visible
                # in the published table, and never dialled: the refusal is
                # known before the first byte.
                raise AdapterError(excluded)
            async with adapter:
                caller = BranchingCaller(adapter, log)
                # The text arm sends no audio at all, so the carrier stays parked:
                # streaming silence into a text-modality session would measure
                # nothing and could be refused outright.
                if modality == "audio":
                    caller.start()
                try:
                    result = await probe.run(
                        ProbeContext(
                            adapter=adapter, caller=caller, corpus=self.corpus,
                            voice=voice, log=log, tools=tools, transform=TRANSFORMS[planned.transform],
                        )
                    )
                finally:
                    await caller.stop()
                if caller.max_slip_ms > VOID_SLIP_MS:
                    # Our own host, not the provider. Voiding is the honest call.
                    result.void = result.void or f"caller pacing slipped {caller.max_slip_ms:.0f} ms"
        except SessionClosed as exc:
            failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            result = ProbeResult(probe.name, void=f"provider closed the session: {exc}")
        except AdapterError as exc:
            if excluded:
                # Declared, not failed: no traceback and no error count. An
                # exclusion is a fact about the provider's surface, and a
                # published table that showed it as an error would be wrong.
                result = ProbeResult(probe.name, void=f"configuration not supported: {exc}")
            else:
                failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                result = ProbeResult(probe.name, void=f"provider refused the session: {exc}")
        except Exception as exc:  # noqa: BLE001
            # The repr alone loses where it happened, and a harness bug found in
            # a run that cost an hour of provider time should not need the run
            # repeated to be diagnosed.
            failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            result = ProbeResult(probe.name, void=f"harness error: {exc!r}")
        finally:
            if adapter.agent_pcm:
                write_wav(directory / "agent.wav", bytes(adapter.agent_pcm), adapter.output_rate)
            if adapter.caller_pcm:
                write_wav(directory / "caller.wav", bytes(adapter.caller_pcm), adapter.input_rate)
            # The timestamps that make the wavs measurable. Without them the
            # artifacts show what was said but not when it became audible, and a
            # published latency could not be recomputed by anyone but us.
            prov.write_json(
                directory / "timelines.json",
                {
                    "caller": adapter.caller_timeline.as_json(),
                    "agent": adapter.agent_timeline.as_json(),
                    "wall_origin": clock.wall_origin,
                },
                indent=None,
            )
            log.close()

        ended = datetime.now(timezone.utc)
        totals = usage(adapter)
        cell = Cell(
            provider=self.spec.provider,
            model=self.model,
            config=config.label,
            voice=voice,
            transform=planned.transform,
            repeat=repeat,
            probe=probe.name,
            variant=probe.slug,
            verdict=result.verdict,
            void=result.void,
            values=result.values,
            artifacts={"dir": str(directory), "slug": slug},
            started_utc=started_utc,
            duration_s=round(clock.now(), 3),
            usage=totals,
            error=None if failure is None else f"{failure['type']}: {failure['message']}",
        )
        self._write_cell_record(
            directory, cell, planned=planned, session=session, adapter=adapter, caller=caller,
            log=log, clock=clock, tools=tools, result=result, failure=failure,
            ended_utc=ended.isoformat(),
        )
        self.cells.append(cell)
        self._append_cell(cell)
        return cell

    def _write_cell_record(
        self, directory: Path, cell: Cell, *, planned: PlannedCell, session: SessionConfig, adapter: Any,
        caller: BranchingCaller | None, log: ev.EventLog, clock: ev.Clock,
        tools: MockToolServer | None, result: ProbeResult, failure: dict[str, Any] | None,
        ended_utc: str,
    ) -> None:
        """One self-contained record per cell, written last so it can inventory the rest.

        Each object describes itself -- the adapter its session, the caller its
        utterances and pacing, the tool server its calls. Assembling those here
        field by field would mean a new piece of state reaches the published cell
        only if someone remembers to edit this function too, which is the exact
        failure this record exists to prevent.
        """
        probe = planned.probe
        state = adapter.state()
        record = {
            "schema": prov.CELL_SCHEMA,
            "methodology_version": METHODOLOGY_VERSION,
            "run_id": self.root.name,
            "cell_id": cell.artifacts["slug"],
            "identity": {
                "provider": self.spec.provider,
                "adapter": adapter.name,
                "model": cell.model,
                "config": cell.config,
                "voice": cell.voice,
                "transform": cell.transform,
                "transform_description": TRANSFORMS[cell.transform].description,
                "sentinel": planned.sentinel,
                "repeat": cell.repeat,
                "probe": cell.probe,
                "variant": cell.variant,
                "suite": self.spec.label,
                "tool_contract": self.spec.suite,
            },
            "probe_params": prov.describe(probe),
            "session": {"requested": prov.session_snapshot(session), **state["session"]},
            "audio": state["audio"],
            "transcripts": state["transcripts"],
            "corpus": {
                "version": self.corpus.version,
                "voice": cell.voice,
                "utterances": [] if caller is None else [u.as_json() for u in caller.utterances],
                "text_sent": [e.data.get("text", "") for e in log.of_kind(ev.CALLER_TEXT)],
            },
            "caller_pacing": None if caller is None else caller.pacing(),
            "tools": {
                **state["tools"],
                "served": [] if tools is None else [c.as_json() for c in tools.calls],
            },
            "usage": cell.usage,
            "counts": {
                "events": len(log.events),
                "raw_frames": log.raw_frames,
                "tool_calls": len(adapter.tool_calls),
                "responses": sum(1 for _ in log.of_kind(ev.RESPONSE_DONE)),
                "provider_errors": sum(1 for _ in log.of_kind(ev.SESSION_ERROR)),
            },
            "result": {"verdict": result.verdict, "void": result.void, "values": result.values},
            "error": failure,
            "timing": {
                "started_utc": cell.started_utc,
                "ended_utc": ended_utc,
                "duration_s": cell.duration_s,
                "wall_origin": clock.wall_origin,
            },
            "harness": self.harness,
            "environment": self.environment,
            "artifacts": prov.file_inventory(directory),
        }
        prov.write_json(directory / "cell.json", record)

    def _append_cell(self, cell: Cell) -> None:
        if self._cells_file is None:
            self.open_run()
        self._cells_file.write(json.dumps(cell.as_json()) + "\n")
        self._cells_file.flush()

    # -- the campaign -----------------------------------------------------

    async def run(self, on_cell=None) -> Path:
        self.open_run()
        try:
            for planned in self.planned:
                cell = await self.run_cell(planned)
                if on_cell:
                    on_cell(cell)
                self.write_summary(status="running")
                await asyncio.sleep(0.5)  # be a polite client, not a load test
        except BaseException as exc:  # noqa: BLE001 -- including Ctrl-C: the record still closes
            self.write_summary(status="interrupted", note=f"{type(exc).__name__}: {exc}")
            self.close()
            raise
        self.write_summary(status="complete")
        self.close()
        return self.root

    def write_summary(self, status: str = "complete", note: str | None = None) -> Path:
        """Rewritten after every cell, so an abandoned run still says what it did."""
        self.root.mkdir(parents=True, exist_ok=True)
        done = {c.artifacts["slug"] for c in self.cells}
        summary = {
            "schema": prov.RUN_SCHEMA,
            "run_id": self.root.name,
            "status": status,
            "note": note,
            "planned_cells": len(self.planned),
            "completed_cells": len(self.cells),
            "missing_cells": [p.cell_id for p in self.planned if p.cell_id not in done],
            "verdicts": self._tally("verdict"),
            "voids": sum(1 for c in self.cells if c.void),
            "errors": sum(1 for c in self.cells if c.error),
            "usage_totals": self._usage_totals(),
            "started_utc": self.started_utc,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        prov.write_json(self.root / "run.json", summary)
        return self.root

    def _tally(self, attribute: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells:
            value = getattr(cell, attribute)
            if value:
                counts[value] = counts.get(value, 0) + 1
        return counts

    def _usage_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for cell in self.cells:
            for key, value in cell.usage.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        return totals

    def close(self) -> None:
        if self._cells_file is not None:
            self._cells_file.close()
            self._cells_file = None
