"""Dev / test / holdout splits of an authored dataset, assigned per template.

A split is a set of *templates* (spec ids such as ``TB-01`` or ``NB-07``), never
of conversations: every voice pairing and every degradation condition of a
template goes to the same split, so a model tuned on dev has never heard a
test conversation's script in any voice or condition.

``assign_templates`` draws the assignment: stratified by family, seeded, with
about ``holdout`` of each family's templates held out first and the rest split
``dev`` / ``1 - dev`` (per family: ``round(n * holdout)`` holdout, then
``round(remaining * dev)`` dev, the rest test; so a family of two templates
has one dev and one test and nothing held out). The result is written as
``splits.json`` (``dev`` and ``test`` ids, the seed, the fractions and the
*count* of holdout templates) and, separately, the holdout ids — the holdout
file lives in a gitignored directory so the held-out templates stay hidden
from the repository.

``split_dataset`` copies a built dataset's rows into one dataset directory per
split (parquet, durations, ground truth and a manifest each), one row at a
time, choosing by each row's ``spec_id`` in ``ground_truth.json``.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from turn_taking_bench.author.package import DatasetWriter
from turn_taking_bench.data import load_local_dataset, local_split, locate_rows, read_row

SPLITS = ("dev", "test", "holdout")
# Templates that must share a split because they share material: the minimal
# pair NB-31 (two-word acks) / NB-32 (two-word stops) uses the same agent lines
# and caller offsets, so a model tuned on one has heard the other's script.
# Each group follows the split drawn for its first member.
LINKED_TEMPLATES: tuple[tuple[str, ...], ...] = (("NB-31", "NB-32"),)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def assign_templates(
    families: dict[str, str],
    *,
    seed: int,
    holdout: float = 0.2,
    dev: float = 0.4,
    linked: tuple[tuple[str, ...], ...] = LINKED_TEMPLATES,
) -> dict[str, list[str]]:
    """``{"dev": [...], "test": [...], "holdout": [...]}`` over the template ids
    of ``families`` (template id -> family), stratified by family and seeded;
    then every ``linked`` group present is moved to its first member's split."""
    by_family: dict[str, list[str]] = defaultdict(list)
    for template, family in sorted(families.items()):
        by_family[family].append(template)
    result: dict[str, list[str]] = {name: [] for name in SPLITS}
    for family in sorted(by_family):
        members = sorted(by_family[family])
        rng = random.Random(f"{seed}:{family}")
        rng.shuffle(members)
        n_holdout = round(len(members) * holdout)
        n_dev = round((len(members) - n_holdout) * dev)
        result["holdout"] += members[:n_holdout]
        result["dev"] += members[n_holdout : n_holdout + n_dev]
        result["test"] += members[n_holdout + n_dev :]
    split_of = {template: name for name, ids in result.items() for template in ids}
    for group in linked:
        present = [template for template in group if template in split_of]
        for template in present[1:]:
            split_of[template] = split_of[present[0]]
    return {name: sorted(t for t, s in split_of.items() if s == name) for name in SPLITS}


def write_assignment(assignment: dict[str, list[str]], path: Path, holdout_path: Path, *, seed: int, holdout: float, dev: float) -> None:
    """``path``: dev and test ids plus the holdout *count*; ``holdout_path``
    (gitignored): the holdout ids."""
    public = {
        "seed": seed,
        "fractions": {"holdout": holdout, "dev": dev, "test": round(1 - dev, 6)},
        "rule": "assigned per template, stratified by family; every pairing and condition of a template is in one split",
        "linked": [list(group) for group in LINKED_TEMPLATES],
        "dev": assignment["dev"],
        "test": assignment["test"],
        "holdout_count": len(assignment["holdout"]),
        "holdout_file": str(holdout_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text(json.dumps({"seed": seed, "holdout": assignment["holdout"]}, indent=2) + "\n", encoding="utf-8")


def load_assignment(path: Path, holdout_path: Path | None = None) -> dict[str, list[str]]:
    public = json.loads(Path(path).read_text(encoding="utf-8"))
    assignment = {"dev": list(public["dev"]), "test": list(public["test"]), "holdout": []}
    holdout_path = Path(holdout_path) if holdout_path else Path(public.get("holdout_file", ""))
    if holdout_path and holdout_path.is_file():
        assignment["holdout"] = list(json.loads(holdout_path.read_text(encoding="utf-8"))["holdout"])
    return assignment


def split_dataset(src: Path, outs: dict[str, Path], assignment: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Copy ``src``'s rows into one dataset directory per split named in
    ``outs`` (``{"dev": Path, "test": Path, "holdout": Path}``); returns each
    split's manifest. Every template of ``src`` must be assigned, and every
    assigned template that is in ``src`` goes to exactly one split."""
    src = Path(src)
    truth = json.loads((src / "ground_truth.json").read_text(encoding="utf-8"))
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    split_of: dict[str, str] = {}
    for name, templates in assignment.items():
        for template in templates:
            if template in split_of:
                raise ValueError(f"template {template} assigned to both {split_of[template]} and {name}")
            split_of[template] = name
    templates_in_src = {record["spec_id"] for record in truth.values()}
    unassigned = sorted(templates_in_src - set(split_of))
    if unassigned:
        raise ValueError(f"templates in {src} with no split: {unassigned}")
    dataset = load_local_dataset(src)
    locations = locate_rows(local_split(src))
    writers = {name: DatasetWriter(path) for name, path in outs.items() if any(split_of[t] == name for t in templates_in_src)}
    counts: dict[str, int] = defaultdict(int)
    for cid in sorted(dataset.index, key=int):
        name = split_of[truth[cid]["spec_id"]]
        if name not in writers:
            continue
        row = read_row(locations[cid])
        writers[name].add(row, truth[cid], dataset.durations[cid], spec=(src_manifest.get("authored", {}).get("specs") or {}).get(cid))
        counts[name] += 1
    manifests = {}
    for name, writer in writers.items():
        manifests[name] = writer.finish(
            generated_by={"source_dataset": {"dir": str(src)}, "split": name, "templates": sorted(t for t in templates_in_src if split_of[t] == name)},
            authored={**{k: v for k, v in src_manifest.get("authored", {}).items() if k != "specs"}, "split": name},
        )
        log(f"{name}: {counts[name]} conversations -> {writer.out}")
    return manifests


def split_summary(src: Path, assignment: dict[str, list[str]]) -> dict[str, Any]:
    """Templates and conversations per split, with family counts, for the report."""
    truth = json.loads((Path(src) / "ground_truth.json").read_text(encoding="utf-8"))
    split_of = {t: name for name, ts in assignment.items() for t in ts}
    summary: dict[str, Any] = {}
    for name in SPLITS:
        records = [r for r in truth.values() if split_of.get(r["spec_id"]) == name]
        families: dict[str, int] = defaultdict(int)
        for template in {r["spec_id"]: r["family"] for r in records}.items():
            families[template[1]] += 1
        summary[name] = {"templates": len({r["spec_id"] for r in records}), "conversations": len(records), "families": dict(sorted(families.items()))}
    return summary
