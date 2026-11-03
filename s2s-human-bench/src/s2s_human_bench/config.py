"""The benchmark's settings, read from a versioned JSON file (``config/real-callers.json``).

Everything that names a model, a skill, a metric, a call type, a language group or a
threshold lives in the file; the code only knows the shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OUTPUT_KINDS = ("benchmark", "insights", "reports", "languages", "skills", "examples")


class ConfigError(SystemExit):
    """The config file is incomplete or inconsistent."""


def site_order(skill_id: str) -> list[int | str]:
    """Sort key for skill ids such as ``1.10`` (after ``1.8``) or ``3.5a`` (before ``3.5b``)."""
    return [int(x) if x.isdigit() else x for x in skill_id.replace("a", ".1").replace("b", ".2").split(".")]


@dataclass(frozen=True)
class Skill:
    id: str
    metric: int
    family: str
    name: str
    description: str
    verdict: str = "boolean"  # boolean | count (problems per call, 0 passes) | explanation (enum read from the text)
    condition: str | None = None


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    vendor: str
    site_id: str
    reference: bool = False
    runs: tuple[str, ...] | None = None  # explicit run folders; otherwise discovered from agent_id
    agent_id: int | None = None


@dataclass
class Config:
    raw: dict[str, Any]
    families: dict[str, str] = field(default_factory=dict)  # key -> label, in display order
    weights: dict[str, float] = field(default_factory=dict)
    skills: list[Skill] = field(default_factory=list)  # family order
    models: list[Model] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def scoring(self) -> dict[str, Any]:
        return self.raw["scoring"]

    @property
    def skills_by_site(self) -> list[Skill]:
        """Skills in site-id order (1.2, 1.3, ... 4.4)."""
        return sorted(self.skills, key=lambda s: site_order(s.id))

    def skill(self, skill_id: str) -> Skill:
        return next(s for s in self.skills if s.id == skill_id)


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("version", "inputs", "scoring", "metrics", "families", "models", "suites", "categories", "languages", "outputs"):
        if key not in raw:
            raise ConfigError(f"{path}: missing {key!r}")
    cfg = Config(raw=raw)
    for fam in raw["families"]:
        cfg.families[fam["key"]] = fam["label"]
        cfg.weights[fam["key"]] = fam["weight"]
        for s in fam["skills"]:
            cfg.skills.append(
                Skill(
                    id=s["id"],
                    metric=s["metric"],
                    family=fam["key"],
                    name=s["name"],
                    description=s["description"],
                    verdict=s.get("verdict", "boolean"),
                    condition=s.get("condition"),
                )
            )
    for m in raw["models"]:
        if ("runs" in m) == ("agentId" in m):
            raise ConfigError(f"model {m['key']}: give exactly one of runs or agentId")
        cfg.models.append(
            Model(
                key=m["key"],
                name=m["name"],
                vendor=m["vendor"],
                site_id=m["siteId"],
                reference=m.get("reference", False),
                runs=tuple(m["runs"]) if "runs" in m else None,
                agent_id=m.get("agentId"),
            )
        )
    ids = [s.id for s in cfg.skills]
    if len(set(ids)) != len(ids) or len({s.metric for s in cfg.skills}) != len(ids):
        raise ConfigError("skill ids and metric ids must be unique")
    for cat in raw["categories"]:
        unknown = set(cat["skills"]) - set(ids)
        if unknown:
            raise ConfigError(f"category {cat['key']}: unknown skills {sorted(unknown)}")
    if raw["languages"]["skill"] not in ids:
        raise ConfigError("languages.skill is not a configured skill")
    missing = set(OUTPUT_KINDS) - set(raw["outputs"])
    if missing:
        raise ConfigError(f"outputs: missing names for {sorted(missing)}")
    return cfg
