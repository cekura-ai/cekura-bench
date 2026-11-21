"""ELO and pass³ per caller-language group.

A scenario switches language when the configured language-switch skill applies to it in
any call. Each switching scenario's language is named in the config, and the configured
lists must cover exactly the scenarios where the skill applies. ``"all"`` groups every
scenario; ``"rest"`` the ones that never switch.
"""

from __future__ import annotations

from typing import Any

from s2s_human_bench.bench import Entry, elo, pass_all, scenario_scores
from s2s_human_bench.config import Config, ConfigError


def switching_scenarios(entries: list[Entry], skill_id: str) -> set[str]:
    return {c.code for e in entries for c in e.calls if c.verdicts.get(skill_id) is not None}


def group_scenarios(cfg: Config, entries: list[Entry]) -> list[tuple[dict[str, Any], set[str]]]:
    conf = cfg["languages"]
    codes = {c.code for e in entries for c in e.calls}
    named = {g["id"]: set(g["scenarios"]) for g in conf["groups"] if isinstance(g["scenarios"], list)}
    listed = set().union(*named.values()) if named else set()
    derived = switching_scenarios(entries, conf["skill"])
    if listed != derived:
        raise ConfigError(
            f"languages: configured switching scenarios differ from where the skill applies: "
            f"unlisted {sorted(derived - listed)}, never switching {sorted(listed - derived)}"
        )
    out = []
    for g in conf["groups"]:
        members = {"all": codes, "rest": codes - listed}.get(g["scenarios"]) if isinstance(g["scenarios"], str) else named[g["id"]]
        if members is None:
            raise ConfigError(f"languages group {g['id']}: scenarios must be a list, 'all' or 'rest'")
        out.append((g, members))
    return out


def languages_doc(cfg: Config, entries: list[Entry]) -> dict[str, Any]:
    groups = []
    for g, codes in group_scenarios(cfg, entries):
        ratings = elo(cfg, entries, scenario_scores(entries, lambda code, s=codes: code in s))
        models = {}
        for e in entries:
            ok = sum(pass_all(cfg, e.runs, c) for c in codes)
            r = ratings.get(e.model.site_id)
            models[e.model.site_id] = dict(elo=round(r) if r is not None else None, passAll=round(100 * ok / len(codes), 1))
        groups.append(dict(id=g["id"], label=g["label"], note=g["note"], calls=len(codes), models=models))
    return {"groups": groups}
