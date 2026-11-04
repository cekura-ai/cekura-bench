"""Example moments (a good and a bad one per skill) and the reviewed example calls per model.

Moments are candidates: the site plays one only once its clip is cut and approved. Example
calls are published only when listed in the reviewed-calls file, after a person has listened
to them and confirmed the caller audio is human.
"""

from __future__ import annotations

import re
from typing import Any

from s2s_human_bench.bench import Entry
from s2s_human_bench.calls import MET, MISSED, Call
from s2s_human_bench.config import Config

TIMESTAMP = re.compile(r"\b(\d{1,2}):(\d{2})\b")
JUDGE_PREFIX = re.compile(r"^S2S [^:]*:\s*(pass|fail)\s*[-:]\s*", re.I)


def clean(text: str, max_chars: int) -> str:
    """The judge's explanation without its verdict prefix, cut at a sentence end near ``max_chars``."""
    text = JUDGE_PREFIX.sub("", (text or "").strip())
    out = ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if len(out) + len(s) > max_chars and out:
            break
        out += (" " if out else "") + s
    return out


def _candidates(cfg: Config, entries: list[Entry]) -> list[tuple[str, Call]]:
    """(site id, call) for every call that connected, in config, run and code order."""
    broken = re.compile(cfg["examples"]["brokenCall"], re.I)
    return [(e.model.site_id, c) for e in entries for c in e.calls if not any(broken.search(x) for x in c.outcome_lines)]


def _verdicts(cfg: Config, call: Call) -> list[tuple[str, bool, str]]:
    return [(s.id, v, call.explanations.get(s.id, "")) for s in cfg.skills_by_site if (v := call.verdicts.get(s.id)) is not None]


def moments(cfg: Config, calls: list[tuple[str, Call]], titles: dict[str, str | None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Per skill a passing and a failing moment: the shortest explanation citing at least two
    timestamps, each skill and model used once per kind."""
    x = cfg["examples"]
    out: dict[str, dict[str, Any]] = {}
    picks, used = [], set()
    for want in (True, False):
        pool = sorted(((model, c, sid, text) for model, c in calls for sid, v, text in _verdicts(cfg, c) if v == want), key=lambda t: len(t[3]))
        kind = "pass" if want else "fail"
        for model, c, sid, text in pool:
            ts = [int(m) * 60 + int(s) for m, s in TIMESTAMP.findall(text)]
            if len(ts) < x["momentMinTimestamps"] or len(text) < x["momentMinChars"] or kind in out.get(sid, {}) or (sid, model) in used:
                continue
            start = max(0, min(ts) - x["clipLeadSeconds"])
            end = min(max(ts) + x["clipTailSeconds"], start + x["clipMaxSeconds"])
            out.setdefault(sid, {})[kind] = dict(
                model=model, scenario=c.code, title=titles.get(c.code), text=clean(text, x["momentMaxChars"]), url=c.url, start_s=start, end_s=end
            )
            picks.append(dict(check=sid, kind=kind, model=model, scenario=c.code, url=c.url, call_id=c.call_id, start_s=start, end_s=end))
            used.add((sid, model))
    return out, picks


def _lines(call: Call) -> tuple[int, int]:
    return sum(x.startswith(MET) for x in call.outcome_lines), sum(x.startswith(MISSED) for x in call.outcome_lines)


def example_calls(cfg: Config, calls: list[tuple[str, Call]], titles: dict[str, str | None], reviewed: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    """Per model, the reviewed calls in the order listed. ``reviewed`` maps site id to call ids."""
    x = cfg["examples"]
    by_id = {(m, str(c.call_id)): c for m, c in calls}
    domain = {s["prefix"]: s["domain"] for s in cfg["suites"]}
    out = {}
    for model, ids in reviewed.items():
        chosen = []
        for call_id in ids:
            c = by_id.get((model, str(call_id)))
            if c is None:
                raise ValueError(f"reviewed call {call_id} is not a usable {model} call in these runs")
            met, missed = _lines(c)
            v = _verdicts(cfg, c)
            chosen.append(
                dict(
                    scenario=c.code,
                    title=titles.get(c.code),
                    domain=next(d for p, d in domain.items() if c.code.startswith(p)),
                    met=met,
                    missed=missed,
                    url=c.url,
                    conditions=[cfg.skill(sid).condition for sid, _, _ in v if cfg.skill(sid).condition][: x["maxConditions"]],
                    handled=[cfg.skill(sid).name for sid, ok, _ in v if ok][: x["maxSkills"]],
                    struggled=[cfg.skill(sid).name for sid, ok, _ in v if not ok][: x["maxSkills"]],
                )
            )
        out[model] = chosen
    return out


def examples_doc(
    cfg: Config, entries: list[Entry], titles: dict[str, str | None], reviewed: dict[str, list[str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The examples file and the clip windows to cut for its moments."""
    calls = _candidates(cfg, entries)
    found, picks = moments(cfg, calls, titles)
    doc = dict(moments=found, examples=example_calls(cfg, calls, titles, reviewed))
    return doc, picks
