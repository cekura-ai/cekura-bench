"""Draft example picks: a good and a bad moment per skill, and example calls per model.

Picks are drafts until a person approves them and confirms the caller audio is human.
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


def example_calls(cfg: Config, calls: list[tuple[str, Call]], titles: dict[str, str | None]) -> dict[str, list[dict[str, Any]]]:
    """Per model and suite, calls spread over the range of outcomes (best, second, thirds, median, worst)."""
    x = cfg["examples"]
    out = {}
    for model in sorted({m for m, _ in calls}):
        mine = [c for m, c in calls if m == model and sum(_lines(c)) >= x["minExpectedLines"]]
        seen, chosen = set(), []
        for suite in cfg["suites"]:
            group = sorted((c for c in mine if c.code.startswith(suite["prefix"]) and c.code not in seen), key=lambda c: -_lines(c)[0] / sum(_lines(c)))
            n = len(group)
            idx = sorted({0, 1, n // 3, n // 2, (2 * n) // 3, n - 1}) if n >= 6 else list(range(n))
            for i in idx[: x["perSuite"]]:
                c = group[i]
                seen.add(c.code)
                met, missed = _lines(c)
                v = _verdicts(cfg, c)
                chosen.append(
                    dict(
                        scenario=c.code,
                        title=titles.get(c.code),
                        domain=suite["domain"],
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


def examples_doc(cfg: Config, entries: list[Entry], titles: dict[str, str | None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The examples file and the clip windows to cut for its moments."""
    calls = _candidates(cfg, entries)
    found, picks = moments(cfg, calls, titles)
    doc = dict(draft=cfg["examples"]["draft"], note=cfg["examples"]["note"], moments=found, examples=example_calls(cfg, calls, titles))
    return doc, picks
