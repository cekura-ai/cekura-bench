"""Benchmark, insights and reports files in the public S2S benchmark page's own shapes,
so that page's components render this benchmark."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from s2s_human_bench.bench import Entry, Run, elo, family_means, pass_all, passed, percentile, published_price, scenario_scores, skill_rates
from s2s_human_bench.calls import Call
from s2s_human_bench.config import Config, site_order

ALL = "all"


def suite_runs(runs: list[Run], prefix: str | None) -> list[Run]:
    return [{c: f for c, f in r.items() if prefix is None or c.startswith(prefix)} for r in runs]


def result_row(cfg: Config, e: Entry, suite: str, sel: list[Run], price: dict[str, Any]) -> dict[str, Any]:
    facts = [f for r in sel for f in r.values()]
    codes = sorted({c for r in sel for c in r})
    lat = [f.latency_ms for f in facts if f.latency_ms]
    entity_id = next(s.id for s in cfg.skills if s.metric == cfg["metrics"]["entity"])
    ent = [v for f in facts if (v := f.verdicts.get(entity_id)) is not None]
    per_min = price.get("costPerMinuteUsd") if price.get("costPublished") else None
    no_price = cfg["benchmark"]["noPriceNote"]
    agent_ended = sum(f.agent_ended for f in facts)
    return dict(
        suite=suite,
        model=e.model.site_id,
        runs=len(facts),
        scenarios=len(codes),
        overallPass=sum(passed(cfg, f) for f in facts),
        taskPass=sum(f.task_done for f in facts),
        toolCalls=dict(expected=len(ent), strict=sum(ent), equivalent=sum(ent), scoredRuns=len(ent), strictPassedRuns=sum(ent), equivalentPassedRuns=sum(ent)),
        infraClean=sum(f.spoke for f in facts),
        latencyP50Ms=percentile(lat, 0.5),
        latencyP90Ms=percentile(lat, 0.9),
        latencyP95Ms=percentile(lat, 0.95),
        integrity=dict(ok=sum(f.spoke for f in facts), repliesDrifting=0, turnsUnanswered=sum(f.stalled for f in facts)),
        closedBy=dict(agent=agent_ended, harness=0, callerOrTimeout=len(facts) - agent_ended),
        toolResolutions=dict(exact=0, fuzzy=0, none=0),
        costPerCallUsd=round(per_min * statistics.mean(f.seconds for f in facts) / 60, 4) if per_min else None,
        costPublished=bool(per_min),
        costNote=price.get("costNote", no_price) if per_min else no_price,
        costPerMinuteUsd=per_min,
        passAll=dict(
            overall=sum(pass_all(cfg, sel, c) for c in codes),
            task=sum(1 for c in codes if len([r for r in sel if c in r]) == len(sel) and all(r[c].task_done for r in sel if c in r)),
            toolStrict=0,
            toolEquivalent=0,
        ),
        stalls=dict(calls=len(facts), stalled=sum(f.stalled for f in facts)),
        families=family_means(cfg, facts),
    )


def defects(cfg: Config, facts: list[Call]) -> tuple[Counter, Counter]:
    """Missed skills counted by family label (plus tasks not completed) and by skill name."""
    kinds: Counter = Counter()
    names: Counter = Counter()
    for f in facts:
        for sid in f.misses:
            skill = cfg.skill(sid)
            names[skill.name] += 1
            kinds[cfg.families[skill.family]] += 1
        if not f.task_done:
            kinds[cfg["insights"]["taskNotCompleted"]] += 1
    return kinds, names


def most_missed(cfg: Config, names: Counter, n: int) -> dict[str, int]:
    """The ``n`` most-missed skills; equal counts in skill order, so ties never depend on file order."""
    order = {s.name: site_order(s.id) for s in cfg.skills}
    return dict(sorted(names.items(), key=lambda kv: (-kv[1], order[kv[0]]))[:n])


def consistency(sel: list[Run], cfg: Config) -> dict[str, int]:
    """How many scenarios passed in 0, 1, ... n of the n runs (scenarios present in every run)."""
    codes = sorted({c for r in sel for c in r})
    per = {c: [passed(cfg, r[c]) for r in sel if c in r] for c in codes}
    return dict(Counter(str(sum(v)) for v in per.values() if len(v) == len(sel)))


def by_category(cfg: Config, sel: list[Run]) -> dict[str, Any]:
    out = {}
    codes = sorted({c for r in sel for c in r})
    for cat in cfg["categories"]:
        skills = set(cat["skills"])
        scen = [c for c in codes if any(set(r[c].tested) & skills for r in sel if c in r)]
        fs = [r[c] for r in sel for c in scen if c in r]
        if fs:
            out[cat["key"]] = dict(
                runs=len(fs),
                passed=sum(passed(cfg, f) for f in fs),
                scenarios=len(scen),
                passedAll=sum(1 for c in scen if all(passed(cfg, r[c]) for r in sel if c in r)),
            )
    return out


def model_notes(
    cfg: Config, families: dict[str, float | None], rates: dict[str, float], avg: dict[str, float], task_done: float
) -> tuple[list[str], dict[str, Any]]:
    """Plain-language notes from a model's own skill rates against the field average."""
    text, n = cfg["insights"]["text"], cfg["insights"]["highlights"]
    fams = {k: v for k, v in families.items() if v is not None}
    delta = sorted(((rates[k] - avg[k], k) for k in rates), reverse=True)

    def line(d: float, k: str) -> str:
        return text["skill"].format(name=cfg.skill(k).name, rate=round(100 * rates[k]), delta=round(100 * d))

    good = [line(d, k) for d, k in delta[:n] if d > 0]
    watch = [line(d, k) for d, k in reversed(delta[-n:]) if d < 0]
    notes = []
    if fams:
        strong, weak = max(fams, key=fams.get), min(fams, key=fams.get)
        notes.append(text["families"].format(strong=strong, strong_score=round(fams[strong]), weak=weak, weak_score=round(fams[weak])))
    if good:
        notes.append(text["strength"].format(skill=good[0]))
    if watch:
        notes.append(text["gap"].format(skill=watch[0]))
    notes.append(text["task"].format(rate=task_done))
    return notes, dict(story=notes[0] if notes else "", good=good, watch=watch)


def s2s_docs(cfg: Config, entries: list[Entry], live: dict[str, Any], date: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    live_models = {m["id"]: m for m in live.get("models", [])}
    suites = [(ALL, None)] + [(s["id"], s["prefix"]) for s in cfg["suites"]]
    top = cfg["insights"]["topArguments"]
    models, results, insight_models, reports, rates = [], [], {}, {}, {}
    for e in entries:
        m = e.model
        base = dict(live_models.get(m.site_id, {}))
        base.update(m.details)
        base.update(id=m.site_id, name=m.name, short=m.name, vendor=m.vendor)
        base.setdefault("modelId", m.key)
        for k, v in cfg["benchmark"]["modelDefaults"].items():
            base.setdefault(k, v)
        base["baseline"] = m.reference
        models.append(base)
        price = published_price(live, m.site_id)
        rows, by_suite = [], {}
        for sid, prefix in suites:
            sel = suite_runs(e.runs, prefix)
            if not any(sel):
                continue
            rows.append(result_row(cfg, e, sid, sel, price))
            if sid != ALL:
                kinds, names = defects(cfg, [f for r in sel for f in r.values()])
                by_suite[sid] = dict(kinds=dict(kinds), arguments=most_missed(cfg, names, top), tools={}, consistency=consistency(sel, cfg))
        results += rows
        facts = e.calls
        kinds, names = defects(cfg, facts)
        lat = [f.latency_ms for f in facts if f.latency_ms]
        overall = rows[0]
        task_done = round(100 * sum(f.task_done for f in facts) / len(facts), 1)
        rates[m.site_id] = skill_rates(cfg, facts)
        agent_ended = sum(f.agent_ended for f in facts)
        replies = sum(f.replies for f in facts)
        insight_models[m.site_id] = dict(
            task_done=task_done,
            families=overall["families"],
            runs=len(facts),
            passed=sum(passed(cfg, f) for f in facts),
            consistency=consistency(e.runs, cfg),
            defects={"kinds": dict(kinds), "arguments": most_missed(cfg, names, top)},
            closedBy=dict(agent=agent_ended, unknown=len(facts) - agent_ended),
            bySuite=by_suite,
            byCategory=by_category(cfg, e.runs),
            replies=dict(
                count=replies,
                p50Ms=percentile(lat, 0.5),
                p90Ms=percentile(lat, 0.9),
                p99Ms=percentile(lat, 0.99),
                over5s=sum(1 for x in lat if x > 5000),
                firstP50Ms=percentile(lat, 0.5),
                laterP50Ms=percentile(lat, 0.5),
            ),
            talk=dict(
                wordsPerReply=round(sum(f.words for f in facts) / max(1, replies), 1),
                repliesPerCall=round(statistics.mean(f.replies for f in facts), 1),
                callSecondsP50=percentile([f.seconds for f in facts], 0.5),
            ),
            endings=dict(calls=len(facts), agentHungUp=agent_ended, harnessHungUp=0, callerHungUp=len(facts) - agent_ended, timedOut=0),
            stalled={r["suite"]: dict(scored=r["stalls"]["calls"], stalled=r["stalls"]["stalled"]) for r in rows},
        )
        # The run's results page: every call's link without its call id.
        first = next(iter(e.runs[0].values())).url.split("?")[0]
        reports[m.site_id] = {sid: first for sid, _ in suites}

    field = defaultdict(list)
    for r in rates.values():
        for k, v in r.items():
            field[k].append(v)
    avg = {k: statistics.mean(v) for k, v in field.items()}
    for site_id, r in rates.items():
        ins = insight_models[site_id]
        ins["notes"], ins["story"] = model_notes(cfg, ins["families"] or {}, r, avg, ins["task_done"])

    for sid, prefix in suites:
        scores = scenario_scores(entries, lambda code, p=prefix: p is None or code.startswith(p))
        ratings = elo(cfg, [e for e in entries if any(m == e.model.site_id for d in scores.values() for m in d)], scores)
        for row in results:
            if row["suite"] == sid and row["model"] in ratings:
                row["elo"] = round(ratings[row["model"]])

    bench = dict(
        schemaVersion=live.get("schemaVersion", 1),
        generatedAt=date,
        repeats=max((len(e.runs) for e in entries), default=0),
        calls=sum(len(e.calls) for e in entries),
        suites=[{k: v for k, v in s.items() if k not in ("prefix", "domain")} | {"tools": []} for s in cfg["suites"]],
        models=models,
        results=results,
    )
    family_labels = list(cfg.families.values()) + [cfg["insights"]["taskNotCompleted"]]
    insights = dict(
        schemaVersion=1,
        note=cfg["insights"]["note"],
        kinds={k: k for k in family_labels},
        models=insight_models,
        categories=[
            dict(
                key=c["key"],
                label=c["label"],
                description=c["description"],
                scenarios=max((m["byCategory"].get(c["key"], {}).get("scenarios", 0) for m in insight_models.values()), default=0),
            )
            for c in cfg["categories"]
        ],
    )
    reports_doc = dict(note=cfg["reports"]["note"], reports=reports)
    return bench, insights, reports_doc
