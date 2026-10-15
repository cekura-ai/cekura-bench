"""Leaderboard site data from a finished full run.

The published site's data file (``turn-taking-benchmark.json``) holds editorial
content (methodology, comparisons, model insights) next to numbers. This module
refreshes the numbers from the run outputs and leaves the editorial content as
it is. It never changes a published number silently: every existing result row
must be reproduced from the run outputs first (ranks excepted, since adding a
model moves them), or nothing is written.

Inputs, per condition ``<c>``, under a full-run root:

* ``t1/<c>/SCORECARD.json``: the end-of-turn leaderboard rows (rank, scored
  calls, the calls excluded as provider failures or pacing voids);
* ``t1/<c>/<slug>/report.json``: each model's per-conversation caller counts;
* ``t2/<c>/TRACK2.json``: the cut-in and "mm-hmm" rows;
* ``datasets/t1-<c>/ground_truth.json``: each call's script, family, call kind
  and voices.

The public-set root has the same layout for the conditions the public set ran.

Added on top of the per-condition results:

* **Overall**: the end-of-turn score averaged over all conditions, its rank,
  and a 95% interval from a cluster bootstrap that resamples base calls (one
  script in one voice pairing) jointly across conditions, because every
  condition plays the same calls under different audio. Draws, seed and
  percentiles are the scorecard's.
* **Cut-in handling**: the mean of stops-on-cut-in and keeps-going-through
  "mm-hmm", averaged over conditions. It is shown beside the rank and is never
  part of it. It is indicative when any condition has under 80% of the planned
  moments scorable.
* **Breakdowns** (a separate file): per model and condition, the end-of-turn
  score by family, call kind and caller voice, over the same scored calls.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from turn_taking_bench.report import authored_groups
from turn_taking_bench.scorecard import CI_PERCENTILES, draw_matrix

LOW_COVERAGE = 0.8
BREAKDOWN_AXES = {"family": "family", "callKind": "subtype", "callerVoice": "caller_voice"}
NONE_GROUP = "(none)"
NOTES = {
    "overall.end_of_turn": (
        "overall rank: end-of-turn score averaged over the six conditions; 95% interval from a bootstrap that "
        "resamples the same calls in every condition (2,000 draws, seed 7). Ranks order point estimates only; "
        "overlapping intervals mean the difference is not established"
    ),
    "overall.cut_in_handling": (
        "mean of stops_on_wait and ignores_mmhmm, averaged over the six conditions; shown beside the rank, not part "
        "of it; indicative when fewer than 80% of planned moments were scorable in any condition"
    ),
    "breakdowns": (
        "turn-bench-breakdowns.json: per condition, end-of-turn score by scenario family, call kind and caller voice, "
        "from the same scored calls as the leaderboard; about 3 calls per call kind and condition, so treat single "
        "values as indicative; score is the mean of the defined parts (turn-ending kinds have no pauses)"
    ),
}


class ExportError(SystemExit):
    """The run outputs do not reproduce the site, or are incomplete; nothing was written."""


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
    return a == b


def _rates(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pause handling, turn taking and score from pooled ``[..., (tp, fn, fp, tn)]``
    counts. The score is the scorer's overall: the mean of the parts that are defined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pause = 1 - counts[..., 2] / (counts[..., 2] + counts[..., 3])
        turn = counts[..., 0] / (counts[..., 0] + counts[..., 1])
    score = np.where(np.isnan(pause), turn, np.where(np.isnan(turn), pause, (pause + turn) / 2))
    return pause, turn, score


def _value(x: Any, digits: int | None = None) -> float | None:
    x = float(x)
    if math.isnan(x):
        return None
    return round(x, digits) if digits is not None else x


# ---- per-condition result rows ---------------------------------------------------


def eot_rows(card: dict[str, Any], condition: str, names: dict[str, str], references: set[str], private: bool) -> list[dict[str, Any]]:
    rows = []
    for r in card["rows"]:
        if r["slug"] not in names:
            continue
        c = r["caller"]
        if c["overall"]["value"] is None:
            raise ExportError(f"{condition} {r['slug']}: no end-of-turn score")
        row = {
            "model": names[r["slug"]], "slug": r["slug"], "condition": condition,
            "overall": c["overall"]["value"], "overallLo": c["overall"]["ci95"]["lo"], "overallHi": c["overall"]["ci95"]["hi"],
            "pauseHandling": c["pause_handling"]["value"], "turnTaking": c["turn_taking"]["value"],
            "latencyP50Ms": c["turn_taking"]["latency_ms"]["p50"]["value"],
            "scored": r["scored_conversations"], "total": r["total_conversations"],
            "status": "reference" if r["slug"] in references else ("ranked" if r.get("rank") else "unranked"),
            "rank": r.get("rank"),
        }
        if private:
            row["note"] = ""
            row["_p90"] = c["turn_taking"]["latency_ms"]["p90"]["value"]
        rows.append(row)
    return rows


def interruption_rows(track2: dict[str, Any], condition: str, names: dict[str, str], private: bool) -> list[dict[str, Any]]:
    rows = []
    for slug, r in track2["rows"].items():
        if slug not in names or r.get("yield_rate") is None:
            continue
        barge, bc = r["yield_n"] / r["barge_planned"], r["backchannel_n"] / r["backchannel_planned"]
        row = {"model": names[slug], "slug": slug, "condition": condition, "yieldRate": r["yield_rate"], "excessYield": r["excess_yield"]}
        if private:
            row.update(excessLo=r["excess_ci95"][0], excessHi=r["excess_ci95"][1])
        row.update(backchannelContinue=r["backchannel_continue"], stopP50Ms=r["stop_ms"]["p50"])
        if private:
            row.update(bargeScored=r["yield_n"], bargePlanned=r["barge_planned"], bcScored=r["backchannel_n"], bcPlanned=r["backchannel_planned"])
        row.update(coverageBarge=barge, coverageBc=bc)
        if private:
            row.update(lowCoverage=min(barge, bc) < LOW_COVERAGE, note="")
        rows.append(row)
    return rows


def mismatches(published: list[dict[str, Any]], derived: list[dict[str, Any]], what: str) -> list[str]:
    """Every published value the run outputs do not reproduce (rank excepted)."""
    index = {(r["slug"], r["condition"]): r for r in derived}
    problems = []
    for old in published:
        new = index.get((old["slug"], old["condition"]))
        if new is None:
            problems.append(f"{what} {old['slug']} {old['condition']}: missing from the run outputs")
            continue
        for key, value in old.items():
            if key != "rank" and not _close(value, new.get(key)):
                problems.append(f"{what} {old['slug']} {old['condition']} {key}: site {value}, runs {new.get(key)}")
    return problems


def _keep_order(published: list[dict[str, Any]], rows: list[dict[str, Any]], conditions: list[str]) -> list[dict[str, Any]]:
    position = {(r["slug"], r["condition"]): i for i, r in enumerate(published)}
    return sorted(rows, key=lambda r: (position.get((r["slug"], r["condition"]), len(position)), conditions.index(r["condition"])))


# ---- scored calls ----------------------------------------------------------------


def scored_counts(root: Path, condition: str, row: dict[str, Any]) -> dict[str, np.ndarray]:
    """The caller-block ``(tp, fn, fp, tn)`` of every call the condition's
    scorecard scored for this model: its report's calls minus the excluded ones."""
    excluded = {x["conversation_id"] for x in row.get("pacing_voided_excluded", [])}
    excluded |= {x["conversation_id"] for x in row.get("failures", [])}
    calls = _read(root / "t1" / condition / row["slug"] / "report.json")["blocks"]["caller"]["per_conversation"]
    counts = {cid: np.array([v["eot"][k] for k in ("tp", "fn", "fp", "tn")], dtype=float) for cid, v in calls.items() if cid not in excluded}
    if len(counts) != row["scored_conversations"]:
        raise ExportError(f"{condition} {row['slug']}: {len(counts)} scored calls in report.json, scorecard says {row['scored_conversations']}")
    return counts


def base_call(record: dict[str, Any]) -> tuple[str, str]:
    """A call's identity across conditions: its script and voice pairing."""
    return record["spec_id"], record["pairing"]


def overall_end_of_turn(per_condition: list[dict[str, np.ndarray]], truths: list[dict[str, Any]]) -> dict[str, Any]:
    """Score averaged over conditions, with a joint cluster-bootstrap interval."""
    bases = sorted({base_call(rec) for truth in truths for rec in truth.values()})
    position = {b: j for j, b in enumerate(bases)}
    counts = np.zeros((len(per_condition), len(bases), 4))
    for i, (calls, truth) in enumerate(zip(per_condition, truths)):
        seen = set()
        for cid, vec in calls.items():
            j = position[base_call(truth[cid])]
            if j in seen:
                raise ExportError(f"two calls of one condition share script and voice pairing {bases[j]}")
            seen.add(j)
            counts[i, j] = vec
    point = _rates(counts.sum(axis=1))[2]
    weights = np.stack([np.bincount(row, minlength=len(bases)) for row in draw_matrix(len(bases))])
    draws = _rates(np.einsum("dj,cjk->dck", weights, counts))[2].mean(axis=1)
    lo, hi = np.percentile(draws[np.isfinite(draws)], CI_PERCENTILES)
    return {"score": float(point.mean()), "ci": [float(lo), float(hi)], "perCondition": point}


def breakdown(calls: dict[str, np.ndarray], groups: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    out = {}
    for key, axis in BREAKDOWN_AXES.items():
        out[key] = {}
        for name, members in groups.get(axis, {}).items():
            if name == NONE_GROUP:
                continue
            ids = [cid for cid in members if cid in calls]
            pause, turn, score = _rates(sum(calls[cid] for cid in ids)) if ids else (np.nan,) * 3
            out[key][name] = {"score": _value(score, 4), "pauseHandling": _value(pause, 4), "turnTaking": _value(turn, 4),
                              "scored": len(ids), "total": len(members)}
    return out


# ---- export ----------------------------------------------------------------------


def export(site: dict[str, Any], runs: Path, public_runs: Path, new_models: dict[str, Any] | None = None,
           aliases: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The refreshed site data and the breakdowns document. ``new_models`` maps
    a slug to ``{name, vendor, insights, setup}`` for models not yet on the
    site; ``aliases`` maps a site model name to its slug where the site's model
    block and result rows name it differently."""
    new_models = new_models or {}
    conditions = [c["id"] for c in site["benchmark"]["conditions"]]
    public_conditions = site["benchmark"]["sets"]["public"]["conditionsRun"]
    results = site["results"]
    names = {r["slug"]: r["model"] for r in results["private"]["endOfTurn"]}
    names.update({slug: meta["name"] for slug, meta in new_models.items()})
    references = {r["slug"] for r in results["private"]["endOfTurn"] if r["status"] == "reference"}
    slug_of = {name: slug for slug, name in names.items()}
    slug_of.update(aliases or {})
    unknown = [name for name in site["models"] if name not in slug_of]
    if unknown:
        raise ExportError(f"no result rows for site models {unknown}; pass --alias NAME=SLUG")

    cards = {c: _read(runs / "t1" / c / "SCORECARD.json") for c in conditions}
    private_eot = [r for c in conditions for r in eot_rows(cards[c], c, names, references, True)]
    private_int = [r for c in conditions for r in interruption_rows(_read(runs / "t2" / c / "TRACK2.json"), c, names, True)]
    public_eot = [r for c in public_conditions for r in eot_rows(_read(public_runs / "t1" / c / "SCORECARD.json"), c, names, references, False)]
    public_int = [r for c in public_conditions for r in interruption_rows(_read(public_runs / "t2" / c / "TRACK2.json"), c, names, False)]
    problems = (mismatches(results["private"]["endOfTurn"], private_eot, "private end of turn")
                + mismatches(results["private"]["interruptions"], private_int, "private interruptions")
                + mismatches(results["public"]["endOfTurn"], public_eot, "public end of turn")
                + mismatches(results["public"]["interruptions"], public_int, "public interruptions"))
    if problems:
        raise ExportError("existing site values not reproduced; nothing written:\n  " + "\n  ".join(problems))

    by_slug: dict[str, dict[str, dict[str, Any]]] = {}
    for r in private_eot:
        by_slug.setdefault(r["slug"], {}).setdefault("eot", {})[r["condition"]] = r
    for r in private_int:
        by_slug.setdefault(r["slug"], {}).setdefault("int", {})[r["condition"]] = r
    for name, block in site["models"].items():
        for c, entry in block["endOfTurn"].items():
            entry["rank"] = by_slug[slug_of[name]]["eot"][c]["rank"]
    for slug, meta in new_models.items():
        site["models"][meta["name"]] = _new_model_block(meta, by_slug[slug]["eot"], by_slug[slug].get("int", {}), conditions)

    truths = [_read(runs / "datasets" / f"t1-{c}" / "ground_truth.json") for c in conditions]
    groups = [authored_groups(runs / "datasets" / f"t1-{c}") for c in conditions]
    rows_by = {c: {r["slug"]: r for r in cards[c]["rows"]} for c in conditions}
    breakdowns, overall = {}, {}
    for name, block in site["models"].items():
        slug = slug_of[name]
        calls = [scored_counts(runs, c, rows_by[c][slug]) for c in conditions]
        result = overall_end_of_turn(calls, truths)
        for c, value in zip(conditions, result.pop("perCondition")):
            if not _close(float(value), block["endOfTurn"][c]["score"]):
                raise ExportError(f"{name} {c}: scored calls give {value}, site has {block['endOfTurn'][c]['score']}")
        breakdowns[name] = {c: breakdown(calls[i], groups[i]) for i, c in enumerate(conditions)}
        if block["status"] == "reference":
            continue
        cut = block["interruptions"]
        low = [c for c in conditions if cut[c]["lowCoverage"]]
        overall[name] = {
            "endOfTurn": result,
            "cutInHandling": {"score": float(np.mean([(cut[c]["stopsOnWait"] + cut[c]["ignoresMmhmm"]) / 2 for c in conditions])),
                              "indicative": bool(low), "indicativeConditions": low},
        }
    order = sorted(overall, key=lambda n: -overall[n]["endOfTurn"]["score"])
    for rank, name in enumerate(order, 1):
        overall[name]["endOfTurn"]["rank"] = rank
        site["models"][name]["overall"] = overall[name]
    results["private"]["leaderboard"] = [
        {"rank": rank, "model": name, "slug": slug_of[name], "endOfTurn": overall[name]["endOfTurn"]["score"],
         "endOfTurnLo": overall[name]["endOfTurn"]["ci"][0], "endOfTurnHi": overall[name]["endOfTurn"]["ci"][1],
         "cutInHandling": overall[name]["cutInHandling"]["score"], "cutInIndicative": overall[name]["cutInHandling"]["indicative"]}
        for rank, name in enumerate(order, 1)
    ]

    for r in private_eot:
        r.pop("_p90", None)
    results["private"]["endOfTurn"] = _keep_order(results["private"]["endOfTurn"], private_eot, conditions)
    results["public"]["endOfTurn"] = _keep_order(results["public"]["endOfTurn"], public_eot, conditions)
    results["private"]["interruptions"] = _keep_order(results["private"]["interruptions"], private_int, conditions)
    results["public"]["interruptions"] = _keep_order(results["public"]["interruptions"], public_int, conditions)
    site["benchmark"]["models"] = sum(1 for m in site["models"].values() if m["status"] != "reference")
    site["modelNotes"].update(NOTES)
    document = {
        "note": NOTES["breakdowns"].split(": ", 1)[1],
        "conditions": conditions,
        "groups": {key: [g for g in groups[0].get(axis, {}) if g != NONE_GROUP] for key, axis in BREAKDOWN_AXES.items()},
        "models": breakdowns,
    }
    return site, document


def _new_model_block(meta: dict[str, Any], eot: dict[str, dict[str, Any]], cut: dict[str, dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    clean = eot["clean"]["overall"]
    drops = {c: round((clean - eot[c]["overall"]) * 100) for c in conditions if c != "clean"}
    worst = max(drops, key=drops.get)
    gaps = [f"{c}: cut-in results indicative (fewer than 80% of planned moments scorable)" for c in conditions if c in cut and cut[c]["lowCoverage"]]
    gaps += [f"{c}: end of turn unranked ({eot[c]['scored']} of {eot[c]['total']} calls scored)" for c in conditions if eot[c]["status"] != "ranked"]
    return {
        "vendor": meta["vendor"],
        "status": "ranked" if all(eot[c]["status"] == "ranked" for c in conditions) else "unranked",
        "insights": meta["insights"], "setup": meta["setup"],
        "endOfTurn": {c: {"score": eot[c]["overall"], "ci": [eot[c]["overallLo"], eot[c]["overallHi"]],
                          "pauseHandling": eot[c]["pauseHandling"], "turnTaking": eot[c]["turnTaking"],
                          "replyP50Ms": eot[c]["latencyP50Ms"], "replyP90Ms": eot[c]["_p90"],
                          "scored": eot[c]["scored"], "total": eot[c]["total"], "rank": eot[c]["rank"]} for c in conditions},
        "interruptions": {c: {"stopsOnWait": cut[c]["yieldRate"], "ignoresMmhmm": cut[c]["backchannelContinue"],
                              "excessYield": cut[c]["excessYield"], "stopP50Ms": cut[c]["stopP50Ms"],
                              "coverageBarge": cut[c]["coverageBarge"], "coverageBc": cut[c]["coverageBc"],
                              "lowCoverage": cut[c]["lowCoverage"]} for c in conditions if c in cut},
        "stability": {"maxDropPts": max(drops[worst], 0), "worstCondition": worst},
        "gaps": gaps,
    }


def _compact(document: dict[str, Any]) -> str:
    """JSON with each breakdown cell on one line, so the file stays reviewable."""
    text = json.dumps(document, indent=2, ensure_ascii=False)
    return re.sub(r'\{\n\s+"score"[^{}]*?\n\s+\}', lambda m: re.sub(r"\s*\n\s*", " ", m.group(0)).replace("{ ", "{").replace(" }", "}"), text) + "\n"


def run_site_export(args: Any) -> int:
    site = _read(args.site)
    new_models = _read(args.add_models) if args.add_models else None
    aliases = {}
    for pair in args.alias:
        name, sep, slug = pair.rpartition("=")
        if not sep:
            raise SystemExit(f"--alias expects NAME=SLUG, got {pair!r}")
        aliases[name] = slug
    site, document = export(site, args.runs, args.public_runs, new_models, aliases)
    args.out.write_text(json.dumps(site, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.breakdowns_out.write_text(_compact(document), encoding="utf-8")
    for row in site["results"]["private"]["leaderboard"]:
        flag = " (indicative)" if row["cutInIndicative"] else ""
        print(f"{row['rank']:>2}  {row['endOfTurn']:.3f} [{row['endOfTurnLo']:.3f}, {row['endOfTurnHi']:.3f}]  "
              f"cut-in {row['cutInHandling']:.2f}{flag}  {row['model']}")
    print(f"wrote {args.out} and {args.breakdowns_out}")
    return 0
