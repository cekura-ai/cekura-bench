"""Two checks that an authored dataset means what its script says.

1. Gold check — load the dataset directory with Sesame's own loader
   (``turnbench.data.resolve_dataset``), run ``turnbench.gold.events_for_conversation``
   on each conversation's annotation tracks and compare, event by event, with
   the gold the script planted (``ground_truth.json``). The table prints
   planted vs derived counts per conversation; any difference is listed.

2. Oracle round trip — replay the derived gold positives as a submission
   (what Sesame's ``baselines/oracle_annotator`` does; the baselines are not part
   of the ``turnbench`` package, so the few lines that build it live here) and
   score it with ``turnbench.score.score_submission``. By construction this must
   give recall 1.0 and fp_rate 0.0 on both tasks, which proves the dataset
   round-trips through the official scorer.

``turn-taking-bench author check --dataset <dir>`` runs both and exits non-zero on
any mismatch. Both read the gold columns and ``durations.json`` only
(``turn_taking_bench.data.load_local_dataset``), so a dataset of any size is checked
without its audio in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from turnbench.data import Dataset, conversation, conversation_ids
from turnbench.gold import AnchorEvent, ConversationEvents, Interval, events_for_conversation
from turnbench.score import ConversationScores, score_submission
from turnbench.submission import SCHEMA_VERSION, ConversationPrediction, SpeakerEvents, Submission

from turn_taking_bench.author.labels import Planted, compare
from turn_taking_bench.data import load_local_dataset

COUNT_KEYS = ("eot_pos", "eot_neg", "int_pos", "int_neg")


def planted_from_json(record: dict[str, Any]) -> Planted:
    planted = record["planted"]
    return Planted(
        eot_positive=[AnchorEvent(e["speaker"], e["time_s"]) for e in planted["eot_positive"]],
        eot_negative=[Interval(e["speaker"], e["start_s"], e["end_s"]) for e in planted["eot_negative"]],
        int_positive=[AnchorEvent(e["speaker"], e["time_s"]) for e in planted["int_positive"]],
        int_negative=[Interval(e["speaker"], e["start_s"], e["end_s"]) for e in planted["int_negative"]],
    )


def derived_counts(events: ConversationEvents) -> dict[str, int]:
    return {
        "eot_pos": len(events.eot_positive_events),
        "eot_neg": len(events.eot_negative_spans),
        "int_pos": len(events.int_positive_events),
        "int_neg": len(events.int_negative_spans),
    }


@dataclass
class GoldCheck:
    rows: list[dict[str, Any]] = field(default_factory=list)
    problems: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def table(self) -> str:
        header = f"{'conversation':>14} {'spec':>8} | " + " ".join(f"{k:>14}" for k in COUNT_KEYS) + " | status"
        lines = [header, "-" * len(header)]
        for row in self.rows:
            cells = " ".join(f"{row['planted'][k]:>6}/{row['derived'][k]:<7}" for k in COUNT_KEYS)
            lines.append(f"{row['conversation_id']:>14} {row['spec_id']:>8} | {cells} | {'ok' if row['ok'] else 'MISMATCH'}")
        lines.append("(cells are planted/derived)")
        for cid, problems in self.problems.items():
            lines.append(f"{cid}:")
            lines.extend(f"  {p}" for p in problems)
        return "\n".join(lines)


def gold_check(dataset_dir: Path, dataset: Dataset | None = None) -> GoldCheck:
    dataset_dir = Path(dataset_dir)
    dataset = dataset or load_local_dataset(dataset_dir)
    truth = json.loads((dataset_dir / "ground_truth.json").read_text(encoding="utf-8"))
    result = GoldCheck()
    for cid in conversation_ids(dataset):
        derived = events_for_conversation(conversation(dataset, cid))
        planted = planted_from_json(truth[cid])
        problems = compare(planted, derived)
        result.rows.append(
            {
                "conversation_id": cid,
                "spec_id": truth[cid]["spec_id"],
                "planted": planted.counts(),
                "derived": derived_counts(derived),
                "ok": not problems,
            }
        )
        if problems:
            result.problems[cid] = problems
    return result


def oracle_submission(dataset: Dataset) -> Submission:
    """The gold positives replayed as predictions (Sesame's oracle baseline)."""
    predictions = []
    for cid in conversation_ids(dataset):
        conv = conversation(dataset, cid)
        events = events_for_conversation(conv)

        def times(anchors: list[AnchorEvent], speaker: int) -> list[float]:
            return sorted(min(a.time_s, conv.duration_s) for a in anchors if a.speaker == speaker)

        predictions.append(
            ConversationPrediction(
                conversation_id=cid,
                speaker_1=SpeakerEvents(eot=times(events.eot_positive_events, 1), interruption=times(events.int_positive_events, 1)),
                speaker_2=SpeakerEvents(eot=times(events.eot_positive_events, 2), interruption=times(events.int_positive_events, 2)),
            )
        )
    return Submission(schema_version=SCHEMA_VERSION, predictions=predictions)


def oracle_scores(dataset: Dataset) -> ConversationScores:
    return score_submission(oracle_submission(dataset), dataset)


def oracle_ok(scores: ConversationScores) -> bool:
    """Recall 1.0 and fp_rate 0.0 on both tasks; a rate with no events behind it
    (NaN: a subset with no positives, or no negatives, of a task) is not a failure."""
    import math

    def fine(value: float, expected: float) -> bool:
        return math.isnan(value) or value == expected

    return all(fine(task.recall, 1.0) and fine(task.fp_rate, 0.0) for task in (scores.task_eot, scores.task_int))


def oracle_table(scores: ConversationScores) -> str:
    lines = [f"{'task':>4} {'recall':>7} {'fp_rate':>8} {'tp':>4} {'fn':>4} {'fp':>4} {'tn':>4}"]
    for name, task in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        lines.append(f"{name:>4} {task.recall:>7.3f} {task.fp_rate:>8.3f} {task.tp:>4} {task.fn:>4} {task.fp:>4} {task.tn:>4}")
    return "\n".join(lines)


def run_check(dataset_dir: Path) -> int:
    """Both checks, printed; 0 when everything matches."""
    dataset = load_local_dataset(Path(dataset_dir))
    gold = gold_check(dataset_dir, dataset)
    print("Gold check: planted (script) vs derived (turnbench.gold)")
    print(gold.table())
    scores = oracle_scores(dataset)
    print("\nOracle round trip (gold replayed through turnbench.score)")
    print(oracle_table(scores))
    ok = gold.ok and oracle_ok(scores)
    print("\nRESULT:", "ok" if ok else "FAILED")
    return 0 if ok else 1
