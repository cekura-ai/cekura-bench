"""Re-score stored provider events; never calls a provider."""
import json
import importlib.metadata
from pathlib import Path
import re

import jiwer
import numpy as np
from whisper_normalizer.english import EnglishTextNormalizer

from .data import sha256, write_json
from .providers import reduce_events
from .streaming import pacing_metrics, read_events

NORMALIZER = EnglishTextNormalizer()
ENTITY_TYPES = {"number", "identifier", "date", "phone", "email", "amount", "spelled_sequence"}


def word_errors(reference: str, hypothesis: str) -> dict:
    ref, hyp = NORMALIZER(reference), NORMALIZER(hypothesis)
    alignment = jiwer.process_words(ref, hyp)
    return dict(substitutions=alignment.substitutions, insertions=alignment.insertions,
                deletions=alignment.deletions, reference_words=len(ref.split()),
                reference_normalized=ref, hypothesis_normalized=hyp)


def aggregate_wer(rows: list[dict]) -> dict:
    counts = {k: sum(r[k] for r in rows) for k in
              ("substitutions", "insertions", "deletions", "reference_words")}
    n = counts["reference_words"]
    return {**counts, "wer": sum(counts[k] for k in ("substitutions", "insertions", "deletions")) / n if n else None}


def entity_errors(reference: str, hypothesis: str, entities) -> dict:
    """Case-sensitive reference-entity accuracy; ignore whitespace only.

    Character alignment preserves fractions, digits embedded in unit labels,
    punctuation and case. Count each annotated reference span once.
    """
    if entities is None:
        return {"status": "not_annotated", "errors": 0, "reference_entities": 0, "details": []}
    ref_positions = [i for i, c in enumerate(reference) if not c.isspace()]
    hyp_positions = [i for i, c in enumerate(hypothesis) if not c.isspace()]
    ref = "".join(reference[i] for i in ref_positions)
    hyp = "".join(hypothesis[i] for i in hyp_positions)
    alignment = jiwer.process_characters(ref, hyp).alignments[0]
    errors, by_type, details = 0, {}, []
    previous_end = 0
    for entity in entities:
        start, end = entity["start"], entity["end"]
        if (entity["type"] not in ENTITY_TYPES or not 0 <= start < end <= len(reference)
                or start < previous_end or reference[start:end] != entity["text"]):
            raise ValueError("Invalid, overlapping or changed entity annotation")
        previous_end = end
        indices = [i for i, pos in enumerate(ref_positions) if start <= pos < end]
        if not indices:
            raise ValueError("Entity span has no non-whitespace characters")
        first, last = min(indices), max(indices) + 1
        bad, predicted_indices, edits = False, [], []
        for a in alignment:
            overlap = a.ref_start_idx < last and a.ref_end_idx > first
            inserted = a.type == "insert" and first < a.ref_start_idx < last
            # A prefix/suffix attached to the entity's raw token belongs to it.
            # Explicit spelled sequences also include adjacent inserted letters.
            if a.type == "insert" and a.ref_start_idx in (first, last):
                lo, hi = a.hyp_start_idx, a.hyp_end_idx
                if a.ref_start_idx == first:
                    attached = hi < len(hyp_positions) and lo < hi and hyp_positions[hi] == hyp_positions[hi - 1] + 1
                else:
                    attached = lo > 0 and lo < hi and hyp_positions[lo] == hyp_positions[lo - 1] + 1
                extension = hyp[lo:hi]
                # Sentence punctuation following an entity is outside its span.
                # Added digits/letters/currency or value signs can extend the entity.
                value_extension = any(c.isalnum() or c in "$€£%@+-" for c in extension)
                inserted = (attached and value_extension) or (entity["type"] == "spelled_sequence" and extension.isalnum())
            if not (overlap or inserted):
                continue
            if a.type != "equal":
                bad = True
                edits.append(a.type)
            if a.type == "equal":
                left, right = max(first, a.ref_start_idx), min(last, a.ref_end_idx)
                predicted_indices.extend(range(a.hyp_start_idx + left - a.ref_start_idx,
                                               a.hyp_start_idx + right - a.ref_start_idx))
            else:
                predicted_indices.extend(range(a.hyp_start_idx, a.hyp_end_idx))
        observed = ""
        if predicted_indices:
            observed = hypothesis[hyp_positions[min(predicted_indices)]:hyp_positions[max(predicted_indices)] + 1]
        errors += int(bad)
        item = by_type.setdefault(entity["type"], {"errors": 0, "reference_entities": 0})
        item["errors"] += int(bad)
        item["reference_entities"] += 1
        details.append({**entity, "incorrect": bad, "observed_aligned_text": observed, "edit_types": sorted(set(edits))})
    return {"status": "annotated", "errors": errors, "reference_entities": len(entities), "by_type": by_type, "details": details}


def percentiles(values: list[float]) -> dict:
    return {"n": len(values), "p50_ms": float(np.percentile(values, 50)) if values else None,
            "p90_ms": float(np.percentile(values, 90)) if values else None}


def score(run_dir: Path, out: Path, review: Path | None = None) -> dict:
    from .report import build_report
    return build_report(run_dir, out, review)


def plot(groups, mode, path):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stt-bench-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    points = [g for g in groups if any(d['wer'] is not None for d in g.get('deadlines', []))]
    for g in points:
        measured = [d for d in g['deadlines'] if d['wer'] is not None]
        ax.plot([d['deadline_ms'] for d in measured], [d['wer'] * 100 for d in measured], marker='o',
                label=f"{g['provider']} / {g['condition']}")
    if not points:
        ax.text(.5, .5, "No measured provider results", transform=ax.transAxes, ha="center")
    ax.set(xlabel="Deadline after speech end (ms)", ylabel="First-attempt corpus WER (%)",
           title=f"Streaming STT — {mode}; diagnostic, includes pacing-invalid attempts")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if points:
        ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
