"""Site-shaped numbers for a collected NVIDIA batch run, recomputed from its attempt journal.

Public clips resample as clips, private recordings and turns as conversations, with the
website's 95% range method (10,000 percentile-bootstrap draws, seed 42) over
summed errors and summed reference words. No provider is contacted.

Words are counted the way the published rows count them: the pinned Whisper
normaliser alone, so punctuation-only tokens it leaves (the reference's
ellipses) stay in the denominator. That is the published private total of
12,561 words; the stricter `word_errors` rule gives 12,555.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import jiwer  # noqa: E402
from stt_bench.score import NORMALIZER, word_errors  # noqa: E402

ITERATIONS, SEED = 10000, 42
KEYS = ('substitutions', 'insertions', 'deletions')


def first_valid(journal):
    rows = {}
    for line in Path(journal).read_text().splitlines():
        r = json.loads(line)
        if r['valid'] and r['clip_id'] not in rows:
            rows[r['clip_id']] = r
    return rows


def site_errors(reference, hypothesis):
    ref, hyp = NORMALIZER(reference), NORMALIZER(hypothesis)
    a = jiwer.process_words(ref, hyp)
    return dict(substitutions=a.substitutions, insertions=a.insertions, deletions=a.deletions, reference_words=len(ref.split()))


def ci95(units, rng):
    units = np.asarray(units, dtype=float)
    draws = units[rng.integers(len(units), size=(ITERATIONS, len(units)))].sum(axis=1)
    rates = draws[:, 0] / draws[:, 1] * 100
    return [round(float(x), 3) for x in np.percentile(rates, [2.5, 97.5])]


def report(run, plan_items, references, conversations=None):
    rows = first_valid(run / 'attempts.jsonl')
    out = {}
    for cohort in ('public', 'private', 'turns'):
        ids = [i['clip_id'] for i in plan_items if i['cohort'] == cohort]
        if not ids:
            continue
        # Rescore from the saved hypothesis so the journal's own counts are checked, not trusted.
        counts = {}
        for cid in ids:
            if cid in rows:
                e = word_errors(references[cid], rows[cid]['hypothesis'])
                assert all(e[k] == rows[cid]['word_errors'][k] for k in (*KEYS, 'reference_words')), cid
                counts[cid] = site_errors(references[cid], rows[cid]['hypothesis'])
        totals = {k: sum(c[k] for c in counts.values()) for k in (*KEYS, 'reference_words')}
        # Private recordings and turns resample by conversation, public clips one by one.
        group = {'private': lambda cid: cid.rsplit('-', 1)[0],
                 'turns': lambda cid: conversations[cid]}.get(cohort, lambda cid: cid)
        grouped = {}
        for cid, c in counts.items():
            g = grouped.setdefault(group(cid), [0, 0])
            g[0] += sum(c[k] for k in KEYS)
            g[1] += c['reference_words']
        rng = np.random.default_rng(SEED)
        trips = sorted(rows[c]['round_trip_ms'] for c in counts)
        out[cohort] = dict(planned=len(ids), usable=len(counts), errors=totals,
                           wer=round(sum(totals[k] for k in KEYS) / totals['reference_words'] * 100, 4),
                           wer_ci95=ci95(list(grouped.values()), rng), groups=len(grouped),
                           round_trip_ms=dict(p50=trips[len(trips) // 2], p90=trips[int(len(trips) * .9)]))
    if {'public', 'private'} <= out.keys():
        both = {k: out['public']['errors'][k] + out['private']['errors'][k] for k in (*KEYS, 'reference_words')}
        out['overall'] = dict(included=out['public']['usable'] + out['private']['usable'], planned=len(plan_items),
                              errors=both, wer=round(sum(both[k] for k in KEYS) / both['reference_words'] * 100, 4))
    return out


def main(run_dir):
    run = Path(run_dir)
    plan = json.loads((run / 'plan.json').read_text())
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    import os
    os.chdir(root)
    from stt_bench.nvidia_batch import TURNS, items
    references = {i['clip_id']: i['reference'] for i in items(plan['scope'])}
    conversations = {c['clip_id']: c['conversation_id'] for c in json.loads((TURNS / 'manifest.json').read_text())['clips']}
    result = report(run, plan['items'], references, conversations)
    (run / 'site-numbers.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
