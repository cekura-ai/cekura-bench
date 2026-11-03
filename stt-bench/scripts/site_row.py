"""Builds one model's website row from a finished full run and turn run. No provider calls.

The website's figures use the first published scoring: the Whisper English normalizer
and jiwer, with no punctuation-token postprocessing, so they differ slightly from this
bench's own reports. An ellipsis character in a transcript is read as a space.
Confidence intervals are a 10,000-draw bootstrap with seed 42: public clips resampled
one by one in clip order, private recordings resampled by conversation.
"""
import argparse
import json
from pathlib import Path
import re
import tarfile

import jiwer
import numpy as np

from stt_bench.score import NORMALIZER

ITERATIONS = 10_000
SEED = 42


def counts(reference, hypothesis):
    ref, hyp = NORMALIZER(reference), NORMALIZER((hypothesis or '').replace('…', ' '))
    if not ref.split():
        return dict(S=0, I=len(hyp.split()), D=0, N=0)
    a = jiwer.process_words(ref, hyp)
    return dict(S=a.substitutions, I=a.insertions, D=a.deletions, N=len(ref.split()))


def full_rows(root, model):
    with tarfile.open(root / 'input.tar.gz') as t:
        plan = json.load(t.extractfile('full-input/plan.json'))
    refs = {i['clip_id']: i['reference'] for i in plan['items']}
    rows = []
    for it in json.loads((root / 'results.json').read_text())['models'][model]['items']:
        att = next(a for a in it['attempts'] if int(a['attempt']) == int(it['selected_attempt']))
        rows.append(dict(clip=it['clip_id'], cohort=it['cohort'], **counts(refs[it['clip_id']], att['transcript'])))
    return len(plan['items']), sorted(rows, key=lambda r: r['clip'])


def latest_report(root):
    reports = sorted(root.glob('report-*/comparison.json'))
    if not reports:
        raise SystemExit(f'No turn report under {root}')
    return reports[-1].parent


def turn_rows(report, model):
    clips = json.loads((report / model / 'results.json').read_text())['clips']
    return [dict(clip=c['clip_id'], **counts(c['reference'], c['transcript'])) for c in clips if c.get('valid')]


def total(rows):
    t = {k: sum(r[k] for r in rows) for k in 'SIDN'}
    t['WER'] = 100 * (t['S'] + t['I'] + t['D']) / t['N']
    return t


def errors(t):
    return dict(deletions=t['D'], insertions=t['I'], referenceWords=t['N'], substitutions=t['S'])


def ci95(groups, seed=SEED, iterations=ITERATIONS):
    err = np.array([sum(r['S'] + r['I'] + r['D'] for r in g) for g in groups], float)
    n = np.array([sum(r['N'] for r in g) for g in groups], float)
    idx = np.random.default_rng(seed).integers(0, len(groups), size=(iterations, len(groups)))
    return [round(float(x), 3) for x in np.percentile(100 * err[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])]


def conversation(clip):
    """conversation-NN-A and conversation-NN-B are the two speakers of one conversation."""
    return re.sub(r'-[AB]$', '', clip.split('/')[-1])


def site_row(full_root, turn_root, model):
    planned, rows = full_rows(Path(full_root), model)
    public = [r for r in rows if r['cohort'] == 'public']
    private = [r for r in rows if r['cohort'] == 'private']
    conversations = {}
    for r in private:
        conversations.setdefault(conversation(r['clip']), []).append(r)
    report = latest_report(Path(turn_root))
    turns = turn_rows(report, model)
    timing = json.loads((report / 'comparison.json').read_text())['models'][model]
    pub, prv, trn = total(public), total(private), total(turns)
    attempted = timing['counts']['attempted']
    row = dict(
        publicErrors=errors(pub), publicWer=pub['WER'], publicWerCi95=ci95([[r] for r in public]),
        privateErrors=errors(prv), privateWer=prv['WER'], privateWerCi95=ci95(list(conversations.values())),
        overallWer=100 * sum(pub[k] + prv[k] for k in 'SID') / (pub['N'] + prv['N']),
        overallPlanned=planned, overallIncluded=len(rows), publicUsable=len(public), privateUsable=len(private),
        turnWer=trn['WER'], referenceWords=trn['N'], attempted=attempted, valid=len(turns),
        failed=attempted - len(turns),
        ttfsP50=round(timing['ttfs']['p50_ms']), ttfsN=timing['ttfs']['n'],
        ttftP50=round(timing['ttft']['p50_ms']), ttftN=len(turns))
    percentiles = {metric: {str(p): round(timing[metric][f'p{p}_ms']) for p in (50, 90, 95)}
                   for metric in ('ttfs', 'ttft')}
    return dict(row=row, percentiles=percentiles)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--full', required=True, help='finished full-run folder (results.json, input.tar.gz)')
    p.add_argument('--turns', required=True, help='finished turn-run folder (report-*/comparison.json)')
    p.add_argument('--model', required=True, help='model id, as in config/models')
    a = p.parse_args()
    print(json.dumps(site_row(a.full, a.turns, a.model), indent=2, sort_keys=True))
