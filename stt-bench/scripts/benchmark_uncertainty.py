"""Paired bootstrap of the frozen pooled ranking; private recordings stay grouped."""
import numpy as np


def add_uncertainty(data, records, *, iterations=4000, seed=42):
    if iterations < 100:
        raise ValueError('At least 100 bootstrap draws required')
    models = sorted((m for m in data['models'] if m.get('rank') is not None), key=lambda m: m['rank'])
    ids = [m['id'] for m in models]
    if not ids:
        return
    datasets = data['ranking']['datasets']
    indices = {mid: {(r['cohort'], r['id']): r['counts'] for r in records[mid]} for mid in ids}
    def unit(cohort, cid):
        counts = [indices[mid].get((cohort, cid)) for mid in ids]
        if any(c is None for c in counts):
            raise ValueError('Uncertainty requires every ranked model on every frozen item')
        words = {c['reference_words'] for c in counts}
        if len(words) != 1:
            raise ValueError('Bootstrap reference denominators differ')
        return [sum(c[k] for k in ('substitutions', 'insertions', 'deletions')) for c in counts] + [words.pop()]
    public = np.asarray([unit('pipecat', cid) for cid in datasets['pipecat']['clip_ids']], dtype=float)
    groups = {}
    for cid in datasets['private']['clip_ids']:
        # The frozen ranking uses conversation-NN-A/B recording identifiers.
        group, side = cid.rsplit('-', 1)
        if side not in ('A', 'B') or not group.startswith('conversation-'):
            raise ValueError('Private ranking item lacks an explicit conversation group')
        groups.setdefault(group, []).append(unit('private', cid))
    if len(groups) < 2:
        raise ValueError('At least two private conversations required')
    private = np.asarray([np.sum(v, axis=0) for v in groups.values()])
    pooled = public.sum(axis=0) + private.sum(axis=0)
    if pooled[-1] <= 0 or any(not np.isclose(pooled[i]/pooled[-1], m['ranking_score']['wer']) for i, m in enumerate(models)):
        raise ValueError('Bootstrap inputs do not reproduce the displayed frozen scores')
    rng = np.random.default_rng(seed)
    scores = np.empty((iterations, len(models)))
    for i in range(iterations):
        draw = public[rng.integers(len(public), size=len(public))].sum(axis=0)
        draw += private[rng.integers(len(private), size=len(private))].sum(axis=0)
        scores[i] = draw[:-1] / draw[-1]
    # Equal scores receive equal competition ranks, without arbitrary ID tie-breaking.
    ranks = 1 + (scores[:, None, :] < scores[:, :, None]).sum(axis=2)
    for i, m in enumerate(models):
        lo, hi = np.percentile(scores[:, i], [2.5, 97.5])
        rlo, rhi = np.percentile(ranks[:, i], [2.5, 97.5], method='inverted_cdf')
        m['ranking_uncertainty'] = dict(confidence=.95, lower_wer=float(lo), upper_wer=float(hi),
            lower_rank=int(rlo), upper_rank=int(rhi), first_place_frequency=float(np.mean(ranks[:, i] == 1)))
    data['ranking']['uncertainty'] = dict(method='Paired stratified bootstrap: public clips and private conversations',
        iterations=iterations, seed=seed, public_groups=len(public), private_groups=len(private),
        warning=f'Only {len(private)} private conversations. These intervals describe resampling sensitivity, not established population tiers. First-place frequency is not the probability a model is universally best.',
        public_grouping_assumption='Public clips are treated as independent; any unrecorded shared speaker or source dependence is not represented.')


def records_from_review(data):
    return {m['id']: [dict(id=c['id'], cohort=c['cohort'], counts=({k: counts[k] for k in ('substitutions', 'insertions', 'deletions', 'reference_words')}
                             if (counts := c['results'].get(m['id'], {}).get('counts')) is not None else None))
                     for c in data['clip_review']['clips']] for m in data['models']}
